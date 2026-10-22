# эксперименты и причинность: A/B тест оффера, uplift модель, difference in differences по повышению цены
# истину из симулятора здесь не используем при расчетах, только в конце сравниваем с ней

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import churn, marts
from .config import HIKE_WEEK, START, TREATED_REGIONS

COVARIATES = ["hours_ratio", "hours_l2", "tenure_weeks", "plan_basic", "plan_premium", "ch_social", "dev_tv"]
UPLIFT_FEATURES = ["hours_l2", "hours_l4", "hours_ratio", "hours_trend", "hours_cv8", "tenure_weeks", "plan_basic",
                   "plan_premium", "dev_tv", "dev_web", "ch_social", "ch_search", "ch_referral", "ch_partner",
                   "activated", "sessions_l4", "titles_l4"]


# ---------------------------------------------------------------- данные эксперимента

def load_experiment(conn):
    e = pd.read_sql_query("SELECT user_id, variant, assigned_date, hours_ratio AS flag_ratio FROM experiment_assignment", conn)
    e["assigned_date"] = pd.to_datetime(e["assigned_date"])
    e["snap_week"] = (e["assigned_date"] - pd.Timestamp(START)).dt.days // 7
    weeks = sorted(e["snap_week"].unique())
    snap = churn.build_snapshots(conn, weeks, drop_offered=False)
    d = e.merge(snap, on=["user_id", "snap_week"], how="inner")
    d["treat"] = (d["variant"] == "offer").astype(int)
    d["retained"] = 1 - d["y"]
    return d, len(e) - len(d)


# ---------------------------------------------------------------- A/B анализ

def srm_check(d):
    # проверка на перекос выборки: ждали 50 на 50
    n = d["treat"].value_counts()
    obs = np.array([n.get(0, 0), n.get(1, 0)])
    chi2 = float(((obs - obs.sum() / 2) ** 2 / (obs.sum() / 2)).sum())
    return {"control": int(obs[0]), "offer": int(obs[1]), "chi2": chi2, "p": float(stats.chi2.sf(chi2, 1))}


def balance(d, cols=COVARIATES):
    # стандартизованная разница средних, по модулю меньше 0,1 считаем нормой
    rows = []
    for c in cols:
        a, b = d.loc[d["treat"] == 1, c], d.loc[d["treat"] == 0, c]
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"covariate": c, "mean_offer": a.mean(), "mean_control": b.mean(), "smd": (a.mean() - b.mean()) / sd})
    return pd.DataFrame(rows)


def itt(d, n_boot=3000, n_perm=5000, seed=17):
    y1 = d.loc[d["treat"] == 1, "y"].values
    y0 = d.loc[d["treat"] == 0, "y"].values
    p1, p0 = y1.mean(), y0.mean()
    diff = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / len(y1) + p0 * (1 - p0) / len(y0))
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(y1, len(y1)).mean() - rng.choice(y0, len(y0)).mean() for _ in range(n_boot)])
    # перестановочный тест: перемешиваем метки группы и смотрим, как часто разница не меньше наблюдаемой
    y = d["y"].values
    t = d["treat"].values
    perm = np.empty(n_perm)
    for i in range(n_perm):
        tp = rng.permutation(t)
        perm[i] = y[tp == 1].mean() - y[tp == 0].mean()
    table = np.array([[y1.sum(), len(y1) - y1.sum()], [y0.sum(), len(y0) - y0.sum()]])
    return {"p_offer": float(p1), "p_control": float(p0), "diff": float(diff), "se": float(se),
            "ci_normal": (float(diff - 1.96 * se), float(diff + 1.96 * se)),
            "ci_boot": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
            "z_p": float(2 * stats.norm.sf(abs(diff / se))),
            "perm_p": float((np.abs(perm) >= abs(diff)).mean()),
            "fisher_p": float(stats.fisher_exact(table)[1]),
            "relative": float(diff / p0), "n_offer": len(y1), "n_control": len(y0), "boot": boots}


def adjusted_effect(d, cols=COVARIATES):
    # регрессионная поправка: линейная вероятностная модель с ковариатами и робастной ошибкой HC1
    X = d[cols].astype(float).values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Z = np.column_stack([np.ones(len(d)), d["treat"].values, X])
    y = d["y"].values.astype(float)
    beta = np.linalg.solve(Z.T @ Z, Z.T @ y)
    res = y - Z @ beta
    n, k = Z.shape
    meat = (Z * res[:, None] ** 2).T @ Z
    bread = np.linalg.inv(Z.T @ Z)
    cov = bread @ meat @ bread * n / (n - k)
    se = float(np.sqrt(cov[1, 1]))
    return {"diff": float(beta[1]), "se": se, "ci": (float(beta[1] - 1.96 * se), float(beta[1] + 1.96 * se))}


def power_table(p0, n_per_arm, effects=(0.02, 0.04, 0.06, 0.08), alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    mde = (za + zb) * np.sqrt(2 * p0 * (1 - p0) / n_per_arm)
    rows = []
    for e in effects:
        p1 = p0 - e
        n = (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / e ** 2
        rows.append({"effect_pp": 100 * e, "n_per_arm": int(np.ceil(n))})
    return float(mde), pd.DataFrame(rows)


def subgroup_effects(d, by, bins=None):
    # эффект в подгруппах, с приблизительным 95% интервалом
    g = d.copy()
    if bins is not None:
        g["grp"] = pd.qcut(g[by], bins, labels=[f"{i + 1}/{bins}" for i in range(bins)])
    else:
        g["grp"] = g[by]
    rows = []
    for k, s in g.groupby("grp", observed=True):
        a, b = s[s["treat"] == 1]["y"], s[s["treat"] == 0]["y"]
        if len(a) < 20 or len(b) < 20:
            continue
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        rows.append({"group": str(k), "n": len(s), "control_rate": b.mean(), "diff": diff,
                     "lo": diff - 1.96 * se, "hi": diff + 1.96 * se})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- uplift

def _make(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
    return HistGradientBoostingClassifier(max_depth=2, max_iter=60, learning_rate=0.06, min_samples_leaf=40,
                                          l2_regularization=2.0, early_stopping=False, random_state=3)


def oof_scores(d, features=UPLIFT_FEATURES, kind="logreg", folds=5, repeats=3, seed=21):
    # T learner с перекрестной подгонкой: модель для оффера и модель для контроля, оценка только на отложенной части
    # uplift здесь это снижение шанса ухода: чем больше, тем лучше действует оффер
    X = d[features].astype(float).fillna(d[features].median()).values
    y = d["y"].values
    t = d["treat"].values
    up = np.zeros(len(d))
    risk = np.zeros(len(d))
    for r in range(repeats):
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed + r)
        strat = t * 2 + y
        for tr, te in skf.split(X, strat):
            m1 = _make(kind).fit(X[tr][t[tr] == 1], y[tr][t[tr] == 1])
            m0 = _make(kind).fit(X[tr][t[tr] == 0], y[tr][t[tr] == 0])
            p1, p0 = m1.predict_proba(X[te])[:, 1], m0.predict_proba(X[te])[:, 1]
            up[te] += (p0 - p1) / repeats
            risk[te] += p0 / repeats          # базовый риск без оффера, им таргетируют по оттоку
    return up, risk


def qini_curve(good, treat, score, points=60):
    # накопленный прирост удержанных, если предлагать оффер по убыванию счета
    order = np.argsort(-score, kind="stable")
    g, t = np.asarray(good)[order], np.asarray(treat)[order]
    n = len(g)
    ks = np.unique(np.linspace(0, n, points).astype(int))
    q = []
    for k in ks:
        nt, nc = t[:k].sum(), (1 - t[:k]).sum()
        gt, gc = (g[:k] * t[:k]).sum(), (g[:k] * (1 - t[:k])).sum()
        q.append(gt - gc * nt / nc if nc > 0 else 0.0)
    q = np.array(q)
    return ks / n, q


def qini_area(x, q):
    # площадь между кривой и линией случайного таргетинга, в единицах удержанных людей
    rand = q[-1] * x
    return float(np.trapezoid(q - rand, x))


def qini_bootstrap(good, treat, s_a, s_b, n_boot=300, seed=5):
    # доверительный интервал для разницы площадей двух стратегий
    rng = np.random.default_rng(seed)
    n = len(good)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        xa, qa = qini_curve(good[i], treat[i], s_a[i])
        xb, qb = qini_curve(good[i], treat[i], s_b[i])
        diffs.append(qini_area(xa, qa) - qini_area(xb, qb))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def oracle_policy_curve(score, true_effect, points=40):
    # здесь нам помогает симулятор: у каждого человека известен настоящий эффект оффера на шанс ухода
    # возвращаем среднее снижение шанса ухода среди тех, кому предложили оффер, по доле охвата
    order = np.argsort(-score, kind="stable")
    eff = np.asarray(true_effect)[order]
    n = len(eff)
    ks = np.unique(np.linspace(max(1, n // 40), n, points).astype(int))
    saved = np.array([-eff[:k].sum() for k in ks])         # спасенные люди, эффект отрицательный
    return ks / n, saved


def uplift_study(d, truth_users, truth):
    d = d.merge(truth_users[["user_id", "utype"]], on="user_id", how="left")
    eff_by_type = {k: v["effect"] for k, v in truth["offer"]["by_type"].items()}
    d["true_effect"] = d["utype"].map(eff_by_type).astype(float)
    good, treat = d["retained"].values, d["treat"].values
    out = {"data": d, "scores": {}, "qini": {}, "policy": {}}
    rng = np.random.default_rng(9)
    scores = {}
    for kind in ["logreg", "gbm"]:
        up, risk = oof_scores(d, kind=kind)
        scores[f"uplift_{kind}"] = up
        if kind == "logreg":
            scores["risk"] = risk
    scores["random"] = rng.random(len(d))
    for name, sc in scores.items():
        x, q = qini_curve(good, treat, sc)
        out["qini"][name] = (x, q, qini_area(x, q))
        out["policy"][name] = oracle_policy_curve(sc, d["true_effect"].values)
    # идеальный выбор: сначала те, у кого эффект оффера самый сильный, то есть самый отрицательный
    out["policy"]["oracle"] = oracle_policy_curve(-d["true_effect"].values, d["true_effect"].values)
    out["scores"] = scores
    out["area_ci_vs_risk"] = qini_bootstrap(good, treat, scores["uplift_logreg"], scores["risk"])
    out["area_ci_vs_random"] = qini_bootstrap(good, treat, scores["uplift_logreg"], scores["random"])
    return out


# ---------------------------------------------------------------- difference in differences

def region_panel(conn):
    r = marts.region_weekly_churn(conn)
    r = r[r["region"] != "unknown"].copy()
    r["treated"] = r["region"].isin(TREATED_REGIONS)
    return r


def group_series(r, first_week=12):
    # недельный уровень ухода в группе регионов: сумма уходов к сумме активных на начало недели
    g = r[r["week_idx"] >= first_week].groupby(["week_idx", "treated"])[["active_at_start", "churned"]].sum().reset_index()
    g["rate"] = g["churned"] / g["active_at_start"]
    p = g.pivot(index="week_idx", columns="treated", values="rate").rename(columns={True: "treated", False: "control"})
    p["diff"] = p["treated"] - p["control"]
    return p


def did_estimate(p, pre=(20, HIKE_WEEK - 1), post=(HIKE_WEEK, 103)):
    pre_m = p.loc[pre[0]:pre[1]].mean()
    post_m = p.loc[post[0]:post[1]].mean()
    naive = post_m["treated"] - pre_m["treated"]
    did = (post_m["treated"] - pre_m["treated"]) - (post_m["control"] - pre_m["control"])
    return float(naive), float(did)


def did_bootstrap(p, n_boot=2000, block=4, seed=13, pre=(20, HIKE_WEEK - 1), post=(HIKE_WEEK, 103)):
    # блочный бутстреп по неделям: недели соседние похожи, поэтому берем блоки по четыре
    rng = np.random.default_rng(seed)
    d_pre = p.loc[pre[0]:pre[1], "diff"].values
    d_post = p.loc[post[0]:post[1], "diff"].values

    def resample(x):
        nb = int(np.ceil(len(x) / block))
        starts = rng.integers(0, len(x) - block + 1, nb)
        return np.concatenate([x[s:s + block] for s in starts])[:len(x)]

    est = np.array([resample(d_post).mean() - resample(d_pre).mean() for _ in range(n_boot)])
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def event_study(p, base=(20, HIKE_WEEK - 1), lo=-24, hi=42):
    # разница между группами по неделям относительно среднего до повышения
    base_mean = p.loc[base[0]:base[1], "diff"].mean()
    base_sd = p.loc[base[0]:base[1], "diff"].std()
    es = (p["diff"] - base_mean).rename("effect").reset_index()
    es["rel_week"] = es["week_idx"] - HIKE_WEEK
    es = es[(es["rel_week"] >= lo) & (es["rel_week"] <= hi)].reset_index(drop=True)
    return es, float(base_sd)


def placebo_did(p, fake_week=40):
    # фальшивая дата повышения внутри периода до него: оценка должна быть около нуля
    return did_estimate(p, pre=(20, fake_week - 1), post=(fake_week, HIKE_WEEK - 5))[1]


def hike_study(conn, truth):
    r = region_panel(conn)
    p = group_series(r)
    naive, did = did_estimate(p)
    lo, hi = did_bootstrap(p)
    es, base_sd = event_study(p)
    return {"series": p, "naive": naive, "did": did, "ci": (lo, hi), "event_study": es, "base_sd": base_sd,
            "placebo": placebo_did(p), "truth_pp": truth["hike"]["true_effect_weekly_churn_pp"]}


def run(conn, truth_users, truth):
    d, dropped = load_experiment(conn)
    res = {"data": d, "dropped": dropped, "srm": srm_check(d), "balance": balance(d), "itt": itt(d),
           "adjusted": adjusted_effect(d)}
    res["mde"], res["power"] = power_table(res["itt"]["p_control"], res["itt"]["n_control"])
    dd = d.merge(truth_users[["user_id", "utype"]], on="user_id", how="left")
    res["by_ratio"] = subgroup_effects(d, "hours_ratio", bins=3)
    res["by_plan"] = subgroup_effects(d.assign(plan=np.where(d["plan_basic"] == 1, "basic",
                                                             np.where(d["plan_premium"] == 1, "premium", "standard"))), "plan")
    res["by_type"] = subgroup_effects(dd, "utype")
    res["uplift"] = uplift_study(d, truth_users, truth)
    res["hike"] = hike_study(conn, truth)
    return res
