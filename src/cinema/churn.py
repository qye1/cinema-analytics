# модель оттока: кто из платящих подписчиков отменит подписку в ближайшие четыре недели
# снимки делаем каждые две недели, признаки считаем только по данным до даты снимка

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CHANNELS, CHURN_HORIZON_WEEKS, HIKE_WEEK, NW, START, TREATED_REGIONS
from . import survival

TRAIN_LAST = 72        # последний снимок обучения, метки по нему доходят до недели 76
TEST_FIRST = 80        # тест начинается позже с запасом, чтобы окна меток не пересекались
LAST_SNAP = NW - CHURN_HORIZON_WEEKS

BEHAVIOR = ["hours_l2", "hours_l4", "hours_ratio", "hours_cv8", "sessions_l4", "titles_l4", "zero_weeks_l4",
            "hours_trend"]
CONTEXT = ["tenure_weeks", "weeks_on_plan", "plan_basic", "plan_premium", "cycle_pos", "hike_active", "summer",
           "activated", "dev_tv", "dev_web"] + [f"ch_{c}" for c in CHANNELS[1:]]
SUPPORT = ["tickets_90d", "tickets_tech_90d", "tickets_price_90d", "cancel_tickets_30d", "failed_pay_90d"]
ALL_FEATURES = BEHAVIOR + CONTEXT + SUPPORT


def _matrices(conn):
    maxid = int(pd.read_sql_query("SELECT MAX(user_id) AS m FROM users", conn)["m"].iloc[0])
    e = pd.read_sql_query("SELECT user_id, week_start, hours, sessions, titles FROM engagement_clean", conn)
    w = ((pd.to_datetime(e["week_start"]) - pd.Timestamp(START)).dt.days // 7).values
    mats = {}
    for col in ["hours", "sessions", "titles"]:
        m = np.full((maxid + 1, NW), np.nan, dtype=np.float32)
        m[e["user_id"].values, w] = e[col].values
        mats[col] = m
    return mats


def _episode_table(sub):
    # у каждой подписки добавляем начало эпизода, конец эпизода и причину конца
    s = sub.copy()
    g = s.groupby(["user_id", "ep"])
    s["ep_start"] = g["start_date"].transform("min")
    s["ep_end"] = g["end_date_clean"].transform("last")
    s["ep_reason"] = g["end_reason"].transform("last")
    s["n_spells_before"] = g.cumcount()
    return s


def build_snapshots(conn, snap_weeks=None, drop_offered=True):
    # у людей без просмотров в окне nanmean ругается на пустой срез, это ожидаемо, значение остается пропуском
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return _build_snapshots(conn, snap_weeks, drop_offered)


def _build_snapshots(conn, snap_weeks=None, drop_offered=True):
    sub = survival.episodes(conn)[1]
    sp = _episode_table(sub)
    sp = sp[sp["billing"] == "monthly"]
    users = pd.read_sql_query("SELECT user_id, channel, region, device, activated_3d FROM users_clean", conn)
    users = users.set_index("user_id")
    mats = _matrices(conn)
    H, S, T = mats["hours"], mats["sessions"], mats["titles"]
    tickets = pd.read_sql_query("SELECT user_id, created_date, category FROM tickets", conn)
    tickets["created_date"] = pd.to_datetime(tickets["created_date"])
    pay = pd.read_sql_query("SELECT user_id, pay_date FROM payments_clean WHERE status = 'failed'", conn)
    pay["pay_date"] = pd.to_datetime(pay["pay_date"])
    exp = pd.read_sql_query("SELECT user_id, variant, assigned_date FROM experiment_assignment", conn)
    exp["assigned_date"] = pd.to_datetime(exp["assigned_date"])
    offered = exp[exp["variant"] == "offer"].set_index("user_id")["assigned_date"]

    if snap_weeks is None:
        snap_weeks = list(range(12, LAST_SNAP + 1, 2))
    frames = []
    for s in snap_weeks:
        D = pd.Timestamp(START) + pd.Timedelta(weeks=s)
        a = sp[(sp["start_date"] <= D) & (sp["end_date_clean"].isna() | (sp["end_date_clean"] > D))].copy()
        a = a[(D - a["ep_start"]).dt.days >= 28]                      # хотя бы один полный цикл оплаты позади
        uid = a["user_id"].values
        d = pd.DataFrame({"user_id": uid, "snap_week": s})
        d["tenure_weeks"] = ((D - a["ep_start"]).dt.days / 7).values
        d["weeks_on_plan"] = ((D - a["start_date"]).dt.days / 7).values
        d["plan_basic"] = (a["plan"] == "basic").astype(int).values
        d["plan_premium"] = (a["plan"] == "premium").astype(int).values
        d["cycle_pos"] = d["tenure_weeks"].astype(int) % 4
        u = users.loc[uid]
        d["activated"] = u["activated_3d"].values
        d["dev_tv"] = (u["device"].values == "tv").astype(int)
        d["dev_web"] = (u["device"].values == "web").astype(int)
        for c in CHANNELS[1:]:
            d[f"ch_{c}"] = (u["channel"].values == c).astype(int)
        d["hike_active"] = (np.isin(u["region"].values, TREATED_REGIONS) & (s >= HIKE_WEEK)).astype(int)
        d["summer"] = int(D.month in (6, 7, 8))

        h = H[uid]
        l2 = np.nanmean(h[:, s - 2:s], axis=1)
        l4 = np.nanmean(h[:, s - 4:s], axis=1)
        p8 = np.nanmean(h[:, s - 10:s - 2], axis=1)
        d["hours_l2"], d["hours_l4"] = l2, l4
        d["hours_ratio"] = np.clip(l2 / (p8 + 0.1), 0, 3)
        w8 = h[:, s - 8:s]
        d["hours_cv8"] = np.nanstd(w8, axis=1) / (np.nanmean(w8, axis=1) + 0.1)
        d["sessions_l4"] = np.nansum(S[uid][:, s - 4:s], axis=1)
        d["titles_l4"] = np.nansum(T[uid][:, s - 4:s], axis=1)
        d["zero_weeks_l4"] = (S[uid][:, s - 4:s] == 0).sum(axis=1)
        # наклон часов за последние 8 недель: отрицательный значит интерес остывает
        x = np.arange(8) - 3.5
        hh = np.where(np.isnan(w8), np.nanmean(w8, axis=1, keepdims=True), w8)
        d["hours_trend"] = (hh * x).sum(axis=1) / (x ** 2).sum()

        for name, mask_fn in [("tickets_90d", lambda t: t["category"].notna()),
                              ("tickets_tech_90d", lambda t: t["category"] == "technical"),
                              ("tickets_price_90d", lambda t: t["category"] == "price")]:
            t = tickets[(tickets["created_date"] < D) & (tickets["created_date"] >= D - pd.Timedelta(days=90))]
            t = t[mask_fn(t)]
            d[name] = d["user_id"].map(t.groupby("user_id").size()).fillna(0).values
        t = tickets[(tickets["created_date"] < D) & (tickets["created_date"] >= D - pd.Timedelta(days=30))
                    & (tickets["category"] == "cancel_request")]
        d["cancel_tickets_30d"] = d["user_id"].map(t.groupby("user_id").size()).fillna(0).values
        p = pay[(pay["pay_date"] < D) & (pay["pay_date"] >= D - pd.Timedelta(days=90))]
        d["failed_pay_90d"] = d["user_id"].map(p.groupby("user_id").size()).fillna(0).values

        # метка: добровольная отмена в ближайшие четыре недели
        end = a["ep_end"]
        lab = ((a["ep_reason"] == "voluntary") & (end >= D) & (end < D + pd.Timedelta(weeks=CHURN_HORIZON_WEEKS)))
        d["y"] = lab.astype(int).values
        # будущие часы, только для демонстрации утечки, в честной модели их нет
        d["future_hours_l2"] = np.nanmean(h[:, s:min(s + 2, NW)], axis=1) if s + 2 <= NW else np.nan
        d["snap_date"] = D
        d["region_treated"] = np.isin(u["region"].values, TREATED_REGIONS)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # убираем тех, кому в эксперименте показывали оффер, на время вокруг показа: это вмешательство в отток
    if not drop_offered:
        return df.reset_index(drop=True)
    adate = df["user_id"].map(offered)
    near = adate.notna() & (df["snap_date"] > adate - pd.Timedelta(days=28)) & (df["snap_date"] < adate + pd.Timedelta(days=56))
    df = df[~near].reset_index(drop=True)
    return df


def temporal_split(df):
    train = df[df["snap_week"] <= TRAIN_LAST]
    test = df[df["snap_week"] >= TEST_FIRST]
    return train, test


def make_models():
    return {
        "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=0.5, max_iter=500)),
        "gbm": HistGradientBoostingClassifier(learning_rate=0.06, max_iter=250, max_depth=5, min_samples_leaf=80,
                                              l2_regularization=1.0, early_stopping=False, random_state=11),
    }


def metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "base_rate": float(np.mean(y))}


def gains_table(y, p, bins=10):
    # доля всех уходящих, которую ловим, если взять верхние q процентов по риску
    order = np.argsort(-p)
    ys = np.asarray(y)[order]
    n = len(ys)
    rows = []
    for i in range(1, bins + 1):
        k = int(round(n * i / bins))
        cap = ys[:k].sum() / ys.sum()
        prec = ys[:k].mean()
        rows.append({"top_share": i / bins, "captured": cap, "precision": prec, "lift": prec / ys.mean()})
    return pd.DataFrame(rows)


def calibration_table(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    d = pd.DataFrame({"y": np.asarray(y), "p": p, "bin": q})
    g = d.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
    return g


def cost_curve(y, p, price, saved_share, contact_cost, months_value=6, n_points=50):
    # прибыль от рассылки оффера верхним q процентам по риску
    # допущения: оффер спасает долю saved_share тех, кто ушел бы, спасенный приносит price * months_value
    order = np.argsort(-p)
    ys = np.asarray(y)[order]
    n = len(ys)
    qs = np.linspace(0.01, 1.0, n_points)
    rows = []
    for q in qs:
        k = int(round(n * q))
        caught = ys[:k].sum()
        profit = caught * saved_share * price * months_value - k * contact_cost
        rows.append({"top_share": q, "caught": int(caught), "profit": float(profit)})
    return pd.DataFrame(rows)


def feature_importance(model, test, features, n=20000, repeats=5, seed=3):
    d = test.sample(min(n, len(test)), random_state=seed)
    r = permutation_importance(model, d[features], d["y"], scoring="roc_auc", n_repeats=repeats, random_state=seed)
    out = pd.DataFrame({"feature": features, "auc_drop": r.importances_mean, "std": r.importances_std})
    return out.sort_values("auc_drop", ascending=False).reset_index(drop=True)


def ablation(train, test):
    # какие группы признаков сколько дают: ранний сигнал против поздних сигналов поддержки
    sets = {"только контекст": CONTEXT, "контекст и поведение": CONTEXT + BEHAVIOR,
            "все признаки": ALL_FEATURES}
    rows = []
    for name, cols in sets.items():
        m = make_models()["gbm"].fit(train[cols], train["y"])
        rows.append({"features": name, **metrics(test["y"], m.predict_proba(test[cols])[:, 1])})
    return pd.DataFrame(rows)


def leakage_demo(df, train, test):
    # три способа обмануть себя: случайное разбиение, признак из будущего, смесь обоих
    rows = []
    gbm = make_models()["gbm"]
    m = gbm.fit(train[ALL_FEATURES], train["y"])
    rows.append({"setup": "честно: делим по времени", "auc": roc_auc_score(test["y"], m.predict_proba(test[ALL_FEATURES])[:, 1])})
    rng = np.random.default_rng(5)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.7)
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    m = make_models()["gbm"].fit(tr[ALL_FEATURES], tr["y"])
    rows.append({"setup": "случайное разбиение: один человек в обеих частях",
                 "auc": roc_auc_score(te["y"], m.predict_proba(te[ALL_FEATURES])[:, 1])})
    leak = ALL_FEATURES + ["future_hours_l2"]
    tr2, te2 = train.dropna(subset=["future_hours_l2"]), test.dropna(subset=["future_hours_l2"])
    m = make_models()["gbm"].fit(tr2[leak], tr2["y"])
    rows.append({"setup": "признак из будущего: часы после снимка",
                 "auc": roc_auc_score(te2["y"], m.predict_proba(te2[leak])[:, 1])})
    return pd.DataFrame(rows)


def run(conn):
    df = build_snapshots(conn)
    train, test = temporal_split(df)
    models = {}
    preds = {}
    res = []
    for name, m in make_models().items():
        m.fit(train[ALL_FEATURES], train["y"])
        p = m.predict_proba(test[ALL_FEATURES])[:, 1]
        models[name], preds[name] = m, p
        res.append({"model": name, **metrics(test["y"], p)})
    # наивное правило без модели: у кого часы за две недели упали сильнее всего
    naive = -test["hours_ratio"].fillna(1.0).values
    from scipy.stats import rankdata
    naive_p = rankdata(naive) / len(naive)
    # у правила нет вероятностей, поэтому честно считаем только ранговые метрики
    rule = {"model": "правило: часы упали", "auc": float(roc_auc_score(test["y"], naive_p)),
            "pr_auc": float(average_precision_score(test["y"], naive_p)), "base_rate": float(test["y"].mean())}
    res.append(rule)
    results = pd.DataFrame(res)
    gbm_p = preds["gbm"]
    out = {
        "data": df, "train": train, "test": test, "results": results, "pred": preds,
        "gains": gains_table(test["y"], gbm_p), "gains_logreg": gains_table(test["y"], preds["logreg"]),
        "gains_rule": gains_table(test["y"], naive_p),
        "calibration": calibration_table(test["y"], gbm_p),
        "importance": feature_importance(models["gbm"], test, ALL_FEATURES),
        "ablation": ablation(train, test),
        "leakage": leakage_demo(df, train, test),
        "models": models,
    }
    return out
