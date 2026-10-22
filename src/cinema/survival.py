# анализ времени жизни подписки: Каплан Майер, логранк и дискретная модель риска
# все три метода написаны руками на numpy, чтобы было видно, что внутри

import numpy as np
import pandas as pd
from scipy import stats

from . import marts
from .config import HIKE_WEEK, START, TREATED_REGIONS

OBS_END = pd.Timestamp("2025-12-29")


# ---------------------------------------------------------------- эпизоды подписки

def episodes(conn):
    # один человек, один первый период платной подписки
    # смену тарифа не считаем уходом, поэтому подряд идущие куски склеиваем
    sub = pd.read_sql_query(
        "SELECT user_id, plan, billing, start_date, end_date_clean, end_reason FROM subscriptions_clean", conn)
    sub["start_date"] = pd.to_datetime(sub["start_date"])
    sub["end_date_clean"] = pd.to_datetime(sub["end_date_clean"])
    sub = sub.sort_values(["user_id", "start_date"]).reset_index(drop=True)
    prev_end = sub.groupby("user_id")["end_date_clean"].shift()
    new_ep = prev_end.isna() | (sub["start_date"] > prev_end)
    sub["ep"] = new_ep.groupby(sub["user_id"]).cumsum()
    first = sub[sub["ep"] == 1]
    g = first.groupby("user_id")
    ep = pd.DataFrame({
        "start": g["start_date"].min(),
        "end": g["end_date_clean"].last(),
        "last_reason": g["end_reason"].last(),
        "plan_start": g["plan"].first(),
        "billing": g["billing"].first(),
    }).reset_index()
    ep["event_voluntary"] = (ep["last_reason"] == "voluntary").astype(int)
    ep["event_payment"] = (ep["last_reason"] == "payment").astype(int)
    # смена тарифа в самый последний день данных оставляет подписку живой, значит это цензура
    ended = ep["last_reason"].isin(["voluntary", "payment"])
    ep["end_obs"] = ep["end"].where(ended, OBS_END)
    ep["weeks"] = (ep["end_obs"] - ep["start"]).dt.days / 7.0

    users = pd.read_sql_query(
        "SELECT user_id, channel, region, device, activated_3d FROM users_clean", conn)
    te = marts.trial_engagement(conn)
    ep = ep.merge(users, on="user_id").merge(te[["user_id", "trial_hours"]], on="user_id", how="left")
    ep["log_trial_hours"] = np.log(ep["trial_hours"].clip(lower=0.05))
    ep["log_trial_hours"] = ep["log_trial_hours"].fillna(ep["log_trial_hours"].median())
    ep["treated_region"] = ep["region"].isin(TREATED_REGIONS)
    return ep[ep["weeks"] > 0].reset_index(drop=True), sub


# ---------------------------------------------------------------- Каплан Майер

def kaplan_meier(t, e, alpha=0.05):
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    times = np.unique(t[e == 1])
    ts = np.sort(t)
    n_risk = len(t) - np.searchsorted(ts, times, side="left")
    d = np.bincount(np.searchsorted(times, t[e == 1]), minlength=len(times))
    surv = np.cumprod(1 - d / n_risk)
    # дисперсия по Гринвуду, интервал строим на шкале log log, чтобы он не вылезал за 0 и 1
    with np.errstate(divide="ignore", invalid="ignore"):
        gv = np.cumsum(d / (n_risk * (n_risk - d)))
        se = np.sqrt(gv) / np.abs(np.log(surv))
    z = stats.norm.ppf(1 - alpha / 2)
    lo = surv ** np.exp(z * se)
    hi = surv ** np.exp(-z * se)
    out = pd.DataFrame({"time": times, "n_risk": n_risk, "events": d, "surv": surv, "lo": lo, "hi": hi})
    zero = pd.DataFrame({"time": [0.0], "n_risk": [len(t)], "events": [0], "surv": [1.0], "lo": [1.0], "hi": [1.0]})
    return pd.concat([zero, out], ignore_index=True)


def km_median(km):
    below = km[km["surv"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else np.nan


def km_at(km, t):
    # значение кривой в момент t, кривая ступенчатая и идет вправо
    i = np.searchsorted(km["time"].values, t, side="right") - 1
    return float(km["surv"].iloc[max(i, 0)])


# ---------------------------------------------------------------- логранк

def logrank(t, e, group):
    # критерий для k групп: сравниваем наблюдаемые уходы с теми, что ждали бы при одинаковом риске
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    group = np.asarray(group)
    labels = list(pd.unique(group))
    k = len(labels)
    times = np.unique(t[e == 1])
    n_i = np.zeros((len(times), k))
    d_i = np.zeros((len(times), k))
    for j, lab in enumerate(labels):
        m = group == lab
        tj = np.sort(t[m])
        n_i[:, j] = m.sum() - np.searchsorted(tj, times, side="left")
        ev = t[m & (e == 1)]
        d_i[:, j] = np.bincount(np.searchsorted(times, ev), minlength=len(times))
    n = n_i.sum(axis=1)
    d = d_i.sum(axis=1)
    exp = d[:, None] * n_i / n[:, None]
    obs_minus_exp = (d_i - exp).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(n > 1, d * (n - d) / (n - 1), 0.0) / n ** 2
    cov = -(n_i * corr[:, None]).T @ n_i
    cov[np.diag_indices(k)] = (corr[:, None] * n_i * (n[:, None] - n_i)).sum(axis=0)
    z = obs_minus_exp[:-1]
    v = cov[:-1, :-1]
    chi2 = float(z @ np.linalg.solve(v, z))
    return {"chi2": chi2, "df": k - 1, "p": float(stats.chi2.sf(chi2, k - 1)),
            "observed": dict(zip(labels, d_i.sum(axis=0))), "expected": dict(zip(labels, exp.sum(axis=0)))}


# ---------------------------------------------------------------- дискретная модель риска

def person_period(ep, spells, calendar_lag_weeks=0):
    # раскладываем каждую подписку на недели, одна строка это одна неделя под риском ухода
    # конкурирующий риск, то есть списание не прошло, считаем цензурой: он от признаков не зависит
    ep = ep.reset_index(drop=True)
    T = np.ceil(ep["weeks"].values).astype(int)
    idx = np.repeat(np.arange(len(ep)), T)
    offset = np.repeat(np.cumsum(T) - T, T)
    k = np.arange(T.sum()) - offset
    start = ep["start"].values[idx]
    date = pd.to_datetime(start) + pd.to_timedelta(7 * k, unit="D")
    pp = pd.DataFrame({"row": idx, "user_id": ep["user_id"].values[idx], "tenure": k, "date": date})
    pp["event"] = ((k == T[idx] - 1) & (ep["event_voluntary"].values[idx] == 1)).astype(int)

    # текущий тариф недели берем из таблицы периодов
    sp = spells[["user_id", "start_date", "plan"]].sort_values("start_date")
    pp = pp.sort_values("date")
    pp = pd.merge_asof(pp, sp, left_on="date", right_on="start_date", by="user_id", direction="backward")
    pp = pp.sort_values(["row", "tenure"]).reset_index(drop=True)

    cal = pp["date"] - pd.to_timedelta(7 * calendar_lag_weeks, unit="D")
    cal_week = ((cal - pd.Timestamp(START)).dt.days // 7).values
    treated = ep["treated_region"].values[pp["row"].values]
    pp["hike"] = (treated & (cal_week >= HIKE_WEEK)).astype(float)
    pp["summer"] = cal.dt.month.isin([6, 7, 8]).astype(float)
    return pp


def design_matrix(pp, ep):
    r = pp["row"].values
    cols = {}
    for ch in ["search", "social", "referral", "partner"]:
        cols[ch] = (ep["channel"].values[r] == ch).astype(float)
    for pl in ["basic", "premium"]:
        cols[pl] = (pp["plan"].values == pl).astype(float)
    for dv in ["tv", "web"]:
        cols[dv] = (ep["device"].values[r] == dv).astype(float)
    cols["activated"] = ep["activated_3d"].values[r].astype(float)
    cols["engagement_per_log_hour"] = ep["log_trial_hours"].values[r] - ep["log_trial_hours"].mean()
    cols["hike"] = pp["hike"].values
    cols["summer"] = pp["summer"].values
    ten = pp["tenure"].values
    # границы недель подобраны под этапы жизни подписки, а не под истинные значения
    for lo, hi in [(4, 8), (8, 16), (16, 10 ** 6)]:
        cols[f"tenure_{lo}_{hi if hi < 10 ** 6 else 'plus'}"] = ((ten >= lo) & (ten < hi)).astype(float)
    names = ["intercept"] + list(cols)
    X = np.column_stack([np.ones(len(pp))] + [cols[c] for c in cols])
    return X, names


def fit_logit(X, y, max_iter=50, tol=1e-8):
    # логистическая регрессия методом Ньютона, он же IRLS
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        grad = X.T @ (y - p)
        hess = (X * w[:, None]).T @ X
        step = np.linalg.solve(hess, grad)
        b = b + step
        if np.abs(step).max() < tol:
            break
    eta = X @ b
    p = 1 / (1 + np.exp(-eta))
    hess = (X * (p * (1 - p))[:, None]).T @ X
    cov = np.linalg.inv(hess)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return b, np.sqrt(np.diag(cov)), ll


def hazard_model(ep, spells, calendar_lag_weeks=0):
    pp = person_period(ep, spells, calendar_lag_weeks)
    X, names = design_matrix(pp, ep)
    y = pp["event"].values.astype(float)
    b, se, ll = fit_logit(X, y)
    res = pd.DataFrame({"term": names, "coef": b, "se": se})
    res["lo"] = res["coef"] - 1.96 * res["se"]
    res["hi"] = res["coef"] + 1.96 * res["se"]
    res["hazard_ratio"] = np.exp(res["coef"])
    return res, {"n_rows": int(len(pp)), "n_events": int(y.sum()), "loglik": ll}


def empirical_hazard(ep, max_week=52):
    # доля ушедших среди тех, кто дожил до недели, для месячной подписки
    m = ep[ep["billing"] == "monthly"]
    t = np.ceil(m["weeks"].values).astype(int) - 1
    ev = m["event_voluntary"].values
    rows = []
    for w in range(max_week):
        at_risk = (t >= w).sum()
        d = ((t == w) & (ev == 1)).sum()
        rows.append((w, at_risk, d, d / at_risk if at_risk else np.nan))
    return pd.DataFrame(rows, columns=["week", "at_risk", "events", "hazard"])


def run(conn):
    # весь блок целиком: кривые, тесты, модель, все в одном словаре для отчета и графиков
    ep, spells = episodes(conn)
    mon = ep[ep["billing"] == "monthly"]
    out = {"episodes": ep, "spells": spells}
    km = {}
    for ch, g in mon.groupby("channel"):
        km[ch] = kaplan_meier(g["weeks"], g["event_voluntary"])
    out["km_channel"] = km
    km_plan = {}
    for pl, g in mon.groupby("plan_start"):
        km_plan[pl] = kaplan_meier(g["weeks"], g["event_voluntary"])
    out["km_plan"] = km_plan
    out["logrank_channel"] = logrank(mon["weeks"], mon["event_voluntary"], mon["channel"])
    out["logrank_plan"] = logrank(mon["weeks"], mon["event_voluntary"], mon["plan_start"])
    summary = []
    for ch, k in km.items():
        summary.append({"channel": ch, "n": int((mon["channel"] == ch).sum()), "median_weeks": km_median(k),
                        "surv_13w": km_at(k, 13), "surv_26w": km_at(k, 26), "surv_52w": km_at(k, 52)})
    out["km_summary"] = pd.DataFrame(summary).sort_values("surv_26w", ascending=False).reset_index(drop=True)
    known = mon[mon["region"] != "unknown"]
    out["hazard_naive"], out["hazard_info"] = hazard_model(known, spells, 0)
    out["hazard_lag"], _ = hazard_model(known, spells, 4)
    out["empirical_hazard"] = empirical_hazard(ep)
    return out


def hazard_by_utype(ep, spells, truth_users, utype="lost_cause", lag=4):
    # проверка причины расхождений, возможная только на симуляции
    # у типа lost_cause спад интереса почти всегда заканчивается уходом, значит наблюдаемый уход почти равен
    # началу спада, и оценки модели должны сойтись с заложенными значениями
    m = ep[(ep["billing"] == "monthly") & (ep["region"] != "unknown")].merge(
        truth_users[["user_id", "utype"]], on="user_id")
    m = m[m["utype"] == utype]
    res, info = hazard_model(m, spells, lag)
    return res, info
