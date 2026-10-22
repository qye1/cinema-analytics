# прогноз ежедневных регистраций на 28 дней: три метода, скользящая проверка на истории и интервалы
# метод Хольта Винтерса написан руками, чтобы не зависеть от лишних библиотек

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

SEASON = 7
HORIZON = 28


# ---------------------------------------------------------------- Хольт Винтерс

def _hw_run(y, alpha, beta, gamma, phi, season=SEASON):
    # один проход по ряду, возвращает ошибки на шаг вперед и последнее состояние
    n = len(y)
    level = y[:season].mean()
    trend = (y[season:2 * season].mean() - y[:season].mean()) / season
    seas = list(y[:season] - level)
    sse = 0.0
    for t in range(season, n):
        s = seas[t - season]
        pred = level + phi * trend + s
        err = y[t] - pred
        sse += err * err
        new_level = level + phi * trend + alpha * err
        trend = phi * trend + beta * err
        seas.append(s + gamma * (1 - alpha) * err)
        level = new_level
    return sse, level, trend, seas


def hw_fit(y, season=SEASON):
    # параметры ищем минимизацией суммы квадратов ошибок на шаг вперед, берем лучший из нескольких стартов
    y = np.asarray(y, float)
    best = None
    for start in ([0.3, 0.05, 0.2, 0.95], [0.1, 0.01, 0.1, 0.9], [0.5, 0.1, 0.4, 0.98]):
        r = minimize(lambda p: _hw_run(y, *p, season=season)[0], start, method="L-BFGS-B",
                     bounds=[(0.01, 0.99), (0.0005, 0.5), (0.01, 0.99), (0.8, 1.0)])
        if best is None or r.fun < best.fun:
            best = r
    alpha, beta, gamma, phi = best.x
    _, level, trend, seas = _hw_run(y, alpha, beta, gamma, phi, season)
    return {"params": (alpha, beta, gamma, phi), "level": level, "trend": trend, "seas": seas[-season:], "n": len(y)}


def hw_forecast(model, h=HORIZON, season=SEASON):
    alpha, beta, gamma, phi = model["params"]
    out = []
    damp = 0.0
    for k in range(1, h + 1):
        damp += phi ** k
        out.append(model["level"] + damp * model["trend"] + model["seas"][(k - 1) % season])
    return np.array(out)


# ---------------------------------------------------------------- календарные признаки и простые методы

def calendar_features(dates, first_date, campaign_days=()):
    d = pd.DatetimeIndex(dates)
    t = ((d - pd.Timestamp(first_date)).days.values) / 365.0
    cols = {"t": t, "t2": t ** 2}
    for k in range(1, 7):
        cols[f"dow{k}"] = (d.dayofweek.values == k).astype(float)
    cols["summer"] = d.month.isin([6, 7, 8]).astype(float)
    cols["campaign"] = d.isin(pd.DatetimeIndex(campaign_days)).astype(float)
    return pd.DataFrame(cols, index=d)


def ridge_forecast(dates_train, y_train, dates_future, first_date, campaign_days=(), alpha=1.0):
    Xtr = calendar_features(dates_train, first_date, campaign_days)
    Xf = calendar_features(dates_future, first_date, campaign_days)
    m = Ridge(alpha=alpha).fit(Xtr.values, np.log(y_train))
    return m.predict(Xf.values), m


def seasonal_naive(y_train, h=HORIZON, season=SEASON):
    # прошлая неделя повторяется
    last = np.asarray(y_train)[-season:]
    return np.array([last[(k) % season] for k in range(h)])


def seasonal_mean(y_train, h=HORIZON, season=SEASON, weeks=4):
    # среднее по последним четырем одноименным дням недели, шум сглаживается
    y = np.asarray(y_train)
    block = y[-weeks * season:].reshape(weeks, season)
    m = block.mean(axis=0)
    return np.array([m[k % season] for k in range(h)])


# ---------------------------------------------------------------- проверка на истории

def backtest(series, origins, h=HORIZON, campaign_days=()):
    # скользящая проверка: в каждой точке обучаемся только на прошлом и прогнозируем следующие h дней
    y = series["signups"].values.astype(float)
    dates = pd.to_datetime(series["day"]).values
    first = dates[0]
    rows = []
    for o in origins:
        ytr, dtr = y[:o], dates[:o]
        dfu = dates[o:o + h]
        act = y[o:o + h]
        if len(act) < h:
            continue
        hw = hw_fit(np.log(ytr))
        fc = {
            "seasonal_naive": seasonal_naive(ytr, h),
            "seasonal_mean4": seasonal_mean(ytr, h),
            "holt_winters": np.exp(hw_forecast(hw, h)),
            "ridge_calendar": np.exp(ridge_forecast(dtr, ytr, dfu, first, campaign_days)[0]),
        }
        for name, f in fc.items():
            for k in range(h):
                rows.append({"origin": o, "horizon": k + 1, "model": name, "date": dfu[k], "actual": act[k], "forecast": f[k]})
    return pd.DataFrame(rows)


def score(bt):
    d = bt.copy()
    d["abs_err"] = (d["actual"] - d["forecast"]).abs()
    d["err"] = d["forecast"] - d["actual"]
    g = d.groupby("model")
    out = pd.DataFrame({"wape": g["abs_err"].sum() / g["actual"].sum(),
                        "mae": g["abs_err"].mean(),
                        "bias_pct": g["err"].sum() / g["actual"].sum()})
    return out.sort_values("wape")


def score_by_horizon(bt):
    d = bt.copy()
    d["abs_err"] = (d["actual"] - d["forecast"]).abs()
    d["week"] = (d["horizon"] - 1) // 7 + 1
    g = d.groupby(["model", "week"])
    return (g["abs_err"].sum() / g["actual"].sum()).unstack("model")


def interval_quantiles(bt, model, lo=0.05, hi=0.95, origins=None):
    # границы интервала берем из логарифмических ошибок прошлых прогнозов по каждому горизонту
    d = bt[bt["model"] == model].copy()
    if origins is not None:
        d = d[d["origin"].isin(origins)]
    d["lerr"] = np.log(d["actual"]) - np.log(d["forecast"])
    q = d.groupby("horizon")["lerr"].quantile([lo, hi]).unstack()
    q.columns = ["lo", "hi"]
    return q


def coverage(bt, model, calib_origins, test_origins, lo=0.05, hi=0.95):
    # честная проверка интервалов: калибруем на ранних точках, проверяем на поздних
    q = interval_quantiles(bt, model, lo, hi, calib_origins)
    d = bt[(bt["model"] == model) & (bt["origin"].isin(test_origins))].copy()
    d = d.join(q, on="horizon")
    low = d["forecast"] * np.exp(d["lo"])
    up = d["forecast"] * np.exp(d["hi"])
    inside = ((d["actual"] >= low) & (d["actual"] <= up)).mean()
    return float(inside), float((up - low).mean())


def final_forecast(series, model="holt_winters", h=HORIZON, bt=None, campaign_days=()):
    y = series["signups"].values.astype(float)
    dates = pd.to_datetime(series["day"])
    fut = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=h)
    if model == "holt_winters":
        f = np.exp(hw_forecast(hw_fit(np.log(y)), h))
    elif model == "ridge_calendar":
        f = np.exp(ridge_forecast(dates.values, y, fut.values, dates.iloc[0], campaign_days)[0])
    else:
        f = seasonal_mean(y, h)
    out = pd.DataFrame({"date": fut, "forecast": f})
    if bt is not None:
        q = interval_quantiles(bt, model)
        out["lo"] = f * np.exp(q["lo"].values[:h])
        out["hi"] = f * np.exp(q["hi"].values[:h])
    return out


def run(conn, campaign_days=()):
    from . import marts
    s = marts.daily_signups(conn)
    n = len(s)
    origins = list(range(210, n - HORIZON + 1, 14))
    bt = backtest(s, origins, campaign_days=campaign_days)
    sc = score(bt)
    best = sc.index[0]
    half = len(origins) // 2
    cov = {}
    for m in ["holt_winters", "ridge_calendar"]:
        cov[m] = coverage(bt, m, origins[:half], origins[half:])
    fin = final_forecast(s, best, bt=bt, campaign_days=campaign_days)
    return {"series": s, "backtest": bt, "score": sc, "by_horizon": score_by_horizon(bt), "coverage": cov,
            "final": fin, "best": best, "origins": origins}
