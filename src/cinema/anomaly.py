# поиск аномалий: сбой платежного шлюза по доле неудачных оплат и скачки регистраций
# три подхода: проверка по биному с поправкой на множественные тесты, CUSUM и изоляционный лес

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from . import marts
from .config import INCIDENT_DAYS


def benjamini_hochberg(p, q=0.01):
    # отбор по контролю доли ложных находок: сортируем p значения и сравниваем с лестницей q * i / m
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    ok = p[order] <= thresh
    keep = np.zeros(m, bool)
    if ok.any():
        k = np.max(np.where(ok)[0])
        keep[order[:k + 1]] = True
    return keep


def rolling_baseline(rate, window=28, gap=1):
    # типичный уровень по предыдущим дням: медиана, он устойчив к самим сбоям
    return rate.shift(gap).rolling(window, min_periods=14).median()


def binomial_test(pf, q=0.01, window=28):
    d = pf.copy()
    d["base"] = rolling_baseline(d["fail_rate"], window)
    d = d.dropna(subset=["base"]).copy()
    # вероятность увидеть столько неудач и больше при обычном уровне
    d["p_value"] = stats.binom.sf(d["failed"].values - 1, d["attempts"].values, d["base"].values)
    d["flag_binom"] = benjamini_hochberg(d["p_value"].values, q)
    return d


def cusum(d, k=0.5, h=6.0):
    # накопительная сумма стандартизованных отклонений, ловит и небольшой, но долгий сдвиг
    z = (d["fail_rate"] - d["base"]) / np.sqrt(d["base"] * (1 - d["base"]) / d["attempts"])
    s = np.zeros(len(d))
    alarm = np.zeros(len(d), bool)
    cur = 0.0
    for i, v in enumerate(z.values):
        cur = max(0.0, cur + v - k)
        s[i] = cur
        if cur > h:
            # после тревоги счетчик обнуляем, иначе он держит флаг еще много дней после конца сбоя
            alarm[i] = True
            cur = 0.0
    d = d.copy()
    d["z"] = z.values
    d["cusum"] = s
    d["flag_cusum"] = alarm
    return d


def isolation(d, seed=0, contamination=0.01):
    X = np.column_stack([d["fail_rate"], np.log(d["attempts"]), pd.to_datetime(d["day"]).dt.dayofweek])
    m = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    d = d.copy()
    d["iso_score"] = -m.score_samples(X)
    d["flag_iso"] = m.predict(X) == -1
    return d


def evaluate(d, flag_col, true_days):
    t = set(pd.Timestamp(x) for x in true_days)
    days = pd.to_datetime(d["day"])
    flagged = set(days[d[flag_col].values])
    tp = len(flagged & t)
    fp = len(flagged - t)
    fn = len(t - flagged)
    # задержка: сколько дней прошло от начала сбоя до первой тревоги внутри него
    first_true = min(t)
    hits = sorted(x for x in flagged if x in t)
    delay = (hits[0] - first_true).days if hits else None
    return {"method": flag_col, "true_positive": tp, "false_positive": fp, "missed": fn,
            "precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / len(t), "delay_days": delay}


def signup_anomalies(series, z_thr=4.0):
    # скачки регистраций: сравниваем с медианой того же дня недели за последние восемь недель, шум берем по MAD
    # медиана по восьми неделям устойчива к тому, что часть из них сама попала в рекламную кампанию
    s = series.copy()
    s["day"] = pd.to_datetime(s["day"])
    y = np.log(s["signups"].values.astype(float))
    base = np.full(len(y), np.nan)
    resid_hist = []
    z = np.full(len(y), np.nan)
    for i in range(56, len(y)):
        ref = y[[i - 7 * k for k in range(1, 9)]]
        base[i] = np.median(ref)
        r = y[i] - base[i]
        if len(resid_hist) >= 28:
            h = np.array(resid_hist[-56:])
            mad = np.median(np.abs(h - np.median(h))) * 1.4826
            z[i] = (r - np.median(h)) / max(mad, 1e-6)
        # аномальные точки в историю шума не кладем, иначе они раздувают MAD
        if not (len(resid_hist) >= 28 and abs(z[i]) > z_thr):
            resid_hist.append(r)
    s["z"] = z
    s["flag"] = np.abs(z) > z_thr
    return s


def run(conn, campaign_days):
    pf = marts.payment_failures(conn)
    d = binomial_test(pf)
    d = cusum(d)
    d = isolation(d)
    ev = pd.DataFrame([evaluate(d, c, INCIDENT_DAYS) for c in ["flag_binom", "flag_cusum", "flag_iso"]])
    sg = signup_anomalies(marts.daily_signups(conn))
    t = set(pd.Timestamp(x) for x in campaign_days)
    flagged = set(sg.loc[sg["flag"], "day"])
    sig_eval = {"flagged": len(flagged), "true_positive": len(flagged & t), "false_positive": len(flagged - t),
                "campaign_days": len(t), "missed": len(t - flagged)}
    return {"failures": d, "eval": ev, "signups": sg, "signup_eval": sig_eval}
