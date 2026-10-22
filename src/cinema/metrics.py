# продуктовые метрики поверх sql витрин: NRR и GRR, матрицы когорт, LTV по каналам, окупаемость

import numpy as np
import pandas as pd

from . import marts
from .config import CHANNELS


def waterfall_metrics(w):
    # добавляем к водопаду MRR относительные метрики, считаем их на каждый месяц
    w = w.copy()
    start = w["start_mrr"].replace(0, np.nan)
    kept = w["start_mrr"] + w["expansion_mrr"] + w["price_uplift_mrr"] + w["contraction_mrr"] + w["churn_mrr"]
    # реактивации в NRR не входят, это возвращение ушедших, а не рост текущей базы
    w["nrr"] = kept / start
    w["grr"] = (w["start_mrr"] + w["contraction_mrr"] + w["churn_mrr"]) / start
    gain = w["new_mrr"] + w["reactivation_mrr"] + w["expansion_mrr"] + w["price_uplift_mrr"]
    loss = -(w["contraction_mrr"] + w["churn_mrr"])
    w["quick_ratio"] = gain / loss.replace(0, np.nan)
    w["mrr_growth"] = w["end_mrr"] / start - 1
    # годовой NRR как произведение последних 12 месячных значений
    w["nrr_12m"] = w["nrr"].rolling(12).apply(np.prod, raw=True)
    return w


def cohort_matrices(coh):
    # из длинной таблицы когорт делаем две матрицы: доля людей и доля денег, строки это когорты
    ret = coh.pivot(index="cohort_idx", columns="age", values="retention_pct")
    rev = coh.pivot(index="cohort_idx", columns="age", values="revenue_retention_pct")
    size = coh.drop_duplicates("cohort_idx").set_index("cohort_idx")["cohort_users"]
    return ret, rev, size


def weighted_retention(coh, max_age=12):
    # средняя кривая удержания по когортам с весом по размеру когорты
    # берем только когорты, у которых этот возраст уже наступил
    out = []
    for age in range(max_age + 1):
        part = coh[coh["age"] == age]
        out.append({"age": age, "retention_pct": np.average(part["retention_pct"], weights=part["cohort_users"]),
                    "revenue_retention_pct": np.average(part["revenue_retention_pct"], weights=part["cohort_users"]),
                    "cohorts": len(part)})
    return pd.DataFrame(out)


def channel_mix(conn):
    # доля каналов в регистрациях по месяцам, тут видно, что social растет
    users = pd.read_sql_query("SELECT signup_date, channel FROM users_clean", conn)
    users["month"] = pd.to_datetime(users["signup_date"]).dt.to_period("M").astype(str)
    mix = users.groupby(["month", "channel"]).size().unstack(fill_value=0)
    mix = mix.reindex(columns=CHANNELS, fill_value=0)
    return mix.div(mix.sum(axis=1), axis=0)


def revenue_curves(conn, horizon_days=360, last_signup="2024-12-31"):
    # накопленная выручка на одну регистрацию по дням жизни и каналам
    # берем только тех, у кого впереди полный горизонт наблюдения
    users = pd.read_sql_query(
        "SELECT user_id, signup_date, channel FROM users_clean WHERE signup_date <= ?", conn, params=[last_signup])
    pay = pd.read_sql_query(
        "SELECT p.user_id, p.pay_date, p.amount FROM payments_clean p WHERE p.status = 'success'", conn)
    users["signup_date"] = pd.to_datetime(users["signup_date"])
    pay["pay_date"] = pd.to_datetime(pay["pay_date"])
    pay = pay.merge(users[["user_id", "signup_date", "channel"]], on="user_id")
    pay["day"] = (pay["pay_date"] - pay["signup_date"]).dt.days
    pay = pay[(pay["day"] >= 0) & (pay["day"] <= horizon_days)]
    n = users.groupby("channel").size()
    daily = pay.groupby(["channel", "day"])["amount"].sum().unstack(fill_value=0)
    daily = daily.reindex(columns=range(horizon_days + 1), fill_value=0)
    curves = daily.cumsum(axis=1).div(n, axis=0)
    return curves.T


def channel_economics(conn, funnel):
    # цена регистрации и выручка за 180 дней на регистрацию, из них строится картинка окупаемости
    f = funnel.copy()
    f["cac_per_signup"] = f["spend"] / f["signups"]
    f["revenue_180_per_signup"] = f["revenue_180_per_paying"] * f["paying"] / f["signups"]
    f["profit_180_per_signup"] = f["revenue_180_per_signup"] - f["cac_per_signup"]
    return f


def kpi_summary(w, coh, funnel):
    # короткий набор цифр для отчета
    last = w.iloc[-1]
    wr = weighted_retention(coh, 6)
    return {
        "end_mrr": float(last["end_mrr"]),
        "end_subscribers": int(last["end_subscribers"]),
        "nrr_12m": float(last["nrr_12m"]),
        "median_monthly_grr": float(w["grr"].median()),
        "retention_m3": float(wr.loc[wr["age"] == 3, "retention_pct"].iloc[0]),
        "retention_m6": float(wr.loc[wr["age"] == 6, "retention_pct"].iloc[0]),
        "best_channel": str(funnel.dropna(subset=["revenue180_to_cac"]).iloc[0]["channel"]),
        "worst_channel": str(funnel.dropna(subset=["revenue180_to_cac"]).iloc[-1]["channel"]),
    }


def build_all(conn):
    # один вызов, который собирает все таблицы блока метрик
    w = waterfall_metrics(marts.mrr_waterfall(conn))
    coh = marts.cohorts(conn)
    funnel = marts.funnel_cac(conn)
    return {
        "waterfall": w,
        "cohorts": coh,
        "funnel": channel_economics(conn, funnel),
        "mix": channel_mix(conn),
        "curves": revenue_curves(conn),
        "retention_curve": weighted_retention(coh, 12),
    }
