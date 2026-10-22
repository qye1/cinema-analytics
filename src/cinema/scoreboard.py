# сводная таблица: что мы оценили по данным и что на самом деле заложено в симуляцию
# это главная проверка проекта, потому что в реальной жизни правды нет, а здесь она есть

import pandas as pd

TERM_NAMES = {"social": "риск ухода, канал social", "referral": "риск ухода, канал referral",
              "partner": "риск ухода, канал partner", "search": "риск ухода, канал search",
              "basic": "риск ухода, тариф basic", "premium": "риск ухода, тариф premium",
              "tv": "риск ухода, устройство tv", "web": "риск ухода, устройство web",
              "engagement_per_log_hour": "риск ухода, вовлеченность", "hike": "риск ухода, повышение цены",
              "summer": "риск ухода, лето"}


def _row(group, name, est, lo, hi, truth, unit):
    covered = None if lo is None else bool(lo <= truth <= hi)
    return {"group": group, "quantity": name, "estimate": est, "ci_low": lo, "ci_high": hi, "truth": truth,
            "unit": unit, "covered": covered}


def build(res, truth):
    rows = []
    planted = truth["hazard_ratios"]["planted_log_hazard"]
    naive = res["survival"]["hazard_naive"].set_index("term")
    lag = res["survival"]["hazard_lag"].set_index("term")
    for term, tv in planted.items():
        src = lag if term in ("hike", "summer") else naive
        r = src.loc[term]
        rows.append(_row("модель риска", TERM_NAMES[term], r["coef"], r["lo"], r["hi"], tv, "лог риска"))
    ex = res["exp"]
    itt = ex["itt"]
    rows.append(_row("A/B оффер", "эффект оффера на уход за 4 недели", 100 * itt["diff"], 100 * itt["ci_boot"][0],
                     100 * itt["ci_boot"][1], 100 * truth["offer"]["true_effect_on_treated_4w_churn"], "п.п."))
    bt = ex["by_type"].set_index("group")
    for k, v in truth["offer"]["by_type"].items():
        if v["effect"] is None or k not in bt.index:
            continue
        r = bt.loc[k]
        rows.append(_row("A/B оффер", f"эффект для типа {k}", 100 * r["diff"], 100 * r["lo"], 100 * r["hi"],
                         100 * v["effect"], "п.п."))
    h = ex["hike"]
    rows.append(_row("повышение цены", "difference in differences, недельный отток", 100 * h["did"], 100 * h["ci"][0],
                     100 * h["ci"][1], truth["hike"]["true_effect_weekly_churn_pp"], "п.п."))
    rows.append(_row("повышение цены", "наивное сравнение до и после", 100 * h["naive"], None, None,
                     truth["hike"]["true_effect_weekly_churn_pp"], "п.п."))
    inj = truth["injected"]
    q = res["quality"].set_index("check_name")["problem_rows"]
    for key, name in [("duplicate_payments", "дубли платежей"), ("zero_amount_success", "успешные платежи с нулем"),
                      ("payment_before_signup", "платеж раньше регистрации"), ("null_region", "пустой регион"),
                      ("negative_hours", "отрицательные часы"), ("overlapping_spells", "пересекающиеся подписки")]:
        rows.append(_row("качество данных", f"найдено: {name}", float(q[key]), None, None, float(inj[key]), "строк"))
    out = pd.DataFrame(rows)
    out["covered"] = out["covered"].astype(object)
    exact = out["group"] == "качество данных"
    out.loc[exact, "covered"] = (out.loc[exact, "estimate"] == out.loc[exact, "truth"])
    return out


def summary_counts(sb):
    d = sb[sb["covered"].notna()]
    return int(d["covered"].sum()), int(len(d))
