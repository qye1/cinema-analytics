# сборка HTML отчета: шаблон jinja2, картинки вшиваются в файл, цифры в тексте берутся из результатов

import base64
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from . import scoreboard
from .plots_churn import LABELS

ROOT = Path(__file__).resolve().parents[2]


def _rub(x, digits=0):
    return f"{x:,.{digits}f}".replace(",", " ")


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f}%".replace(".", ",")


def _num(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def _pval(p):
    return "меньше 0,001" if p < 0.001 else _num(p, 3)


def _table(df, formats=None, rename=None, nan_text=None):
    d = df.copy()
    formats = formats or {}
    nan_text = nan_text or {}
    for c, f in formats.items():
        if c in d.columns:
            d[c] = d[c].map(lambda v, f=f, c=c: nan_text.get(c, "") if pd.isna(v) else f(v))
    if rename:
        d = d.rename(columns=rename)
    return Markup('<div class="tw">' + d.to_html(index=False, border=0, classes="t", escape=True) + "</div>")


def _img_factory(charts_dir):
    def img(name, alt=""):
        data = base64.b64encode((Path(charts_dir) / name).read_bytes()).decode("ascii")
        return Markup(f'<img alt="{alt}" src="data:image/png;base64,{data}">')
    return img


def _cusum_fp(an):
    from .config import INCIDENT_DAYS
    d = an["failures"]
    t = set(pd.Timestamp(x) for x in INCIDENT_DAYS)
    days = pd.to_datetime(d.loc[d["flag_cusum"], "day"])
    extra = [x for x in days if x not in t]
    if not extra:
        return "лишних нет"
    return "лишние дни: " + ", ".join(x.strftime("%d.%m.%Y") for x in extra[:5])


def build_context(res):
    m = res["metrics"]
    w = m["waterfall"]
    kpi = res["kpi"]
    wr = m["retention_curve"].set_index("age")
    f = m["funnel"].copy()
    mix = m["mix"]
    sv = res["survival"]
    seg = res["segments"]
    ch = res["churn"]
    ex = res["exp"]
    hk = ex["hike"]
    fc = res["forecast"]
    an = res["anomaly"]
    truth = res["truth"]
    sb = res["scoreboard"]
    hit, total = scoreboard.summary_counts(sb)

    uplift_month = w.loc[w["price_uplift_mrr"].idxmax()]
    f_order = f.sort_values("revenue180_to_cac", ascending=False, na_position="last")
    funnel_tbl = _table(
        f_order[["channel", "signups", "conversion_pct", "cac_per_signup", "revenue_180_per_signup", "revenue180_to_cac"]],
        {"signups": lambda v: _rub(v), "conversion_pct": lambda v: _num(v), "cac_per_signup": lambda v: _rub(v),
         "revenue_180_per_signup": lambda v: _rub(v), "revenue180_to_cac": lambda v: _num(v, 1)},
        {"channel": "канал", "signups": "регистраций", "conversion_pct": "стали платящими, %",
         "cac_per_signup": "цена регистрации, руб", "revenue_180_per_signup": "выручка за 180 дней на регистрацию, руб",
         "revenue180_to_cac": "окупаемость"}, nan_text={"revenue180_to_cac": "нет затрат"})

    quality = res["quality"].copy()
    inj = truth["injected"]
    quality["injected"] = quality["check_name"].map(inj)
    quality_tbl = _table(quality[["check_name", "problem_rows", "injected"]],
                         {"problem_rows": lambda v: _rub(v), "injected": lambda v: _rub(v)},
                         {"check_name": "проверка", "problem_rows": "найдено", "injected": "подложено в генераторе"})

    km_tbl = _table(sv["km_summary"],
                    {"n": lambda v: _rub(v), "median_weeks": lambda v: "не достигнута" if pd.isna(v) else _num(v, 0),
                     "surv_13w": lambda v: _pct(v, 0), "surv_26w": lambda v: _pct(v, 0), "surv_52w": lambda v: _pct(v, 0)},
                    {"channel": "канал", "n": "подписок", "median_weeks": "медиана, недель",
                     "surv_13w": "живы через 13 недель", "surv_26w": "через 26", "surv_52w": "через 52"},
                    nan_text={"median_weeks": "не достигнута"})

    prof = seg["profile"].copy()
    prof.insert(0, "segment", [seg["names"][i] for i in prof.index])
    seg_tbl = _table(prof[["segment", "users", "mean_hours", "hours_per_session", "titles_per_session", "weekend_share"]],
                     {"users": lambda v: _rub(v), "mean_hours": lambda v: _num(v, 1), "hours_per_session": lambda v: _num(v, 2),
                      "titles_per_session": lambda v: _num(v, 2), "weekend_share": lambda v: _pct(v, 0)},
                     {"segment": "сегмент", "users": "людей", "mean_hours": "часов в неделю",
                      "hours_per_session": "часов за сессию", "titles_per_session": "названий за сессию",
                      "weekend_share": "доля выходных"})
    st = seg["surv_table"].copy()
    st["segment"] = st["segment"].map(lambda i: seg["names"][i])
    seg_surv_tbl = _table(st, {"n": lambda v: _rub(v), "median_weeks": lambda v: _num(v, 0), "surv_26w": lambda v: _pct(v, 0)},
                          {"segment": "сегмент", "n": "подписок", "median_weeks": "медиана, недель", "surv_26w": "живы через 26 недель"})

    cres = ch["results"].copy()
    churn_tbl = _table(cres[["model", "auc", "pr_auc", "brier"]],
                       {"auc": lambda v: _num(v, 3), "pr_auc": lambda v: _num(v, 3), "brier": lambda v: _num(v, 4)},
                       {"model": "модель", "auc": "AUC", "pr_auc": "PR AUC", "brier": "Brier"})
    g = ch["gains"]
    top10 = g.iloc[0]
    imp = ch["importance"].head(5)
    ab = ch["ablation"].set_index("features")["auc"]
    lk = ch["leakage"].set_index("setup")["auc"]
    gbm_row = cres[cres["model"] == "gbm"].iloc[0]
    rule_row = cres[cres["model"].str.startswith("правило")].iloc[0]

    itt = ex["itt"]
    up = ex["uplift"]
    pol = {k: v for k, v in up["policy"].items()}

    def saved_at(name, share):
        x, s = pol[name]
        return float(s[np.searchsorted(x, share)])

    n_exp = len(ex["data"])
    power_tbl = _table(ex["power"], {"effect_pp": lambda v: _num(v, 0), "n_per_arm": lambda v: _rub(v)},
                       {"effect_pp": "эффект, п.п.", "n_per_arm": "нужно людей в каждой группе"})
    ratio_tbl = _table(ex["by_ratio"], {"n": lambda v: _rub(v), "control_rate": lambda v: _pct(v, 0),
                                        "diff": lambda v: _num(100 * v, 1), "lo": lambda v: _num(100 * v, 1),
                                        "hi": lambda v: _num(100 * v, 1)},
                       {"group": "падение часов (терциль)", "n": "людей", "control_rate": "уходят без оффера",
                        "diff": "эффект, п.п.", "lo": "нижняя граница", "hi": "верхняя граница"})

    sc_tbl = fc["score"].reset_index().rename(columns={"model": "model"})
    names = {"ridge_calendar": "регрессия на календаре", "holt_winters": "Хольт Винтерс", "seasonal_mean4": "среднее за 4 недели",
             "seasonal_naive": "прошлая неделя"}
    sc_tbl["model"] = sc_tbl["model"].map(names)
    fc_tbl = _table(sc_tbl, {"wape": lambda v: _pct(v, 1), "mae": lambda v: _num(v, 1), "bias_pct": lambda v: _pct(v, 1)},
                    {"model": "метод", "wape": "WAPE", "mae": "средняя абс. ошибка", "bias_pct": "смещение"})
    an_names = {"flag_binom": "биномиальный тест с поправкой BH", "flag_cusum": "CUSUM", "flag_iso": "изоляционный лес"}
    ev = an["eval"].copy()
    ev["method"] = ev["method"].map(an_names)
    an_tbl = _table(ev[["method", "true_positive", "false_positive", "missed", "delay_days"]],
                    {"delay_days": lambda v: "" if v is None else str(int(v))},
                    {"method": "метод", "true_positive": "найдено дней сбоя из 3", "false_positive": "ложных дней",
                     "missed": "пропущено", "delay_days": "задержка, дней"})
    sg = an["signups"]
    from .config import CAMPAIGN_DAYS
    ny = sg[sg["day"].isin(pd.to_datetime(pd.Series(CAMPAIGN_DAYS[4:])))]

    sbv = sb.copy()
    sbv["covered"] = sbv["covered"].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else ("да" if v else "нет"))
    # знаков после запятой по смыслу строки: счетчики целые, проценты с двумя, логарифмы с тремя
    digits = sbv["unit"].map({"строк": 0, "п.п.": 2, "лог риска": 3})
    for c in ["estimate", "ci_low", "ci_high", "truth"]:
        sbv[c] = [np.nan if pd.isna(v) else f"{v:.{int(dg)}f}".replace(".", ",") for v, dg in zip(sbv[c], digits)]
    sb_tbl = _table(sbv[["group", "quantity", "estimate", "ci_low", "ci_high", "truth", "unit", "covered"]],
                    {c: str for c in ["estimate", "ci_low", "ci_high", "truth"]},
                    {"group": "блок", "quantity": "что оценивали", "estimate": "оценка", "ci_low": "от", "ci_high": "до",
                     "truth": "заложено", "unit": "единицы", "covered": "правда внутри интервала"})
    missed = sb[(sb["covered"] == False) & (sb["group"] == "модель риска")]  # noqa: E712

    return {
        "today": date.today().strftime("%d.%m.%Y"), "kpi": kpi, "hit": hit, "total": total,
        "end_mrr_m": _num(kpi["end_mrr"] / 1e6, 2), "subs": _rub(kpi["end_subscribers"]),
        "nrr_12m": _pct(kpi["nrr_12m"], 0), "grr_med": _pct(kpi["median_monthly_grr"], 1),
        "ret_m1": _pct(wr.loc[1, "retention_pct"] / 100, 0), "ret_m3": _pct(wr.loc[3, "retention_pct"] / 100, 0),
        "ret_m6": _pct(wr.loc[6, "retention_pct"] / 100, 0), "ret_m12": _pct(wr.loc[12, "retention_pct"] / 100, 0),
        "rev_m12": _pct(wr.loc[12, "revenue_retention_pct"] / 100, 0),
        "uplift_month": str(uplift_month["month_end"])[:7], "uplift_mrr": _rub(uplift_month["price_uplift_mrr"]),
        "uplift_nrr": _pct(uplift_month["nrr"], 1),
        "social_first": _pct(mix["social"].iloc[0], 0), "social_last": _pct(mix["social"].iloc[-1], 0),
        "funnel_tbl": funnel_tbl, "quality_tbl": quality_tbl, "km_tbl": km_tbl, "seg_tbl": seg_tbl, "seg_surv_tbl": seg_surv_tbl,
        "churn_tbl": churn_tbl, "power_tbl": power_tbl, "ratio_tbl": ratio_tbl, "fc_tbl": fc_tbl, "an_tbl": an_tbl, "sb_tbl": sb_tbl,
        "best_channel": kpi["best_channel"], "worst_channel": kpi["worst_channel"],
        "best_ratio": _num(f_order["revenue180_to_cac"].dropna().iloc[0], 1),
        "worst_ratio": _num(f_order["revenue180_to_cac"].dropna().iloc[-1], 1),
        "lr_chi2": _num(sv["logrank_channel"]["chi2"], 0), "lr_p": _pval(sv["logrank_channel"]["p"]),
        "hz_events": _rub(sv["hazard_info"]["n_events"]), "hz_rows": _rub(sv["hazard_info"]["n_rows"]),
        "missed_terms": ", ".join(missed["quantity"].str.replace("риск ухода, ", "", regex=False)),
        "seg_ari": _num(seg["agreement"]["ari"], 2), "seg_purity": _pct(seg["agreement"]["purity"], 0),
        "seg_k": seg["k"], "seg_sil": ", ".join(f"{k}: {_num(v, 2)}" for k, v in zip(seg["scan"]["k"], seg["scan"]["silhouette"])),
        "seg_n": _rub(len(seg["features"])),
        "ch_n_train": _rub(len(ch["train"])), "ch_n_test": _rub(len(ch["test"])),
        "ch_base": _pct(ch["test"]["y"].mean(), 1), "ch_auc": _num(gbm_row["auc"], 3), "ch_pr": _num(gbm_row["pr_auc"], 3),
        "ch_rule_auc": _num(rule_row["auc"], 3), "ch_top10": _pct(top10["captured"], 0), "ch_lift": _num(top10["lift"], 1),
        "ch_ctx": _num(ab["только контекст"], 3), "ch_beh": _num(ab["контекст и поведение"], 3),
        "ch_time": _num(lk["честно: делим по времени"], 3), "ch_rand": _num(lk["случайное разбиение: один человек в обеих частях"], 3),
        "ch_leak": _num(lk["признак из будущего: часы после снимка"], 3),
        "imp_top": ", ".join(LABELS.get(x, x) for x in imp["feature"].head(3)),
        "exp_n": _rub(n_exp), "srm_p": _num(ex["srm"]["p"], 2), "itt": itt,
        "itt_diff": _num(100 * itt["diff"], 1), "itt_lo": _num(100 * itt["ci_boot"][0], 1), "itt_hi": _num(100 * itt["ci_boot"][1], 1),
        "itt_p0": _pct(itt["p_control"], 1), "itt_p1": _pct(itt["p_offer"], 1), "itt_rel": _pct(abs(itt["relative"]), 0),
        "itt_perm": _pval(itt["perm_p"]), "adj_diff": _num(100 * ex["adjusted"]["diff"], 1),
        "adj_w": _num(200 * 1.96 * ex["adjusted"]["se"], 1), "itt_w": _num(100 * (itt["ci_normal"][1] - itt["ci_normal"][0]), 1),
        "mde": _num(100 * ex["mde"], 1),
        "truth_itt": _num(100 * truth["offer"]["true_effect_on_treated_4w_churn"], 1),
        "q_up": _num(up["qini"]["uplift_logreg"][2], 1), "q_risk": _num(up["qini"]["risk"][2], 1),
        "q_gbm": _num(up["qini"]["uplift_gbm"][2], 1),
        "q_ci_lo": _num(up["area_ci_vs_random"][0], 1), "q_ci_hi": _num(up["area_ci_vs_random"][1], 1),
        "pol_up": _num(saved_at("uplift_logreg", 0.3), 0), "pol_risk": _num(saved_at("risk", 0.3), 0),
        "pol_rand": _num(saved_at("random", 0.3), 0), "pol_orc": _num(saved_at("oracle", 0.3), 0),
        "pol_all": _num(saved_at("random", 1.0), 0),
        "hk_naive": _num(100 * hk["naive"], 2), "hk_did": _num(100 * hk["did"], 2), "hk_lo": _num(100 * hk["ci"][0], 2),
        "hk_hi": _num(100 * hk["ci"][1], 2), "hk_truth": _num(hk["truth_pp"], 2), "hk_placebo": _num(100 * hk["placebo"], 2),
        "fc_best": names[fc["best"]], "fc_wape": _pct(fc["score"].iloc[0]["wape"], 1),
        "fc_naive_wape": _pct(fc["score"].loc["seasonal_naive", "wape"], 1),
        "fc_floor": _pct(0.8 / np.sqrt(fc["series"]["signups"].tail(365).mean()), 1),
        "cov_hw": _pct(fc["coverage"]["holt_winters"][0], 0), "cov_rg": _pct(fc["coverage"]["ridge_calendar"][0], 0),
        "fc_total": _rub(fc["final"]["forecast"].sum()), "fc_lo": _rub(fc["final"]["lo"].sum()), "fc_hi": _rub(fc["final"]["hi"].sum()),
        "ny_max_z": _num(ny["z"].max(), 1), "sig_eval": an["signup_eval"],
        "cusum_fp": _cusum_fp(an),
    }


def render(res, charts_dir, out_path, template_dir=None):
    env = Environment(loader=FileSystemLoader(str(template_dir or ROOT / "report")),
                      autoescape=select_autoescape(["html", "j2"]))
    tpl = env.get_template("template.html.j2")
    ctx = build_context(res)
    html = tpl.render(img=_img_factory(charts_dir), **ctx)
    Path(out_path).write_text(html, encoding="utf-8")
    return Path(out_path)
