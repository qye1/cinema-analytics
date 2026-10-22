# графики блока оттока: качество, калибровка, выигрыш от рассылки, важность признаков, утечки

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from . import style
from .churn import cost_curve
from .style import AQUA, BLUE, INK2, MUTED, ORANGE, YELLOW, plt

LABELS = {"hours_ratio": "часы: последние 2 недели к прошлым 8", "hours_l2": "часы за 2 недели",
          "tenure_weeks": "возраст подписки", "weeks_on_plan": "недель на текущем тарифе",
          "hours_cv8": "разброс часов за 8 недель", "hours_l4": "часы за 4 недели",
          "hours_trend": "наклон часов за 8 недель", "plan_premium": "тариф premium",
          "ch_social": "канал social", "plan_basic": "тариф basic", "dev_tv": "устройство tv",
          "titles_l4": "названий за 4 недели", "sessions_l4": "сессий за 4 недели", "cycle_pos": "позиция в цикле оплаты",
          "failed_pay_90d": "неудачные оплаты за 90 дней", "hike_active": "повышение цены", "summer": "лето",
          "activated": "активация за 3 дня", "dev_web": "устройство web", "ch_search": "канал search",
          "ch_referral": "канал referral", "ch_partner": "канал partner", "zero_weeks_l4": "недель без сессий",
          "tickets_90d": "обращения за 90 дней", "tickets_tech_90d": "тех обращения", "tickets_price_90d": "обращения про цену",
          "cancel_tickets_30d": "заявки на отмену"}


def roc_pr(test, pred, rule_score, out):
    y = test["y"].values
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    series = [("gbm", "градиентный бустинг", BLUE, pred["gbm"]), ("logreg", "логистическая регрессия", ORANGE, pred["logreg"]),
              ("rule", "правило по падению часов", MUTED, rule_score)]
    for key, name, color, p in series:
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=color, label=name)
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, color=color, label=name)
    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[1].axhline(y.mean(), color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].set_title("ROC")
    axes[0].set_xlabel("доля ложных тревог")
    axes[0].set_ylabel("доля пойманных уходящих")
    axes[1].set_title("Точность и полнота")
    axes[1].set_xlabel("полнота")
    axes[1].set_ylabel("точность")
    axes[1].legend(loc="upper right")
    return style.save(fig, out)


def gains_and_calibration(gains, gains_lr, gains_rule, calib, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    x0 = np.r_[0, gains["top_share"]]
    for g, name, color in [(gains, "градиентный бустинг", BLUE), (gains_lr, "логистическая регрессия", ORANGE),
                           (gains_rule, "правило по падению часов", MUTED)]:
        ax.plot(x0, np.r_[0, g["captured"]], color=color, label=name)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"верхние 10% ловят {gains['captured'].iloc[0]:.0%} уходов", xy=(0.1, gains["captured"].iloc[0]),
                xytext=(0.04, 0.9), fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("доля подписчиков, отсортированных по риску")
    ax.set_ylabel("доля всех уходов, которую поймали")
    ax.set_title("Кривая выигрыша")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(calib["mean_pred"], calib["observed"], color=BLUE, marker="o")
    lo = float(min(calib["mean_pred"].min(), calib["observed"].min()) * 0.8)
    hi = float(max(calib["mean_pred"].max(), calib["observed"].max()) * 1.2)
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    # на логарифмической шкале видны и спокойные децили, и верхний
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("предсказанная вероятность, среднее по децилю")
    ax.set_ylabel("доля реально ушедших")
    ax.set_title("Калибровка: риск у самых спокойных занижен")
    return style.save(fig, out)


def importance(imp, out, top=12):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh([LABELS.get(f, f) for f in d["feature"]], d["auc_drop"], xerr=d["std"], color=BLUE,
            error_kw=dict(ecolor=MUTED, capsize=2, linewidth=1))
    ax.set_xlabel("на сколько падает AUC, если перемешать признак")
    ax.set_title("Что важно модели: главное это падение интереса, а не кто человек")
    ax.grid(axis="y", visible=False)
    return style.save(fig, out)


def profit_curves(test, p, price, out, contact_cost=40.0):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = {0.10: MUTED, 0.25: BLUE, 0.40: AQUA}
    for s, c in colors.items():
        cc = cost_curve(test["y"].values, p, price, s, contact_cost)
        ax.plot(cc["top_share"], cc["profit"] / 1e6, color=c, label=f"оффер спасает {s:.0%} уходящих")
        i = cc["profit"].idxmax()
        ax.scatter(cc.loc[i, "top_share"], cc.loc[i, "profit"] / 1e6, color=c, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("кому отправляем оффер: верхние q по риску")
    ax.set_ylabel("прибыль на тестовом периоде, млн руб")
    ax.set_title("Кому слать оффер: точка максимума зависит от доли спасенных, а она неизвестна")
    ax.legend(loc="upper right")
    return style.save(fig, out)


def ablation_and_leakage(ab, lk, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.barh(ab["features"].iloc[::-1], ab["auc"].iloc[::-1], color=BLUE)
    for i, v in enumerate(ab["auc"].iloc[::-1]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("AUC на тесте")
    ax.set_title("Группы признаков: почти весь сигнал в поведении")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    names = ["по времени", "случайно", "с будущим\nпризнаком"]
    colors = [BLUE, YELLOW, ORANGE]
    ax.bar(names, lk["auc"], color=colors, width=0.55)
    for i, v in enumerate(lk["auc"]):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9, color=INK2)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title("Утечка из будущего раздувает AUC, случайный сплит тут почти нет")
    ax.grid(axis="x", visible=False)
    return style.save(fig, out)


def mean_price(test):
    # средняя цена цикла у тестовых подписчиков, из долей тарифов
    basic, premium = test["plan_basic"].mean(), test["plan_premium"].mean()
    return 299 * basic + 799 * premium + 499 * (1 - basic - premium)


def make_all(res, outdir):
    from pathlib import Path
    out = Path(outdir)
    test = res["test"]
    rule = -test["hours_ratio"].fillna(1.0).values       # чем сильнее упали часы, тем выше счет риска
    return [
        roc_pr(test, res["pred"], rule, out / "18_churn_roc_pr.png"),
        gains_and_calibration(res["gains"], res["gains_logreg"], res["gains_rule"], res["calibration"],
                              out / "19_churn_gains_calibration.png"),
        importance(res["importance"], out / "20_churn_importance.png"),
        profit_curves(test, res["pred"]["gbm"], mean_price(test), out / "21_churn_profit.png"),
        ablation_and_leakage(res["ablation"], res["leakage"], out / "22_churn_ablation_leakage.png"),
    ]
