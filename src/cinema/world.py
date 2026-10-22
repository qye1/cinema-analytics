# из матриц симуляции собираем настоящие таблицы: пользователи, подписки, платежи, просмотры, тикеты и так далее
# затем портим данные известным образом и считаем настоящие эффекты по контрфактическим мирам

from datetime import timedelta

import numpy as np
import pandas as pd

from . import config as C
from . import sim

CUTOFF = pd.Timestamp(C.END)


def _d(day_offsets, base=C.START):
    return pd.Timestamp(base) + pd.to_timedelta(day_offsets, unit="D")


def build_users(pop):
    u = pd.DataFrame({
        "user_id": np.arange(1, pop.n + 1),
        "signup_date": _d(pop.signup_day),
        "channel": np.array(C.CHANNELS)[pop.channel],
        "region": np.array(C.REGIONS)[pop.region],
        "device": np.array(C.DEVICES)[pop.device],
        "plan_at_signup": np.array(C.PLANS)[pop.plan0],
        "billing": np.where(pop.annual, "annual", "monthly"),
        "activated_3d": pop.activated.astype(int),
    })
    return u


def build_subscriptions(pop, w):
    ST, PL, EV = w["ST"], w["PL"], w["EV"]
    u, t = np.nonzero(ST == 2)
    plan = PL[u, t]
    first = np.ones(len(u), bool)
    first[1:] = (u[1:] != u[:-1]) | (t[1:] != t[:-1] + 1) | (plan[1:] != plan[:-1])
    starts = np.where(first)[0]
    ends = np.r_[starts[1:] - 1, len(u) - 1]
    su, st_t, last_t, sp = u[starts], t[starts], t[ends], plan[starts]
    ev = EV[su, last_t]
    obs_end = C.NW - pop.sw[su]
    active = (ev == 0) & (last_t == obs_end - 1)
    reason = np.select([ev == 2, ev == 3, (ev == 4) | (ev == 5), active],
                       ["voluntary", "payment", "plan_change", "active"], default="unknown")
    start_date = _d(pop.signup_day[su] + 7 * st_t)
    end_date = _d(pop.signup_day[su] + 7 * (last_t + 1))
    end_date = pd.Series(end_date).where(~active, pd.NaT)
    subs = pd.DataFrame({
        "spell_id": np.arange(1, len(su) + 1), "user_id": su + 1,
        "plan": np.array(C.PLANS)[sp], "billing": np.where(pop.annual[su], "annual", "monthly"),
        "start_date": start_date, "end_date": end_date.to_numpy(), "end_reason": reason,
        "_start_t": st_t, "_last_t": last_t,
    })
    return subs


def price_at(plan_idx, treated, dates):
    base = np.array([C.PRICE[p] for p in C.PLANS])[plan_idx].astype(float)
    hiked = treated & (dates >= pd.Timestamp(C.HIKE_DATE))
    return np.round(np.where(hiked, base * C.HIKE_FACTOR, base))


def build_payments(pop, subs, rng):
    s = subs
    uid = s["user_id"].to_numpy() - 1
    length = (s["_last_t"] - s["_start_t"] + 1).to_numpy()
    annual = pop.annual[uid]
    step = np.where(annual, 52, C.CYCLE_WEEKS)
    n_pay = np.ceil(length / step).astype(int)
    pay_involuntary = (s["end_reason"] == "payment").to_numpy()
    # у ушедших из за неоплаты последняя неделя занята неудачными попытками, обычный платеж туда не ставим
    off_last = (n_pay - 1) * step
    drop_last = pay_involuntary & (off_last == length - 1)
    n_pay = np.where(drop_last, n_pay - 1, n_pay)

    rep = np.repeat(np.arange(len(s)), n_pay)
    pos = np.arange(rep.size) - np.repeat(np.cumsum(n_pay) - n_pay, n_pay)
    week = s["_start_t"].to_numpy()[rep] + pos * step[rep]
    pdate = _d(pop.signup_day[uid[rep]] + 7 * week)
    plan_i = pd.Categorical(s["plan"], categories=C.PLANS).codes[rep]
    amount = price_at(plan_i, pop.treated_region[uid[rep]], pdate)
    amount = np.where(annual[rep], amount * C.ANNUAL_MONTHS, amount)

    inc = np.isin(pdate.normalize(), pd.to_datetime(C.INCIDENT_DAYS))
    p_fail = np.where(inc, C.INCIDENT_FAIL_P, 0.03)
    fail = rng.random(rep.size) < p_fail
    retry1_ok = rng.random(rep.size) < 0.65

    base = pd.DataFrame({"user_id": uid[rep] + 1, "pay_date": pdate, "amount": amount, "plan_i": plan_i})
    ok_first = base[~fail].assign(status="success", attempt=1)
    f = base[fail]
    r1 = retry1_ok[fail]
    failed1 = f.assign(status="failed", attempt=1)
    ok2 = f[r1].assign(pay_date=f[r1]["pay_date"] + pd.Timedelta(days=3), status="success", attempt=2)
    failed2 = f[~r1].assign(pay_date=f[~r1]["pay_date"] + pd.Timedelta(days=3), status="failed", attempt=2)
    ok3 = f[~r1].assign(pay_date=f[~r1]["pay_date"] + pd.Timedelta(days=6), status="success", attempt=3)

    # неоплата: три неудачные попытки в последнюю неделю подписки
    ip = np.where(pay_involuntary)[0]
    rows = []
    for off, att in ((0, 1), (3, 2), (6, 3)):
        d0 = _d(pop.signup_day[uid[ip]] + 7 * (s["_last_t"].to_numpy()[ip])) + pd.Timedelta(days=off)
        pi = pd.Categorical(s["plan"].iloc[ip], categories=C.PLANS).codes
        am = price_at(pi, pop.treated_region[uid[ip]], d0)
        rows.append(pd.DataFrame({"user_id": uid[ip] + 1, "pay_date": d0, "amount": am,
                                  "plan_i": pi, "status": "failed", "attempt": att}))
    pay = pd.concat([ok_first, failed1, ok2, failed2, ok3] + rows, ignore_index=True)
    pay = pay.sort_values(["pay_date", "user_id", "attempt"], kind="stable").reset_index(drop=True)
    pay["payment_id"] = np.arange(1, len(pay) + 1)
    pay["plan"] = np.array(C.PLANS)[pay["plan_i"].to_numpy()]
    pay = pay[pay["pay_date"] < CUTOFF]
    return pay[["payment_id", "user_id", "pay_date", "amount", "status", "attempt", "plan"]].reset_index(drop=True)


def build_engagement(pop, w, rng):
    HR, ST = w["HR"], w["ST"]
    u, t = np.nonzero(HR > 0)
    hours = HR[u, t].astype(float)
    per = pop.persona[u]
    hps = np.array(C.PERSONA_HPS)[per]
    ses = rng.poisson(hours / hps)
    tit = rng.poisson(ses * np.array(C.PERSONA_TPS)[per])
    wk = np.clip(rng.normal(np.array(C.PERSONA_WEEKEND)[per], 0.08), 0, 1)
    eng = pd.DataFrame({
        "user_id": u + 1, "week_start": _d(7 * (pop.sw[u] + t)),
        "hours": hours.round(2), "sessions": ses, "titles": tit, "weekend_share": wk.round(3),
        "_u": u, "_t": t,
    })
    return eng


def build_tickets(pop, w, subs, rng):
    HR, PH, ST = w["HR"], w["PH"], w["ST"]
    nw = C.NW
    cw = pop.sw[:, None] + np.arange(nw)[None, :]
    live = ST > 0
    base_l = np.where(live, 0.015, 0.0)
    decay_l = np.where(PH >= 2, 0.05, 0.0)
    price_l = np.where(live & pop.treated_region[:, None] & (cw >= C.HIKE_WEEK), 0.012, 0.0)
    out = []
    cats = {"base": (["technical", "content", "billing", "other"], [0.40, 0.25, 0.25, 0.10], base_l),
            "decay": (["content", "cancel_request"], [0.5, 0.5], decay_l),
            "price": (["price"], [1.0], price_l)}
    for name, (cl, cp, lam) in cats.items():
        cnt = rng.poisson(lam)
        u, t = np.nonzero(cnt)
        rep = cnt[u, t]
        u, t = np.repeat(u, rep), np.repeat(t, rep)
        c = rng.choice(cl, size=len(u), p=cp)
        out.append(pd.DataFrame({"user_id": u + 1, "week_i": t, "u": u, "category": c}))
    pay_churn = subs[subs["end_reason"] == "payment"]
    for _ in range(2):
        pick = pay_churn[rng.random(len(pay_churn)) < 0.5]
        out.append(pd.DataFrame({"user_id": pick["user_id"].to_numpy(), "week_i": pick["_last_t"].to_numpy(),
                                 "u": pick["user_id"].to_numpy() - 1, "category": "billing"}))
    tk = pd.concat(out, ignore_index=True)
    tk["created_date"] = _d(pop.signup_day[tk["u"].to_numpy()] + 7 * tk["week_i"].to_numpy()
                            + rng.integers(0, 7, len(tk)))
    tk = tk[tk["created_date"] < CUTOFF].sort_values(["created_date", "user_id"]).reset_index(drop=True)
    tk["ticket_id"] = np.arange(1, len(tk) + 1)
    return tk[["ticket_id", "user_id", "created_date", "category"]]


def build_marketing(users, rng):
    signups = users.groupby(["signup_date", "channel"]).size().rename("n").reset_index()
    signups["spend"] = signups["n"] * signups["channel"].map(C.SPEND_PER_SIGNUP) * rng.lognormal(0, 0.12, len(signups))
    signups = signups.rename(columns={"signup_date": "date"})
    return signups[signups["spend"] > 0][["date", "channel", "spend"]].round({"spend": 2}).reset_index(drop=True)


def build_experiment(pop, w):
    m = w["exp_week"] >= 0
    u = np.where(m)[0]
    t = w["exp_week"][u]
    return pd.DataFrame({
        "user_id": u + 1, "experiment": C.EXP_NAME,
        "variant": np.where(w["exp_treat"][u] == 1, "offer", "control"),
        "assigned_date": _d(pop.signup_day[u] + 7 * t),
        "hours_ratio": w["exp_ratio"][u].round(3),
    })


def build_price_list():
    rows = []
    for r in C.REGIONS:
        for i, p in enumerate(C.PLANS):
            rows.append((r, p, C.START.isoformat(), C.PRICE[p]))
            if r in C.TREATED_REGIONS:
                rows.append((r, p, C.HIKE_DATE.isoformat(), round(C.PRICE[p] * C.HIKE_FACTOR)))
    return pd.DataFrame(rows, columns=["region", "plan", "effective_from", "price"])


# ---------- грязь ----------

def inject_dirt(tables, rng):
    t = {k: v.copy() for k, v in tables.items()}
    inj = {}
    pay = t["payments"]
    # успешные платежи с нулевой суммой
    ok_idx = pay.index[(pay["status"] == "success")]
    z = rng.choice(ok_idx, size=40, replace=False)
    pay.loc[z, "amount"] = 0.0
    touched = set(z.tolist())
    inj["zero_amount_success"] = 40
    # платеж раньше даты регистрации
    users = t["users"].set_index("user_id")
    cand = pay.index[pay["status"] == "success"]
    pick = rng.choice(cand, size=30, replace=False)
    sd = users.loc[pay.loc[pick, "user_id"], "signup_date"].to_numpy()
    pay.loc[pick, "pay_date"] = pd.to_datetime(sd) - pd.Timedelta(days=5)
    touched |= set(pick.tolist())
    inj["payment_before_signup"] = 30
    # дубли платежей: тот же человек, дата, сумма и попытка под другим номером
    n = int(round(0.003 * len(pay)))
    # дубли берем только из нетронутых строк, чтобы каждая проблема считалась один раз
    untouched = pay.loc[~pay.index.isin(touched)]
    dup = untouched.sample(n, random_state=int(rng.integers(1 << 30))).copy()
    dup["payment_id"] = np.arange(pay["payment_id"].max() + 1, pay["payment_id"].max() + 1 + len(dup))
    pay = pd.concat([pay, dup], ignore_index=True)
    inj["duplicate_payments"] = int(len(dup))
    t["payments"] = pay.sort_values(["pay_date", "payment_id"], kind="stable").reset_index(drop=True)

    # пустой регион
    u = t["users"]
    mask = rng.random(len(u)) < 0.015
    u.loc[mask, "region"] = None
    inj["null_region"] = int(mask.sum())
    t["users"] = u

    # отрицательные часы просмотра
    e = t["engagement"]
    neg = rng.choice(len(e), size=25, replace=False)
    e.loc[e.index[neg], "hours"] = -e.loc[e.index[neg], "hours"].abs() - 0.1
    inj["negative_hours"] = 25
    t["engagement"] = e

    # пересекающиеся подписки: конец одной заходит на следующую
    s = t["subscriptions"].sort_values(["user_id", "start_date"]).reset_index(drop=True)
    nxt_same = (s["user_id"].shift(-1) == s["user_id"]) & s["end_date"].notna()
    cand = s.index[nxt_same]
    pick = rng.choice(cand, size=min(20, len(cand)), replace=False)
    s.loc[pick, "end_date"] = s.loc[pick + 1, "start_date"].to_numpy() + pd.Timedelta(days=10)
    inj["overlapping_spells"] = int(len(pick))
    t["subscriptions"] = s
    return t, inj


# ---------- истинные эффекты ----------

def weekly_churn_rate(pop, w, user_mask, week_from, week_to):
    ST, EV = w["ST"], w["EV"]
    cw = pop.sw[:, None] + np.arange(C.NW)[None, :]
    sel = (ST == 2) & (cw >= week_from) & (cw <= week_to) & user_mask[:, None] & (~pop.annual[:, None])
    return float((EV == 2)[sel].sum() / sel.sum())


def compute_truth(pop, dr, w):
    w_no_offer = sim.simulate(pop, dr, offer_on=False)
    w_no_hike = sim.simulate(pop, dr, hike_on=False)
    idx = np.where(w["exp_week"] >= 0)[0]
    wk = w["exp_week"][idx]
    tr = w["exp_treat"][idx] == 1
    def churn4(EV):
        return np.array([(EV[u, t:t + C.CHURN_HORIZON_WEEKS] == 2).any() for u, t in zip(idx, wk)]).astype(int)
    y, y0 = churn4(w["EV"]), churn4(w_no_offer["EV"])
    effect = float((y[tr] - y0[tr]).mean())
    by_type = {}
    for i, name in enumerate(C.USER_TYPES):
        s = tr & (pop.utype[idx] == i)
        by_type[name] = {"n_treated": int(s.sum()), "effect": round(float((y[s] - y0[s]).mean()), 4) if s.any() else None}
    reg = pop.treated_region
    post = (C.HIKE_WEEK, C.NW - 1)
    base_t = weekly_churn_rate(pop, w, reg, *post)
    nohike_t = weekly_churn_rate(pop, w_no_hike, reg, *post)
    return {
        "offer": {"n_assigned": int(len(idx)), "n_treated": int(tr.sum()),
                  "true_effect_on_treated_4w_churn": round(effect, 4), "by_type": by_type,
                  "type_share_in_experiment": {n: round(float((pop.utype[idx] == i).mean()), 3)
                                               for i, n in enumerate(C.USER_TYPES)}},
        "hike": {"treated_regions": C.TREATED_REGIONS, "post_weekly_churn_with_hike": round(base_t, 5),
                 "post_weekly_churn_without_hike": round(nohike_t, 5),
                 "true_effect_weekly_churn_pp": round(100 * (base_t - nohike_t), 4)},
    }


def build_world(seed=42, n_users=60000, with_truth=True):
    pop = sim.make_population(seed, n_users)
    dr = sim.Draws(seed, pop.n)
    w = sim.simulate(pop, dr)
    rng = np.random.default_rng(seed + 7)
    users = build_users(pop)
    subs = build_subscriptions(pop, w)
    pay = build_payments(pop, subs, rng)
    eng = build_engagement(pop, w, rng)
    tickets = build_tickets(pop, w, subs, rng)
    tables = {
        "users": users,
        "subscriptions": subs.drop(columns=["_start_t", "_last_t"]),
        "payments": pay,
        "engagement": eng.drop(columns=["_u", "_t"]),
        "tickets": tickets,
        "marketing_spend": build_marketing(users, rng),
        "experiment_assignment": build_experiment(pop, w),
        "price_list": build_price_list(),
    }
    dirty, injected = inject_dirt(tables, rng)
    truth = {
        "config": {"seed": seed, "n_users": int(pop.n), "weeks": C.NW, "start": C.START.isoformat(),
                   "hike_date": C.HIKE_DATE.isoformat(), "exp_date": C.EXP_DATE.isoformat(),
                   "incident_days": [d.isoformat() for d in C.INCIDENT_DAYS]},
        "injected": injected,
        "hazard_ratios": {
            "planted_log_hazard": {"social": sim.CHANNEL_B["social"], "referral": sim.CHANNEL_B["referral"],
                                   "partner": sim.CHANNEL_B["partner"], "search": sim.CHANNEL_B["search"],
                                   "basic": float(sim.PLAN_B[0]), "premium": float(sim.PLAN_B[2]),
                                   "tv": sim.DEVICE_B["tv"], "web": sim.DEVICE_B["web"],
                                   "engagement_per_log_hour": sim.ENG_B, "hike": C.HIKE_LOG_HAZARD,
                                   "summer": sim.SUMMER_B},
        },
    }
    truth_users = pd.DataFrame({"user_id": np.arange(1, pop.n + 1),
                                "persona": np.array(C.PERSONAS)[pop.persona],
                                "utype": np.array(C.USER_TYPES)[pop.utype]})
    if with_truth:
        truth.update(compute_truth(pop, dr, w))
    return dirty, truth, truth_users, {"pop": pop, "w": w}
