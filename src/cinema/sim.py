# симуляция мира подписочного кинотеатра, помесячно и понедельно, без единого цикла по людям
# главная идея: все случайные числа заготавливаются заранее и не зависят от состояния мира,
# поэтому можно прогнать мир трижды (как есть, без повышения цены, без оффера) на одних и тех же
# случайностях и получить точный эффект, то есть настоящий ответ, с которым потом сверяется анализ

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from . import config as C

# ---- коэффициенты риска ухода, в логарифмах ----
PLAN_B = np.array([0.25, 0.0, -0.25])
CHANNEL_B = {"organic": 0.0, "search": 0.08, "social": 0.40, "referral": -0.35, "partner": 0.18}
DEVICE_B = {"tv": -0.15, "mobile": 0.05, "web": 0.10}
REGION_B = {"Moscow": 0.0, "SPb": -0.03, "Central": 0.02, "Volga": 0.04, "South": -0.02,
            "Ural": 0.0, "Siberia": 0.05, "FarEast": -0.04}
ENG_B = -0.45                  # риск падает с ростом базовой вовлеченности
SUMMER_B = 0.15
H0 = 0.0185                    # базовый недельный риск начала спада интереса
H_INV = 0.0035                 # недельный риск ухода из за неудачной оплаты
P_REN_BASE = 0.55              # для годовых: вероятность спада перед продлением
P_UP, P_DOWN = 0.020, 0.012
P_REACT = 0.10
EXP_WAVES = 4
EXP_FLAG_RATIO = 0.70
ACT_CONV = (0.20, 0.72)        # вероятность оплаты после триала: не активировался, активировался
CHANNEL_CONV = {"organic": 1.0, "search": 1.0, "social": 0.85, "referral": 1.15, "partner": 1.12}

CHANNEL_MIX_START = np.array([0.34, 0.26, 0.14, 0.10, 0.16])
CHANNEL_MIX_END = np.array([0.26, 0.22, 0.28, 0.10, 0.14])
PLAN_P = {"default": [0.35, 0.45, 0.20], "partner": [0.55, 0.35, 0.10], "referral": [0.28, 0.45, 0.27]}


def tenure_mult(k):
    return np.where(k < 4, 2.2, np.where(k < 8, 1.5, np.where(k < 16, 1.15, 0.85)))


@dataclass
class Population:
    n: int
    signup_day: np.ndarray
    sw: np.ndarray
    channel: np.ndarray
    region: np.ndarray
    device: np.ndarray
    persona: np.ndarray
    e_base: np.ndarray
    plan0: np.ndarray
    annual: np.ndarray
    utype: np.ndarray
    activated: np.ndarray
    treated_region: np.ndarray
    beta_static: np.ndarray
    p_conv: np.ndarray


class Draws:
    # все случайные числа мира, заготовленные заранее
    def __init__(self, seed, n):
        rng = np.random.default_rng(seed + 1000)
        f = np.float32
        self.u_conv = rng.random(n, dtype=f)
        shape = (n, C.NW)
        self.noise = rng.standard_normal(shape, dtype=f)
        for name in ["u_inv", "u_onset", "u_comp", "u_up", "u_down", "u_react", "u_rdelay", "u_treat"]:
            setattr(self, name, rng.random(shape, dtype=f))


def _daily_intensity():
    d = np.arange(C.DAYS)
    dates = [C.START + timedelta(days=int(i)) for i in d]
    growth = 1.0 + 0.5 * d / C.DAYS
    dow = np.array([1.0, 0.95, 0.95, 1.0, 1.1, 1.25, 1.2])[[x.weekday() for x in dates]]
    summer = np.array([0.9 if x.month in (6, 7, 8) else 1.0 for x in dates])
    spike = np.ones(C.DAYS)
    bf = (d >= 333) & (d <= 336)              # чёрная пятница
    ny = (d >= 362) & (d <= 371)              # новый год
    spike[bf] = 2.2
    spike[ny] = 1.5
    return growth * dow * summer * spike, bf | ny


def make_population(seed, n_users):
    rng = np.random.default_rng(seed)
    lam, campaign = _daily_intensity()
    lam = lam * (n_users / lam.sum())
    counts = rng.poisson(lam)
    day = np.repeat(np.arange(C.DAYS), counts)
    n = len(day)
    sw = day // 7

    # каналы: доля соцсетей растет со временем, в кампании добавляются партнеры и соцсети
    frac = (day / C.DAYS)[:, None]
    mix = CHANNEL_MIX_START[None, :] * (1 - frac) + CHANNEL_MIX_END[None, :] * frac
    boost = campaign[day]
    mix[:, 2] *= np.where(boost, 1.4, 1.0)
    mix[:, 4] *= np.where(boost, 1.5, 1.0)
    mix = mix / mix.sum(axis=1, keepdims=True)
    channel = (rng.random(n)[:, None] > np.cumsum(mix, axis=1)).sum(axis=1)
    channel = np.minimum(channel, len(C.CHANNELS) - 1)

    region = rng.choice(len(C.REGIONS), size=n, p=C.REGION_P)
    device = rng.choice(len(C.DEVICES), size=n, p=C.DEVICE_P)
    persona = rng.choice(len(C.PERSONAS), size=n, p=C.PERSONA_P)
    chan_e = np.array([1.0, 1.0, 0.9, 1.1, 0.85])[channel]
    e_base = 4.0 * np.array(C.PERSONA_HOURS)[persona] * rng.lognormal(0, 0.35, n) * chan_e

    plan0 = np.zeros(n, dtype=np.int8)
    u = rng.random(n)
    for ci, ch in enumerate(C.CHANNELS):
        p = np.array(PLAN_P.get(ch, PLAN_P["default"]))
        m = channel == ci
        plan0[m] = (u[m][:, None] > np.cumsum(p)[None, :]).sum(axis=1).clip(0, 2)
    annual = rng.random(n) < np.where(plan0 >= 1, 0.15, 0.08)

    le = np.log(e_base / 4.0)
    logits = np.stack([
        0.0 + 0.6 * (plan0 == 0) + 0.2 * (device == 1),
        0.2 + 0.6 * le,
        0.1 - 0.5 * le + 0.3 * (channel == 2),
        -0.6 + 0.7 * (plan0 == 2) + 0.5 * annual,
    ], axis=1)
    pr = np.exp(logits - logits.max(axis=1, keepdims=True))
    pr /= pr.sum(axis=1, keepdims=True)
    utype = (rng.random(n)[:, None] > np.cumsum(pr, axis=1)).sum(axis=1).clip(0, 3)

    p_act = 1 / (1 + np.exp(-(0.2 + 1.1 * le)))
    activated = rng.random(n) < p_act
    base_conv = np.where(activated, ACT_CONV[1], ACT_CONV[0])
    p_conv = np.minimum(base_conv * np.array([CHANNEL_CONV[c] for c in C.CHANNELS])[channel], 0.95)
    treated_region = np.isin(np.array(C.REGIONS)[region], C.TREATED_REGIONS)

    beta = (ENG_B * le
            + np.array([CHANNEL_B[c] for c in C.CHANNELS])[channel]
            + np.array([DEVICE_B[d] for d in C.DEVICES])[device]
            + np.array([REGION_B[r] for r in C.REGIONS])[region])
    return Population(n, day, sw, channel, region, device, persona, e_base, plan0, annual, utype,
                      activated, treated_region, beta, p_conv)


def _calendar():
    dates = [C.START + timedelta(weeks=i) for i in range(C.NW)]
    summer = np.array([SUMMER_B if d.month in (6, 7, 8) else 0.0 for d in dates])
    doy = np.array([d.timetuple().tm_yday for d in dates])
    eng = 1.0 + 0.12 * np.cos(2 * np.pi * (doy - 15) / 365)
    return summer, eng


def simulate(pop, dr, hike_on=True, offer_on=True):
    n, nw = pop.n, C.NW
    summer, eng_season = _calendar()
    ST = np.zeros((n, nw), np.int8)
    PL = np.zeros((n, nw), np.int8)
    PH = np.zeros((n, nw), np.int8)
    EV = np.zeros((n, nw), np.int8)         # событие в конце недели: 1 триал не оплачен, 2 уход, 3 неоплата, 4 апгрейд, 5 даунгрейд
    HR = np.zeros((n, nw), np.float32)

    stat = np.ones(n, np.int8)               # 1 триал, 2 платит, 3 ушел
    k = np.zeros(n, np.int16)
    phase = np.zeros(n, np.int8)
    ucomp = np.ones(n, np.float32)
    plan = pop.plan0.copy()
    react_at = np.full(n, -1, np.int16)
    react_used = np.zeros(n, bool)
    c_eff = np.array(C.COMPLETE_BASE)[pop.utype].astype(np.float32)
    c_offer = np.array(C.COMPLETE_OFFER)[pop.utype].astype(np.float32)

    exp_week = np.full(n, -1, np.int16)
    exp_treat = np.full(n, -1, np.int8)
    exp_ratio = np.full(n, np.nan, np.float32)
    exp_phase = np.full(n, -1, np.int8)
    annual = pop.annual
    obs_end = nw - pop.sw
    pf = np.array(C.PHASE_FACTOR, dtype=np.float32)
    log_h0 = np.log(H0)

    for t in range(nw):
        cw = np.minimum(pop.sw + t, nw - 1)
        obs = t < obs_end

        # возвращение ушедших (реактивация)
        re = (stat == 3) & (react_at == t) & obs
        stat[re] = 2
        k[re] = 0
        phase[re] = 0

        # конец триала
        if t == C.TRIAL_WEEKS:
            tr = stat == 1
            conv = tr & (dr.u_conv < pop.p_conv)
            lost = tr & ~conv
            stat[conv] = 2
            k[conv] = 0
            stat[lost] = 3
            EV[lost & (obs_end > t - 1), t - 1] = 1

        # эксперимент: оффер показываем тем, у кого заметно упала вовлеченность
        wk = pop.sw + t
        in_wave = (wk >= C.EXP_WEEK) & (wk < C.EXP_WEEK + EXP_WAVES)
        if t >= 8 and in_wave.any():
            cand = np.where(in_wave & (stat == 2) & ~annual & (k >= 4) & (exp_week == -1) & obs)[0]
            if len(cand):
                recent = HR[cand, t - 2:t].mean(axis=1)
                before = HR[cand, t - 8:t - 2].mean(axis=1)
                ratio = recent / (before + 1e-6)
                flag = ratio < EXP_FLAG_RATIO
                chosen = cand[flag]
                exp_week[chosen] = t
                exp_ratio[chosen] = ratio[flag]
                exp_phase[chosen] = phase[chosen]
                tr_flag = dr.u_treat[chosen, t] < 0.5
                exp_treat[chosen] = tr_flag.astype(np.int8)
                if offer_on:
                    to_offer = chosen[tr_flag]
                    c_eff[to_offer] = c_offer[to_offer]

        # состояние недели и вовлеченность
        trial = (stat == 1) & obs
        paid = (stat == 2) & obs
        ST[trial, t] = 1
        ST[paid, t] = 2
        PL[trial | paid, t] = plan[trial | paid]
        PH[paid, t] = phase[paid]
        if t == 0:
            nov = np.where(pop.activated, 1.3, 0.4)
        elif t == 1:
            nov = np.where(pop.activated, 1.15, 0.5)
        else:
            nov = np.ones(n)
        fac = np.where(stat == 1, nov, pf[phase])
        hrs = pop.e_base * eng_season[cw] * np.exp(0.18 * dr.noise[:, t]) * fac
        live = trial | paid
        HR[live, t] = hrs[live].astype(np.float32)

        # события конца недели для платящих
        inv = paid & ~annual & (dr.u_inv[:, t] < H_INV)
        stat[inv] = 3
        EV[inv, t] = 3

        phase_before = phase.copy()
        p3 = paid & ~inv & (phase_before == 3)
        quit_v = p3 & (ucomp < c_eff)
        stat[quit_v] = 3
        EV[quit_v, t] = 2
        rr = quit_v & ~react_used & (dr.u_react[:, t] < P_REACT)
        react_at[rr] = (t + 1 + 6 + np.floor(dr.u_rdelay[rr, t] * 15)).astype(np.int16)
        react_used[rr] = True
        recovered = p3 & ~quit_v
        phase[recovered] = 0
        mid = paid & ~inv & ((phase_before == 1) | (phase_before == 2))
        phase[mid] += 1

        # начало спада интереса
        can = paid & ~inv & (phase_before == 0)
        hike = (pop.treated_region & (cw >= C.HIKE_WEEK) & hike_on) * C.HIKE_LOG_HAZARD
        beta = pop.beta_static + PLAN_B[plan] + summer[cw] + hike
        h_month = np.clip(H0 * tenure_mult(k) * np.exp(beta), 0, 0.6)
        p_ren = np.clip(P_REN_BASE * np.exp(beta), 0, 0.95)
        at_renewal = (k % 52) == 48
        p_on = np.where(annual, np.where(at_renewal, p_ren, 0.0), h_month)
        onset = can & (dr.u_onset[:, t] < p_on)
        phase[onset] = 1
        ucomp[onset] = dr.u_comp[onset, t]

        # смена тарифа только на границе платежного цикла
        boundary = ((k + 1) % C.CYCLE_WEEKS == 0)
        elig = paid & ~inv & ~quit_v & ~annual & (phase_before == 0) & ~onset & boundary
        lev = np.clip((pop.e_base / 4.0), 0.3, 3.0)
        up = elig & (plan < 2) & (dr.u_up[:, t] < P_UP * lev ** 0.8)
        down = elig & ~up & (plan > 0) & (dr.u_down[:, t] < P_DOWN * lev ** -0.5)
        EV[up, t] = 4
        EV[down, t] = 5
        plan[up] += 1
        plan[down] -= 1

        k[paid & (stat == 2)] += 1

    return {"ST": ST, "PL": PL, "PH": PH, "EV": EV, "HR": HR, "exp_week": exp_week,
            "exp_treat": exp_treat, "exp_ratio": exp_ratio, "exp_phase": exp_phase}
