import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from _world import small
from cinema import survival as sv


def test_kaplan_meier_hand_example():
    # пять человек: уходы на 2, 3 и 5 неделе, двое выбывают на 3 и 6 (без ухода)
    t = np.array([2, 3, 3, 5, 6])
    e = np.array([1, 1, 0, 1, 0])
    km = sv.kaplan_meier(t, e).set_index("time")
    # на неделе 2 в риске 5 человек, один ушел: 4/5
    assert abs(km.loc[2, "surv"] - 0.8) < 1e-12
    # на неделе 3 в риске 4, один ушел (второй цензурирован): 0.8 * 3/4
    assert abs(km.loc[3, "surv"] - 0.6) < 1e-12
    # на неделе 5 остался один из двух: 0.6 * 1/2
    assert abs(km.loc[5, "surv"] - 0.3) < 1e-12


def test_kaplan_meier_interval_inside_zero_one():
    rng = np.random.default_rng(1)
    t = rng.exponential(20, 500)
    e = (rng.random(500) < 0.7).astype(int)
    km = sv.kaplan_meier(t, e)
    assert (km["lo"] >= 0).all() and (km["hi"] <= 1.0000001).all()
    assert (km["lo"] <= km["surv"] + 1e-12).all() and (km["hi"] >= km["surv"] - 1e-12).all()
    assert (np.diff(km["surv"]) <= 1e-12).all()


def test_logrank_matches_scipy_two_groups():
    rng = np.random.default_rng(2)
    a = rng.exponential(10, 300)
    b = rng.exponential(14, 300)
    ea = (rng.random(300) < 0.8).astype(int)
    eb = (rng.random(300) < 0.8).astype(int)
    ours = sv.logrank(np.r_[a, b], np.r_[ea, eb], np.r_[np.zeros(300), np.ones(300)])
    ref = stats.logrank(stats.CensoredData(uncensored=a[ea == 1], right=a[ea == 0]),
                        stats.CensoredData(uncensored=b[eb == 1], right=b[eb == 0]))
    assert abs(ours["chi2"] - ref.statistic ** 2) < 1e-6
    assert abs(ours["p"] - ref.pvalue) < 1e-9


def test_logrank_same_distribution_is_not_significant():
    rng = np.random.default_rng(3)
    t = rng.exponential(10, 1200)
    g = rng.integers(0, 3, 1200)
    r = sv.logrank(t, np.ones(1200, int), g)
    assert r["df"] == 2 and r["p"] > 0.01


def test_fit_logit_matches_sklearn():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(4000, 3))
    p = 1 / (1 + np.exp(-(-1 + X @ np.array([0.8, -0.5, 0.3]))))
    y = (rng.random(4000) < p).astype(float)
    Z = np.column_stack([np.ones(4000), X])
    b, se, ll = sv.fit_logit(Z, y)
    sk = LogisticRegression(C=1e10, max_iter=2000).fit(X, y)
    assert np.allclose(b[1:], sk.coef_[0], atol=1e-3)
    assert abs(b[0] - sk.intercept_[0]) < 1e-3
    assert (se > 0).all()


def test_episodes_merge_plan_changes_and_censor():
    w = small()
    ep, spells = sv.episodes(w["conn"])
    assert ep["user_id"].is_unique
    assert (ep["weeks"] > 0).all()
    # у живых и у смены тарифа в конце данных конец равен границе наблюдения
    alive = ep[~ep["last_reason"].isin(["voluntary", "payment"])]
    assert (alive["end_obs"] == sv.OBS_END).all()
    # событие и конец согласованы
    assert ((ep["event_voluntary"] == 1) == (ep["last_reason"] == "voluntary")).all()


def test_person_period_rows_equal_weeks_and_one_event_per_leaver():
    w = small()
    ep, spells = sv.episodes(w["conn"])
    m = ep[(ep["billing"] == "monthly") & (ep["region"] != "unknown")].head(800).reset_index(drop=True)
    pp = sv.person_period(m, spells, 0)
    assert len(pp) == int(np.ceil(m["weeks"]).sum())
    assert int(pp["event"].sum()) == int(m["event_voluntary"].sum())
    # событие стоит только в последней неделе человека
    last = pp.groupby("row")["tenure"].transform("max") == pp["tenure"]
    assert (pp.loc[pp["event"] == 1, "tenure"].index.isin(pp.index[last])).all()


def test_hazard_model_recovers_direction():
    w = small()
    ep, spells = sv.episodes(w["conn"])
    m = ep[(ep["billing"] == "monthly") & (ep["region"] != "unknown")]
    res, info = sv.hazard_model(m, spells, 4)
    r = res.set_index("term")
    assert r.loc["social", "coef"] > 0.15 and r.loc["premium", "coef"] < -0.05
    assert r.loc["engagement_per_log_hour", "coef"] < -0.2
    assert info["n_events"] > 500
