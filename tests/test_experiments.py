import numpy as np
import pandas as pd

from _world import small
from cinema import experiments as ex


def _toy(n=4000, p0=0.30, effect=-0.08, seed=1):
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, 2, n)
    y = (rng.random(n) < p0 + effect * treat).astype(int)
    return pd.DataFrame({"treat": treat, "y": y, "retained": 1 - y, "x": rng.normal(size=n)})


def test_srm_balanced_and_unbalanced():
    ok = ex.srm_check(pd.DataFrame({"treat": [0] * 500 + [1] * 505}))
    bad = ex.srm_check(pd.DataFrame({"treat": [0] * 500 + [1] * 650}))
    assert ok["p"] > 0.5 and bad["p"] < 0.001


def test_itt_finds_known_effect():
    d = _toy()
    r = ex.itt(d, n_boot=500, n_perm=500)
    assert abs(r["diff"] - (-0.08)) < 0.03
    assert r["ci_boot"][0] < r["diff"] < r["ci_boot"][1]
    assert r["perm_p"] < 0.01 and r["z_p"] < 0.01


def test_itt_no_effect_not_significant():
    d = _toy(effect=0.0, seed=5)
    r = ex.itt(d, n_boot=300, n_perm=300)
    assert r["z_p"] > 0.01


def test_power_formula_known_value():
    # классический ответ: 30% против 25% при альфа 0,05 и мощности 0,8 это примерно 1250 на группу
    _, tbl = ex.power_table(0.30, 1000, effects=(0.05,))
    n = int(tbl["n_per_arm"].iloc[0])
    assert 1200 < n < 1400


def test_adjusted_effect_close_to_raw_and_hc_se_positive():
    d = _toy()
    a = ex.adjusted_effect(d, cols=["x"])
    assert abs(a["diff"] - (-0.08)) < 0.03 and a["se"] > 0


def test_qini_perfect_ranking_beats_random():
    rng = np.random.default_rng(3)
    n = 6000
    treat = rng.integers(0, 2, n)
    responsive = rng.random(n) < 0.3
    # отзывчивые люди с оффером остаются почти всегда, без оффера почти никогда
    good = np.where(responsive, np.where(treat == 1, rng.random(n) < 0.9, rng.random(n) < 0.1), rng.random(n) < 0.5).astype(int)
    x_p, q_p = ex.qini_curve(good, treat, responsive.astype(float) + rng.random(n) * 0.01)
    x_r, q_r = ex.qini_curve(good, treat, rng.random(n))
    assert ex.qini_area(x_p, q_p) > 5 * abs(ex.qini_area(x_r, q_r)) + 50
    # линия случайного выбора начинается в нуле
    assert q_p[0] == 0


def test_did_on_synthetic_panel_recovers_effect_and_placebo_is_flat():
    rng = np.random.default_rng(8)
    weeks = np.arange(12, 104)
    base = 0.012 + 0.002 * np.sin(weeks / 8)
    treated = base + rng.normal(0, 0.0008, len(weeks)) + np.where(weeks >= 61, 0.004, 0)
    control = base + rng.normal(0, 0.0008, len(weeks))
    p = pd.DataFrame({"treated": treated, "control": control}, index=pd.Index(weeks, name="week_idx"))
    p["diff"] = p["treated"] - p["control"]
    naive, did = ex.did_estimate(p)
    assert abs(did - 0.004) < 0.0006
    lo, hi = ex.did_bootstrap(p, n_boot=400)
    assert lo < did < hi
    assert abs(ex.placebo_did(p)) < 0.0008


def test_load_experiment_matches_assignment_and_outcome_rate():
    w = small()
    d, dropped = ex.load_experiment(w["conn"])
    assert dropped == 0
    assert len(d) == len(w["tables"]["experiment_assignment"])
    # исход сравнивается с настоящим эффектом по знаку
    diff = d.loc[d["treat"] == 1, "y"].mean() - d.loc[d["treat"] == 0, "y"].mean()
    assert diff < 0.02
