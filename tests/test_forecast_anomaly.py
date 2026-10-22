import numpy as np
import pandas as pd

from cinema import anomaly as an
from cinema import forecast as fc


def _series(n=420, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    lam = (60 + 0.05 * t) * np.array([1.0, 0.95, 0.97, 1.0, 1.15, 1.3, 1.25])[t % 7]
    y = rng.poisson(lam)
    days = pd.date_range("2024-01-01", periods=n)
    return pd.DataFrame({"day": days.strftime("%Y-%m-%d"), "signups": y}), lam


def test_holt_winters_recovers_synthetic_pattern():
    rng = np.random.default_rng(1)
    n = 300
    t = np.arange(n)
    seas = np.array([5, 2, 0, -1, -3, 4, 6.0])
    y = 50 + 0.1 * t + seas[t % 7] + rng.normal(0, 1, n)
    m = fc.hw_fit(y)
    f = fc.hw_forecast(m, 14)
    tt = np.arange(n, n + 14)
    truth = 50 + 0.1 * tt + seas[tt % 7]
    assert np.abs(f - truth).mean() < 1.0


def test_seasonal_naive_repeats_last_week():
    y = np.arange(1, 22).astype(float)
    f = fc.seasonal_naive(y, 10)
    assert (f[:7] == y[-7:]).all() and (f[7:] == y[-7:][:3]).all()


def test_backtest_uses_only_the_past():
    s, _ = _series()
    origins = [300, 330]
    a = fc.backtest(s, origins)
    s2 = s.copy()
    s2.loc[330:, "signups"] = s2.loc[330:, "signups"] * 5       # портим будущее после последнего origin
    b = fc.backtest(s2, [300])
    ka = a[(a["origin"] == 300) & (a["horizon"] <= 28)].sort_values(["model", "horizon"])["forecast"].values
    kb = b[(b["origin"] == 300)].sort_values(["model", "horizon"])["forecast"].values
    assert np.allclose(ka, kb)


def test_models_beat_naive_on_stable_series():
    s, _ = _series()
    bt = fc.backtest(s, list(range(250, 392, 14)))
    sc = fc.score(bt)
    assert sc.loc["ridge_calendar", "wape"] < sc.loc["seasonal_naive", "wape"]
    assert sc.loc["holt_winters", "wape"] < sc.loc["seasonal_naive", "wape"]


def test_interval_coverage_close_to_nominal():
    s, _ = _series(n=520)
    origins = list(range(210, 492, 7))
    bt = fc.backtest(s, origins)
    cov, width = fc.coverage(bt, "ridge_calendar", origins[:len(origins) // 2], origins[len(origins) // 2:])
    assert 0.8 < cov < 0.98 and width > 0


def test_benjamini_hochberg_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    keep = an.benjamini_hochberg(p, q=0.05)
    assert keep.tolist() == [True, True] + [False] * 8
    assert not an.benjamini_hochberg(np.array([0.5, 0.6, 0.7]), 0.05).any()


def _payments(seed=4):
    rng = np.random.default_rng(seed)
    n = 200
    att = rng.integers(200, 400, n)
    p = np.full(n, 0.04)
    p[120:123] = 0.45
    failed = rng.binomial(att, p)
    d = pd.DataFrame({"day": pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d"), "attempts": att, "failed": failed})
    d["fail_rate"] = d["failed"] / d["attempts"]
    return d


def test_binomial_test_catches_spike_without_false_alarms():
    d = an.binomial_test(_payments(), q=0.01)
    flagged = set(d.loc[d["flag_binom"], "day"])
    spike = set(pd.date_range("2025-01-01", periods=200)[120:123].strftime("%Y-%m-%d"))
    assert spike <= flagged
    assert len(flagged - spike) == 0


def test_cusum_resets_after_alarm():
    d = an.binomial_test(_payments(), q=0.01)
    d = an.cusum(d)
    spike = pd.date_range("2025-01-01", periods=200)[120:123].strftime("%Y-%m-%d")
    assert d.loc[d["day"].isin(spike), "flag_cusum"].all()
    after = d[pd.to_datetime(d["day"]) > pd.Timestamp(spike[-1]) + pd.Timedelta(days=3)]
    assert after["flag_cusum"].sum() <= 1


def test_signup_anomaly_catches_single_day_spike():
    s, _ = _series()
    s.loc[300, "signups"] = int(s.loc[300, "signups"] * 2.5)
    r = an.signup_anomalies(s)
    assert bool(r.loc[300, "flag"])
    assert r["flag"].sum() <= 3
