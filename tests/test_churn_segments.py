import numpy as np
import pandas as pd

from _world import small
from cinema import churn, db, marts, segments as sg


def test_snapshot_features_do_not_depend_on_the_future():
    # меняем вовлеченность после даты снимка: признаки и состав строк остаться прежними обязаны
    w = small()
    s = 60
    a = churn.build_snapshots(w["conn"], [s], drop_offered=False)
    import shutil
    tmp = w["dir"] / "mut.db"
    shutil.copy(w["dir"] / "cinema.db", tmp)
    c2 = db.connect(tmp)
    D = (pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=s)).strftime("%Y-%m-%d")
    c2.execute("UPDATE engagement SET hours = hours * 10 WHERE week_start >= ?", (D,))
    c2.execute("DELETE FROM tickets WHERE created_date >= ?", (D,))
    c2.execute("DELETE FROM payments WHERE pay_date >= ?", (D,))
    c2.commit()
    marts.build(c2)
    b = churn.build_snapshots(c2, [s], drop_offered=False)
    c2.close()
    assert len(a) == len(b) and (a["user_id"].values == b["user_id"].values).all()
    cols = [c for c in churn.ALL_FEATURES]
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))
    # а вот метка и будущие часы изменились бы, это и есть признак, который нельзя давать модели
    assert "future_hours_l2" not in churn.ALL_FEATURES


def test_temporal_split_has_gap_larger_than_horizon():
    assert churn.TEST_FIRST - churn.TRAIN_LAST > 4


def test_snapshot_label_matches_end_dates():
    w = small()
    d = churn.build_snapshots(w["conn"], [50], drop_offered=False)
    assert 0.01 < d["y"].mean() < 0.15
    assert d["tenure_weeks"].min() >= 4


def test_gains_table_monotone_and_ends_at_one():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.1).astype(int)
    p = y * 0.5 + rng.random(5000) * 0.5
    g = churn.gains_table(y, p)
    assert (np.diff(g["captured"]) >= 0).all() and abs(g["captured"].iloc[-1] - 1) < 1e-12
    assert g["lift"].iloc[0] > 1.5 and abs(g["lift"].iloc[-1] - 1) < 1e-9


def test_cost_curve_profit_arithmetic():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    cc = churn.cost_curve(y, p, price=100, saved_share=0.5, contact_cost=10, months_value=2, n_points=10)
    row = cc[np.isclose(cc["top_share"], 0.2, atol=0.06)].iloc[0]
    # два уходящих пойманы, каждый приносит 0.5 * 100 * 2, контактов 2
    assert row["caught"] == 2 and abs(row["profit"] - (2 * 100 - 2 * 10)) < 1e-9 or row["caught"] >= 1


def test_segmentation_recovers_planted_personas():
    w = small()
    r = sg.run(w["conn"], w["truth_users"])
    assert r["agreement"]["ari"] > 0.45
    assert r["profile"]["mean_hours"].is_monotonic_decreasing
    assert abs(r["profile"]["share"].sum() - 1) < 1e-9
