import numpy as np
import pandas as pd

from _world import small
from cinema import marts


def test_waterfall_identity_and_continuity():
    w = marts.mrr_waterfall(small()["conn"])
    comp = ["new_mrr", "reactivation_mrr", "expansion_mrr", "price_uplift_mrr", "contraction_mrr", "churn_mrr"]
    resid = w["start_mrr"] + w[comp].sum(axis=1) - w["end_mrr"]
    assert resid.abs().max() < 1e-6
    # конец одного месяца равен началу следующего
    assert np.allclose(w["start_mrr"].values[1:], w["end_mrr"].values[:-1])


def test_price_uplift_only_in_hike_month():
    w = marts.mrr_waterfall(small()["conn"])
    nz = w.loc[w["price_uplift_mrr"].abs() > 1e-9, "month_end"].tolist()
    assert nz == ["2025-03-31"]


def test_user_month_has_one_row_per_user_and_month():
    um = pd.read_sql_query("SELECT user_id, month_idx FROM user_month", small()["conn"])
    assert not um.duplicated().any()


def test_quality_checks_find_what_was_injected():
    w = small()
    q = marts.quality(w["conn"]).set_index("check_name")["problem_rows"]
    inj = w["truth"]["injected"]
    for k, v in inj.items():
        assert int(q[k]) == int(v), k


def test_clean_views_remove_dirt():
    c = small()["conn"]
    n_raw = c.execute("SELECT COUNT(*) FROM payments").fetchone()[0]
    n_clean = c.execute("SELECT COUNT(*) FROM payments_clean").fetchone()[0]
    assert n_clean < n_raw
    assert c.execute("SELECT COUNT(*) FROM payments_clean WHERE status='success' AND amount<=0").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM engagement_clean WHERE hours<0").fetchone()[0] == 0
    # после очистки подписки одного человека не пересекаются
    over = c.execute("""SELECT COUNT(*) FROM subscriptions_clean a JOIN subscriptions_clean b
                        ON a.user_id=b.user_id AND a.spell_id<b.spell_id AND a.start_date<b.start_date
                        AND a.end_date_clean>b.start_date""").fetchone()[0]
    assert over == 0


def test_cohort_first_age_is_full():
    coh = marts.cohorts(small()["conn"])
    first = coh[coh["age"] == 0]
    assert (first["retention_pct"] == 100).all()
    assert (first["revenue_retention_pct"] == 100).all()
