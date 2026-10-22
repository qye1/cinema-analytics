import numpy as np

from _world import small


def test_truth_signs_and_counts():
    t = small()["truth"]
    assert t["offer"]["true_effect_on_treated_4w_churn"] < 0
    assert t["hike"]["true_effect_weekly_churn_pp"] > 0
    bt = t["offer"]["by_type"]
    assert bt["persuadable"]["effect"] < -0.1
    assert bt["sleeping_dog"]["effect"] > 0
    assert abs(bt["sure_thing"]["effect"]) < 0.02 and abs(bt["lost_cause"]["effect"]) < 0.02


def test_experiment_is_randomized_about_half():
    e = small()["tables"]["experiment_assignment"]
    share = (e["variant"] == "offer").mean()
    assert 0.4 < share < 0.6
    assert e["user_id"].is_unique


def test_tables_have_expected_keys():
    tb = small()["tables"]
    assert {"users", "subscriptions", "payments", "engagement", "tickets", "marketing_spend",
            "experiment_assignment", "price_list"} <= set(tb)
    assert tb["users"]["user_id"].is_unique
