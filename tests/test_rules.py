"""Unit tests for the rule-based scenarios.

Each test builds a tiny hand-made bank where the right answer is known,
then checks that the rule fires (or does not fire) exactly when it should.
"""
import pandas as pd

from amltm.rules import RuleConfig, evaluate, run_rules

START = pd.Timestamp("2026-01-01")


def make_bank(rows, segment="retail", income=2_000):
    """Two customers: A00000 (the one under test) and A00001 (a quiet control).

    rows: list of (day, src, dst, amount, txn_type). Day 0..180 so that every
    rolling window, including the 120-day spike history, fits.
    """
    customers = pd.DataFrame({
        "customer_id": ["C00000", "C00001"],
        "account_id": ["A00000", "A00001"],
        "segment": [segment, "retail"],
        "declared_monthly_income": [income, 2_000],
        "is_launderer": [True, False],
        "typology": ["test", None],
    })
    ext = {"X1": "IT", "X2": "IT", "XHR": "IR"} | {f"S{i}": "IT" for i in range(50)}
    accounts = pd.DataFrame({
        "account_id": ["A00000", "A00001"] + list(ext),
        "country": ["IT", "IT"] + list(ext.values()),
    })
    # a little background activity on the control account so the panel spans 6 months
    rows = list(rows) + [(0, "A00001", "X1", 50, "WIRE"), (180, "A00001", "X1", 50, "WIRE")]
    tx = pd.DataFrame(rows, columns=["day", "src_account", "dst_account", "amount", "txn_type"])
    tx["timestamp"] = START + pd.to_timedelta(tx.pop("day"), unit="D") + pd.Timedelta(hours=10)
    tx["txn_id"] = [f"T{i}" for i in range(len(tx))]
    return customers, accounts, tx.sort_values("timestamp").reset_index(drop=True)


def scenarios_for(rows, **kw):
    customers, accounts, tx = make_bank(rows, **kw)
    alerts = run_rules(tx, accounts, customers)
    return set(alerts.loc[alerts.account_id == "A00000", "scenario"]), alerts


def test_structuring_fires_on_three_deposits_in_a_week():
    rows = [(d, "CASH", "A00000", 9_500, "CASH_DEPOSIT") for d in (10, 12, 15)]
    assert "S1" in scenarios_for(rows)[0]


def test_structuring_silent_on_two_deposits_or_spread_out():
    two = [(d, "CASH", "A00000", 9_500, "CASH_DEPOSIT") for d in (10, 12)]
    spread = [(d, "CASH", "A00000", 9_500, "CASH_DEPOSIT") for d in (10, 20, 30)]
    assert "S1" not in scenarios_for(two)[0]
    assert "S1" not in scenarios_for(spread)[0]


def test_structuring_ignores_deposits_above_threshold():
    rows = [(d, "CASH", "A00000", 10_500, "CASH_DEPOSIT") for d in (10, 11, 12)]
    assert "S1" not in scenarios_for(rows)[0]


def test_high_risk_country_fires_in_both_directions():
    assert "S2" in scenarios_for([(30, "A00000", "XHR", 5_000, "WIRE")])[0]
    assert "S2" in scenarios_for([(30, "XHR", "A00000", 5_000, "WIRE")])[0]
    assert "S2" not in scenarios_for([(30, "A00000", "XHR", 500, "WIRE")])[0]   # below 1,000


def test_rapid_movement():
    in_out = [(50, "X1", "A00000", 30_000, "WIRE"), (51, "A00000", "X2", 28_000, "WIRE")]
    kept = [(50, "X1", "A00000", 30_000, "WIRE"), (51, "A00000", "X2", 5_000, "WIRE")]
    assert "S3" in scenarios_for(in_out)[0]
    assert "S3" not in scenarios_for(kept)[0]
    # the same flow is normal for a business: higher threshold
    assert "S3" not in scenarios_for(in_out, segment="business", income=200_000)[0]


def test_fan_in_counts_distinct_senders():
    many = [(60 + i % 5, f"S{i}", "A00000", 500, "WIRE") for i in range(8)]
    same = [(60 + i % 5, "S1", "A00000", 500, "WIRE") for i in range(8)]   # one sender, 8 times
    assert "S4" in scenarios_for(many)[0]
    assert "S4" not in scenarios_for(same)[0]


def test_activity_spike_needs_quiet_history():
    quiet_then_burst = [(d, "A00000", "X1", 200, "WIRE") for d in range(5, 120, 10)] + \
                       [(150, "X1", "A00000", 20_000, "WIRE"), (152, "A00000", "X2", 19_000, "WIRE")]
    always_busy = [(d, "A00000", "X1", 15_000, "WIRE") for d in range(5, 180, 7)]
    assert "S5" in scenarios_for(quiet_then_burst)[0]
    assert "S5" not in scenarios_for(always_busy)[0]


def test_cash_vs_income_only_for_retail():
    rows = [(d, "CASH", "A00000", 2_500, "CASH_DEPOSIT") for d in (40, 45, 50)]   # 7,500 vs 2,000 income
    assert "S6" in scenarios_for(rows)[0]
    assert "S6" not in scenarios_for(rows, segment="business", income=2_000)[0]


def test_one_alert_per_account_scenario_month():
    rows = [(d, "A00000", "XHR", 5_000, "WIRE") for d in (32, 40, 50)]       # all in February
    _, alerts = scenarios_for(rows)
    s2 = alerts[(alerts.account_id == "A00000") & (alerts.scenario == "S2")]
    assert len(s2) == 1
    assert s2.alert_date.iloc[0] == START + pd.Timedelta(days=32)


def test_quiet_customer_never_alerted_and_evaluation():
    customers, accounts, tx = make_bank([(d, "CASH", "A00000", 9_500, "CASH_DEPOSIT") for d in (10, 12, 15)])
    alerts = run_rules(tx, accounts, customers, RuleConfig())
    assert "A00001" not in set(alerts.account_id)
    summary, per_scenario, _ = evaluate(alerts, customers)
    assert summary["alert_precision"] == 1.0
    assert summary["customer_recall"] == 1.0
