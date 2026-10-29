"""Tests for the cost function, the threshold calibration and the strategies."""
import numpy as np
import pandas as pd
import pytest

from amltm.calibration import (GRID, STRATEGIES, AlertGrid, CostModel, Evaluator, RiskAppetite,
                               calibrate_thresholds, compare_strategies, production_values,
                               split_customers, strategy_cases, tune_ml_strategies)
from amltm.config import SimConfig
from amltm.features import build_features
from amltm.generate import generate
from amltm.models import out_of_fold_scores
from amltm.rules import prepare_panel


def toy_rows():
    # customer A: launderer in month 1 only; customer B: launderer in months 0 and 1; C: legit
    return pd.DataFrame({
        "customer_id": ["A", "A", "B", "B", "C", "C"],
        "account_id": ["a", "a", "b", "b", "c", "c"],
        "month": [0, 1, 0, 1, 0, 1],
        "y": [0, 1, 1, 1, 0, 0],
    })


def test_cost_function_counts_cases_and_missed_launderers():
    rows = toy_rows()
    ev = Evaluator(rows, CostModel(review_cost=10, miss_cost=1_000))
    every = np.ones(6, dtype=bool)
    # review A month 0 (not a laundering month) and C month 0: nobody detected
    r = ev(np.array([1, 0, 0, 0, 1, 0], dtype=bool), every)
    assert (r["cases"], r["launderers_detected"], r["launderers_total"]) == (2, 0, 2)
    assert r["total_cost"] == 2 * 10 + 2 * 1_000
    # review A month 1 and B month 0: both detected
    r = ev(np.array([0, 1, 1, 0, 0, 0], dtype=bool), every)
    assert r["launderers_detected"] == 2 and r["total_cost"] == 20


def test_cost_is_measured_only_on_the_subset():
    rows = toy_rows()
    ev = Evaluator(rows, CostModel(review_cost=10, miss_cost=1_000))
    only_c = rows.customer_id.eq("C").to_numpy()
    r = ev(np.ones(6, dtype=bool), only_c)
    assert (r["cases"], r["launderers_total"], r["total_cost"]) == (2, 0, 20)


def test_split_is_half_and_stratified():
    customers, _, _ = generate(SimConfig(n_customers=600, n_months=2, seed=2))
    cal = split_customers(customers, seed=0)
    assert abs(cal.mean() - 0.5) < 0.02
    for t, ids in customers[customers.is_launderer].groupby("typology").customer_id:
        if len(ids) < 4:            # too few customers to split meaningfully
            continue
        share = cal[ids.to_numpy()].mean()
        assert 0.3 <= share <= 0.7, t


@pytest.fixture(scope="module")
def bank():
    customers, accounts, tx = generate(SimConfig(n_customers=800, n_months=6, seed=9))
    feats = build_features(tx, accounts, customers)
    oof, _ = out_of_fold_scores(feats, n_splits=3)
    grid = AlertGrid(prepare_panel(tx, accounts, customers), customers, feats, tx.timestamp.min())
    return customers, feats, grid, oof["xgboost"]


def test_production_setting_reproduces_the_rules(bank):
    customers, feats, grid, _ = bank
    for sid, v in production_values().items():
        assert (sid, v) in grid.flags
    assert grid.cases(production_values()).sum() > 0


def test_calibration_keeps_every_scenario_on(bank):
    _, feats, grid, _ = bank
    every = np.ones(len(feats), dtype=bool)
    setting = calibrate_thresholds(grid, every, CostModel())
    assert set(setting) == set(GRID)
    assert all(v is not None for v in setting.values())
    # calibrated thresholds never cost more than production on the data they were tuned on
    ev = Evaluator(grid.rows, CostModel())
    assert ev(grid.cases(setting), every)["total_cost"] <= ev(grid.cases(production_values()), every)["total_cost"]


def test_hybrid_contains_the_filtered_rules(bank):
    _, feats, grid, score = bank
    every = np.ones(len(feats), dtype=bool)
    calibrated = calibrate_thresholds(grid, every, CostModel())
    ev = Evaluator(grid.rows, CostModel())
    t = tune_ml_strategies(grid.cases(calibrated), score, feats.y.to_numpy().astype(bool), ev, every,
                           RiskAppetite())
    cases = strategy_cases(grid, score, calibrated, t)
    c, d = cases[STRATEGIES[2]], cases[STRATEGIES[3]]
    assert not (c & ~d).any()                         # D = C plus extra alerts
    assert not (c & ~cases[STRATEGIES[1]]).any()      # C only reviews calibrated rule cases


def test_compare_strategies_runs(bank):
    customers, _, grid, score = bank
    mean, std, settings = compare_strategies(grid, score, customers, CostModel(), n_repeats=2)
    assert list(mean.index) == STRATEGIES
    assert mean.loc[STRATEGIES[0], "cost_vs_production"] == 0
    assert len(settings) == 2
