"""Tests for the feature table, the evaluation helpers and the model pipeline."""
import numpy as np
import pandas as pd
import pytest

from amltm.config import SimConfig
from amltm.features import FEATURES, build_features
from amltm.generate import generate
from amltm.models import (alerts_needed, out_of_fold_scores, rule_scores, top_k)
from amltm.rules import run_rules


@pytest.fixture(scope="module")
def small():
    customers, accounts, tx = generate(SimConfig(n_customers=600, n_months=3, seed=5))
    feats = build_features(tx, accounts, customers)
    return customers, accounts, tx, feats


def test_one_row_per_account_and_month(small):
    customers, _, _, feats = small
    assert len(feats) == len(customers) * 3
    assert not feats.duplicated(["account_id", "month"]).any()


def test_no_missing_or_infinite_features(small):
    _, _, _, feats = small
    X = feats[FEATURES].to_numpy(dtype=float)
    assert np.isfinite(X).all()


def test_label_matches_laundering_transactions(small):
    """y = 1 exactly for the account-months that touch a laundering transaction."""
    customers, _, tx, feats = small
    lab = tx[tx.is_laundering]
    start = tx.timestamp.min()
    month = (lab.timestamp.dt.year * 12 + lab.timestamp.dt.month) - (start.year * 12 + start.month)
    internal = set(customers.account_id)
    pairs = set()
    for side in ("src_account", "dst_account"):
        mask = lab[side].isin(internal)
        pairs |= set(zip(lab.loc[mask, side], month[mask]))
    flagged = set(zip(feats.loc[feats.y == 1, "account_id"], feats.loc[feats.y == 1, "month"]))
    assert flagged == pairs


def test_cycle_members_have_large_scc(small):
    customers, _, _, feats = small
    cyc = customers.loc[customers.typology == "cycle", "account_id"]
    rows = feats[feats.account_id.isin(cyc) & (feats.y == 1)]
    assert (rows.scc_size >= 3).any()
    legit = feats[~feats.account_id.isin(customers.loc[customers.is_launderer, "account_id"])]
    assert (legit.scc_size == 1).mean() > 0.99


def test_top_k_and_alerts_needed():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([.9, .8, .7, .1, .2, .3])
    p, r, idx = top_k(y, s, 3)
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)
    assert alerts_needed(y, s, 2 / 3) == 3     # .9 (hit), .8, .7 (hit)
    assert alerts_needed(y, s, 1.0) == 4       # needs to reach .3


def test_rule_scores_align_with_feature_months(small):
    customers, accounts, tx, feats = small
    alerts = run_rules(tx, accounts, customers)
    rs = rule_scores(feats, alerts, tx.timestamp.min())
    assert len(rs) == len(feats)
    # every alert lands on a row: number of flagged rows = distinct (account, month) alerted
    d = pd.to_datetime(alerts.alert_date)
    start = tx.timestamp.min()
    m = (d.dt.year * 12 + d.dt.month) - (start.year * 12 + start.month)
    assert (rs > 0).sum() == len(set(zip(alerts.account_id, m)))


def test_out_of_fold_scores_are_probabilities(small):
    _, _, _, feats = small
    oof, models = out_of_fold_scores(feats, n_splits=3)
    for s in oof.values():
        assert len(s) == len(feats)
        assert ((s >= 0) & (s <= 1)).all()
    assert len(models) == 3
    # the supervised model must beat random ranking on held-out customers
    from sklearn.metrics import average_precision_score
    assert average_precision_score(feats.y, oof["xgboost"]) > 5 * feats.y.mean()
