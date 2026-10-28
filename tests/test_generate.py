"""Generator tests: make sure the data makes sense before building on top of it."""
import pandas as pd
import pytest

from amltm.config import SimConfig
from amltm.generate import TYPOLOGIES, generate


@pytest.fixture(scope="module")
def data():
    # small version so the test suite runs in a few seconds
    return generate(SimConfig(n_customers=600, n_months=3, seed=1))


def test_reproducible():
    """Same seed -> exactly the same data."""
    cfg = SimConfig(n_customers=500, n_months=2, seed=3)
    _, _, a = generate(cfg)
    _, _, b = generate(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_basic_integrity(data):
    customers, accounts, tx = data
    assert tx.txn_id.is_unique
    assert (tx.amount > 0).all()
    assert tx.timestamp.is_monotonic_increasing
    known = set(accounts.account_id) | {"CASH"}
    assert set(tx.src_account).issubset(known)
    assert set(tx.dst_account).issubset(known)
    assert (tx.src_account != tx.dst_account).all()


def test_all_typologies_present(data):
    _, _, tx = data
    assert set(tx.loc[tx.is_laundering, "typology"]) == set(TYPOLOGIES)


def test_labels_consistent(data):
    """Every labelled transaction has a typology, and vice versa."""
    _, _, tx = data
    assert (tx.is_laundering == tx.typology.notna()).all()


def test_laundering_is_rare(data):
    """Laundering must be a rare event, as in real portfolios."""
    _, _, tx = data
    assert 0.0005 < tx.is_laundering.mean() < 0.02


def test_structuring_below_threshold(data):
    """Structuring deposits always stay below the cash threshold."""
    _, _, tx = data
    s = tx[(tx.typology == "structuring") & (tx.txn_type == "CASH_DEPOSIT")]
    assert len(s) > 0
    assert (s.amount < SimConfig().cash_threshold).all()


def test_every_launderer_has_labelled_activity(data):
    customers, _, tx = data
    lab = tx[tx.is_laundering]
    involved = set(lab.src_account) | set(lab.dst_account)
    for acc in customers.loc[customers.is_launderer, "account_id"]:
        assert acc in involved
