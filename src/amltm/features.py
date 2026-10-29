"""Customer-month feature table for machine learning.

One row per (account, calendar month). Each row describes how the account
behaved in that month, compared with:
  * its own declared profile (income / turnover),
  * its own history (previous months),
  * its peers (same segment, same month),
  * the transaction network (PageRank, strongly connected components).

The label `y` is 1 when the account sent or received at least one laundering
transaction in that month. This is the same unit as the monthly rule alerts of
step 2, so rules and models can be compared on exactly the same rows.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .rules import RuleConfig, build_ledger

_FUND_MOVES = ["WIRE", "CASH_DEPOSIT", "CASH_WITHDRAWAL"]
_KYC = {"low": 0, "medium": 1, "high": 2}

# Features fed to the models, grouped by family (the groups are used in the README)
FEATURE_GROUPS = {
    "volume": ["n_in", "n_out", "amt_in", "amt_out", "max_in", "max_out", "avg_in"],
    "cash": ["cash_dep_n", "cash_dep_amt", "n_cash_near_thr"],
    "counterparties": ["n_senders", "n_receivers", "n_countries"],
    "geography": ["amt_foreign_in", "amt_foreign_out", "amt_high_risk"],
    "profile": ["total_to_income", "cash_to_income", "in_out_balance", "is_business", "kyc_risk",
                "foreign_national"],
    "history_and_peers": ["vol_vs_history", "vol_vs_peers"],
    "network": ["pagerank", "scc_size"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


def _month_index(led: pd.DataFrame) -> pd.Series:
    start = led.attrs["start"]
    dates = start + pd.to_timedelta(led.day, unit="D")
    return (dates.dt.year * 12 + dates.dt.month) - (start.year * 12 + start.month)


def _network_features(moves: pd.DataFrame, tx_internal: set, min_cycle_amount: float) -> pd.DataFrame:
    """PageRank and strongly-connected-component size, one graph per month.

    Graph: nodes are accounts (inside and outside the bank), edges are fund
    transfers weighted by amount. Cash is left out because it would be a hub.
    SCCs use only internal-to-internal transfers of at least `min_cycle_amount`:
    an account in a non-trivial SCC is part of a round trip of large amounts.
    """
    rows = []
    wires = moves[(moves.txn_type == "WIRE") & (moves.direction == "out")]
    for m, w in wires.groupby("month"):
        edges = w.groupby(["account_id", "counterparty"]).amount.sum()
        # include transfers *received* from external accounts (they appear only as 'in' rows)
        inc = moves[(moves.month == m) & (moves.txn_type == "WIRE") & (moves.direction == "in")
                    & ~moves.counterparty.isin(tx_internal)]
        edges_in = inc.groupby(["counterparty", "account_id"]).amount.sum()
        g = nx.DiGraph()
        g.add_weighted_edges_from((s, d, a) for (s, d), a in edges.items())
        g.add_weighted_edges_from((s, d, a) for (s, d), a in edges_in.items())
        pr = nx.pagerank(g, alpha=0.85, weight="weight")
        n = g.number_of_nodes()

        big = edges[(edges >= min_cycle_amount)]
        big = big[[d in tx_internal for d in big.index.get_level_values(1)]]
        h = nx.DiGraph(list(big.index))
        scc = {node: len(c) for c in nx.strongly_connected_components(h) for node in c}

        accs = sorted(tx_internal)
        rows.append(pd.DataFrame({
            "account_id": accs, "month": m,
            "pagerank": [pr.get(a, 0.0) * n for a in accs],   # scaled: 1 = average node
            "scc_size": [scc.get(a, 1) for a in accs],
        }))
    return pd.concat(rows, ignore_index=True)


def build_features(tx: pd.DataFrame, accounts: pd.DataFrame, customers: pd.DataFrame,
                   cfg: RuleConfig | None = None, min_cycle_amount: float = 5_000) -> pd.DataFrame:
    """Return the customer-month table with FEATURES, label `y` and identifiers."""
    cfg = cfg or RuleConfig()
    led = build_ledger(tx, accounts, customers, cfg.home_country)
    led["month"] = _month_index(led)
    led["is_laundering"] = led.txn_id.map(tx.set_index("txn_id").is_laundering).astype(bool)
    n_months = int(led.month.max()) + 1

    grid = pd.MultiIndex.from_product([customers.account_id, range(n_months)],
                                      names=["account_id", "month"])
    moves = led[led.txn_type.isin(_FUND_MOVES)]
    inn, out = moves[moves.direction == "in"], moves[moves.direction == "out"]
    foreign = moves.cp_country != cfg.home_country
    cash_dep = inn[inn.txn_type == "CASH_DEPOSIT"]
    lo = cfg.structuring_band * cfg.cash_threshold

    key = ["account_id", "month"]
    parts = [
        inn.groupby(key).agg(n_in=("amount", "size"), amt_in=("amount", "sum"), max_in=("amount", "max"),
                             n_senders=("counterparty", "nunique")),
        out.groupby(key).agg(n_out=("amount", "size"), amt_out=("amount", "sum"), max_out=("amount", "max"),
                             n_receivers=("counterparty", "nunique")),
        cash_dep.groupby(key).agg(cash_dep_n=("amount", "size"), cash_dep_amt=("amount", "sum")),
        cash_dep[(cash_dep.amount >= lo) & (cash_dep.amount < cfg.cash_threshold)]
        .groupby(key).size().rename("n_cash_near_thr"),
        moves[foreign & (moves.direction == "in")].groupby(key).amount.sum().rename("amt_foreign_in"),
        moves[foreign & (moves.direction == "out")].groupby(key).amount.sum().rename("amt_foreign_out"),
        moves[moves.cp_country.isin(cfg.high_risk_countries)].groupby(key).amount.sum().rename("amt_high_risk"),
        moves[foreign].groupby(key).cp_country.nunique().rename("n_countries"),
        led.groupby(key).is_laundering.any().rename("y"),
    ]
    df = pd.concat(parts, axis=1).reindex(grid).fillna(0).reset_index()
    df["y"] = df.y.astype(int)

    # profile
    c = customers.set_index("account_id")
    df["customer_id"] = df.account_id.map(c.customer_id)
    df["segment"] = df.account_id.map(c.segment)
    df["is_business"] = (df.segment == "business").astype(int)
    df["kyc_risk"] = df.account_id.map(c.kyc_risk).map(_KYC).fillna(0).astype(int)
    df["foreign_national"] = (df.account_id.map(c.nationality) != cfg.home_country).astype(int)
    income = df.account_id.map(c.declared_monthly_income).clip(lower=1)
    total = df.amt_in + df.amt_out
    df["avg_in"] = df.amt_in / df.n_in.replace(0, np.nan)
    df["total_to_income"] = total / income
    df["cash_to_income"] = df.cash_dep_amt / income
    hi = np.maximum(df.amt_in, df.amt_out)
    df["in_out_balance"] = np.where(hi > 0, np.minimum(df.amt_in, df.amt_out) / hi, 0.0)

    # own history: log-volume this month minus the mean of previous months (0 in month 0)
    df = df.sort_values(key).reset_index(drop=True)
    logv = np.log1p(df.amt_in + df.amt_out)
    past = logv.groupby(df.account_id).transform(lambda s: s.shift(1).expanding().mean())
    df["vol_vs_history"] = (logv - past).fillna(0.0)
    # peers: same segment, same month
    df["vol_vs_peers"] = logv - logv.groupby([df.segment, df.month]).transform("median")

    # network
    net = _network_features(led, set(customers.account_id), min_cycle_amount)
    df = df.merge(net, on=key, how="left").fillna({"pagerank": 0.0, "scc_size": 1})
    df["avg_in"] = df.avg_in.fillna(0.0)

    cols = ["account_id", "customer_id", "month", "segment"] + FEATURES + ["y"]
    return df[cols]
