"""Rule-based transaction monitoring scenarios.

Six scenarios of the kind found in production TxM systems. Each one is a
simple, auditable condition on a rolling time window, with fixed thresholds
set from business logic (not fitted on the labels).

| id  | scenario                 | condition (default thresholds)                                        |
|-----|--------------------------|-----------------------------------------------------------------------|
| S1  | Structuring              | >= 3 cash deposits between 8,000 and 10,000 EUR within 7 days         |
| S2  | High-risk jurisdiction   | any wire >= 1,000 EUR to/from a high-risk country                     |
| S3  | Rapid movement of funds  | >= 20,000 EUR in (50,000 business) within 3 days, >= 80% out again   |
| S4  | Fan-in                   | >= 6 distinct senders in 7 days (retail), >= 40 (business)            |
| S5  | Activity spike           | 30-day volume >= 5x the previous 90-day average and >= 25,000 EUR     |
| S6  | Cash vs declared income  | 30-day cash deposits >= 2x declared monthly income and >= 5,000 EUR   |

Alerts are generated once per account, scenario and calendar month (as in a
monthly alert batch). Each alert keeps the value of the metric that triggered
it: that value is what threshold calibration (ATL/BTL) works on later.

Implementation: transactions are turned into an account "ledger" (one row per
internal account side), then into daily account x day matrices. Rolling windows
become cumulative-sum differences, so the whole bank runs in a few seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SimConfig

SCENARIOS = {
    "S1": "Structuring",
    "S2": "High-risk jurisdiction",
    "S3": "Rapid movement of funds",
    "S4": "Fan-in",
    "S5": "Activity spike",
    "S6": "Cash vs declared income",
}

# transaction types that move funds (card payments, bills, salary and taxes excluded)
_FUND_MOVES = ["WIRE", "CASH_DEPOSIT", "CASH_WITHDRAWAL"]


@dataclass
class RuleConfig:
    home_country: str = "IT"
    high_risk_countries: list = field(default_factory=lambda: list(SimConfig().high_risk_countries))

    cash_threshold: float = 10_000
    structuring_band: float = 0.8           # deposits in [0.8 * threshold, threshold)
    structuring_window: int = 7
    structuring_min_count: int = 3

    high_risk_min_amount: float = 1_000

    rapid_window: int = 3
    rapid_min_inflow_retail: float = 20_000
    rapid_min_inflow_business: float = 50_000
    rapid_min_out_ratio: float = 0.8

    fanin_window: int = 7
    fanin_min_senders_retail: int = 6
    fanin_min_senders_business: int = 40

    spike_window: int = 30
    spike_baseline: int = 90
    spike_ratio: float = 5.0
    spike_min_volume: float = 25_000
    spike_baseline_floor: float = 1_000     # avoids dividing by ~0 on new/quiet accounts

    cash_income_window: int = 30
    cash_income_ratio: float = 2.0
    cash_income_min: float = 5_000


# ---------------------------------------------------------------------- ledger
def build_ledger(tx: pd.DataFrame, accounts: pd.DataFrame, customers: pd.DataFrame,
                 home_country: str = "IT") -> pd.DataFrame:
    """One row per (internal account, transaction) with direction and counterparty.

    A transfer between two clients of the bank appears twice: as 'out' for the
    sender and as 'in' for the receiver.
    """
    internal = set(customers.account_id)
    country = accounts.set_index("account_id")["country"]
    start = tx.timestamp.min().normalize()

    parts = []
    for direction, own, other in [("out", "src_account", "dst_account"),
                                  ("in", "dst_account", "src_account")]:
        side = tx[tx[own].isin(internal)]
        parts.append(pd.DataFrame({
            "txn_id": side.txn_id.to_numpy(),
            "account_id": side[own].to_numpy(),
            "counterparty": side[other].to_numpy(),
            "direction": direction,
            "amount": side.amount.to_numpy(),
            "txn_type": side.txn_type.to_numpy(),
            "day": (side.timestamp.dt.normalize() - start).dt.days.to_numpy(),
        }))
    led = pd.concat(parts, ignore_index=True)
    led["cp_country"] = led.counterparty.map(country).fillna(home_country)   # CASH -> home
    led.attrs["start"] = start
    return led


# --------------------------------------------------------------- matrix tools
class _Panel:
    """Daily account x day matrices and rolling-window helpers."""

    def __init__(self, led: pd.DataFrame, customers: pd.DataFrame):
        self.accounts = customers.account_id.to_numpy()
        self.idx = pd.Series(np.arange(len(self.accounts)), index=self.accounts)
        self.n_days = int(led.day.max()) + 1
        self.led = led.assign(acc=led.account_id.map(self.idx).to_numpy())
        start = led.attrs.get("start", pd.Timestamp("2026-01-01"))
        dates = start + pd.to_timedelta(np.arange(self.n_days), unit="D")
        self.dates = dates
        self.month = (dates.year * 12 + dates.month).to_numpy()

    def daily(self, mask, value="amount"):
        """Sum (value='amount') or count (value=None) per account and day."""
        d = self.led[mask]
        m = np.zeros((len(self.accounts), self.n_days))
        np.add.at(m, (d.acc.to_numpy(), d.day.to_numpy()),
                  1.0 if value is None else d[value].to_numpy())
        return m

    @staticmethod
    def rolling(m, window):
        """Sum over the last `window` days, including the current one."""
        c = np.concatenate([np.zeros((m.shape[0], 1)), np.cumsum(m, axis=1)], axis=1)
        hi = np.arange(1, m.shape[1] + 1)
        lo = np.maximum(hi - window, 0)
        return c[:, hi] - c[:, lo]

    def to_alerts(self, flag, value, scenario):
        """One alert per account, scenario and month: first flagged day, max metric value."""
        a, d = np.nonzero(flag)
        if len(a) == 0:
            return pd.DataFrame(columns=["account_id", "scenario", "alert_date", "value"])
        df = pd.DataFrame({"acc": a, "day": d, "value": value[a, d], "month": self.month[d]})
        g = df.groupby(["acc", "month"]).agg(day=("day", "min"), value=("value", "max")).reset_index()
        return pd.DataFrame({
            "account_id": self.accounts[g.acc],
            "scenario": scenario,
            "alert_date": self.dates[g.day],
            "value": g.value.round(2),
        })


# ------------------------------------------------------------------ scenarios
def s1_structuring(p: _Panel, cfg: RuleConfig):
    L = p.led
    near = (L.direction == "in") & (L.txn_type == "CASH_DEPOSIT") & \
        (L.amount >= cfg.structuring_band * cfg.cash_threshold) & (L.amount < cfg.cash_threshold)
    count = p.rolling(p.daily(near, None), cfg.structuring_window)
    return p.to_alerts(count >= cfg.structuring_min_count, count, "S1")


def s2_high_risk(p: _Panel, cfg: RuleConfig):
    L = p.led
    hr = (L.txn_type == "WIRE") & L.cp_country.isin(cfg.high_risk_countries) & \
        (L.amount >= cfg.high_risk_min_amount)
    amt = p.daily(hr)
    return p.to_alerts(amt > 0, amt, "S2")


def s3_rapid_movement(p: _Panel, cfg: RuleConfig, customers: pd.DataFrame):
    L = p.led
    moves = L.txn_type.isin(_FUND_MOVES)
    inflow = p.rolling(p.daily(moves & (L.direction == "in")), cfg.rapid_window)
    outflow = p.rolling(p.daily(moves & (L.direction == "out")), cfg.rapid_window)
    is_biz = (customers.set_index("account_id").loc[p.accounts, "segment"] == "business").to_numpy()
    thr = np.where(is_biz, cfg.rapid_min_inflow_business, cfg.rapid_min_inflow_retail)[:, None]
    flag = (inflow >= thr) & (outflow >= cfg.rapid_min_out_ratio * inflow)
    return p.to_alerts(flag, inflow, "S3")


def s4_fan_in(p: _Panel, cfg: RuleConfig, customers: pd.DataFrame):
    L = p.led
    inc = L[(L.direction == "in") & (L.txn_type == "WIRE")][["acc", "counterparty", "day"]]
    inc = inc.drop_duplicates()
    # rolling distinct count: each (sender, day) is "visible" for the next `window` days
    shifted = pd.concat([inc.assign(day=inc.day + k) for k in range(cfg.fanin_window)])
    shifted = shifted[shifted.day < p.n_days]
    nun = shifted.groupby(["acc", "day"]).counterparty.nunique()
    senders = np.zeros((len(p.accounts), p.n_days))
    senders[nun.index.get_level_values(0), nun.index.get_level_values(1)] = nun.to_numpy()
    is_biz = (customers.set_index("account_id").loc[p.accounts, "segment"] == "business").to_numpy()
    thr = np.where(is_biz, cfg.fanin_min_senders_business, cfg.fanin_min_senders_retail)[:, None]
    return p.to_alerts(senders >= thr, senders, "S4")


def s5_activity_spike(p: _Panel, cfg: RuleConfig):
    L = p.led
    vol = p.daily(L.txn_type.isin(_FUND_MOVES))
    recent = p.rolling(vol, cfg.spike_window)
    # average 30-day volume over the 90 days *before* the recent window
    past = p.rolling(vol, cfg.spike_baseline)
    past = np.concatenate([np.zeros((vol.shape[0], cfg.spike_window)), past[:, :-cfg.spike_window]], axis=1)
    baseline = np.maximum(past * cfg.spike_window / cfg.spike_baseline, cfg.spike_baseline_floor)
    ratio = recent / baseline
    flag = (ratio >= cfg.spike_ratio) & (recent >= cfg.spike_min_volume)
    flag[:, : cfg.spike_window + cfg.spike_baseline - 1] = False   # no full history yet
    return p.to_alerts(flag, ratio, "S5")


def s6_cash_vs_income(p: _Panel, cfg: RuleConfig, customers: pd.DataFrame):
    L = p.led
    cash = p.rolling(p.daily((L.direction == "in") & (L.txn_type == "CASH_DEPOSIT")), cfg.cash_income_window)
    c = customers.set_index("account_id").loc[p.accounts]
    income = c.declared_monthly_income.to_numpy()[:, None]
    retail = (c.segment == "retail").to_numpy()[:, None]
    ratio = cash / income
    flag = retail & (ratio >= cfg.cash_income_ratio) & (cash >= cfg.cash_income_min)
    return p.to_alerts(flag, ratio, "S6")


# ----------------------------------------------------------------------- run
def run_rules(tx: pd.DataFrame, accounts: pd.DataFrame, customers: pd.DataFrame,
              cfg: RuleConfig | None = None) -> pd.DataFrame:
    """Run all scenarios and return one alert per row, with the customer attached."""
    cfg = cfg or RuleConfig()
    led = build_ledger(tx, accounts, customers, cfg.home_country)
    p = _Panel(led, customers)
    alerts = pd.concat([
        s1_structuring(p, cfg),
        s2_high_risk(p, cfg),
        s3_rapid_movement(p, cfg, customers),
        s4_fan_in(p, cfg, customers),
        s5_activity_spike(p, cfg),
        s6_cash_vs_income(p, cfg, customers),
    ], ignore_index=True)
    alerts = alerts.merge(customers[["account_id", "customer_id", "segment"]], on="account_id", how="left")
    alerts = alerts.sort_values(["alert_date", "scenario", "account_id"]).reset_index(drop=True)
    alerts.insert(0, "alert_id", [f"AL{i:06d}" for i in range(len(alerts))])
    return alerts


# ------------------------------------------------------------------ evaluate
def evaluate(alerts: pd.DataFrame, customers: pd.DataFrame):
    """Alert-level and customer-level performance of the rules.

    An alert is counted as *productive* when the alerted customer is a known
    launderer in the ground truth (a proxy for an alert escalated to a SAR).
    """
    truth = customers.set_index("customer_id")[["is_launderer", "typology"]]
    a = alerts.join(truth, on="customer_id")

    per_scenario = a.groupby("scenario").agg(
        alerts=("alert_id", "size"),
        customers=("customer_id", "nunique"),
        productive=("is_launderer", "sum"),
    )
    per_scenario["precision"] = per_scenario.productive / per_scenario.alerts
    per_scenario.insert(0, "name", per_scenario.index.map(SCENARIOS))

    launderers = customers[customers.is_launderer]
    by_scen = a[a.is_launderer].groupby(["typology", "scenario"]).customer_id.nunique().unstack(fill_value=0)
    by_scen = by_scen.reindex(columns=list(SCENARIOS), fill_value=0)
    n_typ = launderers.typology.value_counts()
    caught_any = a[a.is_launderer].groupby("typology").customer_id.nunique()
    recall = by_scen.div(n_typ, axis=0).reindex(n_typ.index).fillna(0)
    recall["Any rule"] = (caught_any / n_typ).reindex(n_typ.index).fillna(0)

    alerted = a.customer_id.nunique()
    caught = a.loc[a.is_launderer, "customer_id"].nunique()
    summary = {
        "alerts": len(a),
        "productive_alerts": int(a.is_launderer.sum()),
        "alert_precision": a.is_launderer.mean(),
        "false_positive_rate": 1 - a.is_launderer.mean(),
        "customers_alerted": alerted,
        "launderers_caught": caught,
        "launderers_total": len(launderers),
        "customer_recall": caught / len(launderers),
    }
    return summary, per_scenario, recall
