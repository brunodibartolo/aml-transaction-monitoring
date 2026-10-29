"""Step 4: threshold calibration with a cost function, and alerting strategies.

The question a bank has to answer is not "which model is best" but "how many
alerts do we review, and which ones". Every reviewed case costs analyst time;
every launderer that goes undetected costs much more (regulatory sanctions,
reputational damage). This module makes that trade-off explicit:

    total cost = review_cost x cases reviewed + miss_cost x launderers not detected

*Case*: one customer-month with at least one alert (alerts of different scenarios
on the same customer and month are reviewed together, as in a case manager).
*Detected*: the launderer has at least one reviewed case in a month in which they
actually moved illicit funds.

Above-the-Line / Below-the-Line (ATL/BTL) testing: for every scenario the key
threshold is moved below (BTL) and above (ATL) its production value, and for each
setting we measure how many cases it creates and how many launderers it detects.
Thresholds are then chosen to minimise the total cost (coordinate descent, since
scenarios overlap: a launderer caught by two rules only needs one).

To avoid tuning and reporting on the same data, customers are split in two halves
(stratified by typology): thresholds are chosen on the **calibration** half and
every number in the README is measured on the **holdout** half.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .rules import RuleConfig, run_scenario

# key threshold of each scenario and the values tested (None = scenario switched off)
GRID = {
    "S1": ("structuring_min_count", [None, 2, 3, 4, 5, 6]),
    "S2": ("high_risk_min_amount", [None, 500, 1_000, 2_500, 5_000, 10_000, 20_000]),
    "S3": ("rapid_min_inflow_retail", [None, 10_000, 15_000, 20_000, 30_000, 50_000, 75_000]),
    "S4": ("fanin_min_senders_retail", [None, 4, 5, 6, 8, 10, 12, 15]),
    "S5": ("spike_ratio", [None, 2, 3, 5, 7, 10, 15]),
    "S6": ("cash_income_ratio", [None, 1.5, 2, 3, 4, 6, 8]),
}
# for S3 the business threshold moves with the retail one (production ratio 50k / 20k)
_S3_BUSINESS_RATIO = 2.5


@dataclass
class CostModel:
    review_cost: float = 100.0         # EUR per case reviewed (analyst time, illustrative)
    miss_cost: float = 100_000.0       # EUR per launderer not detected (illustrative)


def production_values(base: RuleConfig | None = None) -> dict:
    base = base or RuleConfig()
    return {sid: getattr(base, param) for sid, (param, _) in GRID.items()}


def split_customers(customers: pd.DataFrame, seed: int = 42) -> pd.Series:
    """True = calibration half, False = holdout half. Stratified by typology."""
    rng = np.random.default_rng(seed)
    strata = customers.typology.fillna("legit").astype(str) + "_" + customers.segment
    cal = pd.Series(False, index=customers.customer_id.to_numpy())
    for _, ids in customers.groupby(strata.to_numpy()).customer_id:
        ids = ids.to_numpy().copy()
        rng.shuffle(ids)
        cal[ids[: (len(ids) + 1) // 2]] = True
    return cal


class AlertGrid:
    """Pre-computes, for every scenario and threshold, which customer-months it flags.

    After this, any combination of thresholds is evaluated with a few boolean ORs,
    which makes the search over thresholds instantaneous.
    """

    def __init__(self, panel, customers: pd.DataFrame, rows: pd.DataFrame, start,
                 base: RuleConfig | None = None):
        self.base = base or RuleConfig()
        self.rows = rows.reset_index(drop=True)
        start = pd.Timestamp(start)
        index = pd.MultiIndex.from_frame(self.rows[["account_id", "month"]])
        self.flags = {}
        for sid, (param, values) in GRID.items():
            for v in values:
                if v is None:
                    self.flags[(sid, None)] = np.zeros(len(self.rows), dtype=bool)
                    continue
                cfg = replace(self.base, **{param: v})
                if sid == "S3":
                    cfg = replace(cfg, rapid_min_inflow_business=v * _S3_BUSINESS_RATIO)
                a = run_scenario(panel, cfg, customers, sid)
                d = pd.to_datetime(a.alert_date)
                month = (d.dt.year * 12 + d.dt.month) - (start.year * 12 + start.month)
                hit = pd.MultiIndex.from_arrays([a.account_id.to_numpy(), month.to_numpy()])
                self.flags[(sid, v)] = index.isin(hit)

    def cases(self, setting: dict) -> np.ndarray:
        """Boolean mask of customer-months flagged by at least one scenario."""
        out = np.zeros(len(self.rows), dtype=bool)
        for sid, v in setting.items():
            out |= self.flags[(sid, v)]
        return out


class Evaluator:
    """Fast cost evaluation: customers are integer-coded once, then every candidate
    set of cases is scored with a couple of vectorised operations."""

    def __init__(self, rows: pd.DataFrame, cost: CostModel):
        self.y = rows.y.to_numpy().astype(bool)
        self.code, self.ids = pd.factorize(rows.customer_id)
        self.cost = cost

    def __call__(self, cases: np.ndarray, subset: np.ndarray) -> dict:
        n = len(self.ids)
        pos = np.bincount(self.code[subset & self.y], minlength=n) > 0
        hit = np.bincount(self.code[subset & self.y & cases], minlength=n) > 0
        n_cases = int((cases & subset).sum())
        missed = int(pos.sum() - hit.sum())
        return {
            "cases": n_cases,
            "productive_cases": int((cases & subset & self.y).sum()),
            "launderers_detected": int(hit.sum()),
            "launderers_total": int(pos.sum()),
            "review_cost": n_cases * self.cost.review_cost,
            "miss_cost": missed * self.cost.miss_cost,
            "total_cost": n_cases * self.cost.review_cost + missed * self.cost.miss_cost,
        }


def evaluate_cases(rows: pd.DataFrame, cases: np.ndarray, subset: np.ndarray, cost: CostModel) -> dict:
    """Workload, detection and cost of reviewing `cases`, measured on `subset` rows."""
    return Evaluator(rows, cost)(cases, subset)


def calibrate_thresholds(grid: AlertGrid, subset: np.ndarray, cost: CostModel,
                         start_setting: dict | None = None, allow_switch_off: bool = False,
                         max_rounds: int = 10) -> dict:
    """Coordinate descent: one scenario at a time, pick the threshold that minimises
    the total cost given the other scenarios; repeat until nothing changes.

    By default no scenario may be switched off: supervisors expect every typology
    in the bank's risk assessment to stay covered by at least one scenario, even when
    another scenario happens to catch the same customers."""
    ev = Evaluator(grid.rows, cost)
    setting = dict(start_setting or production_values(grid.base))
    for _ in range(max_rounds):
        changed = False
        for sid, (_, values) in GRID.items():
            options = [v for v in values if allow_switch_off or v is not None]
            best = min(options, key=lambda v: (
                ev(grid.cases({**setting, sid: v}), subset)["total_cost"],
                v is None, 0 if v is None else -v))      # ties: keep the scenario on, stricter threshold
            if best != setting[sid]:
                setting[sid], changed = best, True
        if not changed:
            break
    return setting


def atl_btl_curve(grid: AlertGrid, setting: dict, sid: str, subset: np.ndarray, cost: CostModel) -> pd.DataFrame:
    """For one scenario, every tested threshold with the others held fixed."""
    out = []
    for v in GRID[sid][1]:
        if v is None:
            continue
        alone = evaluate_cases(grid.rows, grid.flags[(sid, v)], subset, cost)
        system = evaluate_cases(grid.rows, grid.cases({**setting, sid: v}), subset, cost)
        out.append({"threshold": v, "cases": alone["cases"], "launderers_detected": alone["launderers_detected"],
                    "system_total_cost": system["total_cost"]})
    return pd.DataFrame(out)


def _thresholds(score: np.ndarray, subset: np.ndarray, n: int = 100) -> np.ndarray:
    """Candidate score thresholds: quantiles of the scores on the subset (plus 'never')."""
    q = np.unique(np.quantile(score[subset], np.linspace(0, 1, n)))
    return np.append(q, np.inf)


@dataclass
class RiskAppetite:
    """Policy choices, fixed in advance (not optimised on the data).

    filter_keep:           when the ML score filters rule cases, keep at least this
                           share of the productive cases seen on the calibration half
                           (a safety margin: a threshold tuned to the very edge of the
                           calibration data loses launderers on new data)
    ml_only_min_precision: an extra alert raised by the model (outside the filtered
                           rule cases) is accepted only above the score where, on the
                           calibration half, at least this share of such alerts were
                           laundering
    """
    filter_keep: float = 0.99
    ml_only_min_precision: float = 0.5


def tune_ml_strategies(rule_cases, score, y, ev: Evaluator, subset, appetite: RiskAppetite):
    """Score thresholds of strategies C, D, E, chosen on the calibration rows only.

    C  ML filter:  review a rule case only if score >= t_filter
    D  Hybrid:     C, plus any other case whose score >= t_ml_only, whether or not a
                   rule fired (high-confidence model alerts)
    E  ML only:    every case with score >= t, rules ignored; t minimises the cost
                   (a reference point, see the README for why it is not the answer)
    """
    productive = score[rule_cases & subset & y]
    t_filter = np.quantile(productive, 1 - appetite.filter_keep) if len(productive) else np.inf

    # additional alerts: rows the filtered rules would not review
    not_reviewed = ~(rule_cases & (score >= t_filter)) & subset
    s, lab = score[not_reviewed], y[not_reviewed]
    order = np.argsort(-s, kind="stable")
    precision = np.cumsum(lab[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(precision >= appetite.ml_only_min_precision)
    t_ml_only = s[order][ok.max()] if len(ok) else np.inf

    candidates = np.append(np.unique(np.quantile(score[subset], np.linspace(0, 1, 200))), np.inf)
    t_e = min(candidates, key=lambda t: ev(score >= t, subset)["total_cost"])
    return {"filter": t_filter, "ml_only": t_ml_only, "ml_only_reference": t_e}


STRATEGIES = [
    "A. Rules, production thresholds",
    "B. Rules, calibrated thresholds",
    "C. Calibrated rules + ML filter",
    "D. Hybrid: C + high-confidence ML-only alerts",
    "E. ML only (reference)",
]


def strategy_cases(grid: AlertGrid, score: np.ndarray, calibrated: dict, t: dict) -> dict:
    rule_prod, rule_cal = grid.cases(production_values(grid.base)), grid.cases(calibrated)
    c = rule_cal & (score >= t["filter"])
    return dict(zip(STRATEGIES, [
        rule_prod,
        rule_cal,
        c,
        c | (score >= t["ml_only"]),
        score >= t["ml_only_reference"],
    ]))


def compare_strategies(grid: AlertGrid, score: np.ndarray, customers: pd.DataFrame,
                       cost: CostModel, appetite: RiskAppetite | None = None,
                       n_repeats: int = 20, seed: int = 42):
    """Repeated calibration / holdout splits.

    For each repeat: customers are split in half, every strategy is tuned on the
    calibration half and measured on the holdout half. Results are averaged over
    the repeats, so that a single lucky or unlucky split cannot drive the conclusion.
    Returns (mean results, std of total cost, per-repeat calibrated thresholds).
    """
    appetite = appetite or RiskAppetite()
    ev = Evaluator(grid.rows, cost)
    y = grid.rows.y.to_numpy().astype(bool)
    cust = grid.rows.customer_id
    results, settings = [], []
    for r in range(n_repeats):
        cal = cust.map(split_customers(customers, seed + r)).to_numpy()
        hold = ~cal
        calibrated = calibrate_thresholds(grid, cal, cost)
        t = tune_ml_strategies(grid.cases(calibrated), score, y, ev, cal, appetite)
        for name, m in strategy_cases(grid, score, calibrated, t).items():
            results.append({"strategy": name, "repeat": r, **ev(m, hold)})
        settings.append(calibrated)
    df = pd.DataFrame(results)
    mean = df.groupby("strategy", sort=False).mean(numeric_only=True).drop(columns="repeat")
    std = df.groupby("strategy", sort=False).total_cost.std()
    base = mean.loc[STRATEGIES[0], "total_cost"]
    mean["cost_vs_production"] = mean.total_cost / base - 1
    mean["detection_rate"] = mean.launderers_detected / mean.launderers_total
    return mean, std, pd.DataFrame(settings)
