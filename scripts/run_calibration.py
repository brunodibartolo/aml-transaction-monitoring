"""Step 4: ATL/BTL threshold calibration, cost function and alerting strategies.

Outputs:
    reports/calibration_results.md                 tables used in the README
    reports/figures/calibration_atl_btl.png        trade-off curve of every scenario
    reports/figures/strategy_costs.png             total cost of the five strategies

Usage (after scripts/run_rules.py and scripts/run_models.py):
    python scripts/run_calibration.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amltm.calibration import (GRID, STRATEGIES, AlertGrid, CostModel, Evaluator,  # noqa: E402
                               RiskAppetite, calibrate_thresholds, compare_strategies,
                               production_values)
from amltm.io import load_data  # noqa: E402
from amltm.rules import SCENARIOS, prepare_panel  # noqa: E402

INK, MUTED, GRID_C, SURFACE = "#2b2b29", "#6b6a63", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, LINE = "#2a78d6", "#eb6834", "#86b6ef"
PARAM_LABEL = {
    "S1": "deposits in 7 days", "S2": "min. amount (EUR)", "S3": "min. inflow, retail (EUR)",
    "S4": "distinct senders, retail", "S5": "volume vs history (x)", "S6": "cash / income (x)",
}


def _fmt(v):
    if v is None:
        return "off"
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        v = int(v)
        return f"{v // 1000}k" if v >= 1000 else str(v)
    return f"{v:g}"


def _load(name):
    pq, csv = ROOT / "data" / f"{name}.parquet", ROOT / "data" / f"{name}.csv.gz"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"{name} not found: run the previous scripts first")


def plot_atl_btl(grid, ev, prod, calibrated, out):
    """Small multiples: for each scenario, cases created vs launderers detected (scenario alone)."""
    every = np.ones(len(grid.rows), dtype=bool)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), dpi=150, facecolor=SURFACE)
    for ax, (sid, (_, values)) in zip(axes.flat, GRID.items()):
        ax.set_facecolor(SURFACE)
        pts = [(v, ev(grid.flags[(sid, v)], every)) for v in values if v is not None]
        x = [r["cases"] for _, r in pts]
        y = [r["launderers_detected"] for _, r in pts]
        ax.plot(x, y, color=LINE, linewidth=2, zorder=1)
        ax.scatter(x, y, s=22, color=LINE, zorder=2)
        # label points, skipping any label that would overlap one already placed
        xr, yr = max(max(x), 1), max(max(y), 1)
        placed = []
        keep = {prod[sid], calibrated[sid]}
        for v, r in sorted(pts, key=lambda p: p[0] not in keep):
            px, py = r["cases"] / xr, r["launderers_detected"] / yr
            if any(abs(px - qx) < 0.07 and abs(py - qy) < 0.08 for qx, qy in placed):
                continue
            placed.append((px, py))
            ax.annotate(_fmt(v), (r["cases"], r["launderers_detected"]), textcoords="offset points",
                        xytext=(4, -10), fontsize=7, color=MUTED)
        for v, color, label in [(prod[sid], INK, "production"), (calibrated[sid], ORANGE, "calibrated")]:
            if v is None:
                continue
            r = ev(grid.flags[(sid, v)], every)
            ax.scatter(r["cases"], r["launderers_detected"], s=90, facecolor="none", edgecolor=color,
                       linewidth=2, zorder=3, label=label)
        ax.set_title(f"{sid} {SCENARIOS[sid]}", loc="left", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xlabel(f"cases  ·  labels = {PARAM_LABEL[sid]}", fontsize=7.5, color=MUTED)
        ax.set_ylim(0, max(y) * 1.18)
        ax.set_xlim(0, max(x) * 1.12)
        ax.grid(True, color=GRID_C, linewidth=0.7)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(GRID_C)
        ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    axes[0, 0].set_ylabel("launderers detected", fontsize=8, color=MUTED)
    axes[1, 0].set_ylabel("launderers detected", fontsize=8, color=MUTED)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK, ncol=2)
    fig.suptitle("ATL/BTL: what each threshold costs (cases) and buys (launderers detected)",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_strategies(mean, cost, out):
    m = mean.loc[STRATEGIES][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 4.3), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(m))
    ax.barh(y, m.review_cost / 1000, color=BLUE, edgecolor=SURFACE, linewidth=2, height=0.62,
            label=f"Review cost ({cost.review_cost:,.0f} EUR per case)")
    ax.barh(y, m.miss_cost / 1000, left=m.review_cost / 1000, color=ORANGE, edgecolor=SURFACE,
            linewidth=2, height=0.62, label=f"Missed launderers ({cost.miss_cost:,.0f} EUR each)")
    for yi, (_, r) in zip(y, m.iterrows()):
        ax.text(r.total_cost / 1000 + 3, yi,
                f"{r.total_cost / 1000:,.0f}k  ·  {r.cases:,.0f} cases  ·  "
                f"{r.launderers_detected:.1f}/{r.launderers_total:.0f} detected",
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(m.index, fontsize=9, color=INK)
    ax.set_xlim(0, m.total_cost.max() / 1000 * 1.75)
    ax.set_xlabel("Expected cost on the holdout half (thousand EUR, mean of 20 splits)", color=MUTED)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID_C)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK)
    ax.set_title("Alerting strategies: workload, detection and total cost", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    customers, accounts, tx = load_data(ROOT / "data")
    scores = _load("scores")
    score = scores.xgboost.to_numpy()
    cost, appetite = CostModel(), RiskAppetite()

    print("pre-computing alerts for every threshold (ATL/BTL grid)...")
    panel = prepare_panel(tx, accounts, customers)
    grid = AlertGrid(panel, customers, scores, tx.timestamp.min())
    ev = Evaluator(grid.rows, cost)
    every = np.ones(len(scores), dtype=bool)

    prod = production_values()
    calibrated = calibrate_thresholds(grid, every, cost)                     # on all customers
    unconstrained = calibrate_thresholds(grid, every, cost, allow_switch_off=True)

    print("comparing strategies on 20 calibration / holdout splits...")
    mean, std, settings = compare_strategies(grid, score, customers, cost, appetite, n_repeats=20)
    sens = {}
    for mc in (10_000, 1_000_000):
        print(f"sensitivity: missed launderer = {mc:,} EUR...")
        sens[mc] = compare_strategies(grid, score, customers, CostModel(miss_cost=mc), appetite,
                                      n_repeats=20)[0]
    sens[cost.miss_cost] = mean

    figs = ROOT / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plot_atl_btl(grid, ev, prod, calibrated, figs / "calibration_atl_btl.png")
    plot_strategies(mean, cost, figs / "strategy_costs.png")

    # ----------------------------------------------------------- markdown tables
    md = ["| Scenario | Threshold | Production | Calibrated | Cases (prod → cal) | Launderers detected (prod → cal) |",
          "|---|---|---:|---:|---:|---:|"]
    for sid, (param, _) in GRID.items():
        a, b = ev(grid.flags[(sid, prod[sid])], every), ev(grid.flags[(sid, calibrated[sid])], every)
        md.append(f"| {sid} {SCENARIOS[sid]} | {PARAM_LABEL[sid]} | {_fmt(prod[sid])} | "
                  f"{_fmt(calibrated[sid])} | {a['cases']} → {b['cases']} | "
                  f"{a['launderers_detected']} → {b['launderers_detected']} |")
    sys_prod, sys_cal = ev(grid.cases(prod), every), ev(grid.cases(calibrated), every)
    md.append(f"| **All scenarios** | | | | **{sys_prod['cases']} → {sys_cal['cases']}** | "
              f"**{sys_prod['launderers_detected']} → {sys_cal['launderers_detected']}** "
              f"(of {sys_prod['launderers_total']}) |")
    off = [sid for sid, v in unconstrained.items() if v is None]
    md += ["", f"Without the coverage constraint the optimiser would switch off: "
               f"{', '.join(f'{s} {SCENARIOS[s]}' for s in off) or 'nothing'}.", ""]

    md += ["| Strategy | Cases reviewed | Launderers detected | Review cost | Missed-launderer cost | "
           "Total cost | vs production |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in mean.loc[STRATEGIES].iterrows():
        md.append(f"| {name} | {r.cases:,.0f} | {r.launderers_detected:.1f} / {r.launderers_total:.0f} | "
                  f"{r.review_cost:,.0f} | {r.miss_cost:,.0f} | {r.total_cost:,.0f} | "
                  f"{'–' if name == STRATEGIES[0] else f'{r.cost_vs_production:+.0%}'} |")
    md += ["", "| Strategy | Missed launderer = 10,000 EUR | 100,000 EUR | 1,000,000 EUR |",
           "|---|---:|---:|---:|"]
    for name in STRATEGIES:
        cells = []
        for mc in (10_000, cost.miss_cost, 1_000_000):
            v = sens[mc].loc[name, "cost_vs_production"]
            cells.append("–" if name == STRATEGIES[0] else f"{v:+.0%}")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    (ROOT / "reports" / "calibration_results.md").write_text("\n".join(md) + "\n")

    pd.set_option("display.width", 220)
    print()
    print("calibrated thresholds:", {k: _fmt(v) for k, v in calibrated.items()})
    print("switched off without coverage constraint:", off or "nothing")
    print()
    print(mean[["cases", "launderers_detected", "launderers_total", "total_cost",
                "cost_vs_production"]].to_string(float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
