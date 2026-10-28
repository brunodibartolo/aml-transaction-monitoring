"""Run the six rule-based scenarios and evaluate them against the ground truth.

Outputs:
    data/alerts.(parquet|csv.gz)            one row per alert
    reports/rules_results.md                tables used in the README
    reports/figures/rules_alerts_by_scenario.png
    reports/figures/rules_recall_by_typology.png

Usage:
    python scripts/run_rules.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amltm.io import load_data  # noqa: E402
from amltm.rules import SCENARIOS, evaluate, run_rules  # noqa: E402

INK, MUTED, GRID, SURFACE = "#2b2b29", "#6b6a63", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# sequential blue ramp (light -> dark) for the recall heatmap
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TYPO_LABELS = {
    "structuring": "Structuring", "fan_in_out": "Fan-in / fan-out", "cycle": "Cycle",
    "high_risk_country": "High-risk country", "pass_through": "Pass-through",
    "dormant_burst": "Dormant burst",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def plot_alerts(per_scenario, out):
    """Horizontal stacked bars: productive vs false-positive alerts per scenario."""
    ps = per_scenario.iloc[::-1]
    labels = [f"{k}  {v}" for k, v in zip(ps.index, ps.name)]
    fp = ps.alerts - ps.productive
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.barh(labels, ps.productive, color=ORANGE, edgecolor=SURFACE, linewidth=2, label="Productive (launderer)")
    ax.barh(labels, fp, left=ps.productive, color=BLUE, edgecolor=SURFACE, linewidth=2,
            label="False positive")
    for i, (tot, prec) in enumerate(zip(ps.alerts, ps.precision)):
        ax.text(tot + 8, i, f"{tot:,} alerts · {prec:.0%} precision", va="center", color=INK, fontsize=8.5)
    ax.set_xlim(0, ps.alerts.max() * 1.45)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Alerts over 6 months", color=MUTED)
    ax.set_title("Rule-based scenarios: alert volume and precision", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_recall(recall, out):
    """Heatmap: share of launderers of each typology caught by each scenario."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    m = recall.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(m.shape[1]))
    short = {"S1": "Structuring", "S2": "High-risk", "S3": "Rapid move", "S4": "Fan-in",
             "S5": "Spike", "S6": "Cash/income"}
    ax.set_xticklabels([f"{c}\n{short[c]}" if c in short else c for c in recall.columns],
                       color=INK, fontsize=8.5)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels([TYPO_LABELS.get(t, t) for t in recall.index], color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # white gridlines between cells
    ax.set_xticks(np.arange(-.5, m.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, m.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.axvline(m.shape[1] - 1.5, color=INK, linewidth=1.2)   # separate the "Any rule" column
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, f"{v:.0%}" if v > 0 else "–", ha="center", va="center", fontsize=9,
                    color="white" if v >= 0.55 else INK)
    ax.set_title("Share of launderers caught, by typology and scenario", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    ax.xaxis.tick_top()
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    customers, accounts, tx = load_data(ROOT / "data")
    alerts = run_rules(tx, accounts, customers)
    summary, per_scenario, recall = evaluate(alerts, customers)

    try:
        import pyarrow  # noqa: F401
        alerts.to_parquet(ROOT / "data" / "alerts.parquet", index=False)
    except ImportError:
        alerts.to_csv(ROOT / "data" / "alerts.csv.gz", index=False)

    figs = ROOT / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plot_alerts(per_scenario, figs / "rules_alerts_by_scenario.png")
    plot_recall(recall, figs / "rules_recall_by_typology.png")

    # markdown tables for the README
    lines = ["| Scenario | Alerts | Customers alerted | Productive alerts | Precision |",
             "|---|---:|---:|---:|---:|"]
    for k, r in per_scenario.iterrows():
        lines.append(f"| {k} {SCENARIOS[k]} | {r.alerts:,} | {r.customers:,} | {r.productive:,} | {r.precision:.0%} |")
    lines.append(f"| **All rules** | **{summary['alerts']:,}** | **{summary['customers_alerted']:,}** | "
                 f"**{summary['productive_alerts']:,}** | **{summary['alert_precision']:.0%}** |")
    (ROOT / "reports" / "rules_results.md").write_text("\n".join(lines) + "\n")

    print(f"alerts: {summary['alerts']:,} on {summary['customers_alerted']:,} customers")
    print(f"productive alerts: {summary['productive_alerts']:,} "
          f"(precision {summary['alert_precision']:.1%}, false positives {summary['false_positive_rate']:.1%})")
    print(f"launderers caught: {summary['launderers_caught']}/{summary['launderers_total']} "
          f"(recall {summary['customer_recall']:.0%})")
    print()
    print(per_scenario.to_string(formatters={"precision": "{:.0%}".format}))


if __name__ == "__main__":
    main()
