"""Figure for the README: why a simple cash-threshold rule is not enough.

Plots the distribution of cash deposits between 2,000 and 10,000 EUR,
legitimate vs structuring (stacked). Structuring clusters just below the
threshold; legitimate cash-intensive businesses also fall in that range.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, SURFACE = "#2b2b29", "#6b6a63", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def main():
    tx = pd.read_parquet(ROOT / "data" / "transactions.parquet",
                         columns=["amount", "txn_type", "typology"])
    cash = tx[(tx.txn_type == "CASH_DEPOSIT") & tx.amount.between(2_000, 10_000)]
    legit = cash.loc[cash.typology.isna(), "amount"]
    struct = cash.loc[cash.typology == "structuring", "amount"]
    bins = np.arange(2_000, 10_001, 250)

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    kw = dict(bins=bins, edgecolor=SURFACE, linewidth=1.5)
    ax.hist([legit, struct], stacked=True, color=[BLUE, ORANGE],
            label=[f"Legitimate ({len(legit):,})", f"Structuring ({len(struct):,})"], **kw)
    ax.axvline(10_000, color=INK, linewidth=1.2, linestyle="--")
    ax.text(9_930, ax.get_ylim()[1] * 0.95, "Cash threshold\n10,000 EUR", ha="right", va="top",
            color=INK, fontsize=9)

    ax.set_title("Cash deposits between 2,000 and 10,000 EUR", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Deposit amount (EUR)", color=MUTED)
    ax.set_ylabel("Number of deposits", color=MUTED)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="upper center", labelcolor=INK)
    near = cash[cash.amount >= 8_000]
    fig.text(0.01, 0.01, f"Structuring clusters just below the threshold. A rule on 8,000-10,000 EUR "
             f"deposits would still flag {near.typology.isna().sum()} legitimate deposits "
             f"(cash-intensive businesses).", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = ROOT / "reports" / "figures" / "cash_deposits_near_threshold.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
