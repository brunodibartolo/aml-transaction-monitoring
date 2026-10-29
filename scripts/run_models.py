"""Step 3: customer-month features, ML models and comparison with the rules.

Outputs:
    data/features.(parquet|csv.gz)          feature table (one row per account and month)
    data/scores.(parquet|csv.gz)            out-of-fold scores of every model + rule score
    reports/ml_results.md                   tables used in the README
    reports/figures/ml_alerts_needed.png
    reports/figures/ml_precision_recall.png
    reports/figures/ml_feature_importance.png

Usage (after scripts/run_rules.py):
    python scripts/run_models.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amltm.features import build_features  # noqa: E402
from amltm.io import load_data  # noqa: E402
from amltm.models import (compare, feature_importance, isolation_forest_scores,  # noqa: E402
                          out_of_fold_scores, recall_by_typology, rule_scores, top_k,
                          unseen_typology_scores)

INK, MUTED, GRID, SURFACE = "#2b2b29", "#6b6a63", "#e6e5e0", "#fcfcfb"
NEUTRAL = "#a3a29b"
SERIES = {"Isolation Forest": "#1baf7a", "Logistic regression": "#eb6834", "XGBoost": "#2a78d6"}
TYPO_LABELS = {
    "structuring": "Structuring", "fan_in_out": "Fan-in / fan-out", "cycle": "Cycle",
    "high_risk_country": "High-risk country", "pass_through": "Pass-through",
    "dormant_burst": "Dormant burst",
}


def _save(df, name):
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(ROOT / "data" / f"{name}.parquet", index=False)
    except ImportError:
        df.to_csv(ROOT / "data" / f"{name}.csv.gz", index=False)


def _load_alerts():
    pq, csv = ROOT / "data" / "alerts.parquet", ROOT / "data" / "alerts.csv.gz"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=["alert_date"])
    raise FileNotFoundError("alerts not found: run scripts/run_rules.py first")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def plot_alerts_needed(res, target_recall, out):
    names = ["Rules (any scenario)", "Isolation Forest", "Logistic regression", "XGBoost"]
    vals = res.loc[names, "alerts_for_rule_recall"].to_numpy()
    colors = [NEUTRAL] + [SERIES[n] for n in names[1:]]
    fig, ax = plt.subplots(figsize=(9, 3.9), dpi=150, facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, edgecolor=SURFACE, linewidth=2, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK)
    for yi, v, n in zip(y, vals, names):
        red = res.loc[n, "alert_reduction_vs_rules"]
        txt = f"{v:,}" + ("" if n.startswith("Rules") else f"   (−{red:.0%})")
        ax.text(v + vals.max() * 0.01, yi, txt, va="center", color=INK, fontsize=9)
    ax.set_xlim(0, vals.max() * 1.22)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Customer-months to review", color=MUTED)
    ax.set_title(f"Alerts needed to detect {target_recall:.0%} of laundering months "
                 "(the rules' detection rate)", loc="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_pr(y, scores, rule_score, out):
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(GRID)
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(r, p, color=SERIES[name], linewidth=2, label=f"{name} (PR-AUC {ap:.2f})")
    # rules: one point per "at least n scenarios fired"
    for n in range(1, int(rule_score.max()) + 1):
        f = rule_score >= n
        if f.sum() == 0:
            continue
        ax.scatter(y[f].sum() / y.sum(), y[f].mean(), s=60, color=INK, edgecolor=SURFACE,
                   linewidth=2, zorder=5, label="Rules (≥ n scenarios)" if n == 1 else None)
        ax.annotate(f"≥{n}", (y[f].sum() / y.sum(), y[f].mean()), textcoords="offset points",
                    xytext=(7, 4), color=INK, fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall (share of laundering months detected)", color=MUTED)
    ax.set_ylabel("Precision (share of alerts that are laundering)", color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK, fontsize=9)
    ax.set_title("Precision-recall: models vs rules", loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_importance(imp, out, top=12):
    imp = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.barh(imp.index, imp.values, color=SERIES["XGBoost"], edgecolor=SURFACE, linewidth=2, height=0.65)
    for i, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, i, f"{v:.0%}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, color=INK, fontsize=9)
    ax.set_xlim(0, imp.max() * 1.15)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("XGBoost: what drives the score (share of total gain)", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    customers, accounts, tx = load_data(ROOT / "data")
    alerts = _load_alerts()

    print("building features...")
    feats = build_features(tx, accounts, customers)
    _save(feats, "features")
    y = feats.y.to_numpy()
    typology = customers.set_index("customer_id").typology

    print("training models (5-fold cross-validation grouped by customer)...")
    rs = rule_scores(feats, alerts, tx.timestamp.min())
    iso = isolation_forest_scores(feats)
    oof, fitted = out_of_fold_scores(feats)
    scores = {"Isolation Forest": iso, "Logistic regression": oof["logistic"], "XGBoost": oof["xgboost"]}
    res = compare(feats, scores, rs)
    k = res.attrs["k"]
    imp = feature_importance(fitted)

    print("stress test: one typology at a time removed from training...")
    unseen = unseen_typology_scores(feats, typology)
    typ_rows = feats.customer_id.map(typology).to_numpy()
    stress = []
    for t, s in unseen.items():
        m = (typ_rows == t) | (y == 0)          # this typology's rows + all legitimate rows
        stress.append({
            "typology": t,
            "rules_customer_recall": recall_by_typology(feats, typology, np.flatnonzero(rs > 0)).get(t, 0),
            "pr_auc_seen": average_precision_score(y[m], oof["xgboost"][m]),
            "pr_auc_unseen": average_precision_score(y[m], s[m]),
            "customer_recall_unseen": recall_by_typology(feats, typology, top_k(y, s, k)[2]).get(t, 0),
        })
    stress = pd.DataFrame(stress).set_index("typology")

    out_scores = feats[["account_id", "customer_id", "month", "segment", "y"]].copy()
    out_scores["rule_score"] = rs
    for name, s in scores.items():
        out_scores[name.lower().replace(" ", "_")] = s
    _save(out_scores, "scores")

    figs = ROOT / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rule_recall = res.loc["Rules (any scenario)", "recall_at_k"]
    plot_alerts_needed(res, rule_recall, figs / "ml_alerts_needed.png")
    plot_pr(y, scores, rs, figs / "ml_precision_recall.png")
    plot_importance(imp, figs / "ml_feature_importance.png")

    md = [f"Rows: {len(feats):,} customer-months, of which {y.sum():,} with laundering "
          f"({y.mean():.2%}). Rules flag {k:,} of them.", "",
          "| Model | PR-AUC | Precision at rules' workload | Recall at rules' workload | "
          "Alerts to match rules' detection | Alert reduction |",
          "|---|---:|---:|---:|---:|---:|"]
    for name, r in res.iterrows():
        md.append(f"| {name} | {r.pr_auc:.2f} | {r.precision_at_k:.0%} | {r.recall_at_k:.0%} | "
                  f"{int(r.alerts_for_rule_recall):,} | "
                  f"{'–' if name.startswith('Rules') else f'−{r.alert_reduction_vs_rules:.0%}'} |")
    md += ["", "| Typology | Rules: launderers caught | XGBoost PR-AUC, typology seen | "
               "XGBoost PR-AUC, typology never seen |", "|---|---:|---:|---:|"]
    for t, r in stress.iterrows():
        md.append(f"| {TYPO_LABELS.get(t, t)} | {r.rules_customer_recall:.0%} | {r.pr_auc_seen:.2f} | "
                  f"{r.pr_auc_unseen:.2f} |")
    md += ["", "| Feature | Share of gain |", "|---|---:|"]
    md += [f"| `{f}` | {v:.1%} |" for f, v in imp.head(10).items()]
    (ROOT / "reports" / "ml_results.md").write_text("\n".join(md) + "\n")

    pd.set_option("display.width", 200)
    print()
    print(res.drop(columns=["customer_recall_at_k"]).to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print(stress.to_string(float_format=lambda v: f"{v:.2f}"))
    print()
    print("top features:", ", ".join(f"{f} {v:.0%}" for f, v in imp.head(5).items()))


if __name__ == "__main__":
    main()
