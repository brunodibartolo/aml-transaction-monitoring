"""Models that score customer-months, and their comparison with the rules.

Three models, from simplest to strongest:
  * Isolation Forest   unsupervised: no labels, flags what is unusual
  * Logistic regression supervised, linear, fully explainable baseline
  * XGBoost            supervised, gradient-boosted trees

Supervised models are evaluated with 5-fold cross-validation **grouped by
customer**: all the months of a customer are either in training or in test, so
the model is never scored on a customer it has already seen. Every row ends up
with an out-of-fold score, which makes the comparison with the rules fair: same
rows, same labels, no row scored by a model that was trained on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURES

# amounts and counts are heavy-tailed: log-transform them (trees do not need it,
# linear models and Isolation Forest do; the transform is monotone, so it is harmless)
_LOG = [f for f in FEATURES if f.startswith(("n_", "amt_", "max_", "avg_", "cash_dep", "total_", "cash_to"))]


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].astype(float).copy()
    X[_LOG] = np.log1p(X[_LOG].clip(lower=0))
    return X


def _xgb(pos_weight: float, seed: int):
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=2, scale_pos_weight=pos_weight,
                         eval_metric="aucpr", n_jobs=4, random_state=seed)


def _logit(seed: int):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=0.5, max_iter=5_000,
                                            random_state=seed))


def isolation_forest_scores(df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Anomaly score (higher = more unusual). Labels are never used."""
    X = design_matrix(df)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=4)
    model.fit(X)
    return -model.score_samples(X)


def out_of_fold_scores(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """Out-of-fold scores for logistic regression and XGBoost, plus the fitted XGBoost
    models (one per fold) for feature importance."""
    X, y, groups = design_matrix(df), df.y.to_numpy(), df.customer_id.to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {"logistic": np.zeros(len(df)), "xgboost": np.zeros(len(df))}
    fitted = []
    for train, test in cv.split(X, y, groups):
        pos_weight = (y[train] == 0).sum() / max((y[train] == 1).sum(), 1)
        lr = _logit(seed).fit(X.iloc[train], y[train])
        oof["logistic"][test] = lr.predict_proba(X.iloc[test])[:, 1]
        xgb = _xgb(pos_weight, seed).fit(X.iloc[train], y[train])
        oof["xgboost"][test] = xgb.predict_proba(X.iloc[test])[:, 1]
        fitted.append(xgb)
    return oof, fitted


def rule_scores(df: pd.DataFrame, alerts: pd.DataFrame, start) -> np.ndarray:
    """Number of distinct scenarios that fired on each account-month (0 = no alert).

    `start` is the first day of the data, so that month 0 matches the feature table."""
    start = pd.Timestamp(start)
    d = pd.to_datetime(alerts.alert_date)
    month = (d.dt.year * 12 + d.dt.month) - (start.year * 12 + start.month)
    n = alerts.assign(month=month).groupby(["account_id", "month"]).scenario.nunique()
    idx = pd.MultiIndex.from_frame(df[["account_id", "month"]])
    return n.reindex(idx).fillna(0).to_numpy()


def feature_importance(models) -> pd.Series:
    """Mean gain-based importance across the cross-validation models, normalised to 1."""
    imp = pd.concat([pd.Series(m.get_booster().get_score(importance_type="gain")) for m in models],
                    axis=1).fillna(0).mean(axis=1)
    imp = imp.reindex(FEATURES).fillna(0)
    return (imp / imp.sum()).sort_values(ascending=False)


# ----------------------------------------------------------------- evaluation
def top_k(y: np.ndarray, score: np.ndarray, k: int, seed: int = 0):
    """Precision and recall when analysts review the k highest-scored rows.
    Ties are broken at random so that coarse scores (like rule counts) are not favoured."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(score)), -score))[:k]
    tp = y[order].sum()
    return tp / k, tp / y.sum(), order


def alerts_needed(y: np.ndarray, score: np.ndarray, target_recall: float) -> int:
    """Smallest number of reviewed rows (highest scores first) that reaches the target recall."""
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(y[order])
    return int(np.searchsorted(hits, np.ceil(target_recall * y.sum() - 1e-9)) + 1)


def compare(df: pd.DataFrame, scores: dict, rule_score: np.ndarray) -> pd.DataFrame:
    """Compare every scorer with the rules at the rules' own workload and detection rate."""
    y = df.y.to_numpy()
    customers_pos = df.loc[df.y == 1, "customer_id"].nunique()
    flagged = rule_score > 0
    k = int(flagged.sum())
    rule_recall = y[flagged].sum() / y.sum()

    def cust_recall(idx):
        hit = df.iloc[idx]
        return hit.loc[hit.y == 1, "customer_id"].nunique() / customers_pos

    rows = [{
        "model": "Rules (any scenario)",
        "pr_auc": average_precision_score(y, rule_score),
        "precision_at_k": y[flagged].mean(),
        "recall_at_k": rule_recall,
        "customer_recall_at_k": cust_recall(np.flatnonzero(flagged)),
        "alerts_for_rule_recall": k,
    }]
    for name, s in scores.items():
        p, r, idx = top_k(y, s, k)
        rows.append({
            "model": name,
            "pr_auc": average_precision_score(y, s),
            "precision_at_k": p,
            "recall_at_k": r,
            "customer_recall_at_k": cust_recall(idx),
            "alerts_for_rule_recall": alerts_needed(y, s, rule_recall),
        })
    out = pd.DataFrame(rows).set_index("model")
    out["alert_reduction_vs_rules"] = 1 - out.alerts_for_rule_recall / k
    out.attrs["k"] = k
    return out


def unseen_typology_scores(df: pd.DataFrame, typology: pd.Series, n_splits: int = 5, seed: int = 42):
    """Stress test: can XGBoost detect a laundering scheme it has never seen?

    For each typology T, the same grouped cross-validation is run, but customers of
    typology T are removed from every *training* fold. Their rows are still scored,
    by models that never saw a single example of T. Returns {T: out-of-fold scores}.
    """
    X, y, groups = design_matrix(df), df.y.to_numpy(), df.customer_id.to_numpy()
    typ = df.customer_id.map(typology).to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y, groups))
    out = {}
    for t in sorted(pd.Series(typ).dropna().unique()):
        s = np.zeros(len(df))
        for train, test in splits:
            train = train[typ[train] != t]
            pos_weight = (y[train] == 0).sum() / max((y[train] == 1).sum(), 1)
            m = _xgb(pos_weight, seed).fit(X.iloc[train], y[train])
            s[test] = m.predict_proba(X.iloc[test])[:, 1]
        out[t] = s
    return out


def recall_by_typology(df: pd.DataFrame, typology: pd.Series, flagged_idx) -> pd.Series:
    """Share of launderers of each typology with at least one flagged laundering month."""
    pos = df[df.y == 1]
    hit = df.iloc[flagged_idx]
    hit = set(hit.loc[hit.y == 1, "customer_id"])
    t = pos.drop_duplicates("customer_id").customer_id
    return t.map(lambda c: c in hit).groupby(t.map(typology).values).mean()
