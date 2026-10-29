Rows: 30,000 customer-months, of which 289 with laundering (0.96%). Rules flag 1,391 of them.

| Model | PR-AUC | Precision at rules' workload | Recall at rules' workload | Alerts to match rules' detection | Alert reduction |
|---|---:|---:|---:|---:|---:|
| Rules (any scenario) | 0.48 | 19% | 89% | 1,391 | – |
| Isolation Forest | 0.79 | 20% | 96% | 632 | −55% |
| Logistic regression | 0.88 | 21% | 99% | 365 | −74% |
| XGBoost | 0.98 | 21% | 99% | 259 | −81% |

| Typology | Rules: launderers caught | XGBoost PR-AUC, typology seen | XGBoost PR-AUC, typology never seen |
|---|---:|---:|---:|
| Cycle | 100% | 0.87 | 0.29 |
| Dormant burst | 100% | 1.00 | 1.00 |
| Fan-in / fan-out | 88% | 0.98 | 0.83 |
| High-risk country | 100% | 0.89 | 0.86 |
| Pass-through | 80% | 1.00 | 0.98 |
| Structuring | 100% | 0.99 | 0.86 |

| Feature | Share of gain |
|---|---:|
| `total_to_income` | 48.1% |
| `vol_vs_peers` | 14.0% |
| `max_out` | 5.7% |
| `max_in` | 3.2% |
| `amt_foreign_in` | 3.2% |
| `amt_high_risk` | 2.8% |
| `cash_dep_amt` | 2.8% |
| `amt_in` | 2.8% |
| `avg_in` | 2.3% |
| `n_countries` | 2.1% |
