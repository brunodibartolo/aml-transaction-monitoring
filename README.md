# AML Transaction Monitoring: rules + machine learning on synthetic bank data

End-to-end anti-money-laundering (AML) transaction monitoring pipeline, built on a
fully synthetic bank so that every step can be published and reproduced.

> **Status: work in progress.** Step 1 (synthetic data generator) is complete.
> Rule-based scenarios, feature engineering, ML models and threshold calibration
> are being added in the next steps (see roadmap).

## Why this project

Transaction monitoring in banks still relies mostly on rule-based scenarios with
fixed thresholds. They are explainable and auditable, but generate a very high share
of false positives. This project reproduces, on public synthetic data, the approach
of my Master's thesis (*A Machine Learning Approach to Financial Crime Detection*):

1. classic rule-based scenarios, as used in production TxM systems;
2. behavioural and network features at customer level;
3. ML models (Isolation Forest, gradient boosting) to score and prioritise alerts;
4. threshold calibration with Above-the-Line / Below-the-Line testing and an
   explicit cost function.

## The synthetic bank

`src/amltm/generate.py` simulates a retail and corporate bank over 6 months.
Default run (`seed=42`):

| | |
|---|---|
| Customers | 5,000 (85% retail, 15% business) |
| Accounts (incl. external counterparties) | 13,270 |
| Transactions | 1,066,553 |
| Laundering transactions | 3,090 (0.29%) |
| Laundering customers | 100 (2.0%) |

**Normal behaviour:** salaries and pensions, card spending, bills and rent, P2P
transfers inside a friendship network, cash withdrawals; for businesses, client
receipts, suppliers, payroll and monthly taxes.

**Legitimate but "noisy" profiles**, included on purpose to create realistic false
positives: monthly remittances to family abroad, cash-intensive businesses with daily
cash deposits, one-off large operations (e.g. a house sale).

**Six laundering typologies** (FATF / FIU red flags), each labelled at transaction
and customer level:

| Typology | Pattern | Customers | Transactions |
|---|---|---:|---:|
| Structuring | many cash deposits just below the threshold, then an outgoing wire | 25 | 757 |
| Fan-in / fan-out (money mule) | many unrelated senders, funds leave within days | 25 | 1,657 |
| Cycle (round-tripping) | A → B → C → A with small fees at each hop | 15 | 58 |
| High-risk jurisdiction | wires to/from high-risk countries, inconsistent with profile | 15 | 151 |
| Pass-through | large inflows forwarded abroad within 1-3 days, split up | 10 | 148 |
| Dormant account burst | near-dormant account suddenly moves large amounts | 10 | 319 |

![Cash deposits near the threshold](reports/figures/cash_deposits_near_threshold.png)

*The high-risk country list in `config.py` is illustrative, not the official FATF list.*

## Quick start

```bash
pip install -r requirements.txt
python scripts/generate_data.py          # writes data/*.parquet (~20 MB, ~20 s)
python scripts/plot_data_overview.py     # writes reports/figures/
pytest -q                                # 7 tests on data integrity and labels
```

## Project structure

```
aml-transaction-monitoring/
├── src/amltm/
│   ├── config.py        # simulation parameters
│   └── generate.py      # synthetic bank: customers, accounts, transactions
├── scripts/             # command-line entry points
├── tests/               # pytest suite
├── reports/figures/     # charts used in this README
└── data/                # generated data (not versioned)
```

## Roadmap

- [x] Synthetic data generator with six labelled typologies
- [ ] Rule-based scenarios (structuring, high-risk countries, rapid movement, spikes)
- [ ] Customer-level features: rolling statistics, z-scores, peer comparison, PageRank
- [ ] Models: Isolation Forest (unsupervised) and gradient boosting (supervised), PR-AUC
- [ ] Threshold calibration: ATL/BTL testing and cost function
- [ ] Results and discussion

## Author

Bruno Di Bartolo, MSc Financial Engineering (University of Padua / ESILV Paris).
[LinkedIn](https://www.linkedin.com/in/brunodibartolo)

## License

MIT
