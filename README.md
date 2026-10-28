# AML Transaction Monitoring: rules + machine learning on synthetic bank data

End-to-end anti-money-laundering (AML) transaction monitoring pipeline, built on a
fully synthetic bank so that every step can be published and reproduced.

> **Status: work in progress.** Steps 1-2 (synthetic data, rule-based scenarios) are
> complete. Feature engineering, ML models and threshold calibration are next
> (see roadmap).

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

## Step 1: the synthetic bank

`src/amltm/generate.py` simulates a retail and corporate bank over 6 months.
Default run (`seed=42`):

| | |
|---|---|
| Customers | 5,000 (85% retail, 15% business) |
| Accounts (incl. external counterparties) | 15,236 |
| Transactions | 1,076,090 |
| Laundering transactions | 2,946 (0.27%) |
| Laundering customers | 100 (2.0%) |

**Normal behaviour:** salaries and pensions, card spending, bills and rent, P2P
transfers inside a friendship network, cash withdrawals; for businesses, client
receipts, suppliers and monthly taxes.

**Legitimate look-alikes**, included on purpose to create realistic false positives:

| Profile | What it does | Customers |
|---|---|---:|
| Remitters | monthly transfers to family abroad, occasionally larger | 439 |
| Cash-intensive businesses | bars, shops: daily cash deposits | 177 |
| Collectors | club treasurers, group trips: many small incoming transfers, then one payment | 126 |
| Cash side income | tips, side jobs: cash deposits well above declared income | 136 |
| One-off large operation | e.g. a house sale | 74 |
| High-risk trade | businesses trading with a partner in a high-risk country | 63 |

**Six laundering typologies** (FATF / FIU red flags), labelled at transaction and
customer level:

| Typology | Pattern | Customers | Transactions |
|---|---|---:|---:|
| Structuring | many cash deposits just below the threshold, then an outgoing wire | 25 | 735 |
| Fan-in / fan-out (money mule) | many unrelated senders, funds leave within days | 25 | 1,531 |
| Cycle (round-tripping) | A → B → C → A with small fees at each hop | 15 | 51 |
| High-risk jurisdiction | wires to/from high-risk countries, inconsistent with profile | 15 | 172 |
| Pass-through | large inflows forwarded abroad within 1-3 days, split up | 10 | 133 |
| Dormant account burst | near-dormant account suddenly moves large amounts | 10 | 324 |

![Cash deposits near the threshold](reports/figures/cash_deposits_near_threshold.png)

*The high-risk country list in `config.py` is illustrative, not the official FATF list.*

## Step 2: rule-based scenarios

`src/amltm/rules.py` implements six scenarios of the kind found in production
TxM systems. Thresholds are fixed and set from business logic, with separate
thresholds for retail and business customers where volumes differ.

| ID | Scenario | Condition |
|---|---|---|
| S1 | Structuring | ≥ 3 cash deposits between 8,000 and 10,000 EUR within 7 days |
| S2 | High-risk jurisdiction | any wire ≥ 1,000 EUR to/from a high-risk country |
| S3 | Rapid movement of funds | ≥ 20,000 EUR in (50,000 for businesses) within 3 days and ≥ 80% out again |
| S4 | Fan-in | ≥ 6 distinct senders in 7 days (40 for businesses) |
| S5 | Activity spike | 30-day volume ≥ 5× the previous 90-day average and ≥ 25,000 EUR |
| S6 | Cash vs declared income | 30-day cash deposits ≥ 2× declared monthly income and ≥ 5,000 EUR (retail) |

Alerts are generated once per account, scenario and month, as in a monthly alert
batch. An alert counts as **productive** when the customer is a launderer in the
ground truth (a proxy for an alert escalated to a suspicious activity report).

### Results

| Scenario | Alerts | Customers alerted | Productive alerts | Precision |
|---|---:|---:|---:|---:|
| S1 Structuring | 76 | 27 | 73 | 96% |
| S2 High-risk jurisdiction | 321 | 121 | 84 | 26% |
| S3 Rapid movement of funds | 121 | 68 | 97 | 80% |
| S4 Fan-in | 328 | 154 | 78 | 24% |
| S5 Activity spike | 119 | 65 | 57 | 48% |
| S6 Cash vs declared income | 625 | 146 | 99 | 16% |
| **All rules** | **1,590** | **460** | **488** | **31%** |

![Alert volume and precision by scenario](reports/figures/rules_alerts_by_scenario.png)

![Share of launderers caught by typology and scenario](reports/figures/rules_recall_by_typology.png)

**What the rules show**

- **Coverage is high, precision is not.** The rules catch 95 of the 100 launderers,
  but 69% of the 1,590 alerts are false positives. Each one costs analyst time.
- **The noisiest rules are the broadest.** S6 (cash vs income) and S4 (fan-in) produce
  60% of all alerts, mostly on legitimate collectors and customers with cash side income.
  S2 fires on every legitimate trade with a high-risk country.
- **Precise rules are narrow.** S1 (structuring) is 96% precise because it targets one
  very specific pattern; it says nothing about the other five typologies.
- **The 5 launderers no rule catches are all businesses** (3 fan-in/fan-out, 2
  pass-through). Their suspicious flows are hidden inside normal business volumes,
  which segment-level fixed thresholds cannot separate.
- **Precision depends on prevalence.** Launderers are 2% of customers here, far more
  than in a real portfolio, so real-world precision of the same rules would be much
  lower. This is why the next steps focus on **ranking alerts by risk**, not only on
  raising alerts.

## Quick start

```bash
git clone https://github.com/brunodibartolo/aml-transaction-monitoring.git
cd aml-transaction-monitoring
pip install -r requirements.txt
python scripts/generate_data.py          # writes data/ (~20 MB, ~30 s)
python scripts/plot_data_overview.py     # step 1 figure
python scripts/run_rules.py              # step 2: alerts, tables and figures
pytest -q                                # 17 tests
```

## Project structure

```
aml-transaction-monitoring/
├── src/amltm/
│   ├── config.py        # simulation parameters
│   ├── generate.py      # synthetic bank: customers, accounts, transactions
│   ├── io.py            # load generated data (parquet or csv.gz)
│   └── rules.py         # six rule-based scenarios and their evaluation
├── scripts/             # command-line entry points
├── tests/               # pytest suite (data integrity, one test per rule)
├── reports/             # figures and result tables used in this README
└── data/                # generated data (not versioned)
```

## Roadmap

- [x] Synthetic data generator with six labelled typologies and legitimate look-alikes
- [x] Rule-based scenarios with evaluation by scenario and typology
- [ ] Customer-level features: rolling statistics, z-scores, peer comparison, PageRank
- [ ] Models: Isolation Forest (unsupervised) and gradient boosting (supervised), PR-AUC
- [ ] Threshold calibration: ATL/BTL testing and cost function
- [ ] Results and discussion

## Author

Bruno Di Bartolo, MSc Financial Engineering (University of Padua / ESILV Paris).
[GitHub](https://github.com/brunodibartolo) · [LinkedIn](https://www.linkedin.com/in/brunodibartolo)

## License

MIT
