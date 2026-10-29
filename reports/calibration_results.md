| Scenario | Threshold | Production | Calibrated | Cases (prod → cal) | Launderers detected (prod → cal) |
|---|---|---:|---:|---:|---:|
| S1 Structuring | deposits in 7 days | 3 | 6 | 76 → 55 | 25 → 25 |
| S2 High-risk jurisdiction | min. amount (EUR) | 1k | 20k | 321 → 102 | 35 → 34 |
| S3 Rapid movement of funds | min. inflow, retail (EUR) | 20k | 15k | 121 → 208 | 57 → 75 |
| S4 Fan-in | distinct senders, retail | 6 | 8 | 328 → 294 | 30 → 26 |
| S5 Activity spike | volume vs history (x) | 5 | 2 | 119 → 189 | 34 → 57 |
| S6 Cash vs declared income | cash / income (x) | 2 | 8 | 625 → 85 | 24 → 22 |
| **All scenarios** | | | | **1391 → 712** | **95 → 100** (of 100) |

Without the coverage constraint the optimiser would switch off: S1 Structuring.

| Strategy | Cases reviewed | Launderers detected | Review cost | Missed-launderer cost | Total cost | vs production |
|---|---:|---:|---:|---:|---:|---:|
| A. Rules, production thresholds | 683 | 44.4 / 46 | 68,320 | 160,000 | 228,320 | – |
| B. Rules, calibrated thresholds | 368 | 44.9 / 46 | 36,805 | 110,000 | 146,805 | -36% |
| C. Calibrated rules + ML filter | 118 | 44.2 / 46 | 11,750 | 175,000 | 186,750 | -18% |
| D. Hybrid: C + high-confidence ML-only alerts | 160 | 46.0 / 46 | 15,960 | 0 | 15,960 | -93% |
| E. ML only (reference) | 138 | 45.5 / 46 | 13,830 | 45,000 | 58,830 | -74% |

| Strategy | Missed launderer = 10,000 EUR | 100,000 EUR | 1,000,000 EUR |
|---|---:|---:|---:|
| A. Rules, production thresholds | – | – | – |
| B. Rules, calibrated thresholds | -43% | -36% | -32% |
| C. Calibrated rules + ML filter | -65% | -18% | +6% |
| D. Hybrid: C + high-confidence ML-only alerts | -81% | -93% | -99% |
| E. ML only (reference) | -78% | -74% | -72% |
