"""Simulation parameters.

Everything that controls the size and realism of the synthetic data lives here,
so it can be changed in one place without touching the generator logic.
"""
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    seed: int = 42                      # same seed -> same data -> reproducible results
    n_customers: int = 5_000
    share_business: float = 0.15        # 15% businesses, 85% retail customers
    start_date: str = "2026-01-01"
    n_months: int = 6

    # Share of customers involved in laundering schemes (~2%: high, to have enough
    # positives for evaluation; real portfolios have far fewer)
    share_launderers: float = 0.02

    # Legitimate customers that *look* suspicious. They generate false positives;
    # without them the problem would be trivial and results unrealistic.
    share_remitters: float = 0.08       # send money to family abroad every month
    share_cash_business: float = 0.25   # among businesses: bars, restaurants, shops
    share_one_off_large: float = 0.02   # one large legitimate operation (e.g. house sale)
    share_collectors: float = 0.03      # collect money from many people (club treasurer, group trip)
    share_cash_side_income: float = 0.03  # undeclared-looking but legitimate cash income (tips, side jobs)
    share_hr_trade: float = 0.08        # among businesses: legitimate trade with high-risk countries

    # Reference cash threshold (illustrative) used by the structuring typology
    cash_threshold: float = 10_000.0

    # Countries as ISO codes. The high-risk list is ILLUSTRATIVE, not the official
    # FATF list in force (which changes several times a year).
    home_country: str = "IT"
    eu_countries: list = field(default_factory=lambda: ["DE", "FR", "ES", "NL", "BE", "AT", "PT", "IE"])
    other_countries: list = field(default_factory=lambda: ["GB", "CH", "US", "AE", "CN", "RO", "AL", "MA"])
    high_risk_countries: list = field(default_factory=lambda: ["KP", "IR", "MM", "SY", "YE", "HT"])
    remittance_countries: list = field(default_factory=lambda: ["RO", "AL", "MA", "HT", "PH", "BD"])
