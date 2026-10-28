"""Synthetic bank generator: customers, accounts and transactions.

Same idea as the IBM AML dataset on Kaggle: simulate normal customer behaviour,
then inject known money-laundering patterns, each with its own label. Because we
know exactly which transactions are suspicious, we can measure how well rules
and models find them.

Three ingredients:
1. Normal behaviour (salaries, card spending, bills, cash withdrawals, P2P
   transfers; for businesses: client receipts, suppliers, taxes).
2. Legitimate "noisy" customers that look suspicious: family remittances abroad,
   cash-intensive businesses, one-off large operations, collectors (club
   treasurers, group trips), cash side income, trade with high-risk countries.
   Without them any rule would look perfect.
3. Six laundering typologies from FATF / FIU guidance: structuring,
   fan-in/fan-out (money mules), cycles (round-tripping), high-risk
   jurisdictions, pass-through (rapid movement of funds), dormant account burst.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimConfig

DAY = 86_400          # seconds per day: times are seconds since the start
HOUR = 3_600

# Hour-of-day distribution: most activity between 8:00 and 21:00
_HOUR_W = np.array([1, 1, 1, 1, 1, 2, 4, 8, 14, 16, 16, 15, 14, 14, 15, 15,
                    15, 14, 12, 10, 7, 5, 3, 2], dtype=float)
_HOUR_P = _HOUR_W / _HOUR_W.sum()

TYPOLOGIES = ["structuring", "fan_in_out", "cycle", "high_risk_country",
              "pass_through", "dormant_burst"]


class SyntheticBank:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        start = pd.Timestamp(self.cfg.start_date)
        bounds = pd.date_range(start, periods=self.cfg.n_months + 1, freq="MS")
        self.start = start
        # start of each month, in seconds since the start of the simulation
        self.month_starts = ((bounds - start).total_seconds()).astype(np.int64).to_numpy()
        self.horizon = int(self.month_starts[-1])

        self._cols = {k: [] for k in ["t", "src", "dst", "amount", "txn_type",
                                      "is_laundering", "typology"]}
        self.accounts: dict[str, dict] = {}
        self._ext_counter = 0

    # ---------------------------------------------------------------- helpers
    def _add(self, t, src, dst, amount, txn_type, laundering=False, typology=None):
        """Append one or more transactions. Scalars are broadcast."""
        t = np.atleast_1d(np.asarray(t, dtype=np.int64))
        n = len(t)
        if n == 0:
            return

        def rep(x):
            return list(x) if isinstance(x, (list, np.ndarray)) else [x] * n

        amount = np.round(np.atleast_1d(np.asarray(amount, dtype=float)), 2)
        if len(amount) == 1:
            amount = np.repeat(amount, n)
        c = self._cols
        c["t"].extend(t.tolist())
        c["src"].extend(rep(src))
        c["dst"].extend(rep(dst))
        c["amount"].extend(amount.tolist())
        c["txn_type"].extend(rep(txn_type))
        c["is_laundering"].extend([laundering] * n)
        c["typology"].extend([typology] * n)

    def _times(self, start_s, end_s, n):
        """n random timestamps between start_s and end_s with realistic hours."""
        start_s, end_s = max(0, int(start_s)), min(self.horizon - 1, int(end_s))
        if n <= 0 or end_s <= start_s:
            return np.array([], dtype=np.int64)
        days = self.rng.integers(start_s // DAY, max(start_s // DAY + 1, end_s // DAY), n)
        hours = self.rng.choice(24, size=n, p=_HOUR_P)
        t = days * DAY + hours * HOUR + self.rng.integers(0, HOUR, n)
        return np.clip(np.sort(t), start_s, end_s)

    def _ext(self, country: str, kind: str = "ext") -> str:
        """Create an external account (a counterparty at another bank)."""
        self._ext_counter += 1
        acc = f"X{self._ext_counter:06d}"
        self.accounts[acc] = {"account_id": acc, "customer_id": None,
                              "country": country, "is_internal": False, "kind": kind}
        return acc

    def _lognormal(self, median, sigma, n):
        return median * np.exp(self.rng.normal(0, sigma, n))

    # -------------------------------------------------------------- customers
    def _build_customers(self):
        cfg, rng = self.cfg, self.rng
        n = cfg.n_customers
        ids = [f"C{i:05d}" for i in range(n)]
        segment = np.where(rng.random(n) < cfg.share_business, "business", "retail")
        is_retail = segment == "retail"

        income = np.where(is_retail,
                          np.clip(self._lognormal(1_800, 0.40, n), 700, 9_000),
                          np.clip(self._lognormal(40_000, 0.80, n), 5_000, 600_000))

        # nationality: mostly domestic, ~10% foreign
        foreign = rng.random(n) < 0.10
        residence = np.where(foreign, rng.choice(cfg.remittance_countries + cfg.eu_countries, n),
                             cfg.home_country)

        cust = pd.DataFrame({
            "customer_id": ids,
            "account_id": [f"A{i:05d}" for i in range(n)],
            "segment": segment,
            "declared_monthly_income": np.round(income, 0),
            "nationality": residence,
            "age": np.where(is_retail, rng.integers(19, 85, n), rng.integers(1, 40, n)),
        })

        # legitimate "noisy" profiles
        cust["is_remitter"] = is_retail & foreign & (rng.random(n) < 0.8) | \
            (is_retail & (rng.random(n) < cfg.share_remitters * 0.3))
        cust["is_cash_business"] = (~is_retail) & (rng.random(n) < cfg.share_cash_business)
        cust["has_one_off"] = is_retail & (rng.random(n) < cfg.share_one_off_large)
        cust["is_collector"] = is_retail & (rng.random(n) < cfg.share_collectors)
        cust["has_cash_side_income"] = is_retail & (rng.random(n) < cfg.share_cash_side_income)
        cust["is_hr_trader"] = (~is_retail) & (rng.random(n) < cfg.share_hr_trade)

        # launderers: random customers, typologies assigned by fixed shares
        n_l = int(round(n * cfg.share_launderers))
        laund_idx = rng.choice(n, n_l, replace=False)
        shares = np.array([0.25, 0.25, 0.15, 0.15, 0.10, 0.10])
        counts = np.maximum(np.floor(shares * n_l).astype(int), 1)
        counts[2] = max(counts[2], 3)                 # a cycle needs at least 3 accounts
        counts[0] += n_l - counts.sum()               # remainder goes to structuring
        if counts[0] < 1:
            raise ValueError("Too few customers to include every typology: increase n_customers")
        typ = np.repeat(TYPOLOGIES, counts)
        cust["typology"] = None
        cust.loc[laund_idx, "typology"] = typ
        cust["is_launderer"] = cust["typology"].notna()

        # KYC risk rating at onboarding: informative but weak, as in real life
        score = (0.8 * (cust["nationality"] != cfg.home_country)
                 + 0.7 * cust["is_cash_business"]
                 + 0.4 * cust["is_launderer"]
                 + rng.normal(0, 0.6, n))
        cust["kyc_risk"] = pd.cut(score, [-np.inf, 0.6, 1.2, np.inf],
                                  labels=["low", "medium", "high"]).astype(str)

        for r in cust.itertuples():
            self.accounts[r.account_id] = {"account_id": r.account_id, "customer_id": r.customer_id,
                                           "country": self.cfg.home_country, "is_internal": True,
                                           "kind": r.segment}
        self.customers = cust

    def _build_counterparty_pools(self):
        cfg, rng = self.cfg, self.rng
        self.merchants = [self._ext(cfg.home_country, "merchant") for _ in range(1_500)]
        self.merchants += [self._ext(rng.choice(cfg.eu_countries), "merchant") for _ in range(200)]
        self.employers = [self._ext(cfg.home_country, "employer") for _ in range(400)]
        self.utilities = [self._ext(cfg.home_country, "utility") for _ in range(25)]
        self.clients_b2b = [self._ext(cfg.home_country, "company") for _ in range(2_000)]
        self.foreign_b2b = [self._ext(rng.choice(cfg.eu_countries + cfg.other_countries), "company")
                            for _ in range(300)]
        self.pension_fund = self._ext(cfg.home_country, "pension")
        self.tax_office = self._ext(cfg.home_country, "tax")
        self.landlords = [self._ext(cfg.home_country, "person") for _ in range(800)]

        cust = self.customers
        retail = cust.loc[cust.segment == "retail", "account_id"].to_numpy()
        business = cust.loc[cust.segment == "business", "account_id"].to_numpy()
        # friendship network among retail customers (for P2P transfers)
        self.friends = {a: rng.choice(retail[retail != a], 5, replace=False) for a in retail}
        # employer: 30% a business client of the bank, the rest external
        self.employer_of = {}
        for a in retail:
            if rng.random() < 0.30 and len(business):
                self.employer_of[a] = rng.choice(business)
            else:
                self.employer_of[a] = rng.choice(self.employers)

    # -------------------------------------------------------- normal behaviour
    def _normal_retail(self, r, activity=1.0):
        rng, acc = self.rng, r.account_id
        inc = r.declared_monthly_income
        pensioner = r.age >= 67
        renter = rng.random() < 0.4
        fav_merchants = rng.choice(self.merchants, 30, replace=False)
        for m in range(self.cfg.n_months):
            ms, me = self.month_starts[m], self.month_starts[m + 1]
            # income: salary or pension near month end
            src = self.pension_fund if pensioner else self.employer_of[acc]
            self._add(ms + rng.integers(24, 28) * DAY + 9 * HOUR, src, acc,
                      inc * rng.normal(1, 0.02), "SALARY")
            # card: many small payments
            k = rng.poisson(25 * activity)
            self._add(self._times(ms, me, k), acc, list(rng.choice(fav_merchants, k)),
                      np.clip(self._lognormal(30, 0.9, k), 1, 2_500), "CARD")
            # bills and rent
            self._add(self._times(ms, ms + 10 * DAY, 3), acc, list(rng.choice(self.utilities, 3)),
                      self._lognormal(70, 0.5, 3), "BILL")
            if renter:
                self._add(ms + rng.integers(0, 5) * DAY + 10 * HOUR, acc,
                          self.landlords[int(acc[1:]) % len(self.landlords)],
                          round(inc * 0.30, -1), "WIRE")
            # transfers to friends
            k = rng.poisson(1.5 * activity)
            self._add(self._times(ms, me, k), acc, list(rng.choice(self.friends[acc], k)),
                      np.round(self._lognormal(60, 0.8, k), -1) + 10, "WIRE")
            # cash withdrawals (multiples of 50)
            k = rng.poisson(2 * activity)
            self._add(self._times(ms, me, k), acc, "CASH",
                      rng.choice([50, 100, 150, 200, 250, 300, 500], k,
                                 p=[.15, .25, .1, .2, .1, .1, .1]), "CASH_WITHDRAWAL")
            # small cash deposits
            k = rng.poisson(0.15)
            self._add(self._times(ms, me, k), "CASH", acc,
                      np.round(self._lognormal(300, 0.6, k), -1), "CASH_DEPOSIT")
            # remittances to family abroad (legitimate, but they trigger rules)
            if r.is_remitter:
                k = rng.integers(1, 3)
                country = r.nationality if r.nationality != self.cfg.home_country \
                    else rng.choice(self.cfg.remittance_countries)
                dst = self._family_account(acc, country)
                self._add(self._times(ms, me, k), acc, dst,
                          inc * rng.uniform(0.08, 0.20, k), "WIRE")
                if rng.random() < 0.15:                      # occasional larger family transfer
                    self._add(self._times(ms, me, 1), acc, dst, rng.uniform(800, 3_000), "WIRE")
            # irregular but legitimate cash income (tips, side jobs, cash rent)
            if r.has_cash_side_income:
                k = rng.integers(2, 7)
                total = inc * rng.uniform(1.5, 3.5)
                self._add(self._times(ms, me, k), "CASH", acc,
                          np.round(total * rng.dirichlet(np.ones(k)), -1) + 50, "CASH_DEPOSIT")

        # one legitimate large operation: e.g. selling a house, then buying another
        if r.has_one_off:
            t0 = rng.integers(10, self.horizon // DAY - 40) * DAY + 11 * HOUR
            amt = rng.uniform(80_000, 300_000)
            self._add(t0, self._ext(self.cfg.home_country, "notary"), acc, amt, "WIRE")
            if rng.random() < 0.7:
                self._add(t0 + rng.integers(2, 30) * DAY, acc, self._ext(self.cfg.home_country, "person"),
                          amt * rng.uniform(0.85, 1.0), "WIRE")

        # collector: gathers small amounts from many people, then pays one supplier
        # (sports club treasurer, group holiday, wedding gift). Looks like a money mule.
        if r.is_collector:
            for _ in range(rng.integers(1, 4)):
                t0 = rng.integers(0, self.horizon // DAY - 15) * DAY
                k = rng.integers(6, 21)
                senders = [rng.choice(self.friends[acc]) if rng.random() < 0.4
                           else self._ext(self.cfg.home_country, "person") for _ in range(k)]
                amts = np.round(rng.uniform(20, 300, k), -1)
                ts = self._times(t0, t0 + rng.integers(2, 8) * DAY, k)
                self._add(ts, senders, acc, amts, "WIRE")
                self._add(ts[-1] + rng.integers(1, 6) * DAY, acc, rng.choice(self.merchants),
                          amts.sum() * rng.uniform(0.9, 1.0), "WIRE")

    def _family_account(self, acc, country):
        if not hasattr(self, "_family"):
            self._family = {}
        key = (acc, country)
        if key not in self._family:
            self._family[key] = self._ext(country, "person")
        return self._family[key]

    def _normal_business(self, r, activity=1.0):
        rng, acc = self.rng, r.account_id
        turnover = r.declared_monthly_income
        clients = rng.choice(self.clients_b2b, 40, replace=False)
        suppliers = rng.choice(self.clients_b2b, 15, replace=False)
        foreign_sup = rng.choice(self.foreign_b2b, 3, replace=False) if rng.random() < 0.3 else []
        for m in range(self.cfg.n_months):
            ms, me = self.month_starts[m], self.month_starts[m + 1]
            share_cash = 0.4 if r.is_cash_business else 0.0
            # receipts from clients
            k = max(1, rng.poisson(30 * activity))
            amt = turnover * (1 - share_cash) / k * self._lognormal(1, 0.5, k)
            self._add(self._times(ms, me, k), list(rng.choice(clients, k)), acc, amt, "WIRE")
            # daily cash takings (bars, shops): false positives for structuring rules
            if r.is_cash_business:
                days = np.arange(ms // DAY, me // DAY)
                days = days[rng.random(len(days)) < 0.75]
                daily = turnover * share_cash / max(1, len(days))
                self._add(days * DAY + 18 * HOUR + rng.integers(0, 2 * HOUR, len(days)),
                          "CASH", acc, daily * self._lognormal(1, 0.35, len(days)), "CASH_DEPOSIT")
            # suppliers
            k = rng.poisson(12 * activity)
            self._add(self._times(ms, me, k), acc, list(rng.choice(suppliers, k)),
                      turnover * 0.55 / max(k, 1) * self._lognormal(1, 0.5, k), "WIRE")
            # legitimate trade with a partner in a high-risk country
            if r.is_hr_trader and rng.random() < 0.6:
                partner = self._family_account(acc, self.cfg.high_risk_countries[
                    int(acc[1:]) % len(self.cfg.high_risk_countries)])
                src, dst = (acc, partner) if rng.random() < 0.7 else (partner, acc)
                self._add(self._times(ms, me, 1), src, dst, rng.uniform(2_000, 25_000), "WIRE")
            for fs in foreign_sup:
                if rng.random() < 0.5:
                    self._add(self._times(ms, me, 1), acc, fs,
                              turnover * rng.uniform(0.05, 0.2), "WIRE")
            # taxes (paid on the 16th, as with the Italian F24)
            self._add(ms + 15 * DAY + 10 * HOUR, acc, self.tax_office,
                      turnover * rng.uniform(0.06, 0.12), "TAX")

    # ---------------------------------------------------- laundering typologies
    def _structuring(self, r):
        """Many cash deposits just below the threshold, then an outgoing wire."""
        rng, acc, thr = self.rng, r.account_id, self.cfg.cash_threshold
        for _ in range(rng.integers(2, 5)):
            t0 = rng.integers(0, self.horizon // DAY - 20) * DAY
            k = rng.integers(5, 13)
            amts = thr * (1 - rng.beta(1.2, 6, k) * 0.5)     # clustered just below the threshold
            ts = self._times(t0, t0 + rng.integers(2, 10) * DAY, k)
            self._add(ts, "CASH", acc, amts, "CASH_DEPOSIT", True, "structuring")
            country = rng.choice(self.cfg.other_countries + self.cfg.high_risk_countries)
            self._add(ts[-1] + rng.integers(1, 5) * DAY, acc, self._ext(country),
                      amts.sum() * rng.uniform(0.85, 0.97), "WIRE", True, "structuring")

    def _fan_in_out(self, r):
        """Money mule: many different senders, then funds leave to a few accounts."""
        rng, acc = self.rng, r.account_id
        for _ in range(rng.integers(2, 6)):
            t0 = rng.integers(0, self.horizon // DAY - 15) * DAY
            k = rng.integers(8, 26)
            senders = [self._ext(rng.choice([self.cfg.home_country] + self.cfg.eu_countries))
                       for _ in range(k)]
            amts = rng.uniform(300, 3_000, k)
            ts = self._times(t0, t0 + rng.integers(2, 8) * DAY, k)
            self._add(ts, senders, acc, amts, "WIRE", True, "fan_in_out")
            n_out = rng.integers(1, 4)
            dst = self._ext(rng.choice(self.cfg.other_countries))
            out = amts.sum() * rng.uniform(0.85, 0.97) * rng.dirichlet(np.ones(n_out))
            self._add(ts[-1] + np.sort(rng.integers(HOUR, 3 * DAY, n_out)), acc, dst, out,
                      "WIRE", True, "fan_in_out")

    def _cycles(self, members):
        """Round-tripping: A -> B -> C -> A, with a small fee taken at each hop."""
        rng = self.rng
        rng.shuffle(members)
        groups, i = [], 0
        while len(members) - i >= 3:                 # groups of 3-5, nobody left out
            size = min(int(rng.integers(3, 6)), len(members) - i)
            if len(members) - i - size in (1, 2):    # never leave a group that is too small
                size = len(members) - i
            groups.append(members[i:i + size])
            i += size
        for g in groups:
            accs = list(self.customers.set_index("customer_id").loc[g, "account_id"])
            for _ in range(rng.integers(2, 5)):
                amt = rng.uniform(15_000, 120_000)
                t = rng.integers(0, self.horizon // DAY - 20) * DAY + 10 * HOUR
                # funds enter the first account from abroad
                self._add(t, self._ext(rng.choice(self.cfg.other_countries)), accs[0], amt,
                          "WIRE", True, "cycle")
                for j in range(len(accs)):
                    t += int(rng.uniform(0.5, 3) * DAY)
                    amt *= rng.uniform(0.97, 0.995)
                    self._add(t, accs[j], accs[(j + 1) % len(accs)], amt, "WIRE", True, "cycle")

    def _high_risk(self, r):
        """Wires to/from high-risk jurisdictions, inconsistent with the profile."""
        rng, acc = self.rng, r.account_id
        for _ in range(rng.integers(2, 7)):
            t = rng.integers(0, self.horizon // DAY - 5) * DAY + 11 * HOUR
            amt = rng.uniform(3_000, 40_000)
            self._add(t, self._ext(self.cfg.home_country), acc, amt * rng.uniform(1, 1.05),
                      "WIRE", True, "high_risk_country")
            hr = self._ext(rng.choice(self.cfg.high_risk_countries))
            self._add(t + rng.integers(HOUR, 4 * DAY), acc, hr, amt, "WIRE", True, "high_risk_country")
            if rng.random() < 0.3:
                self._add(t + rng.integers(5, 20) * DAY, hr, acc, amt * rng.uniform(0.3, 0.8),
                          "WIRE", True, "high_risk_country")

    def _pass_through(self, r):
        """Pass-through: large amounts come in and leave within 1-3 days, split up."""
        rng, acc = self.rng, r.account_id
        for _ in range(rng.integers(2, 6)):
            t = rng.integers(0, self.horizon // DAY - 5) * DAY + 10 * HOUR
            amt = rng.uniform(20_000, 150_000)
            self._add(t, rng.choice(self.foreign_b2b), acc, amt, "WIRE", True, "pass_through")
            n_out = rng.integers(2, 5)
            out = amt * rng.uniform(0.9, 0.99) * rng.dirichlet(np.ones(n_out) * 2)
            dsts = [self._ext(rng.choice(self.cfg.eu_countries + self.cfg.other_countries))
                    for _ in range(n_out)]
            self._add(t + np.sort(rng.integers(2 * HOUR, 3 * DAY, n_out)), acc, dsts, out,
                      "WIRE", True, "pass_through")

    def _dormant_burst(self, r):
        """Near-dormant account for months, then a burst of activity in the last weeks."""
        rng, acc = self.rng, r.account_id
        t0 = self.horizon - rng.integers(20, 45) * DAY
        k = rng.integers(20, 41)
        ts = self._times(t0, self.horizon - DAY, k)
        incoming = rng.random(k) < 0.5
        amts = rng.uniform(2_000, 20_000, k)
        for t, inc, a in zip(ts, incoming, amts):
            other = self._ext(rng.choice(self.cfg.eu_countries + self.cfg.other_countries))
            src, dst = (other, acc) if inc else (acc, other)
            self._add(t, src, dst, a, "WIRE", True, "dormant_burst")

    # ------------------------------------------------------------------ main
    def run(self):
        self._build_customers()
        self._build_counterparty_pools()
        cust = self.customers
        for r in cust.itertuples():
            activity = 0.15 if r.typology == "dormant_burst" else 1.0
            if r.segment == "retail":
                self._normal_retail(r, activity)
            else:
                self._normal_business(r, activity)

        dispatch = {"structuring": self._structuring, "fan_in_out": self._fan_in_out,
                    "high_risk_country": self._high_risk, "pass_through": self._pass_through,
                    "dormant_burst": self._dormant_burst}
        for r in cust[cust.is_launderer].itertuples():
            if r.typology in dispatch:
                dispatch[r.typology](r)
        self._cycles(list(cust.loc[cust.typology == "cycle", "customer_id"]))

        tx = pd.DataFrame(self._cols)
        tx = tx[(tx.t >= 0) & (tx.t < self.horizon)].sort_values("t", kind="stable")
        tx.insert(0, "timestamp", self.start + pd.to_timedelta(tx.pop("t"), unit="s"))
        tx.insert(0, "txn_id", [f"T{i:08d}" for i in range(len(tx))])
        tx = tx.rename(columns={"src": "src_account", "dst": "dst_account"}).reset_index(drop=True)

        acc = pd.DataFrame(self.accounts.values())
        return cust, acc, tx


def generate(cfg: SimConfig | None = None):
    """Convenience wrapper: returns (customers, accounts, transactions)."""
    return SyntheticBank(cfg).run()
