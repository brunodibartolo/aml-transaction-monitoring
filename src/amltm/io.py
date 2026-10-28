"""Load the generated data, whichever format it was saved in (parquet or csv.gz)."""
from pathlib import Path

import pandas as pd

NAMES = ("customers", "accounts", "transactions")


def load_data(data_dir):
    """Return (customers, accounts, transactions) from data_dir."""
    data_dir = Path(data_dir)
    out = []
    for name in NAMES:
        pq, csv = data_dir / f"{name}.parquet", data_dir / f"{name}.csv.gz"
        if pq.exists():
            df = pd.read_parquet(pq)
        elif csv.exists():
            df = pd.read_csv(csv)
        else:
            raise FileNotFoundError(f"{name} not found in {data_dir}: run scripts/generate_data.py first")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "typology" in df.columns:
            df["typology"] = df["typology"].astype(object).where(df["typology"].notna(), None)
        out.append(df)
    return tuple(out)
