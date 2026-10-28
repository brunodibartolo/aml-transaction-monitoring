"""Generate the synthetic dataset and save it to data/.

Usage:
    python scripts/generate_data.py                 # 5,000 customers, 6 months
    python scripts/generate_data.py --customers 1000 --seed 7
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amltm.config import SimConfig  # noqa: E402
from amltm.generate import generate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--customers", type=int, default=5_000)
    p.add_argument("--months", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT / "data")
    args = p.parse_args()

    cfg = SimConfig(seed=args.seed, n_customers=args.customers, n_months=args.months)
    customers, accounts, tx = generate(cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    try:                                   # parquet is smaller and faster, when available
        import pyarrow  # noqa: F401
        ext, save = "parquet", lambda df, f: df.to_parquet(f, index=False)
    except ImportError:
        ext, save = "csv.gz", lambda df, f: df.to_csv(f, index=False)
    for name, df in [("customers", customers), ("accounts", accounts), ("transactions", tx)]:
        save(df, args.out / f"{name}.{ext}")

    print(f"{len(customers):,} customers | {len(accounts):,} accounts | {len(tx):,} transactions")
    print(f"laundering transactions: {tx.is_laundering.sum():,} ({tx.is_laundering.mean():.2%})")
    print(f"laundering customers: {customers.is_launderer.sum()} ({customers.is_launderer.mean():.1%})")
    print(f"saved to {args.out}/ as .{ext}")


if __name__ == "__main__":
    main()
