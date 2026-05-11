#!/usr/bin/env python3
"""
Blue-chip consistency check.

Companion to bluechip_paradox_check.py; documents within-solver consistency.
Aggregated output: artifact/tables/bluechip_paradox_check.csv
Full parquet required for recomputation.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE   = ROOT / "tables" / "bluechip_paradox_check.csv"
DEFAULT_PARQUET = Path("data/processed/auctions_full_usd.parquet")

def main():
    if DEFAULT_PARQUET.exists():
        print("Full parquet found; delegating to bluechip_paradox_check.py...")
        from bluechip_paradox_check import main as run
        run()
    elif DEFAULT_TABLE.exists():
        print("Full parquet not in lightweight artifact; showing aggregated output:")
        print(pd.read_csv(DEFAULT_TABLE).to_string(index=False))
    else:
        raise FileNotFoundError(f"Neither full parquet nor aggregated table found: {DEFAULT_TABLE}")

if __name__ == "__main__":
    main()
