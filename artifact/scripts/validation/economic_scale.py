#!/usr/bin/env python3
"""
Economic scale: accounting-gap estimation.

Aggregated output: artifact/tables/economic_scale.csv
Full parquet required for recomputation.
"""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE   = ROOT / "tables" / "economic_scale.csv"
DEFAULT_PARQUET = Path("data/processed/auctions_full_usd.parquet")

def recompute(parquet_path):
    df = pd.read_parquet(parquet_path,
        columns=["fragility","gap_bps","notional_usd","has_usd"])
    sub = df[df["has_usd"]].copy()
    return pd.DataFrame([{
        "sample":               "USD-covered main auctions",
        "n_auctions":           len(sub),
        "frac_fragile":         round((sub["fragility"]>0).mean(),4),
        "total_notional_usd":   round(sub["notional_usd"].sum(),2),
        "gap_per_auction_bps":  round(sub["gap_bps"].mean(),4),
        "total_gap_usd":        round((sub["gap_bps"]/10000*sub["notional_usd"]).sum(),2),
    }])

def main():
    if DEFAULT_PARQUET.exists():
        print("Full parquet found; recomputing economic scale...")
        print(recompute(DEFAULT_PARQUET).to_string(index=False))
    elif DEFAULT_TABLE.exists():
        print("Full parquet not in lightweight artifact; showing aggregated output:")
        print(pd.read_csv(DEFAULT_TABLE).to_string(index=False))
    else:
        raise FileNotFoundError(f"Neither full parquet nor aggregated table found: {DEFAULT_TABLE}")

if __name__ == "__main__":
    main()
