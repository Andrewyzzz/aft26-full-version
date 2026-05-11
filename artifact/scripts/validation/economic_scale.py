"""
Economic scale: accounting-gap estimation.

Estimates the aggregate USD gap between fragile auctions' actual execution
and the counterfactual shadow-reserve benchmark.

Aggregated output is provided in:
  artifact/tables/economic_scale.csv

Columns:
  - sample:                 USD sample label
  - n_auctions:             number of auctions in sample
  - frac_fragile:           fraction with fragility > 0
  - total_notional_usd:     sum of order notional (USD)
  - gap_per_auction_bps:    mean gap in basis points
  - total_gap_usd:          estimated total USD gap
  - covered_notional_usd:   notional covered by sane-volume denominator

Usage (requires full dataset):
  python economic_scale.py
  # reads: data/processed/auctions_full_usd.parquet
  # writes: results/tables/economic_scale.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

def run(parquet_path="data/processed/auctions_full_usd.parquet",
        out_path="results/tables/economic_scale.csv"):
    df = pd.read_parquet(parquet_path,
        columns=["fragility", "gap_bps", "notional_usd", "has_usd"])
    sub = df[df["has_usd"]].copy()
    result = pd.DataFrame([{
        "sample":               "USD-covered main auctions",
        "n_auctions":           len(sub),
        "frac_fragile":         round((sub["fragility"] > 0).mean(), 4),
        "total_notional_usd":   round(sub["notional_usd"].sum(), 2),
        "gap_per_auction_bps":  round(sub["gap_bps"].mean(), 4),
        "total_gap_usd":        round((sub["gap_bps"] / 10000 * sub["notional_usd"]).sum(), 2),
    }])
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))

if __name__ == "__main__":
    run()
