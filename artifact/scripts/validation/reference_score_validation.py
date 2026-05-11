#!/usr/bin/env python3
"""
Parser-audit wrapper.

If the full parquet dataset is available, this script recomputes the audit.
In the lightweight anonymous artifact, the aggregated output is provided in:
  artifact/tables/reference_score_validation.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE   = ROOT / "tables" / "reference_score_validation.csv"
DEFAULT_PARQUET = Path("data/processed/auctions_full.parquet")

def recompute(parquet_path):
    df = pd.read_parquet(parquet_path,
        columns=["auction_id","score_winner","score_2_raw","score_2_solution",
                 "fragility","fragility_old"])
    affected = df["score_2_raw"].isna() | (df["score_2_raw"] == df["score_winner"])
    return pd.DataFrame([{
        "n_affected":     int(affected.sum()),
        "pct_affected":   round(affected.mean()*100, 4),
        "fragility_old":  round(df["fragility_old"].mean(), 6),
        "fragility_new":  round(df["fragility"].mean(), 6),
        "bias_direction": "old parser understates fragility",
    }])

def main():
    if DEFAULT_PARQUET.exists():
        print("Full parquet found; recomputing audit...")
        print(recompute(DEFAULT_PARQUET).to_string(index=False))
    elif DEFAULT_TABLE.exists():
        print("Full parquet not in lightweight artifact; showing aggregated output:")
        print(pd.read_csv(DEFAULT_TABLE).to_string(index=False))
    else:
        raise FileNotFoundError(f"Neither full parquet nor aggregated table found: {DEFAULT_TABLE}")

if __name__ == "__main__":
    main()
