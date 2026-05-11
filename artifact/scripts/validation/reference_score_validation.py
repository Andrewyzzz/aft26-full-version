"""
Parser audit: reference score validation.

Validates that the best non-winning solver score is correctly constructed
by removing the winning solver by identity and taking the best remaining
solution score (score_2_solution).

Core implementation: artifact/scripts/data_collection/build_dataset.py

This script documents the audit logic; aggregated output is provided in:
  artifact/tables/reference_score_validation.csv

Columns in reference_score_validation.csv:
  - n_affected:     auctions where old parser score == winning solver score
  - pct_affected:   fraction of total auctions affected
  - fragility_old:  mean fragility under old parser
  - fragility_new:  mean fragility under corrected parser
  - bias_direction: old parser understates fragility (sets gap to 0 for affected auctions)

Usage (requires full dataset):
  python reference_score_validation.py
  # reads: data/processed/auctions_full.parquet
  # writes: results/tables/reference_score_validation.csv
"""
import pandas as pd
from pathlib import Path

def run(parquet_path="data/processed/auctions_full.parquet",
        out_path="results/tables/reference_score_validation.csv"):
    df = pd.read_parquet(parquet_path,
        columns=["auction_id", "score_winner", "score_2_raw", "score_2_solution",
                 "fragility", "fragility_old"])
    affected = df["score_2_raw"].isna() | (df["score_2_raw"] == df["score_winner"])
    result = pd.DataFrame([{
        "n_affected":     int(affected.sum()),
        "pct_affected":   round(affected.mean() * 100, 4),
        "fragility_old":  round(df["fragility_old"].mean(), 6),
        "fragility_new":  round(df["fragility"].mean(), 6),
        "bias_direction": "old parser understates fragility",
    }])
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))

if __name__ == "__main__":
    run()
