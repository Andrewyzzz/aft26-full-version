#!/usr/bin/env python3
"""
HHI robustness: minimum-auctions filter.

Aggregated output: artifact/tables/hhi_robustness_min_auctions.csv
Full parquet required for recomputation.
"""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE   = ROOT / "tables" / "hhi_robustness_min_auctions.csv"
DEFAULT_PARQUET = Path("data/processed/auctions_full_usd.parquet")

def recompute(parquet_path):
    df = pd.read_parquet(parquet_path,
        columns=["market_cell","winner_solver","score_winner"])
    rows = []
    for min_a in [5, 10, 20, 50]:
        cell_counts = df.groupby("market_cell")["score_winner"].count()
        valid_cells = cell_counts[cell_counts >= min_a].index
        sub = df[df["market_cell"].isin(valid_cells)]
        share = (sub.groupby(["market_cell","winner_solver"])["score_winner"]
                    .sum().reset_index(name="vol"))
        tot = share.groupby("market_cell")["vol"].transform("sum")
        share["s2"] = (share["vol"]/tot)**2
        hhi = share.groupby("market_cell")["s2"].sum()
        rows.append({"min_auctions":min_a,"n_cells":len(hhi),
            "mean_hhi":round(hhi.mean(),4),
            "frac_hhi_gt50":round((hhi>0.5).mean(),4),
            "frac_hhi_gt25":round((hhi>0.25).mean(),4)})
    return pd.DataFrame(rows)

def main():
    if DEFAULT_PARQUET.exists():
        print("Full parquet found; recomputing HHI robustness...")
        print(recompute(DEFAULT_PARQUET).to_string(index=False))
    elif DEFAULT_TABLE.exists():
        print("Full parquet not in lightweight artifact; showing aggregated output:")
        print(pd.read_csv(DEFAULT_TABLE).to_string(index=False))
    else:
        raise FileNotFoundError(f"Neither full parquet nor aggregated table found: {DEFAULT_TABLE}")

if __name__ == "__main__":
    main()
