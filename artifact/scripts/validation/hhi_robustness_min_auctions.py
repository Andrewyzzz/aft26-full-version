"""
HHI robustness: minimum-auctions filter.

Recomputes cell-level HHI restricting to cells with >= min_auctions
auction observations to check whether HHI results are driven by
thin cells with very few auctions.

Aggregated output is provided in:
  artifact/tables/hhi_robustness_min_auctions.csv

Columns:
  - min_auctions:   filter threshold (5, 10, 20, 50)
  - n_cells:        number of cells passing the filter
  - mean_hhi:       mean cell HHI
  - frac_hhi_gt50:  fraction of cells with HHI > 0.50
  - frac_hhi_gt25:  fraction of cells with HHI > 0.25

Usage (requires full dataset):
  python hhi_robustness_min_auctions.py
  # reads: data/processed/auctions_full_usd.parquet
  # writes: results/tables/hhi_robustness_min_auctions.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

def run(parquet_path="data/processed/auctions_full_usd.parquet",
        out_path="results/tables/hhi_robustness_min_auctions.csv"):
    df = pd.read_parquet(parquet_path,
        columns=["market_cell", "winner_solver", "score_winner"])

    rows = []
    for min_a in [5, 10, 20, 50]:
        cell_counts = df.groupby("market_cell")["score_winner"].count()
        valid_cells = cell_counts[cell_counts >= min_a].index
        sub = df[df["market_cell"].isin(valid_cells)]
        share = (sub.groupby(["market_cell", "winner_solver"])["score_winner"]
                    .sum().reset_index(name="vol"))
        tot = share.groupby("market_cell")["vol"].transform("sum")
        share["s2"] = (share["vol"] / tot) ** 2
        hhi = share.groupby("market_cell")["s2"].sum()
        rows.append({
            "min_auctions":  min_a,
            "n_cells":       len(hhi),
            "mean_hhi":      round(hhi.mean(), 4),
            "frac_hhi_gt50": round((hhi > 0.5).mean(), 4),
            "frac_hhi_gt25": round((hhi > 0.25).mean(), 4),
        })
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))

if __name__ == "__main__":
    run()
