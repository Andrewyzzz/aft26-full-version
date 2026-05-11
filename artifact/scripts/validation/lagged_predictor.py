"""
Lagged fragility predictor regression.

Tests whether past fragility (lagged by one week) predicts current fragility
at the cell level, supporting the forward-looking design motivation.

Aggregated output is provided in:
  artifact/tables/lagged_predictor_regression.csv

Columns:
  - lag_weeks:    lag length (1, 2, 4)
  - coef:         OLS coefficient on lagged fragility
  - se:           heteroskedasticity-robust SE
  - pval:         p-value
  - R2:           R-squared
  - N:            number of cell-week observations

Usage (requires full dataset):
  python lagged_predictor.py
  # reads: data/processed/auctions_full_usd.parquet
  # writes: results/tables/lagged_predictor_regression.csv
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

def run(parquet_path="data/processed/auctions_full_usd.parquet",
        out_path="results/tables/lagged_predictor_regression.csv"):
    df = pd.read_parquet(parquet_path,
        columns=["market_cell", "block_timestamp", "fragility"])
    df["week"] = df["block_timestamp"].dt.to_period("W").astype(str)
    cw = df.groupby(["market_cell", "week"])["fragility"].mean().reset_index()
    cw = cw.sort_values(["market_cell", "week"])

    rows = []
    for lag in [1, 2, 4]:
        cw[f"lag{lag}"] = cw.groupby("market_cell")["fragility"].shift(lag)
        sub = cw.dropna(subset=[f"lag{lag}"])
        m = smf.ols(f"fragility ~ lag{lag} + C(market_cell)",
                    data=sub).fit(cov_type="HC3")
        rows.append({
            "lag_weeks": lag,
            "coef":  round(m.params[f"lag{lag}"], 6),
            "se":    round(m.bse[f"lag{lag}"], 6),
            "pval":  round(m.pvalues[f"lag{lag}"], 6),
            "R2":    round(m.rsquared, 4),
            "N":     int(m.nobs),
        })
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))

if __name__ == "__main__":
    run()
