#!/usr/bin/env python3
"""
Lagged fragility predictor regression.

Aggregated output: artifact/tables/lagged_predictor_regression.csv
Full parquet required for recomputation.
"""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE   = ROOT / "tables" / "lagged_predictor_regression.csv"
DEFAULT_PARQUET = Path("data/processed/auctions_full_usd.parquet")

def recompute(parquet_path):
    import statsmodels.formula.api as smf
    df = pd.read_parquet(parquet_path,
        columns=["market_cell","block_timestamp","fragility"])
    df["week"] = df["block_timestamp"].dt.to_period("W").astype(str)
    cw = df.groupby(["market_cell","week"])["fragility"].mean().reset_index()
    cw = cw.sort_values(["market_cell","week"])
    rows = []
    for lag in [1, 2, 4]:
        cw[f"lag{lag}"] = cw.groupby("market_cell")["fragility"].shift(lag)
        sub = cw.dropna(subset=[f"lag{lag}"])
        m = smf.ols(f"fragility ~ lag{lag} + C(market_cell)",
                    data=sub).fit(cov_type="HC3")
        rows.append({"lag_weeks":lag,
            "coef":round(m.params[f"lag{lag}"],6),
            "se":round(m.bse[f"lag{lag}"],6),
            "pval":round(m.pvalues[f"lag{lag}"],6),
            "R2":round(m.rsquared,4),"N":int(m.nobs)})
    return pd.DataFrame(rows)

def main():
    if DEFAULT_PARQUET.exists():
        print("Full parquet found; recomputing lagged predictor regression...")
        print(recompute(DEFAULT_PARQUET).to_string(index=False))
    elif DEFAULT_TABLE.exists():
        print("Full parquet not in lightweight artifact; showing aggregated output:")
        print(pd.read_csv(DEFAULT_TABLE).to_string(index=False))
    else:
        raise FileNotFoundError(f"Neither full parquet nor aggregated table found: {DEFAULT_TABLE}")

if __name__ == "__main__":
    main()
