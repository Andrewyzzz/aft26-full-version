import pandas as pd, numpy as np
from pathlib import Path
Path("results/tables").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/auctions_main_usd.parquet")
n_main = len(df)
n_reg  = (df["solver_rent_usd"].notna() & df["fragility"].notna() & df["volume_raw"].notna() & df["n_valid"].notna()).sum()
n_gap  = (df["score_gap_bps"].notna() & df["fragility"].notna() & df["volume_raw"].notna() & df["n_valid"].notna()).sum()
n_vol  = (df["volume_usd"].notna() & df["volume_usd"].between(1, 1e7)).sum()
n_rep  = int(n_main * 0.80)

rows = [
    ("1. Main auction sample",      n_main, "fragility, N_comp, HHI, all non-USD tables",    "Parsed post-CIP-67 auctions with valid solutions"),
    ("2. Score-to-USD sample",      n_reg,  "solver_rent_usd regression",                     "solver_rent_usd + fragility + controls non-null"),
    ("3. USD notional sample",      n_gap,  "score_gap_bps regression",                       "score_gap_bps non-null (sell-token price available)"),
    ("4. Sane-volume denominator",  n_vol,  "economic scale bps / aggregate rent",            "volume_usd in [$1, $10M]"),
    ("5. Replay eval sample",       n_rep,  "shadow-reserve counterfactual replay",           "First 20% excluded for reserve warm-up"),
]
result = pd.DataFrame(rows, columns=["step","n","used_for","filter"])
result.to_csv("results/tables/sample_flow.csv", index=False)
print(result[["step","n","filter"]].to_string(index=False))
