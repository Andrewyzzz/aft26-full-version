import pandas as pd, numpy as np
from pathlib import Path
Path("results/tables").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/auctions_main_usd.parquet")

covered   = df["score_gap_bps"].notna() & df["volume_usd"].notna()
df["usd_covered"] = covered

def stats(sub, label):
    return {
        "sample": label,
        "n": len(sub),
        "share": len(sub)/len(df),
        "median_fragility": sub["fragility"].median(),
        "frac_fragile_50": (sub["fragility"]>0.5).mean(),
        "mean_n_competitive": sub["n_competitive"].mean(),
        "blue_chip_share": sub["is_blue_chip"].mean(),
        "small_share": (sub["size_bucket"]=="small").mean(),
        "large_share": (sub["size_bucket"]=="large").mean(),
        "median_score_diff_usd": sub["score_diff_usd"].median() if "score_diff_usd" in sub else np.nan,
    }

rows = [
    stats(df, "full_sample"),
    stats(df[covered], "usd_covered"),
    stats(df[~covered], "usd_uncovered"),
    stats(df[covered & (df["is_blue_chip"]==True)],  "usd_covered_blue_chip"),
    stats(df[covered & (df["is_blue_chip"]==False)], "usd_covered_long_tail"),
    stats(df[~covered & (df["is_blue_chip"]==True)], "usd_uncovered_blue_chip"),
    stats(df[~covered & (df["is_blue_chip"]==False)],"usd_uncovered_long_tail"),
]
result = pd.DataFrame(rows)
result.to_csv("results/tables/usd_missingness.csv", index=False)
print(result[["sample","n","share","median_fragility","frac_fragile_50","mean_n_competitive","blue_chip_share"]].to_string(index=False))
