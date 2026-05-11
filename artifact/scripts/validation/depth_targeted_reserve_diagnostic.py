"""
P1-2: Depth-targeted reserve diagnostic

Tests fragility-threshold triggers vs the baseline quality-floor reserve.

Depth_a = 1 - Fragility_a = Score_2_other / Score_1

Three trigger rules compared to baseline quality floor:
  1. Quality floor (baseline):  score_1 < reserve_benchmark
  2. Depth < 0.50 (frag > 0.5): fraction of auctions where replacement depth < 50%
  3. Depth < 0.25 (frag > 0.75): stricter fragility threshold
  4. Cell historical depth quantile: cell-specific depth trigger (tau=0.25 quantile)

Output: results/tables/depth_targeted_reserve.csv
        paper/tables/depth_targeted_reserve.tex
"""
import pandas as pd, numpy as np
from pathlib import Path

OUT_CSV = Path("results/tables/depth_targeted_reserve.csv")
OUT_TEX = Path("paper/tables/depth_targeted_reserve.tex")

df = pd.read_parquet("data/processed/auctions_full_usd.parquet",
    columns=["auction_id","market_cell","block_timestamp","fragility",
             "score_winner","score_runner_up","n_competitive","size_bucket",
             "is_blue_chip","solver_rent_usd"])
df = df.sort_values("block_timestamp").reset_index(drop=True)

# Depth = 1 - fragility = score_2/score_1
df["depth"] = 1 - df["fragility"].clip(0,1)

# Cell historical depth quantile (using first 20% as warm-up)
cut = int(len(df)*0.20)
warm = df.iloc[:cut]; eval_df = df.iloc[cut:].copy()

# Compute cell-level 25th percentile of depth from warm-up
cell_depth_q25 = warm.groupby("market_cell")["depth"].quantile(0.25).rename("cell_depth_q25")
eval_df = eval_df.merge(cell_depth_q25, on="market_cell", how="left")
# Fill missing cells with global 25th percentile
global_q25 = warm["depth"].quantile(0.25)
eval_df["cell_depth_q25"] = eval_df["cell_depth_q25"].fillna(global_q25)

n_eval = len(eval_df)

# Load baseline quality-floor activation from replay
mech = pd.read_csv("results/tables/mechanical_replay.csv",
    usecols=["auction_id","reserve_active"])
eval_df = eval_df.merge(mech, on="auction_id", how="left")
eval_df["reserve_active"] = eval_df["reserve_active"].fillna(False)

# Define triggers
eval_df["trigger_depth_50"]    = eval_df["depth"] < 0.50   # fragility > 0.5
eval_df["trigger_depth_25"]    = eval_df["depth"] < 0.25   # fragility > 0.75
eval_df["trigger_cell_q25"]    = eval_df["depth"] < eval_df["cell_depth_q25"]

def summarize(triggered, label):
    n_trig = triggered.sum()
    if n_trig == 0:
        return {"trigger": label, "activation_rate": 0, "n_triggered": 0,
                "top_frag_decile_activation": 0, "overlap_with_qfloor": 0,
                "median_gap_proxy_usd_triggered": 0}

    top_dec = eval_df["fragility"] >= eval_df["fragility"].quantile(0.90)
    overlap = (triggered & eval_df["reserve_active"]).sum() / max(triggered.sum(), 1)
    med_rent = eval_df.loc[triggered, "solver_rent_usd"].dropna()
    med_rent_w = med_rent.clip(med_rent.quantile(0.01), med_rent.quantile(0.99)).median()

    return {
        "trigger": label,
        "activation_rate": triggered.mean(),
        "n_triggered": int(n_trig),
        "top_frag_decile_activation": triggered[top_dec].mean(),
        "overlap_with_qfloor": overlap,
        "median_gap_proxy_usd_triggered": med_rent_w,
    }

rows = [
    summarize(eval_df["reserve_active"],           "Quality floor (baseline)"),
    summarize(eval_df["trigger_depth_50"],          "Depth < 0.50 (fragility > 0.5)"),
    summarize(eval_df["trigger_depth_25"],          "Depth < 0.25 (fragility > 0.75)"),
    summarize(eval_df["trigger_cell_q25"],          "Cell historical depth Q25"),
]

result = pd.DataFrame(rows)
result.to_csv(OUT_CSV, index=False)
print("=== Depth-targeted reserve diagnostic ===")
print(f"Eval sample: {n_eval:,} auctions")
print(result.to_string(index=False))

# LaTeX table
def pct(v): return f"{v*100:.1f}\\%"
def usd(v): return f"\\${v:.2f}" if abs(v) < 1000 else f"\\${v/1e3:.1f}K"

tex = r"""\begin{table}[t]
\centering\footnotesize
\caption{Depth-targeted reserve diagnostic. Compares the baseline quality-floor trigger
($\text{score}_1 < r_{c,t}$) with three fragility/depth-based triggers.
``Top decile activation'' is the share of the highest-fragility 10\% of auctions that
would be flagged. ``Overlap'' is the fraction of depth-triggered auctions also flagged
by the quality floor.}
\label{tab:depth-reserve}
\begin{tabular}{lrrrr}
\toprule
Trigger & Activation & Top decile act. & Overlap w/ QF & Med. gap proxy \\
\midrule
"""
for _, row in result.iterrows():
    med_gap = row["median_gap_proxy_usd_triggered"]
    tex += (f"{row['trigger']} & "
            f"{pct(row['activation_rate'])} & "
            f"{pct(row['top_frag_decile_activation'])} & "
            f"{pct(row['overlap_with_qfloor'])} & "
            f"{usd(med_gap)} \\\\\n")
tex += r"""\bottomrule
\end{tabular}
\footnotesize Quality floor and depth-targeted triggers flag different failure modes:
quality floor targets low-quality auctions overall; depth triggers specifically flag
fragile auctions with weak runner-up replacement.
\end{table}
"""
OUT_TEX.write_text(tex)
print(f"\n✓ {OUT_CSV}")
print(f"✓ {OUT_TEX}")
