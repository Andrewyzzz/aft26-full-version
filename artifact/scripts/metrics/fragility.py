"""
Chapter 3: Core Competition Fragility Metrics

Computes the four key auction-level indicators:
  1. N_competitive  — effective solver count
  2. Gap            — normalized score gap (winner vs runner-up)
  3. HHI            — local market concentration
  4. Fragility      — winner indispensability (main indicator)

Usage:
    python fragility.py \\
        --input  data/processed/auctions.parquet \\
        --output results/tables/fragility_summary.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Competitive proximity threshold
EPSILON_DEFAULT = 0.05
EPSILON_ROBUSTNESS = [0.01, 0.03, 0.05, 0.10]


# -------------------------------------------------------------------
# Auction-level indicators  (Chapter 3.4 – 3.7)
# -------------------------------------------------------------------

def compute_n_competitive(df: pd.DataFrame,
                           epsilon: float = EPSILON_DEFAULT) -> pd.Series:
    """
    N^competitive_a = |{s : Score_{s,a} >= (1 - epsilon) * Score_{1,a}}|
    Already pre-computed in build_dataset.py; this allows re-computation
    with different epsilon for robustness checks.
    """
    return df["n_competitive"]  # placeholder — override if raw bids available


def compute_score_gap(df: pd.DataFrame) -> pd.Series:
    """Gap_a = (Score_{1,a} - Score_{2,a}) / Volume_a"""
    return df["score_gap"]


def compute_fragility(df: pd.DataFrame) -> pd.Series:
    """
    Fragility_a = 1 - ReferenceScore_{winner,a} / TotalScore_a

    ~0  => winner replaceable, healthy competition
    ~1  => winner indispensable, fragile competition
    """
    return df["fragility"]


def compute_hhi(df: pd.DataFrame,
                time_window: str = "W",
                share_col: str = "score_winner") -> pd.DataFrame:
    """
    HHI_{c,t} = sum_s share_{s,c,t}^2

    Groups by (market_cell, time window). Returns a DataFrame with columns:
      market_cell | period_start | hhi | n_solvers_active
    """
    df = df.copy()
    df["period"] = df["block_timestamp"].dt.to_period(
        time_window   # e.g. "7D", "W", "ME"
    )
    # Share of each solver within (market_cell, period)
    grp = df.groupby(["market_cell", "period", "winner_solver"])[share_col].sum()
    total = grp.groupby(level=["market_cell", "period"]).transform("sum")
    share = (grp / total) ** 2
    hhi = share.groupby(level=["market_cell", "period"]).sum().reset_index()
    hhi.columns = ["market_cell", "period", "hhi"]

    n_active = (df.groupby(["market_cell", "period"])["winner_solver"]
                  .nunique()
                  .reset_index(name="n_solvers_active"))
    hhi = hhi.merge(n_active, on=["market_cell", "period"])
    return hhi


# -------------------------------------------------------------------
# Summary statistics  (Table 1 in paper)
# -------------------------------------------------------------------

def fragility_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Use is_blue_chip column (set by build_dataset.py from token addresses)
    # Fallback: if column missing, classify by token_pair short names
    if "is_blue_chip" in df.columns:
        blue_mask = df["is_blue_chip"] == True
    else:
        blue_mask = df["token_pair"].isin(BLUE_CHIP_PAIRS)

    stats = {
        "all":        df,
        "blue_chip":  df[blue_mask],
        "long_tail":  df[~blue_mask],
        "small":      df[df["size_bucket"] == "small"],
        "large":      df[df["size_bucket"] == "large"],
    }
    rows = []
    for label, sub in stats.items():
        if sub.empty:
            continue
        rows.append({
            "subset":               label,
            "n_auctions":           len(sub),
            "frac_fragile_50":      (sub["fragility"] > 0.5).mean(),
            "frac_fragile_75":      (sub["fragility"] > 0.75).mean(),
            "median_fragility":     sub["fragility"].median(),
            "mean_n_competitive":   sub["n_competitive"].mean(),
            "median_score_gap":     sub["score_gap"].median(),
        })
    return pd.DataFrame(rows)


# Blue-chip pairs (extend as needed)
BLUE_CHIP_PAIRS = {
    "WETH/USDC", "USDC/WETH",
    "WETH/USDT", "USDT/WETH",
    "WBTC/WETH", "WETH/WBTC",
    "DAI/USDC",  "USDC/DAI",
}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epsilon", type=float, default=EPSILON_DEFAULT)
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d auctions", len(df))

    summary = fragility_summary(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))

    # HHI by market cell
    hhi = compute_hhi(df)
    hhi_out = Path(args.output).parent / "hhi_by_cell.csv"
    hhi.to_csv(hhi_out, index=False)
    logger.info("HHI table saved → %s", hhi_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main()
