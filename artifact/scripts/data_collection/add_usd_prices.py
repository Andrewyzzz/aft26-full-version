"""
Join token price data to auction dataset, adding USD-normalized columns.

New columns added:
  - eth_price_usd    : ETH/USD price on auction date
  - score_winner_usd : score_winner / 1e18 * eth_price  (USD)
  - ref_score_usd    : ref_score / 1e18 * eth_price      (USD)
  - solver_rent_usd  : solver_rent / 1e18 * eth_price    (USD)
  - score_diff_usd   : (score_winner - score_runner_up) / 1e18 * eth_price  (USD, absolute)
  - volume_usd       : sell_amount_human * sell_token_price_usd  (USD notional)
  - score_gap_bps    : 1e4 * score_diff_usd / volume_usd  (basis points, main gap metric)
  - exec_quality     : ref_score_usd / score_winner_usd  (= 1 - fragility)

Paper metrics:
  fragility      = 1 - score_runner_up / score_winner      (unchanged, from build_dataset)
  score_gap_bps  = 1e4 * score_diff_usd / volume_usd      (NEW: volume-normalized gap)
  score_diff_usd = absolute USD difference                  (supplementary)

Note on score units:
  CoW Protocol scores represent user surplus in Wei (1 ETH = 1e18 Wei).
  Dividing by 1e18 converts to ETH; multiplying by ETH/USD gives USD value.

Usage:
    python add_usd_prices.py \\
        --auctions data/processed/auctions_main.parquet \\
        --prices   data/processed/token_prices.parquet \\
        --output   data/processed/auctions_main_usd.parquet
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEI_PER_ETH = 1e18

# Token decimals (sell-side amount conversion)
TOKEN_DECIMALS = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 18,  # WETH
    "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee": 18,  # ETH (native)
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,   # USDT
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,  # DAI
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": 8,   # WBTC
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0": 18,  # wstETH
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": 18,  # stETH
    "0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee": 18,  # weETH
    "0x514910771af9ca656af840dff83e8264ecf986ca": 18,  # LINK
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": 18,  # UNI
}


def add_usd_columns(auctions: pd.DataFrame,
                    prices: pd.DataFrame) -> pd.DataFrame:
    """
    Join ETH price to auctions and compute USD-normalized score metrics.
    """
    # Build ETH daily price lookup: date → price_usd
    eth_prices = (prices[prices["token_addr"] == "eth"]
                  .copy()
                  .assign(date=lambda x: pd.to_datetime(x["date"]).dt.date)
                  .set_index("date")["price_usd"])

    # Auction date
    df = auctions.copy()
    df["_date"] = df["block_timestamp"].dt.date

    # Fill price gaps with forward/backward fill (handles weekend gaps etc.)
    all_dates = pd.date_range(
        df["_date"].min(), df["_date"].max(), freq="D"
    ).date
    eth_prices_full = eth_prices.reindex(all_dates).ffill().bfill()

    # Map ETH price by date
    df["eth_price_usd"] = df["_date"].map(eth_prices_full)

    missing = df["eth_price_usd"].isna().mean()
    if missing > 0.05:
        logger.warning("%.1f%% of auctions missing ETH price — check date range",
                       missing * 100)
    else:
        logger.info("ETH price coverage: %.1f%%", (1 - missing) * 100)

    # ---- Score metrics (Wei → USD via ETH price) ----
    df["score_winner_usd"] = df["score_winner"] / WEI_PER_ETH * df["eth_price_usd"]
    df["ref_score_usd"]    = df["ref_score"]    / WEI_PER_ETH * df["eth_price_usd"]
    df["solver_rent_usd"]  = df["solver_rent"]  / WEI_PER_ETH * df["eth_price_usd"]

    # Absolute score difference in USD (supplementary metric)
    score_diff_col = "score_diff" if "score_diff" in df.columns else None
    if score_diff_col:
        df["score_diff_usd"] = df["score_diff"] / WEI_PER_ETH * df["eth_price_usd"]
    else:
        df["score_diff_usd"] = (
            (df["score_winner"] - df["score_runner_up"]) / WEI_PER_ETH
            * df["eth_price_usd"]
        )

    # ---- Volume in USD (sell token notional) ----
    # Build per-token price lookup: (token_addr, date) → price_usd
    tok_prices = (prices[prices["token_addr"] != "eth"]
                  .copy()
                  .assign(date=lambda x: pd.to_datetime(x["date"]).dt.date))
    tok_lookup = tok_prices.set_index(["token_addr", "date"])["price_usd"]

    sell_col = "sell_token_addr" if "sell_token_addr" in df.columns else "token_addr_0"

    def compute_volume_usd(row):
        raw   = row.get("volume_raw", 0)
        token = str(row.get(sell_col, "")).lower()
        date  = row["_date"]
        if not raw or raw == 0 or not token:
            return np.nan
        decimals = TOKEN_DECIMALS.get(token, 18)
        amount   = raw / (10 ** decimals)
        # Look up token price; fall back to ETH price for ETH-like tokens
        try:
            price = tok_lookup.loc[(token, date)]
        except KeyError:
            if token in ("0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                         "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"):
                price = row["eth_price_usd"]
            else:
                return np.nan
        return float(amount) * float(price)

    logger.info("Computing volume_usd (this may take a moment)…")
    df["volume_usd"] = df.apply(compute_volume_usd, axis=1)

    vol_coverage = df["volume_usd"].notna().mean()
    logger.info("volume_usd coverage: %.1f%%", vol_coverage * 100)

    # ---- score_gap_bps: main paper gap metric ----
    # score_gap_bps = 10^4 * score_diff_usd / volume_usd
    df["score_gap_bps"] = np.where(
        (df["volume_usd"] > 0) & df["score_diff_usd"].notna(),
        1e4 * df["score_diff_usd"] / df["volume_usd"],
        np.nan,
    )
    # Cap extreme values (>10,000 bps = >100% is economically implausible)
    df["score_gap_bps"] = df["score_gap_bps"].clip(upper=10_000)

    # Execution quality proxy (= 1 - fragility, but now in USD terms)
    df["exec_quality"] = np.where(
        df["score_winner_usd"] > 0,
        df["ref_score_usd"] / df["score_winner_usd"],
        np.nan,
    )

    df = df.drop(columns=["_date"])

    logger.info("USD columns added. Sample:")
    sample = df[["score_winner_usd", "solver_rent_usd",
                 "eth_price_usd"]].describe().round(2)
    logger.info("\n%s", sample)

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== USD-Normalized Dataset Summary ===")
    print(f"Auctions:   {len(df):,}")
    print(f"Date range: {df['block_timestamp'].min().date()} → "
          f"{df['block_timestamp'].max().date()}")

    # Win winsorization for display
    for col in ["solver_rent_usd", "score_winner_usd", "score_diff_usd",
                "score_gap_bps", "volume_usd"]:
        if col in df.columns:
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col + "_w"] = df[col].clip(lo, hi)

    print()
    print("─── Core metrics (winsorized at 1%/99%) ───")
    print(f"{'Metric':<25} {'Median':>12} {'Mean':>12} {'Coverage':>10}")
    print("─" * 62)

    rows = [
        ("score_winner_usd", "Auction value (USD)"),
        ("solver_rent_usd",  "Solver rent (USD)"),
        ("score_diff_usd",   "Score diff USD (abs)"),
        ("score_gap_bps",    "Score gap (bps)"),
        ("volume_usd",       "Volume USD"),
    ]
    for col, label in rows:
        wcol = col + "_w"
        if wcol not in df.columns:
            continue
        s = df[wcol].dropna()
        cov = df[col].notna().mean()
        print(f"  {label:<23} {s.median():>12,.2f} {s.mean():>12,.2f} {cov:>9.1%}")

    print()
    print("─── Fragility × solver rent (winsorized median) ───")
    df3 = df.dropna(subset=["fragility", "solver_rent_usd_w"])
    df3 = df3.copy()
    df3["frag_q"] = pd.qcut(df3["fragility"], q=5,
                             labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"],
                             duplicates="drop")
    grp = df3.groupby("frag_q", observed=True).agg(
        median_rent=("solver_rent_usd_w", "median"),
        median_gap_bps=("score_gap_bps_w", "median") if "score_gap_bps_w" in df3 else ("fragility", "count"),
        n=("auction_id", "count"),
    )
    print(grp.round(2).to_string())
    print()
    if "solver_rent_usd_w" in df3.columns:
        q1 = df3[df3["frag_q"] == "Q1(低)"]["solver_rent_usd_w"].median()
        q5 = df3[df3["frag_q"] == "Q5(高)"]["solver_rent_usd_w"].median()
        if q1 > 0:
            print(f"  Q5 / Q1 rent ratio: {q5/q1:.1f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--auctions", required=True)
    p.add_argument("--prices",   required=True)
    p.add_argument("--output",   required=True)
    args = p.parse_args()

    logger.info("Loading auctions…")
    auctions = pd.read_parquet(args.auctions)

    logger.info("Loading prices…")
    prices = pd.read_parquet(args.prices)

    result = add_usd_columns(auctions, prices)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    logger.info("Saved → %s", args.output)

    print_summary(result)


if __name__ == "__main__":
    main()
