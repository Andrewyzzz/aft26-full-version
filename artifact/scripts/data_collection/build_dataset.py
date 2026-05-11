"""
Build the analysis-ready auction-level dataset from raw JSONL.

Handles CIP-67 era CoW Protocol API format (2025-06 onwards):
  - solutions is a LIST, not a dict
  - score is a STRING, must be int()-converted
  - referenceScores field does NOT exist → use runner-up (ranking=2) score
  - timestamp estimated from auctionStartBlock

Output columns (per auction):
  auction_id, chain, block_timestamp, token_pair, volume_raw,
  n_valid, n_competitive, n_filtered,
  score_winner, score_runner_up, ref_score,
  fragility, score_gap, winner_solver,
  surplus_proxy, solver_rent_proxy,
  size_bucket, market_cell

Usage:
    python build_dataset.py \\
        --input  data/raw/cow_mainnet_pilot.jsonl \\
        --output data/processed/auctions_pilot.parquet
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# within this fraction of winner score → "competitive"
EPSILON_COMPETITIVE = 0.05

# Order size buckets in raw token units (approx; will refine with USD prices)
# Using score magnitude as proxy until USD prices are added
SIZE_QUANTILES = [0, 0.25, 0.75, 0.90, 1.0]
SIZE_LABELS    = ["small", "medium", "large", "xl"]

BLUE_CHIP_ADDRESSES = {
    # WETH
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    # USDC
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    # USDT
    "0xdac17f958d2ee523a2206206994597c13d831ec7",
    # WBTC
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
    # DAI
    "0x6b175474e89094c44da98b954eedeac495271d0f",
}


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default=np.nan) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def parse_record(raw: dict) -> Optional[dict]:
    """Parse one raw JSONL record into a flat analysis row."""
    try:
        auction_id = raw.get("_auction_id") or raw.get("auctionId")
        chain      = raw.get("_chain", "mainnet")
        block_ts   = raw.get("_block_timestamp", 0)

        solutions = raw.get("solutions") or []
        if not solutions:
            return None

        # Filter out filteredOut solutions
        valid_solutions = [s for s in solutions if not s.get("filteredOut", False)]
        if not valid_solutions:
            return None

        # Sort by score descending — in CIP-67 combinatorial auction,
        # `ranking` field != score order (co-winners can have ranking=2 with
        # lower score than ranking=3). We use score order for fragility.
        scored = sorted(
            [{"score": _safe_int(s.get("score", "0")),
              "solver": s.get("solver", ""),
              "is_winner": s.get("isWinner", False),
              "ranking": s.get("ranking", 999)}
             for s in valid_solutions
             if _safe_int(s.get("score", "0")) > 0],
            key=lambda x: x["score"],
            reverse=True,
        )

        if not scored:
            return None

        score_1       = scored[0]["score"]
        winner_solver = scored[0]["solver"]

        # score_2_other: best score from a DIFFERENT solver (not winner).
        # This is the exact CoW reference score for single-winner auctions:
        #   ReferenceScore_a = max_{s != s*} Score_{s,a}
        # We also keep score_2_solution (second-highest overall) for validation.
        other_scores = [s["score"] for s in scored if s["solver"] != winner_solver]
        score_2_other    = max(other_scores) if other_scores else 0
        score_2_solution = scored[1]["score"] if len(scored) > 1 else 0

        # ref_score uses score_2_other (audit-defensible)
        ref_score = score_2_other
        # Legacy field kept for validation (may equal score_2_other or be winner's 2nd solution)
        score_2   = score_2_other

        n_valid       = len(scored)       # non-filtered solutions with score > 0
        n_filtered    = len(solutions) - len(valid_solutions)
        n_competitive = sum(
            1 for s in scored
            if s["score"] >= (1 - EPSILON_COMPETITIVE) * score_1
        )

        # Fragility
        fragility = (1 - ref_score / score_1) if score_1 > 0 else np.nan

        # Token pair: try clearingPrices of best solution first,
        # then fall back to auction.prices keys (batch-level token universe).
        # clearingPrices is often empty in CIP-67; use auction.prices as proxy.
        winner_sol = scored[0]  # highest score solution
        winner_sol_raw = next(
            (s for s in valid_solutions
             if s.get("solver") == winner_sol["solver"]), {}
        )
        clearing_prices = winner_sol_raw.get("clearingPrices") or {}
        token_addrs = sorted(k.lower() for k in clearing_prices.keys())

        # Fallback: use pre-filtered 2-token price dict saved during fetch
        if len(token_addrs) < 2:
            slim_prices = raw.get("auction_prices_2tok") or {}
            batch_tokens = sorted(k.lower() for k in slim_prices.keys())
            if len(batch_tokens) == 2:
                token_addrs = batch_tokens

        if len(token_addrs) >= 2:
            t0, t1 = token_addrs[0], token_addrs[1]
            token_pair   = f"{t0}/{t1}"
            is_blue_chip = t0 in BLUE_CHIP_ADDRESSES and t1 in BLUE_CHIP_ADDRESSES
        elif len(token_addrs) == 1:
            token_pair   = f"{token_addrs[0]}/?"
            is_blue_chip = False
        else:
            token_pair   = "multi"   # multi-pair batch
            is_blue_chip = False

        # Volume: sum of sellAmounts across winner's orders (raw token units)
        # Fix: use winner_sol_raw (full API response), not winner_sol (simplified dict)
        winner_orders = winner_sol_raw.get("orders") or []
        volume_raw = sum(_safe_int(o.get("sellAmount", "0")) for o in winner_orders)

        # Sell token address (first token in clearing prices = sell side)
        sell_token_addr = token_addrs[0] if token_addrs else ""

        # Score gap NOTE: currently score_gap == fragility because ref_score = score_2.
        # The true gap_bps = score_diff_usd / volume_usd is computed in add_usd_prices.py
        # once ETH prices and token decimals are available.
        # Here we only store the raw components.
        score_diff = score_1 - score_2   # in raw score units (Wei)

        # Solver rent proxy (in score units = Wei)
        solver_rent = score_1 - ref_score if ref_score > 0 else np.nan

        # n_orders in this auction batch
        n_orders_batch = len(raw.get("auction", {}).get("orders", []))

        return {
            "auction_id":       auction_id,
            "chain":            chain,
            "block_timestamp":  block_ts,
            "token_pair":       token_pair,
            "is_blue_chip":     is_blue_chip,
            "token_addr_0":     token_addrs[0].lower() if token_addrs else "",
            "token_addr_1":     token_addrs[1].lower() if len(token_addrs) > 1 else "",
            "sell_token_addr":  sell_token_addr,
            "volume_raw":       volume_raw,      # raw sell-token units
            "n_orders_batch":   n_orders_batch,
            "n_valid":          n_valid,
            "n_filtered":       n_filtered,
            "n_competitive":    n_competitive,
            "score_winner":          score_1,
            "score_runner_up":       score_2,        # = score_2_other (best non-winner)
            "score_2_solution":      score_2_solution,  # validation: second-highest overall
            "ref_score_exact_match": int(score_2_solution == score_2_other),  # 1 if identical
            "has_other_solver":      int(len(other_scores) > 0),
            "ref_score":             ref_score,
            "fragility":             fragility,
            "score_diff":            score_diff,
            "winner_solver":         winner_solver,
            "solver_rent":           solver_rent,
        }

    except Exception as e:
        logger.debug("Parse error: %s | raw=%s", e, str(raw)[:200])
        return None


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    # Size bucket based on score_winner quantile (proxy until USD prices added)
    df["size_bucket"] = pd.qcut(
        df["score_winner"].rank(pct=True),
        q=4, labels=SIZE_LABELS, duplicates="drop"
    )
    # Market cell
    df["market_cell"] = (df["chain"] + "|" +
                         df["token_pair"] + "|" +
                         df["size_bucket"].astype(str))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, total = [], 0
    with in_path.open() as fh:
        for line in tqdm(fh, desc="Parsing"):
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = parse_record(raw)
            if row:
                rows.append(row)

    logger.info("Parsed %d / %d records (%.1f%%)",
                len(rows), total, 100 * len(rows) / max(total, 1))

    df = pd.DataFrame(rows)
    # Score columns are large integers (up to ~10^16); store as float64
    for col in ["score_winner", "score_runner_up", "ref_score",
                "solver_rent", "volume_raw", "score_diff",
                "score_2_solution"]:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    df["block_timestamp"] = pd.to_datetime(
        df["block_timestamp"], unit="s", utc=True
    )
    df = df.sort_values("block_timestamp").reset_index(drop=True)
    df = assign_buckets(df)
    df.to_parquet(out_path, index=False)

    logger.info("Saved → %s  shape=%s", out_path, df.shape)

    print("\n=== Dataset Summary ===")
    print(f"Auctions:           {len(df):,}")
    print(f"Date range:         {df['block_timestamp'].min()} → {df['block_timestamp'].max()}")
    print(f"Unique solvers:     {df['winner_solver'].nunique()}")
    print(f"Unique pairs:       {df['token_pair'].nunique()}")
    print(f"n_valid >= 2:       {(df['n_valid'] >= 2).mean():.1%}")
    print(f"n_competitive >= 2: {(df['n_competitive'] >= 2).mean():.1%}")
    print(f"Fragility valid:    {df['fragility'].notna().mean():.1%}")
    print("\nFragility distribution:")
    print(df["fragility"].describe().round(4))
    print("\nFragility > 0.5:   ", (df["fragility"] > 0.5).mean())
    print("Fragility > 0.75:  ", (df["fragility"] > 0.75).mean())


if __name__ == "__main__":
    main()
