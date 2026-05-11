"""
Chapter 7.2: Mechanical Replay

Replays the historical auction dataset with the shadow reserve added,
but WITHOUT changing any solver bids. This is the upper bound on
the mechanism's user-surplus improvement.

    B'_a = B_a ∪ {r_a}   (bids unchanged, reserve added)

Outputs:
  - results/tables/mechanical_replay.csv
  - Per-auction delta_surplus, delta_rent, reserve_active flag
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Allow imports from sibling directory
sys.path.insert(0, str(Path(__file__).parent.parent / "mechanism"))
from shadow_reserve import ShadowReserveBank, ReserveParams
from sr_auction import ShadowReserveAuction, ReserveMode, AuctionResult

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str) -> ReserveParams:
    with open(path) as f:
        cfg = yaml.safe_load(f)["reserve"]
    return ReserveParams(**cfg)


def run_mechanical_replay(df: pd.DataFrame,
                          params: ReserveParams,
                          init_frac: float = 0.20) -> pd.DataFrame:
    """
    Main loop: for each auction in chronological order,
    1. Retrieve current reserve for its market cell
    2. Run SR auction (original bids unchanged)
    3. Update reserve with new data point

    Returns a DataFrame with original and SR outcomes side by side.
    """
    bank    = ShadowReserveBank(params)
    auction = ShadowReserveAuction(bank, mode=ReserveMode.HARD)

    # Warm-up phase (excluded from evaluation)
    cutoff   = int(len(df) * init_frac)
    warm_df  = df.iloc[:cutoff]
    eval_df  = df.iloc[cutoff:].copy()

    for _, row in warm_df.iterrows():
        bank.update(row.to_dict())

    rows = []
    for _, row in eval_df.iterrows():
        a_dict = row.to_dict()
        orig, sr = auction.run(a_dict)
        bank.update(a_dict)

        rows.append({
            "auction_id":         orig.auction_id,
            "market_cell":        a_dict.get("market_cell"),
            "fragility":          a_dict.get("fragility"),
            "reserve":            sr.reserve,
            "reserve_active":     sr.reserve_active,
            # Surplus
            "surplus_orig":       orig.user_surplus,
            "surplus_sr":         sr.user_surplus,
            "delta_surplus":      sr.user_surplus - orig.user_surplus,
            # Rent
            "rent_orig":          orig.solver_rent,
            "rent_sr":            sr.solver_rent,
            "delta_rent":         sr.solver_rent - orig.solver_rent,
            # Winner change
            "winner_changed":     sr.winner_solver != orig.winner_solver,
            "fallback":           sr.fallback_triggered,
        })

    results = pd.DataFrame(rows)
    return results


def summarize(results: pd.DataFrame) -> dict:
    active = results[results["reserve_active"]]
    return {
        "n_eval":               len(results),
        "n_reserve_active":     len(active),
        "reserve_activation_rate": len(active) / len(results),
        "mean_delta_surplus":   results["delta_surplus"].mean(),
        "mean_delta_surplus_active": active["delta_surplus"].mean()
                                     if len(active) else np.nan,
        "mean_delta_rent":      results["delta_rent"].mean(),
        "winner_change_rate":   results["winner_changed"].mean(),
        "fallback_rate":        results["fallback"].mean(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",         required=True)
    p.add_argument("--reserve-params",
                   default="experiments/mechanism/params.yaml")
    p.add_argument("--output",        required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    df = df.sort_values("block_timestamp").reset_index(drop=True)

    params  = load_params(args.reserve_params)
    results = run_mechanical_replay(df, params)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    summary = summarize(results)
    for k, v in summary.items():
        logger.info("  %-35s %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    logger.info("Mechanical replay done → %s", args.output)


if __name__ == "__main__":
    main()
