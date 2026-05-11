"""
Chapter 7.3: Behavior-Aware Replay

Models how rational solvers adjust their bids after learning the reserve:

    b'_{s,a} = max(b_{s,a}, r_a + epsilon)  if v_{s,a} >= r_a + kappa_{s,a}
             = ∅                             if v_{s,a} <  r_a + kappa_{s,a}

Parameters:
  - epsilon : minimum margin above reserve needed to win
  - kappa   : solver execution cost buffer (gas + risk + capital)

This gives a more realistic (intermediate) estimate of the mechanism's effect.
The mechanical replay is the upper bound; adversarial replay is the lower bound.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "mechanism"))
from shadow_reserve import ShadowReserveBank, ReserveParams
from sr_auction import ShadowReserveAuction, ReserveMode, AuctionResult

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str) -> tuple[ReserveParams, dict]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return ReserveParams(**cfg["reserve"]), cfg


def adjust_bids(auction: dict, reserve: float,
                epsilon: float, kappa: float) -> dict:
    """
    Apply behavioral adjustment to solver bids.

    We model this at the auction level:
      - winner_score is the max bid any solver would submit
      - If winner cannot profitably beat reserve (score < reserve + kappa),
        they exit → use runner-up as winner (if runner-up beats reserve).
      - Otherwise winner adjusts bid upward to max(original, reserve + epsilon).
    """
    a = auction.copy()
    score_1 = float(a.get("score_winner") or 0)
    score_2 = float(a.get("score_runner_up") or 0)

    can_profitably_win = score_1 >= (reserve + kappa)

    if can_profitably_win:
        # Winner stays; bid adjusted upward to beat reserve if needed
        a["score_winner_adjusted"] = max(score_1, reserve + epsilon)
    else:
        # Winner exits; runner-up takes over (if profitable)
        if score_2 >= (reserve + kappa):
            a["score_winner_adjusted"] = max(score_2, reserve + epsilon)
            a["winner_exited"] = True
        else:
            # Nobody beats reserve profitably → fallback
            a["score_winner_adjusted"] = np.nan
            a["winner_exited"] = True

    return a


def run_behavior_aware_replay(df: pd.DataFrame, params: ReserveParams,
                               epsilon: float = 1e-6, kappa: float = 0.0,
                               init_frac: float = 0.20) -> pd.DataFrame:
    bank = ShadowReserveBank(params)

    cutoff  = int(len(df) * init_frac)
    warm_df = df.iloc[:cutoff]
    eval_df = df.iloc[cutoff:].copy()

    for _, row in warm_df.iterrows():
        bank.update(row.to_dict())

    rows = []
    for _, row in eval_df.iterrows():
        a_dict   = row.to_dict()
        reserve  = bank.get_reserve(a_dict.get("market_cell", ""))

        a_adj    = adjust_bids(a_dict, reserve, epsilon, kappa)
        score_adj = a_adj.get("score_winner_adjusted", np.nan)
        exited    = a_adj.get("winner_exited", False)

        # Original
        surplus_orig = float(a_dict.get("surplus_actual") or 0)
        rent_orig    = float(a_dict.get("solver_rent") or np.nan)

        # Adjusted outcome
        if np.isnan(score_adj):
            surplus_sr = surplus_orig * (reserve / max(
                float(a_dict.get("score_winner") or 1), 1e-10))
            rent_sr    = 0.0
            active     = True
        elif score_adj >= reserve:
            ratio      = score_adj / max(
                float(a_dict.get("score_winner") or 1), 1e-10)
            surplus_sr = surplus_orig * ratio
            rent_sr    = rent_orig * (
                (score_adj - reserve) /
                max(score_adj - float(a_dict.get("reference_score") or 0), 1e-10)
            )
            active = score_adj > float(a_dict.get("score_winner") or score_adj)
        else:
            surplus_sr = surplus_orig
            rent_sr    = rent_orig
            active     = False

        rows.append({
            "auction_id":      a_dict.get("auction_id"),
            "market_cell":     a_dict.get("market_cell"),
            "fragility":       a_dict.get("fragility"),
            "reserve":         reserve,
            "reserve_active":  active,
            "winner_exited":   exited,
            "surplus_orig":    surplus_orig,
            "surplus_sr":      surplus_sr,
            "delta_surplus":   surplus_sr - surplus_orig,
            "rent_orig":       rent_orig,
            "rent_sr":         rent_sr,
            "delta_rent":      rent_sr - rent_orig,
        })

        bank.update(a_dict)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",          required=True)
    p.add_argument("--reserve-params",
                   default="experiments/mechanism/params.yaml")
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--kappa",   type=float, default=0.0)
    p.add_argument("--output",         required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    df = df.sort_values("block_timestamp").reset_index(drop=True)

    params, _ = load_params(args.reserve_params)
    results   = run_behavior_aware_replay(df, params,
                                          epsilon=args.epsilon,
                                          kappa=args.kappa)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    logger.info("Behavior-aware replay done → %s  shape=%s",
                args.output, results.shape)
    logger.info("mean delta_surplus = %.4f", results["delta_surplus"].mean())
    logger.info("mean delta_rent    = %.4f", results["delta_rent"].mean())


if __name__ == "__main__":
    main()
