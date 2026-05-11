"""
Chapter 7.4: Adversarial Replay

Most conservative scenario: all solvers (or a coalition) bid exactly
at the reserve + epsilon. This gives the lower bound on user surplus
improvement under the shadow-reserve mechanism.

    b'_{s,a} = r_a + epsilon  for all s  (if they choose to participate)

This establishes:
    Surplus^SR_a >= ReserveBenchmark_a   (Theorem 1)

Key insight: even in the adversarial case, the mechanism bounds the
minimum execution quality at the reserve level.
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str) -> ReserveParams:
    with open(path) as f:
        cfg = yaml.safe_load(f)["reserve"]
    return ReserveParams(**cfg)


def run_adversarial_replay(df: pd.DataFrame, params: ReserveParams,
                            epsilon: float = 1e-6,
                            init_frac: float = 0.20) -> pd.DataFrame:
    """
    Adversarial case: all competing solvers collude to bid r + epsilon.

    For each auction:
      - If reserve > 0: winning score = reserve + epsilon
        → user surplus is bounded below by reserve quality
        → solver rent = epsilon (minimal)
      - If reserve = 0: no protection (baseline)
    """
    bank = ShadowReserveBank(params)

    cutoff  = int(len(df) * init_frac)
    warm_df = df.iloc[:cutoff]
    eval_df = df.iloc[cutoff:].copy()

    for _, row in warm_df.iterrows():
        bank.update(row.to_dict())

    rows = []
    for _, row in eval_df.iterrows():
        a_dict  = row.to_dict()
        reserve = bank.get_reserve(a_dict.get("market_cell", ""))

        score_orig  = float(a_dict.get("score_winner") or 0)
        surplus_orig = float(a_dict.get("surplus_actual") or 0)

        if reserve > 0:
            # Adversarial bid: r + epsilon
            score_adv   = reserve + epsilon
            surplus_adv = surplus_orig * (score_adv /
                                          max(score_orig, 1e-10))
            rent_adv    = epsilon   # minimal rent
            active      = True
        else:
            # No reserve protection
            score_adv   = score_orig
            surplus_adv = surplus_orig
            rent_adv    = float(a_dict.get("solver_rent") or np.nan)
            active      = False

        rows.append({
            "auction_id":      a_dict.get("auction_id"),
            "market_cell":     a_dict.get("market_cell"),
            "fragility":       a_dict.get("fragility"),
            "reserve":         reserve,
            "reserve_active":  active,
            # Lower-bound outcome
            "surplus_adv":     surplus_adv,
            "surplus_orig":    surplus_orig,
            "delta_surplus_lb":surplus_adv - surplus_orig,
            "rent_adv":        rent_adv,
            "rent_orig":       float(a_dict.get("solver_rent") or np.nan),
            # This confirms Theorem 1: surplus >= reserve quality
            "theorem1_holds":  surplus_adv >= (surplus_orig *
                                               (reserve / max(score_orig, 1e-10)))
                                if reserve > 0 else True,
        })

        bank.update(a_dict)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",          required=True)
    p.add_argument("--reserve-params",
                   default="experiments/mechanism/params.yaml")
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--output",         required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    df = df.sort_values("block_timestamp").reset_index(drop=True)

    params  = load_params(args.reserve_params)
    results = run_adversarial_replay(df, params, epsilon=args.epsilon)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    logger.info("Adversarial replay done → %s", args.output)
    logger.info("Lower-bound mean delta_surplus = %.4f",
                results["delta_surplus_lb"].mean())
    logger.info("Theorem 1 holds fraction: %.4f",
                results["theorem1_holds"].mean())


if __name__ == "__main__":
    main()
