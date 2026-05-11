"""
Chapter 5: Shadow-Reserve Auction

Extends the base solver auction with shadow reserve bids:

    B'_a = B_a ∪ R_a
    W^SR_a = argmax_{W ⊆ B'_a} Σ Score(b)

The reserve acts as a minimum-quality floor:
  - If real solvers beat the reserve → auction unchanged (Theorem 3)
  - If real solvers fall below reserve → reserve triggers protection

Implements:
  - Hard reserve  (bid below reserve cannot win)     [main paper]
  - Soft reserve  (triggers audit / reward reduction) [Discussion]
  - Fallback      (order delayed / rerouted)         [Discussion]
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from shadow_reserve import ShadowReserveBank, ReserveParams

logger = logging.getLogger(__name__)


class ReserveMode(Enum):
    HARD     = "hard"      # winning bid must beat reserve
    SOFT     = "soft"      # penalize reward if below reserve
    FALLBACK = "fallback"  # delay/reroute if no bid beats reserve


@dataclass
class AuctionResult:
    auction_id:       str | int
    mode:             str            # "original" or "shadow_reserve"
    winner_solver:    Optional[str]
    winning_score:    float
    reference_score:  float
    reserve:          float
    reserve_active:   bool           # True iff reserve changed the outcome
    user_surplus:     float
    solver_rent:      float
    fallback_triggered: bool


class ShadowReserveAuction:
    """
    Wraps the base solver competition and applies shadow-reserve logic.

    Parameters
    ----------
    reserve_bank : ShadowReserveBank
        Provides per-cell reserve values.
    mode : ReserveMode
        Hard / Soft / Fallback.
    """

    def __init__(self,
                 reserve_bank: ShadowReserveBank,
                 mode: ReserveMode = ReserveMode.HARD):
        self.bank = reserve_bank
        self.mode = mode

    def run(self, auction: dict) -> tuple[AuctionResult, AuctionResult]:
        """
        Run the auction both with and without the shadow reserve.

        Returns (original_result, sr_result).
        """
        cell   = auction["market_cell"]
        r      = self.bank.get_reserve(cell)

        # ---- Original auction outcome ----
        orig = self._original_result(auction)

        # ---- Shadow-reserve auction ----
        if self.mode == ReserveMode.HARD:
            sr = self._hard_reserve(auction, orig, r)
        elif self.mode == ReserveMode.SOFT:
            sr = self._soft_reserve(auction, orig, r)
        else:
            sr = self._fallback_reserve(auction, orig, r)

        return orig, sr

    # ------------------------------------------------------------------
    # Original auction
    # ------------------------------------------------------------------

    def _original_result(self, auction: dict) -> AuctionResult:
        score_1 = float(auction.get("score_winner") or 0)
        ref     = float(auction.get("reference_score") or
                        auction.get("score_runner_up") or 0)
        surplus = float(auction.get("surplus_actual") or 0)
        rent    = score_1 - ref if not np.isnan(ref) else np.nan

        return AuctionResult(
            auction_id      = auction.get("auction_id"),
            mode            = "original",
            winner_solver   = auction.get("winner_solver"),
            winning_score   = score_1,
            reference_score = ref,
            reserve         = 0.0,
            reserve_active  = False,
            user_surplus    = surplus,
            solver_rent     = rent,
            fallback_triggered = False,
        )

    # ------------------------------------------------------------------
    # Hard reserve  (Theorem 1 + 2 + 3)
    # ------------------------------------------------------------------

    def _hard_reserve(self, auction: dict,
                      orig: AuctionResult, r: float) -> AuctionResult:
        score_1 = orig.winning_score
        active  = score_1 < r

        if not active:
            # Reserve inactive → identical to original (Theorem 3)
            return AuctionResult(
                **{**orig.__dict__,
                   "mode": "shadow_reserve",
                   "reserve": r,
                   "reserve_active": False}
            )

        # Reserve triggers: use reserve benchmark as outcome
        # In practice, fallback execution at reference quality.
        # Here we model it as: user gets surplus at reserve quality level.
        surplus_sr = float(auction.get("surplus_ref") or
                           orig.user_surplus * (r / max(score_1, 1e-10)))
        rent_sr    = r - r   # reserve bids earn zero rent by design

        return AuctionResult(
            auction_id      = auction.get("auction_id"),
            mode            = "shadow_reserve",
            winner_solver   = None,    # reserve "solver" (protocol-owned)
            winning_score   = r,
            reference_score = r,
            reserve         = r,
            reserve_active  = True,
            user_surplus    = surplus_sr,
            solver_rent     = 0.0,
            fallback_triggered = False,
        )

    # ------------------------------------------------------------------
    # Soft reserve  (reward penalty only)
    # ------------------------------------------------------------------

    def _soft_reserve(self, auction: dict,
                      orig: AuctionResult, r: float) -> AuctionResult:
        score_1 = orig.winning_score
        active  = score_1 < r

        # Original winner still executes, but reward is penalized
        penalty_factor = min(1.0, score_1 / r) if (active and r > 0) else 1.0
        rent_sr = orig.solver_rent * penalty_factor

        return AuctionResult(
            auction_id      = auction.get("auction_id"),
            mode            = "shadow_reserve",
            winner_solver   = orig.winner_solver,
            winning_score   = score_1,
            reference_score = orig.reference_score,
            reserve         = r,
            reserve_active  = active,
            user_surplus    = orig.user_surplus,
            solver_rent     = rent_sr,
            fallback_triggered = False,
        )

    # ------------------------------------------------------------------
    # Fallback reserve  (order delayed / rerouted)
    # ------------------------------------------------------------------

    def _fallback_reserve(self, auction: dict,
                          orig: AuctionResult, r: float) -> AuctionResult:
        score_1 = orig.winning_score
        active  = score_1 < r

        return AuctionResult(
            auction_id      = auction.get("auction_id"),
            mode            = "shadow_reserve",
            winner_solver   = orig.winner_solver if not active else None,
            winning_score   = score_1 if not active else r,
            reference_score = orig.reference_score,
            reserve         = r,
            reserve_active  = active,
            user_surplus    = orig.user_surplus,
            solver_rent     = orig.solver_rent if not active else 0.0,
            fallback_triggered = active,
        )
