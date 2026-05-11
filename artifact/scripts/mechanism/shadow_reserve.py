"""
Chapter 5: Robust Shadow-Reserve Construction

Implements the endogenous reserve update rule:

    r_{c,t+1} = max{
        (1 - delta_down) * r_{c,t},
        min[
            (1 + delta_up) * r_{c,t},
            (1 - eta) * r_{c,t} + eta * hat_r_{c,t}
        ]
    }

where hat_r_{c,t} is the tau-quantile of winsorized historical
high-competition benchmarks within window [t-L, t-H].

Parameter selection principles (Section 5.5):
  - L    : history window — long enough to cover multiple regimes
  - H    : delay window  — prevents current bids from influencing reserve
  - k    : min competitive solvers required for benchmark sample
  - tau  : must satisfy tau > alpha (fraction attacker controls)
  - eta  : smoothing speed
  - delta_down : downward rate limit (anti-manipulation core)
  - delta_up   : upward rate limit (prevents over-tightening)
"""

import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReserveParams:
    """
    All hyper-parameters for the shadow reserve with their rationale.
    Theorem 4 requires: tau > alpha.
    """
    # History window: number of auctions (or time periods) to look back
    L: int = 200
    # Delay window: exclude the most recent H auctions from benchmark
    H: int = 10
    # Minimum number of competitive solvers for a sample to count
    k: int = 2
    # Quantile for robust benchmark (must be > attacker's share alpha)
    tau: float = 0.25
    # Smoothing / learning rate
    eta: float = 0.10
    # Maximum downward adjustment per period (anti-manipulation)
    delta_down: float = 0.05
    # Maximum upward adjustment per period (anti-tightening)
    delta_up: float = 0.10
    # Winsorization bounds for benchmark samples
    winsor_lo: float = 0.05
    winsor_hi: float = 0.95
    # Epsilon for "beat reserve" (Behavior-aware replay)
    beat_epsilon: float = 1e-6

    def validate(self, alpha_max_attacker: float = 0.20):
        assert self.tau > alpha_max_attacker, (
            f"tau={self.tau} must be > alpha={alpha_max_attacker} "
            "to satisfy Theorem 4"
        )
        assert 0 < self.eta <= 1
        assert 0 < self.delta_down < 1
        assert 0 < self.delta_up


@dataclass
class ReserveState:
    """Mutable state for one market cell's reserve."""
    cell: str
    current: float = 0.0
    history: deque = field(default_factory=deque)   # (timestamp, value, n_comp)
    last_updated: Optional[pd.Timestamp] = None


class ShadowReserveBank:
    """
    Manages per-cell reserve values and updates them incrementally
    as new auction data arrives.

    Usage in replay:
        bank = ShadowReserveBank(params)
        for auction in sorted_auctions:
            r = bank.get_reserve(auction["market_cell"], auction["block_timestamp"])
            # ... use r as shadow bid ...
            bank.update(auction)
    """

    def __init__(self, params: ReserveParams):
        self.params = params
        self._states: dict[str, ReserveState] = defaultdict(
            lambda: ReserveState(cell="unknown")
        )

    def _init_state(self, cell: str, seed_value: float = 0.0) -> ReserveState:
        s = ReserveState(cell=cell, current=seed_value)
        self._states[cell] = s
        return s

    def get_reserve(self, cell: str,
                    timestamp: Optional[pd.Timestamp] = None) -> float:
        return self._states[cell].current

    def _compute_hat_r(self, state: ReserveState) -> float:
        """
        hat_r_{c,t} = Q_tau( Winsorize( H_{c,t} ) )

        Only uses samples where n_competitive >= k and
        with the [H, L] delay applied.
        """
        p = self.params
        hist = list(state.history)
        # Apply delay window: skip the most recent H entries
        hist = hist[:-p.H] if len(hist) > p.H else []
        # Apply history window: keep only the last L entries
        hist = hist[-p.L:]
        # Filter to high-competition samples
        valid = [v for (ts, v, n_comp) in hist if n_comp >= p.k]
        if len(valid) < 3:
            return state.current   # Not enough data; hold steady

        arr = np.array(valid, dtype=float)
        lo = np.quantile(arr, p.winsor_lo)
        hi = np.quantile(arr, p.winsor_hi)
        arr = np.clip(arr, lo, hi)
        return float(np.quantile(arr, p.tau))

    def update(self, auction: dict) -> float:
        """
        Process one auction and update the relevant cell's reserve.

        auction must contain:
          - market_cell: str
          - block_timestamp: pd.Timestamp
          - reference_score: float   (benchmark value for this auction)
          - n_competitive: int
        """
        cell       = auction["market_cell"]
        ts         = auction["block_timestamp"]
        bench_val  = float(auction.get("reference_score") or
                           auction.get("score_runner_up") or 0.0)
        n_comp     = int(auction.get("n_competitive", 0))

        if cell not in self._states:
            self._init_state(cell, seed_value=bench_val)
        state = self._states[cell]

        # Append to history
        state.history.append((ts, bench_val, n_comp))
        # Trim to L + H
        while len(state.history) > self.params.L + self.params.H + 10:
            state.history.popleft()

        # Compute target
        hat_r = self._compute_hat_r(state)
        p     = self.params
        r_old = state.current

        # Smooth update
        r_target = (1 - p.eta) * r_old + p.eta * hat_r

        # Apply rate limits (key for Theorem 4)
        r_new = max(
            (1 - p.delta_down) * r_old,
            min((1 + p.delta_up) * r_old, r_target)
        )

        state.current    = r_new
        state.last_updated = ts
        return r_new

    def bulk_initialize(self, df: pd.DataFrame,
                        init_frac: float = 0.2) -> None:
        """
        Warm up reserve states from the first init_frac of the dataset
        before the evaluation window.
        """
        cutoff = int(len(df) * init_frac)
        warm_df = df.iloc[:cutoff]
        for _, row in warm_df.iterrows():
            self.update(row.to_dict())
        logger.info("Reserve bank initialized with %d auctions (%d cells)",
                    cutoff, len(self._states))

    def get_all_states(self) -> pd.DataFrame:
        rows = [{"cell": s.cell, "reserve": s.current,
                 "n_history": len(s.history)}
                for s in self._states.values()]
        return pd.DataFrame(rows)
