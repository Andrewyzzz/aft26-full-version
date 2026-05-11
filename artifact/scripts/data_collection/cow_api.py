"""
CoW Protocol API wrapper — CIP-67 era (2025-06 onwards).

Actual response format (verified 2026-05):
  /solver_competition/{auctionId}  →  {
      auctionId: int,
      transactionHashes: [str],
      auctionStartBlock: int,
      competitionSimulationBlock: int,
      auction: { orders: [str] },
      solutions: [
          { solver, solverAddress, score: str, ranking: int,
            isWinner: bool, filteredOut: bool,
            clearingPrices: {}, orders: [{id, sellAmount, buyAmount}] }
      ]
  }

NOTE: referenceScores field does NOT exist; we derive from runner-up score.
NOTE: score is a STRING; convert with int().
"""

import time
import threading
import logging
from typing import Optional

import requests


class GlobalRateLimiter:
    """Token-bucket rate limiter shared across all threads."""
    def __init__(self, max_per_second: float):
        self.interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now  = time.monotonic()
            wait = self._last + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

logger = logging.getLogger(__name__)

COW_API_BASE = {
    "mainnet": "https://api.cow.fi/mainnet/api/v1",
    "gnosis":  "https://api.cow.fi/xdai/api/v1",
    "arbitrum": "https://api.cow.fi/arbitrum_one/api/v1",
}

# Linear block-to-timestamp reference (Merge era, ~12 s/block)
# Calibrated: block 25015688 ≈ 2026-05-03 17:00:00 UTC (verified from API)
_REF_BLOCK = 25_015_688
_REF_TS    = 1_777_830_000   # 2026-05-03 17:00:00 UTC
_SECS_PER_BLOCK = 12


def block_to_timestamp(block: int) -> int:
    """Estimate UTC timestamp from mainnet block number (±minutes accuracy)."""
    return _REF_TS - (_REF_BLOCK - block) * _SECS_PER_BLOCK


class CowApiClient:
    """Thin wrapper around the CoW Protocol REST API."""

    _global_limiter: Optional["GlobalRateLimiter"] = None
    _limiter_lock = threading.Lock()

    def __init__(self, chain: str = "mainnet", max_retries: int = 6,
                 backoff: float = 1.0, max_req_per_sec: float = 5.0):
        self.base = COW_API_BASE[chain]
        self.chain = chain
        self.max_retries = max_retries
        self.backoff = backoff
        # Shared rate limiter across ALL instances / threads
        with CowApiClient._limiter_lock:
            if CowApiClient._global_limiter is None:
                CowApiClient._global_limiter = GlobalRateLimiter(max_req_per_sec)
        self._limiter = CowApiClient._global_limiter
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "AFTpaper/0.1"
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=8,
            max_retries=0,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://",  adapter)

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None):
        url = f"{self.base}{path}"
        for attempt in range(self.max_retries):
            try:
                self._limiter.wait()               # global rate limit across all threads
                resp = self.session.get(url, params=params, timeout=(5, 8))
                if resp.status_code == 404:
                    return None          # auction not found / not settled
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                if resp.status_code == 429:
                    wait = self.backoff * (2 ** attempt)
                    logger.debug("Rate-limited, sleeping %.1fs", wait)
                    time.sleep(wait)
                elif resp.status_code >= 500:
                    wait = self.backoff * (2 ** attempt)
                    logger.debug("Server error %d, sleeping %.1fs",
                                 resp.status_code, wait)
                    time.sleep(wait)
                else:
                    raise
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.backoff * (2 ** attempt))
        return None

    # ------------------------------------------------------------------
    # Solver competition (primary endpoint)
    # ------------------------------------------------------------------

    def get_solver_competition(self, auction_id: int) -> Optional[dict]:
        """
        Fetch full competition record for a given auction ID.
        Returns None if auction not found or not yet settled.
        """
        return self._get(f"/solver_competition/{auction_id}")

    def get_latest_competition(self) -> Optional[dict]:
        """Fetch the most recent solver competition record."""
        return self._get("/solver_competition/latest")

    def get_latest_auction_id(self) -> Optional[int]:
        """Return the most recent auctionId."""
        rec = self.get_latest_competition()
        if rec:
            return rec.get("auctionId")
        return None

    def get_auction_by_tx(self, tx_hash: str) -> Optional[dict]:
        """Fetch competition record by settlement tx hash (fallback)."""
        return self._get(f"/solver_competition/by_tx_hash/{tx_hash}")

    # ------------------------------------------------------------------
    # Settlements (used to build block→timestamp mapping)
    # ------------------------------------------------------------------

    def get_settlements(self, limit: int = 100,
                        before_tx: Optional[str] = None) -> list:
        params: dict = {"limit": limit}
        if before_tx:
            params["beforeTxHash"] = before_tx
        result = self._get("/settlements", params=params)
        return result if isinstance(result, list) else []
