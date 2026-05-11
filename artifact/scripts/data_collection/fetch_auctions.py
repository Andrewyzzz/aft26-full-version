"""
Fetch CoW Protocol solver competition data by enumerating auction IDs.

Strategy:
  1. Get latest auctionId from /solver_competition/latest
  2. Binary-search the start ID corresponding to --start-date
     (using auctionStartBlock → timestamp linear estimate)
  3. Enumerate IDs from start to end, fetch each concurrently
  4. Skip unsettled auctions (transactionHashes == [])
  5. Write JSONL with checkpoint for resuming interrupted runs

Usage:
    python fetch_auctions.py \\
        --chain mainnet \\
        --start-date 2025-06-01 \\
        --end-date   2025-06-30 \\
        --concurrency 8 \\
        --out  data/raw/cow_mainnet_pilot.jsonl \\
        --checkpoint data/raw/cow_mainnet_pilot.ckpt
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from cow_api import CowApiClient, block_to_timestamp

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chain", default="mainnet",
                   choices=["mainnet", "gnosis", "arbitrum"])
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD UTC")
    p.add_argument("--end-date",   required=True, help="YYYY-MM-DD UTC")
    p.add_argument("--start-id",   type=int, default=None,
                   help="Override auto-detected start auction ID")
    p.add_argument("--end-id",     type=int, default=None,
                   help="Override auto-detected end auction ID")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--rate-limit",  type=float, default=20.0,
                   help="Max requests per second (global, default 20)")
    p.add_argument("--out",        required=True)
    p.add_argument("--checkpoint", default=None,
                   help="File to store last written ID for resuming")
    return p.parse_args()


def date_to_ts(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def binary_search_id(client: CowApiClient,
                     target_ts: int,
                     lo: int, hi: int,
                     find_lower: bool = True) -> int:
    """
    Binary-search for the auction ID whose block timestamp is closest to
    target_ts. find_lower=True → smallest ID with ts >= target_ts.
    """
    while lo < hi:
        mid = (lo + hi) // 2
        rec = client.get_solver_competition(mid)
        if rec is None:
            # ID not found; search nearby
            for offset in range(1, 200):
                for delta in [+offset, -offset]:
                    rec = client.get_solver_competition(mid + delta)
                    if rec is not None:
                        mid = mid + delta
                        break
                if rec is not None:
                    break
            if rec is None:
                # Give up on this mid; collapse range
                lo = mid + 1
                continue

        block = rec.get("auctionStartBlock", 0)
        ts    = block_to_timestamp(block)

        if find_lower:
            if ts < target_ts:
                lo = mid + 1
            else:
                hi = mid
        else:
            if ts > target_ts:
                hi = mid
            else:
                lo = mid + 1

    return lo


def load_checkpoint(path: Optional[str]) -> Optional[int]:
    if path and os.path.exists(path):
        with open(path) as f:
            val = f.read().strip()
            return int(val) if val else None
    return None


def save_checkpoint(path: Optional[str], auction_id: int):
    if path:
        with open(path, "w") as f:
            f.write(str(auction_id))


def _slim_record(rec: dict, ts: int, chain: str) -> dict:
    """
    Strip large fields not needed for analysis, keeping only:
      - auctionId, auctionStartBlock, competitionSimulationBlock
      - transactionHashes
      - solutions[] (full)
      - auction.prices  only if it has exactly 2 tokens (single-pair batch)
      - metadata: _auction_id, _block_timestamp, _chain
    """
    auction = rec.get("auction") or {}
    prices  = auction.get("prices") or {}
    # Only keep prices for exactly-2-token batches (pure single-pair)
    slim_prices = prices if len(prices) == 2 else {}

    return {
        "auctionId":                    rec.get("auctionId"),
        "auctionStartBlock":            rec.get("auctionStartBlock"),
        "competitionSimulationBlock":   rec.get("competitionSimulationBlock"),
        "transactionHashes":            rec.get("transactionHashes", []),
        "solutions":                    rec.get("solutions", []),
        "auction_prices_2tok":          slim_prices,   # token pair for single-pair batches
        "_auction_id":                  rec.get("auctionId"),
        "_block_timestamp":             ts,
        "_chain":                       chain,
    }


def fetch_one(client: CowApiClient, auction_id: int,
              start_ts: int, end_ts: int) -> Optional[dict]:
    """Fetch one competition record; return None if out of window or unsettled."""
    rec = client.get_solver_competition(auction_id)
    if rec is None:
        return None

    # Skip unsettled (no transaction)
    if not rec.get("transactionHashes"):
        return None

    # Check timestamp window
    block = rec.get("auctionStartBlock", 0)
    ts    = block_to_timestamp(block)
    if ts < start_ts or ts > end_ts:
        return None

    return _slim_record(rec, ts, client.chain)


def main():
    args = parse_args()
    start_ts = date_to_ts(args.start_date)
    end_ts   = date_to_ts(args.end_date) + 86399   # inclusive

    # Reset class-level rate limiter so new rate takes effect
    CowApiClient._global_limiter = None
    client = CowApiClient(chain=args.chain, max_req_per_sec=args.rate_limit)
    logger.info("Rate limit: %.0f req/s", args.rate_limit)

    # ---- Determine ID range ----
    logger.info("Detecting latest auction ID …")
    latest_id = client.get_latest_auction_id()
    if latest_id is None:
        raise RuntimeError("Cannot reach CoW API")
    logger.info("Latest auction ID: %d", latest_id)

    end_id = args.end_id or latest_id

    if args.start_id:
        start_id = args.start_id
    else:
        logger.info("Binary-searching start ID for %s …", args.start_date)
        # Search range: ±4M IDs around latest covers >1 year of history
        start_id = binary_search_id(client, start_ts,
                                    lo=max(0, latest_id - 4_000_000),
                                    hi=latest_id,
                                    find_lower=True)
        logger.info("Estimated start ID: %d", start_id)

    # ---- Checkpoint (resume) ----
    ckpt_id = load_checkpoint(args.checkpoint)
    if ckpt_id and ckpt_id > start_id:
        logger.info("Resuming from checkpoint ID %d", ckpt_id + 1)
        start_id = ckpt_id + 1

    total_range = end_id - start_id + 1
    logger.info("Fetching IDs %d → %d  (%d total)", start_id, end_id, total_range)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    last_ckpt_id = start_id - 1

    open_mode = "a" if ckpt_id else "w"

    with open(out_path, open_mode) as fh, \
         ThreadPoolExecutor(max_workers=args.concurrency) as pool, \
         tqdm(total=total_range, desc="Auctions", unit="id") as pbar:

        # Keep a rolling window of in-flight futures.
        # Write completed results in strict ID order using a pending buffer.
        chunk_size   = args.concurrency * 4   # smaller chunks → write sooner
        current_id   = start_id
        pending: dict[int, object] = {}       # aid → Future
        write_cursor = start_id               # next ID to write

        FUTURE_TIMEOUT = 15  # seconds; kills hung requests

        def drain_pending():
            nonlocal write_cursor, written, last_ckpt_id
            while write_cursor in pending:
                fut = pending.pop(write_cursor)
                try:
                    rec = fut.result(timeout=FUTURE_TIMEOUT)
                except Exception as e:
                    logger.debug("Skipping ID %d: %s", write_cursor, e)
                    rec = None
                if rec is not None:
                    fh.write(json.dumps(rec) + "\n")
                    written += 1
                last_ckpt_id = write_cursor
                pbar.update(1)
                write_cursor += 1

        while current_id <= end_id:
            # Submit next chunk
            batch_end = min(current_id + chunk_size - 1, end_id)
            for aid in range(current_id, batch_end + 1):
                pending[aid] = pool.submit(fetch_one, client, aid,
                                           start_ts, end_ts)
            current_id = batch_end + 1

            # Drain completed futures in order (write what's ready)
            # Wait for the oldest pending future (with timeout to avoid hangs)
            if pending:
                oldest = min(pending)
                try:
                    pending[oldest].result(timeout=FUTURE_TIMEOUT)
                except Exception:
                    pass
                drain_pending()

            # Checkpoint every chunk
            save_checkpoint(args.checkpoint, last_ckpt_id)

            if written > 0 and written % 5000 == 0:
                logger.info("Written %d records …", written)

        # Final drain
        for fut in pending.values():
            try:
                fut.result()
            except Exception:
                pass
        drain_pending()

    logger.info("Done. Records written: %d  →  %s", written, out_path)


if __name__ == "__main__":
    main()
