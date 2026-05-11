"""
Blue-chip consistency check.

Verifies that the blue-chip paradox (higher fragility on blue-chip token pairs)
holds within solver identity, ruling out single-solver artifacts.

This is a companion to bluechip_paradox_check.py. That script runs the
full 5-specification LPM. This script documents the within-solver consistency
check specifically.

Aggregated output is provided in:
  artifact/tables/bluechip_paradox_check.csv

Usage (requires full dataset):
  python bluechip_consistency.py
  # reads: data/processed/auctions_full_usd.parquet
  # writes: results/tables/bluechip_paradox_check.csv
  # (delegates to bluechip_paradox_check.py)
"""
from bluechip_paradox_check import run

if __name__ == "__main__":
    run()
