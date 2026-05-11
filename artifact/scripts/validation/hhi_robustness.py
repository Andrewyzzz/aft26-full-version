#!/usr/bin/env python3
"""
Review-time wrapper for HHI robustness.

Full-version Table 30 refers to this object as hhi_robustness.py.
The support-filtered implementation and output are provided under:
  artifact/scripts/validation/hhi_robustness_min_auctions.py
  artifact/tables/hhi_robustness_min_auctions.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "tables" / "hhi_robustness_min_auctions.csv"

def main():
    if not TABLE.exists():
        raise FileNotFoundError(f"Missing expected output: {TABLE}")
    print(pd.read_csv(TABLE).to_string(index=False))

if __name__ == "__main__":
    main()
