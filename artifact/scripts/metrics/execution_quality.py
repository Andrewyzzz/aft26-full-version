"""
Chapter 4: Economic Consequences of Competition Fragility

Regression analysis:
    ExecutionQuality_a = alpha_0 + alpha_1 * Fragility_a + alpha_2 * Controls_a + eps
    SolverRent_a       = gamma_0 + gamma_1 * Fragility_a + gamma_2 * Controls_a + eps

Outputs:
  - regression tables (LaTeX + CSV)
  - Figure 4 data: fragility decile vs execution quality / solver rent
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


CONTROLS = [
    "np.log1p(volume_raw)",    # order size proxy (raw token units until USD prices added)
    "n_valid",                  # nominal competition
]

# Raw column names needed for dropna (formulae stripped)
CONTROL_COLS = ["volume_raw", "n_valid"]

# Fixed effects absorbed via entity dummies (token_pair, chain)
FE_COLS = ["token_pair", "chain"]


def winsorize(s: pd.Series, lower: float = 0.01,
              upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)


def run_regression(df: pd.DataFrame,
                   outcome: str,
                   fe_cols: Optional[list] = None,
                   suffix: str = "") -> dict:
    """
    OLS with optional FE via within-demeaning (avoids huge dummy matrices).
    For large panels, use within-estimator: demean by group instead of C() dummies.
    """
    df = df.dropna(subset=[outcome, "fragility"] + CONTROL_COLS).copy()
    for c in CONTROL_COLS + [outcome, "fragility"]:
        if c in df.columns and df[c].dtype in [np.float64, np.float32]:
            df[c] = winsorize(df[c])

    if fe_cols:
        # Within-estimator: demean by combined FE group (handles 10k+ groups)
        fe_key = "_fe_group"
        df[fe_key] = df[fe_cols[0]].astype(str)
        for col in fe_cols[1:]:
            df[fe_key] = df[fe_key] + "_" + df[col].astype(str)

        demean_cols = [outcome, "fragility"] + CONTROL_COLS
        for col in demean_cols:
            df[col] = df[col] - df.groupby(fe_key)[col].transform("mean")

    rhs     = " + ".join(["fragility"] + CONTROLS)
    formula = f"{outcome} ~ {rhs}"
    model   = smf.ols(formula, data=df).fit(
        cov_type="HC3"
    )
    return {
        "outcome": outcome,
        "suffix":  suffix,
        "n":       int(model.nobs),
        "coef_fragility": model.params.get("fragility", np.nan),
        "se_fragility":   model.bse.get("fragility", np.nan),
        "pval_fragility": model.pvalues.get("fragility", np.nan),
        "r2":      model.rsquared,
        "model":   model,
    }


def fragility_decile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Figure 4 / Figure 6: bin auctions by fragility decile and
    compute mean execution quality and solver rent within each decile.
    """
    keep_cols = ["fragility", "solver_rent"]
    if "exec_quality" in df.columns and df["exec_quality"].notna().sum() > 100:
        keep_cols.append("exec_quality")
    df = df.copy().dropna(subset=keep_cols)
    df["frag_decile"] = pd.qcut(df["fragility"], q=10,
                                labels=False, duplicates="drop")
    agg_dict = {
        "mean_fragility":   ("fragility",   "mean"),
        "mean_solver_rent": ("solver_rent", "mean"),
        "n_auctions":       ("auction_id",  "count"),
    }
    if "exec_quality" in keep_cols:
        agg_dict["mean_exec_qual"] = ("exec_quality", "mean")
    agg = df.groupby("frag_decile").agg(**agg_dict).reset_index()
    return agg


def make_regression_table(results: list[dict],
                           output_path: Path) -> None:
    """Export regression summary as CSV and LaTeX stub."""
    rows = []
    for r in results:
        rows.append({
            "Outcome":           r["outcome"],
            "Spec":              r["suffix"],
            "Coef (Fragility)":  f"{r['coef_fragility']:.4f}",
            "SE":                f"({r['se_fragility']:.4f})",
            "p-value":           f"{r['pval_fragility']:.3f}",
            "R²":                f"{r['r2']:.3f}",
            "N":                 r["n"],
        })
    tbl = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(output_path, index=False)
    logger.info("Regression table → %s", output_path)


def main(input_path: str, output_dir: str):
    df = pd.read_parquet(input_path)
    out = Path(output_dir)

    # exec_quality requires USD prices (not yet available); use solver_rent proxy
    available_outcomes = [c for c in ["exec_quality", "solver_rent"]
                          if c in df.columns and df[c].notna().sum() > 100]
    results = []
    for outcome in available_outcomes:
        # Spec 1: no FE
        r1 = run_regression(df, outcome, fe_cols=None, suffix="no_fe")
        # Spec 2: pair + chain FE
        r2 = run_regression(df, outcome, fe_cols=FE_COLS, suffix="fe")
        results.extend([r1, r2])
        logger.info("%s  coef=%.4f  p=%.3f",
                    outcome, r2["coef_fragility"], r2["pval_fragility"])

    make_regression_table(results, out / "regression_main.csv")

    # Decile table for figures
    dec = fragility_decile_analysis(df)
    dec.to_csv(out / "fragility_decile.csv", index=False)
    logger.info("Decile table → %s", out / "fragility_decile.csv")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output-dir", default="results/tables")
    args = p.parse_args()
    main(args.input, args.output_dir)
