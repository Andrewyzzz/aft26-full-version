#!/usr/bin/env python3
"""
Generate LaTeX table files for all main paper tables.
Output: paper/tables/*.tex  (ready for \\input{tables/xxx})

Usage:
    cd ~/AFTpaper && python scripts/make_latex_tables.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

IN  = Path("results/tables")
OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)

def pct(v, dec=1):
    return f"{v*100:.{dec}f}\\%"

def usd(v, dec=0):
    if abs(v) >= 1e6: return f"\\${v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"\\${v/1e3:.1f}K"
    return f"\\${v:.{dec}f}"

def fmt(v, col=""):
    if pd.isna(v): return "---"
    if isinstance(v, str): return v.replace("_", "\\_").replace("%", "\\%").replace("$","\\$")
    col = col.lower()
    if "share" in col or "frac" in col or "rate" in col or col.endswith("_pct"): return pct(v)
    if "usd" in col and "total" not in col and abs(v) < 1e5: return f"\\${v:.2f}"
    if "usd" in col: return usd(v)
    if "hhi" in col and "frac" not in col: return f"{v:.3f}"
    if "r2" in col or "rsquared" in col: return f"{v:.3f}"
    if "p_lag" in col or "p-value" in col or "pval" in col:
        if v < 0.001: return "<0.001"
        return f"{v:.3f}"
    if "coef" in col or "se" in col or "median" in col or "mean" in col:
        if abs(v) >= 1e6: return f"{v/1e6:.2f}M"
        if abs(v) >= 1e3: return f"{v/1e3:.1f}K"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    if isinstance(v, float): return f"{v:.4f}" if abs(v) < 100 else f"{v:,.0f}"
    if isinstance(v, int): return f"{v:,}"
    return str(v)

def to_tex(df, caption, label, col_fmt=None, notes=None, rename=None):
    if rename: df = df.rename(columns=rename)
    ncol = len(df.columns)
    if col_fmt is None:
        col_fmt = "l" + "r" * (ncol - 1)
    header = " & ".join(f"\\textbf{{{c}}}" for c in df.columns) + " \\\\"
    rows = []
    for _, row in df.iterrows():
        cells = [fmt(row[c], col=c) for c in df.columns]
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n        ".join(rows)
    note_str = f"\n    \\footnotesize {notes}" if notes else ""
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_fmt}}}
\\toprule
{header}
\\midrule
        {body}
\\bottomrule
\\end{{tabular}}{note_str}
\\end{{table}}
"""

# ══════════════════════════════════════════════════════
# Table 1: Fragility Summary by Segment
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"fragility_summary_usd.csv")
df = df[["subset","n_auctions","frac_fragile_50","frac_fragile_75",
         "median_fragility","mean_n_competitive","median_score_gap_bps","coverage_volume_usd"]]
tex = to_tex(df,
    caption="Competition fragility summary by market segment (Jun 2025 -- Jan 2026, $N=802{,}816$ auctions).",
    label="tab:fragility-summary",
    col_fmt="lrrrrrrr",
    rename={"subset":"Segment","n_auctions":"$N$","frac_fragile_50":"Frag$>$0.5",
            "frac_fragile_75":"Frag$>$0.75","median_fragility":"Med. Frag.",
            "mean_n_competitive":"Mean $N^{\\text{comp}}$",
            "median_score_gap_bps":"Med. Gap (bps)","coverage_volume_usd":"USD cov."},
    notes="Fragility $= 1 - \\text{score}_2/\\text{score}_1$. Score gap in basis points of notional volume.")
(OUT/"fragility_summary.tex").write_text(tex)
print("✓ fragility_summary.tex")

# ══════════════════════════════════════════════════════
# Table 2: HHI Summary
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"hhi_summary.csv")
n_sol_col = "median_n_solvers_active" if "median_n_solvers_active" in df.columns else "median_n_solvers"
df = df[["segment","n_cells","median_hhi","frac_hhi_gt_025","frac_hhi_eq_1",
         "median_n_auctions", n_sol_col]]
tex = to_tex(df,
    caption="Local market concentration (HHI) by segment. Weekly cell-level concentration.",
    label="tab:hhi-summary",
    col_fmt="lrrrrrr",
    rename={"segment":"Segment","n_cells":"Cells","median_hhi":"Med. HHI",
            "frac_hhi_gt_025":"HHI$>$0.25","frac_hhi_eq_1":"HHI$=1$",
            "median_n_auctions":"Med. auctions",
            "median_n_solvers_active":"Med. solvers","median_n_solvers":"Med. solvers"})
(OUT/"hhi_summary.tex").write_text(tex)
print("✓ hhi_summary.tex")

# ══════════════════════════════════════════════════════
# Table 3: Regression
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"regression_main_usd.csv")
df["stars"] = df["p-value"].apply(lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")
df["coef_str"] = df.apply(lambda r: f"{r['Coef(Fragility)']:.4f}{r['stars']}", axis=1)
df["se_str"]   = df.apply(lambda r: f"({r['SE']:.4f})", axis=1)
out = df[["Outcome","Spec","coef_str","se_str","R2","N"]].rename(
    columns={"Outcome":"Outcome","Spec":"Specification",
             "coef_str":"Coef. (Fragility)","se_str":"(SE)","R2":"$R^2$","N":"$N$"})
tex = to_tex(out,
    caption="Regression of solver rent and score gap on fragility. HC3 robust SE in parentheses. *** $p<0.001$.",
    label="tab:regression",
    col_fmt="llrrrr",
    notes="Controls: $\\log(1+\\text{volume\\_raw})$, $N^{\\text{valid}}$. FE: token-pair $\\times$ chain (within-demeaning).")
(OUT/"regression_main.tex").write_text(tex)
print("✓ regression_main.tex")

# ══════════════════════════════════════════════════════
# Table 4: Fragility Decile
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"fragility_decile_usd.csv")
df["D"] = ["D"+str(int(d)+1) for d in df["frag_dec"]]
out = df[["D","median_fragility","median_solver_rent_usd","mean_solver_rent_usd",
          "median_score_gap_bps","n_auctions"]].rename(
    columns={"D":"Decile","median_fragility":"Med. Fragility",
             "median_solver_rent_usd":"Med. Rent (\\$)","mean_solver_rent_usd":"Mean Rent (\\$)",
             "median_score_gap_bps":"Med. Gap (bps)","n_auctions":"$N$"})
tex = to_tex(out,
    caption="Solver rent and score gap by fragility decile (winsorized P1--P99).",
    label="tab:fragility-decile",
    col_fmt="lrrrrr")
(OUT/"fragility_decile.tex").write_text(tex)
print("✓ fragility_decile.tex")

# ══════════════════════════════════════════════════════
# Table 5: Counterfactual Summary
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"counterfactual_summary_usd.csv")
# Only keep informative columns
keep = [c for c in df.columns if "replay_mode" in c or "activation" in c or
        "winner_change" in c or "delta_rent" in c or "theorem1" in c]
df = df[keep].rename(columns={"replay_mode":"Replay",
    "reserve_activation_rate":"Activation",
    "winner_change_rate":"Winner change",
    "mean_delta_rent_usd":"Mean $\\Delta$Rent",
    "median_delta_rent_usd":"Med. $\\Delta$Rent",
    "mean_delta_rent_usd_active":"Active $\\Delta$Rent",
    "theorem1_holds_frac":"Thm. 1"})
tex = to_tex(df,
    caption="Counterfactual shadow-reserve evaluation. $\\Delta$Rent in USD per auction.",
    label="tab:counterfactual",
    notes="Mechanical: original bids unchanged. Behavior-aware: solvers respond to reserve. Adversarial: all bid $r+\\epsilon$.")
(OUT/"counterfactual_main.tex").write_text(tex)
print("✓ counterfactual_main.tex")

# ══════════════════════════════════════════════════════
# Table 6: Reserve Targeting
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"reserve_targeting.csv")
df["D"] = ["D"+str(int(d)+1) for d in df["frag_dec"]]
out = df[["D","mean_frag","act_rate","win_change","n"]].rename(
    columns={"D":"Decile","mean_frag":"Mean Frag.","act_rate":"Activation",
             "win_change":"Winner change","n":"$N$"})
tex = to_tex(out,
    caption="Shadow-reserve activation rate by fragility decile. Reserve activates less in high-fragility auctions, confirming the mechanism targets low absolute quality, not relative winner dominance.",
    label="tab:reserve-targeting",
    col_fmt="lrrrr")
(OUT/"reserve_targeting.tex").write_text(tex)
print("✓ reserve_targeting.tex")

# ══════════════════════════════════════════════════════
# Table 7: Reserve Sensitivity
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"reserve_sensitivity.csv")
out = df.rename(columns={"config":"Parameter config","activation_rate":"Activation",
    "mean_delta_rent_usd":"Mean $\\Delta$Rent","active_delta_rent":"Active $\\Delta$Rent"})
tex = to_tex(out,
    caption="Shadow-reserve parameter sensitivity (mechanical replay). Baseline: $\\tau=0.25$, $\\delta_\\downarrow=0.05$, $\\eta=0.10$.",
    label="tab:reserve-sensitivity",
    col_fmt="lrrr",
    notes="Single-dimension sensitivity; other parameters held at baseline.")
(OUT/"reserve_sensitivity.tex").write_text(tex)
print("✓ reserve_sensitivity.tex")

# ══════════════════════════════════════════════════════
# Table 8: HHI Robustness (min auctions)
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"hhi_robustness_min_auctions.csv")
out = df[df["segment"]=="all"].drop(columns=["segment"]).rename(
    columns={"min_auctions":"Min. auctions/cell","n_cells":"Cells",
             "median_hhi":"Med. HHI","frac_hhi_gt_025":"HHI$>$0.25",
             "frac_hhi_eq_1":"HHI$=1$","median_n_solvers":"Med. solvers"})
tex = to_tex(out,
    caption="HHI concentration after requiring minimum auction support per cell (all segments). Concentration persists after filtering sparse cells.",
    label="tab:hhi-robustness",
    col_fmt="rrrrrr")
(OUT/"hhi_robustness.tex").write_text(tex)
print("✓ hhi_robustness.tex")

# ══════════════════════════════════════════════════════
# Table 9: Sample Flow
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"sample_flow.csv")
out = df[["step","n","filter"]].rename(
    columns={"step":"Sample","n":"$N$","filter":"Filter / Reason"})
tex = to_tex(out,
    caption="Sample construction and sizes cited in paper.",
    label="tab:sample-flow",
    col_fmt="lrl")
(OUT/"sample_flow.tex").write_text(tex)
print("✓ sample_flow.tex")

# ══════════════════════════════════════════════════════
# Table 10: Reference Score Validation
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"reference_score_validation.csv")
out = df.rename(columns={"check":"Check","value":"Value"})
tex = to_tex(out,
    caption="Reference score construction validation. Confirms that $\\text{score}_2$ equals the best non-winning solver score in 83.9\\% of auctions; remaining 16.1\\% are auctions where the winning solver submitted multiple solutions.",
    label="tab:ref-score-validation",
    col_fmt="lr")
(OUT/"reference_score_validation.tex").write_text(tex)
print("✓ reference_score_validation.tex")

# ══════════════════════════════════════════════════════
# Table 11: USD Missingness
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"usd_missingness.csv")
out = df[["sample","n","share","median_fragility","frac_fragile_50",
          "mean_n_competitive","blue_chip_share"]].rename(
    columns={"sample":"Sample","n":"$N$","share":"Share",
             "median_fragility":"Med. Frag.","frac_fragile_50":"Frag$>$0.5",
             "mean_n_competitive":"Mean $N^{\\text{comp}}$","blue_chip_share":"Blue-chip"})
tex = to_tex(out,
    caption="USD coverage: comparison of covered vs uncovered auctions. USD-uncovered auctions are slightly less fragile, indicating USD-based results are not upward-biased.",
    label="tab:usd-missingness",
    col_fmt="lrrrrrrr")
(OUT/"usd_missingness.tex").write_text(tex)
print("✓ usd_missingness.tex")

# ══════════════════════════════════════════════════════
# Table 12: Replay Decomposition
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"replay_decomposition.csv")
keep_cols = [c for c in df.columns if c in [
    "replay_mode","n_eval","activation_rate",
    "overall_delta_rent_usd","activated_delta_rent_usd",
    "non_activated_delta_rent_usd","theorem1_holds_frac"]]
out = df[keep_cols].rename(columns={
    "replay_mode":"Replay","n_eval":"$N$",
    "activation_rate":"Activation",
    "overall_delta_rent_usd":"Overall $\\Delta$R",
    "activated_delta_rent_usd":"Active $\\Delta$R",
    "non_activated_delta_rent_usd":"Non-active $\\Delta$R",
    "theorem1_holds_frac":"Thm. 1"})
tex = to_tex(out,
    caption="Counterfactual replay decomposition: activated vs non-activated auctions. Behavior-aware non-activated mean $\\approx -\\$5.17$ confirms equilibrium bid-adjustment effect.",
    label="tab:replay-decomposition",
    col_fmt="lrrrrrrr")
(OUT/"replay_decomposition.tex").write_text(tex)
print("✓ replay_decomposition.tex")

# ══════════════════════════════════════════════════════
# Table 13: Economic Scale
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"economic_scale.csv")
out = df.rename(columns={"metric":"Metric","value":"Value"})
tex = to_tex(out,
    caption="Economic scale of visible replacement rent (Jun 2025 -- Jan 2026, 244 days).",
    label="tab:economic-scale",
    col_fmt="lr")
(OUT/"economic_scale.tex").write_text(tex)
print("✓ economic_scale.tex")

# ══════════════════════════════════════════════════════
# Table 14: Fragility Predictability
# ══════════════════════════════════════════════════════
df = pd.read_csv(IN/"fragility_predictability.csv")
out = df.rename(columns={"spec":"Specification","N":"$N$","R2":"$R^2$",
    "coef_lag1":"$\\beta_1$ (lag1)","se_lag1":"SE","p_lag1":"$p$-value"})
tex = to_tex(out,
    caption="Fragility predictability regression: $\\text{Fragility}_a \\sim \\text{lag1\\_cell\\_fragility} + \\text{controls}$. Lag-1 coefficient 0.799 confirms structural persistence.",
    label="tab:fragility-predictability",
    col_fmt="lrrrrrr")
(OUT/"fragility_predictability.tex").write_text(tex)
print("✓ fragility_predictability.tex")

print(f"\n✓ All 14 LaTeX tables saved to {OUT}/")
print("\nUsage in paper: \\input{tables/fragility_summary}")
