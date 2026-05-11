# Anonymous Full Version and Lightweight Artifact

This repository contains the anonymous full version and a lightweight review-time
artifact for the AFT 2026 submission.

## Contents

- `full_version.pdf`: anonymous full version with appendices.
- `artifact/tables/`: aggregated CSV summaries used for the main paper and selected appendix tables.
- `artifact/scripts/`: analysis, validation, reserve, and replay scripts documenting the pipeline.
- `requirements.txt`: Python dependencies.

## Scope

This is a lightweight review-time artifact. It is designed to document the
measurement and table-generation pipeline and to expose the aggregated outputs
reported in the paper.

Large raw CoW solver-competition JSONL files, large intermediate parquet files,
and external price-cache files are not included in this anonymous repository
because of size and data-source constraints. The camera-ready artifact will
include the complete replication package subject to data-source and archival
constraints.

## Key tables

| File | Description |
|------|-------------|
| `artifact/tables/sample_flow.csv` | Sample denominators (802,816 main auctions → 493,653 USD sample) |
| `artifact/tables/reference_score_validation.csv` | Best-non-winning solver parser audit (95,069 affected, 11.84%) |
| `artifact/tables/usd_missingness.csv` | USD coverage diagnostics |
| `artifact/tables/fragility_summary_usd.csv` | Main visible-depth fragility summary |
| `artifact/tables/hhi_robustness_min_auctions.csv` | Support-filtered HHI robustness |
| `artifact/tables/economic_scale.csv` | Accounting-gap economic scale |
| `artifact/tables/regression_main_usd.csv` | Main regression results (USD sample) |
| `artifact/tables/depth_targeted_reserve.csv` | Quality-floor vs depth-trigger diagnostic |
| `artifact/tables/vulnerability_diagnostic.csv` | Reserve benchmark vulnerability diagnostic |
| `artifact/tables/reserve_sensitivity.csv` | Reserve calibration sensitivity |
| `artifact/tables/router_benchmark.csv` | Approximate router sanity check |
| `artifact/tables/replay_decomposition.csv` | Counterfactual replay decomposition |
| `artifact/tables/counterfactual_summary_usd.csv` | Counterfactual summary (USD) |
| `artifact/tables/bluechip_paradox_check.csv` | Blue-chip paradox mechanism consistency check |
| `artifact/tables/artifact_map.csv` | Full mapping of paper objects to scripts and outputs |

## Script overview

```
artifact/scripts/
  data_collection/
    cow_api.py               CoW Protocol API client
    fetch_auctions.py        Auction-ID-based JSONL collection with checkpointing
    build_dataset.py         Core parser: best non-winning solver score construction
    add_usd_prices.py        USD price enrichment

  metrics/
    fragility.py             Visible replacement depth (fragility) metric
    execution_quality.py     Execution quality metrics

  mechanism/
    shadow_reserve.py        Shadow-reserve benchmark construction
    sr_auction.py            Reserve-augmented auction simulator
    params.yaml              Baseline reserve parameters (L=200, H=10, k=2, tau=0.25)

  counterfactual/
    mechanical_replay.py     Mechanical counterfactual replay
    behavior_aware_replay.py Behavior-aware replay
    adversarial_replay.py    Adversarial replay

  validation/
    sample_flow.py                        Sample denominator audit
    usd_missingness.py                    USD missingness diagnostics
    reserve_sensitivity.py                Reserve calibration sensitivity
    depth_targeted_reserve_diagnostic.py  Depth-trigger diagnostic
    router_benchmark.py                   Router sanity check
    replay_decomposition.py               Replay decomposition
    bluechip_paradox_check.py             Blue-chip paradox mechanism check

  make_latex_tables.py       Reads results/tables/*.csv → generates paper/tables/*.tex (preserves original project paths; review-time aggregated outputs are in artifact/tables/)
```

## Parser note

The key implementation that constructs the best non-winning solver score is in
`artifact/scripts/data_collection/build_dataset.py`. This script removes the
winning solver by identity and takes the best remaining solution score, retaining
`score_2_solution` for validation. The parser audit result is in
`artifact/tables/reference_score_validation.csv`.
