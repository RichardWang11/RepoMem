# Dual-Path Agreement Gating: Confound-Controlled Choice B Pilot

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan -> run -> validate
- Origin Date: 2026-07-16
- Verification Status: VERIFIED
- Version Label: dual_path_agreement_v1

## Experiment Overview

- **Objective:** Test whether disagreement between matched Baseline and Memory
  localization paths is a useful post-localization uncertainty signal.
- **Hypothesis:** Cross-arm disagreement predicts Memory outcome instability and
  stability degradation beyond Baseline-only internal disagreement, a proxy for
  general case difficulty.
- **Scope:** Retrospective analysis of the frozen 20-case Random pilot only. No
  new model/API calls and no claim of pre-execution cost savings.
- **Gate timing:** Both localization paths finish first. High disagreement may
  trigger an additional verification step in a future confirmatory experiment.

## Setup

- **Working directory:** `/data/wjl/RepoMem`
- **Environment:** source `openaigpt4o_env`; execute with
  `/data/conda_envs/locagent/bin/python`
- **Entry command:**
  `MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=. /data/conda_envs/locagent/bin/python scripts/analyze_dual_path_agreement.py`
- **Seed / resampling:** seed 42; 10,000 case bootstrap replicates and 10,000
  label permutations.

## Inputs

| Input | Path | Role |
|---|---|---|
| Frozen protocol | `results/paper2_pilot/preregistration.json` | Case and batch membership |
| Raw predictions | `results/paper2_pilot/runs/*/loc_trajs.repaired.jsonl` | Ordered top-5 files |
| Locked outcomes | `results/paper2_pilot/analysis/per_case_metrics.jsonl` | Accuracy/stability audit |
| Verified dataset | Local SWE-bench Verified Arrow cache | Gold-file sensitivity only |

## Locked Signal Definitions

For top-5 file sets, `D(A,B) = 1 - |A intersect B| / |A union B|`. Two empty
sets have distance 0; exactly one empty set has distance 1.

- `D_cross_arm`: mean of four repeat-matched Baseline/Memory Jaccard distances.
- `D_baseline_internal`: mean over six Baseline repeat pairs.
- `D_memory_internal`: mean over six Memory repeat pairs.
- Sensitivities: maximum matched cross-arm distance, all-16-cross-product mean,
  intersection@5 distance, and finite normalized RBO@5 distance (`p=0.9`).

## Locked Targets and Pre-execution Audit

- Co-primary continuous target: `stability_gap = baseline_stability - memory_stability`.
- Co-primary binary target: Memory case flip, meaning the four Memory Acc@5
  outcomes are not unanimous (locked count 6/20).
- Sensitivity: stability degradation after excluding ties. Locked counts are
  5 degradation, 14 tie, and 1 improvement; the 70% tie rate triggers
  `SPARSE_SIGNAL`, leaving only six effective cases.
- Diagnostic only: matched-repeat paired harm (Baseline correct, Memory wrong).
  This target is structurally coupled to cross-arm outcomes and is not a
  deployment-ready label.

## Confound-Controlled Analyses

1. Compare `D_cross_arm`, `D_baseline_internal`, and `D_memory_internal` using
   AUROC/AP for case flip and Spearman correlation for stability gap.
2. Estimate partial Spearman correlation between cross-arm disagreement and
   stability gap while controlling for Baseline-internal disagreement: average
   rank all variables, OLS-residualize signal and outcome on the control plus an
   intercept, then correlate residuals.
3. Use ordinary case bootstrap CIs and a 10,000-draw outcome permutation test.
   Bootstrap the paired AUROC difference between cross-arm and Baseline-internal
   signals.
4. Report high-disagreement quartile recall, leave-one-case-out influence,
   leave-one-repository-out results, leave-Django-out results, and inverse-repo-
   frequency weighted AUROC.
5. Report gold-file-removed distances as mechanism sensitivity, not as a usable
   gate, because gold files are unavailable at inference.
6. At trajectory level, test matched-repeat disagreement against paired harm
   with case-cluster bootstrap. Also construct leave-one-repeat-out instability
   targets from the other three repeats to reduce self-inclusion.

## Decision Rule

Secondary metrics may weaken but never upgrade the verdict.

- **STRONG_PROMISING** requires all: case-flip AUROC >= 0.70 with lower 95% CI
  >= 0.55; raw rho >= 0.30 with lower CI > 0; partial rho >= 0.30 with lower CI
  > 0 and permutation p <= 0.05; cross-arm AUROC strictly exceeds Baseline-
  internal AUROC; maximum case-deletion AUROC drop <= 0.15.
- **MODERATE_PROMISING** requires all: case-flip AUROC >= 0.62; partial rho > 0;
  cross-arm AUROC >= Baseline-internal AUROC; maximum case-deletion drop <= 0.20.
- **NO_USEFUL_SIGNAL** requires all: case-flip AUROC <= 0.55; both raw and
  partial absolute rho <= 0.10; Baseline-internal AUROC >= cross-arm AUROC.
- All other outcomes are **WEAK_INCONCLUSIVE** and motivate a separate 30-case
  replication before committing to Choice B.

## Expected Outputs

All outputs are under `results/paper2_pilot/analysis/dual_path_agreement/`:

- `matched_repeat_metrics.csv` (80 matched repeat pairs)
- `case_metrics.csv` (20 cases)
- `bootstrap_metrics.csv`, `permutation_metrics.csv`
- `case_influence.csv`, `repository_sensitivity.csv`
- `dual_path_summary.json`, `dual_path_report.md`
- `validation_report.md` (11/11 statistical-fallacy scan)
- `agreement_diagnostics.png`

## Reproducibility Gate

Run unit tests, execute the complete analysis twice into separate output
directories, and require identical SHA-256 hashes for JSON/Markdown/CSV outputs.
The raw pilot logs are never modified.
