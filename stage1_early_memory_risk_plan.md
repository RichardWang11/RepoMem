# Stage 1: Early Memory Risk Detection

## Goal and framing

Use execution-early signals from V2/V3/V4 and the Random pilot to predict risky Memory trajectories before the final localization outcome. Stage 1 is diagnostic and does not implement or claim benefits from a gating controller.

- Development: 15 outcome-selected cases, 180 Memory trajectories from V2/V3/V4.
- External validation: 20 random cases, 80 Memory trajectories.
- All splits, weighting, bootstrap, permutation, preprocessing, and calibration operate at case level.
- Primary target: paired harm at matched `(case, source/version, repeat)`, defined as Baseline Acc@5=1 and Memory Acc@5=0.
- Secondary targets: Memory failure, case flip/non-unanimity, and stability degradation relative to Baseline.

Paired harm intentionally uses a Baseline outcome that is unavailable in single-arm deployment. It is a diagnostic mechanism target. Memory failure is baseline-independent and deployment-relevant, but no model is deployment-ready until prospective Stage 2 validation.

Development paired-harm prevalence is 61/180 (33.9%); Random pilot prevalence is 6/80 (7.5%), with positives concentrated in three cases. Every external result must disclose this base-rate shift and limited effective positive sample.

## Early checkpoints and features

### T0: first SearchCommit result, before exposure

- BM25 top-1 score from the first executed query.
- Relative top-1/top-2 margin.
- Mean pairwise changed-file Jaccard among each query's top five commits.
- Mean token Jaccard among executed query texts.
- Mean top-five SHA-set Jaccard across queries.
- Fraction of query pairs with disjoint immediate-parent-module sets.
- Missing indicators for single query, absent top-2, or absent retrieval.

Cross-call retrieval Jaccard is availability-audit only because only one of 260 trajectories has multiple SearchCommit calls.

### T1: first current-code evidence observation

For `search_code_snippets`, `get_entity_contents`, or `explore_tree_structure`:

- Jaccard between Memory top-five files and first code-evidence candidate files.
- Whether first code evidence touches top-1 commit files.
- Unique commits examined before code evidence.
- Rapid unverified top-1 adoption: top-1 is examined before current-code evidence, no alternative SHA is examined first, and first code evidence touches top-1 files.
- Prespecified T0-risk × rapid-adoption interaction.

No gold files, final predictions, Baseline output, post-checkpoint messages, or whole-trajectory statistics may enter features.

## Models and evaluation

- Primary model: L2 logistic regression, C=1.
- Secondary model: balanced decision tree, max depth 2, minimum leaf size 10, seed 42.
- Training gives every case equal total weight and then balances classes.
- Development diagnostic uses leave-one-case-out across V2/V3/V4; every version/repeat of the held-out case stays in the test fold.
- T1 stacking uses nested case-level cross-fitting for T0 risk.
- Models are frozen on development before one-time Random pilot evaluation.
- Pooled 35-case LOCO is sensitivity-only.

Report AUROC, AP, AP/prevalence lift, Brier, Brier skill, and precision/recall among the top 20% predicted risk. Use 10,000 case-cluster bootstrap replicates and 1,000 development case-level label permutations.

## Calibration and verdict

- Report raw predictions, development-OOF isotonic calibration, and an oracle pilot-prior sensitivity using intercept-only log-odds correction.
- Never fit a calibration slope on pilot labels.
- Produce ten equal-frequency reliability bins, ECE, Brier, and Brier skill.
- Prior-adjusted results are descriptive because the 7.5% target prior is observed from pilot labels.

`PROMISING_EARLY_SIGNAL` requires all of:

- External T0 logistic AUROC >= 0.70 and cluster-bootstrap lower 95% bound >= 0.55.
- External AP lift >= 2.0 and lower 95% bound >= 1.25.
- External top-20% harm recall >= 0.50.
- Development LOCO AUROC >= 0.60.

`NO_USEFUL_SIGNAL` requires external AUROC upper 95% bound <= 0.55, AP-lift upper bound < 1.25, and no T1 AUROC/AP improvement. All other results are `INCONCLUSIVE_PILOT`.

T1 provides incremental over-adoption evidence only if AUROC improves by at least 0.05 or AP lift by at least 0.5 and the risk×rapid-adoption coefficient is positive. This is predictive association, not mediation or causality.

## Outputs and acceptance

Write trajectory and case feature tables, development LOCO and external predictions, model interpretation, calibration tables/plots, cluster bootstrap, permutation results, and a final Markdown report under `results/context_safe_m3a/replication_v4/analysis/early_memory_risk/`.

Acceptance counts are development=180, external=80, paired harm=61/180 and 6/80, and three external positive cases. Tests cover feature math, cutoff leakage, single/multi/no retrieval, missing top-2, duplicate SHA, empty code evidence, malformed arguments, case-isolated fitting, case-cluster bootstrap, calibration monotonicity, prior correction, reliability bins, and deterministic reruns.

## Case-flip directional consistency and bootstrap addendum

The exploratory case-flip result is evaluated on 20 Random-pilot cases (6 positive, 14 negative), with a frozen-model AUROC of 0.76190476. Its development LOCO AUROC is 0.3667 across 45 case-version observations from 15 unique cases. Because these estimates lie on opposite sides of chance, directional consistency is checked before any confidence-interval rule.

If development LOCO AUROC is below 0.55 while external AUROC is above 0.65, set `DIRECTIONAL_FLIP` and return `REPLICATION_REQUIRED_DIRECTIONAL_FLIP` regardless of the bootstrap interval. The interval then quantifies point-estimate uncertainty conditional on the frozen model and observed 20-case pilot only. Paper 2 commitment requires a separately evaluated second random pilot with at least 20 additional cases. No Stage 2 gating experiment is authorized.

Use 10,000 ordinary one-row-per-case bootstrap replicates as primary, allowing class counts to vary and discarding single-class samples. Report the discard rate and flag `CLASS_IMBALANCE_FRAGILITY` above 15%. Also report 10,000 stratified replicates with the observed 6/14 balance. The counterfactual continuation rule, available only when there is no directional flip, requires external AUROC at least 0.70 and the smaller of the two percentile lower bounds to exceed 0.50.

For both distributions report the 2.5, 25, 50, 75, and 97.5 percentiles, standard deviation, and counts below 0.5, in [0.5, 0.6), [0.6, 0.7), [0.7, 0.8), and at least 0.8. Flag `SIGNAL_FRAGILITY` if more than 20% of ordinary replicates fall below 0.5. Diagnose bimodality with one- versus two-component Gaussian mixtures: require a two-component BIC improvement of at least 10, component means separated by at least 0.15, and minimum component weight at least 0.10. If ordinary-bootstrap SD exceeds 0.15 or bimodality is detected, report an ordinary BCa interval as an alternative description; it does not override the primary decision.

## Case-flip influence audit addendum

The influence audit is a post-hoc, metric-only sensitivity analysis of the frozen 20-case pilot predictions. Compute every AUROC with `sklearn.metrics.roc_auc_score`, using its default average-rank handling of tied scores. Before deletion, require 20 unique cases, 6 positives, 14 negatives, and AUROC 0.76190476 within 1e-6. The model is never refit, and this audit can only reduce confidence in the pilot result.

Delete each case once and report its signed drop, `original AUROC - AUROC without case`. Report positive-case and negative-case deletion maxima, minima, and means separately, while retaining the overall maximum for the decision. Classify maximum single-case drop below 0.10 as no dominance, 0.10 through 0.20 as moderate sensitivity, and above 0.20 as single-case dominance. A minimum signed drop below -0.15 identifies a noise suppressor whose removal materially improves AUROC. As a stress test, enumerate all 190 two-case deletions and flag pair dominance above 0.20.

Check five hard stops independently: development/pilot directional flip (`dev <= 0.55` and `pilot >= 0.65`), ordinary BCa lower bound at or below 0.50, maximum single-case drop above 0.20, maximum pair drop above 0.20, and minimum signed drop below -0.15. Any trigger returns `STOP_PREDICT_AND_GATE`, with every trigger preserved in `stop_reasons`. With no hard stop, moderate sensitivity returns `REPLICATION_ONLY_MODERATE_SENSITIVITY`; otherwise return `ELIGIBLE_FOR_INDEPENDENT_REPLICATION`. No outcome authorizes Stage 2 gating.
