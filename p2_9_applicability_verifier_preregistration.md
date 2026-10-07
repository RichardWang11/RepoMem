## Material Passport

- Protocol: `p2_9_applicability_verifier_pilot_v1`
- Status: **LOCKED BEFORE P2-9 FEATURE EXTRACTION**
- Date locked: 2026-07-20
- Repository root: `/data/wjl/RepoMem`
- Runtime: `/data/conda_envs/locagent/bin/python`
- External API: none

# P2-9 Applicability-Aware Verifier Offline Pilot

## Scope and interpretation boundary

This is a method-development pilot, not a confirmatory evaluation. The current
Discovery-30 and Fresh-30 labels and the three-sample P1-5 feature audit have
already been inspected. The former Fresh-30 cohort is therefore called the
`pilot evaluation cohort`, not an untouched holdout. A future confirmatory
claim requires a newly drawn evaluation set.

The human adjudicated applicability label is an oracle-informed evaluation
label only. Verifier inputs are restricted to `(issue, Memory, current repo)`.
Gold files, Gold patch, human/GPT labels, retrieval rank, Agent actions, and
outcomes are forbidden from feature computation. GPT labels are sensitivity
artifacts and are never used for parameter selection or the primary gate.

## Units and cohorts

- Unit: unique `(case, memory SHA)` candidate.
- Development: Discovery-30; Unclear rows excluded from primary metrics.
- Pilot evaluation: Fresh-30; all rows currently usable.
- Pooled-60: descriptive sensitivity only.
- Case-cluster bootstrap: 10,000 ordinary replicates, seed `20260720`.

## Locked hunk-local anchor extraction

Only added/deleted patch lines and hunk-header scope names may create anchors.
Pure context lines are excluded. Candidate order is deterministic:

1. function/class names parsed from hunk headers;
2. names defined on changed lines;
3. called or attribute names on changed lines;
4. other identifiers on changed lines.

Within a tier, rank by descending changed-line frequency then lexical order.
Python keywords, literals, one-character names, and a locked generic-name list
are excluded. This replaces whole-patch symbol extraction, which produced up to
313 candidates in P1-5.

Candidate budgets are `{5, 10, 20, 40}`. The budget is selected on
Discovery-30 only. For V-Struct, select among budgets retaining at least 85% of
Discovery applicable memories; maximize irrelevant-exposure reduction, then
choose the smaller budget. If no budget satisfies retention, V-Struct receives
`NO_DEVELOPMENT_SOLUTION` and no favorable pilot gate is available.

## Non-oracle features

1. `temporal_ancestry`
   - Prefer commit-catalog history order.
   - If the base commit is absent from the catalog, use a local Git checkout
     and `git merge-base --is-ancestor memory_sha base_sha`.
   - If neither source resolves both commits: `UNKNOWN`, pass-through.
   - `AFTER_OR_DIVERGENT` may reject; missing data may not reject.
2. `file_survival`
   - Resolve Memory changed files in the current-base graph.
3. `hunk_anchor_resolution`
   - Resolve budgeted anchors in the same file or at one unique moved location.
4. `current_relation_evidence`
   - Report current `imports`/`invokes`/`inherits` edges around resolved anchors.
   - This is explicitly a proxy and does not establish memory-era consistency.
5. `repository_evolution_proxy`
   - File survival plus same-file/moved-anchor survival. It does not detect
     semantic API lifecycle compatibility.

## Verifier variants

- `BM25`: retain every candidate.
- `V-Relaxed`: reject only when temporal ancestry is contradictory, or when all
  Memory files are absent and no anchor resolves uniquely elsewhere. Missing
  temporal/symbol evidence passes through.
- `V-Struct`: apply V-Relaxed, then reject when anchors exist but none resolve
  either in the original file or uniquely elsewhere. A patch with zero
  extractable anchors is an abstention/pass-through, not a rejection.

Current relation evidence is reported and ablated but is not a hard rejection
condition because the current graph is not a historical graph.

## Metrics

For cohort size `N`, labels exclude Unclear unless stated otherwise:

- Irrelevant Exposure = irrelevant candidates retained / `N`.
- Exposure reduction = BM25 Irrelevant Exposure minus verifier exposure.
- Useful Memory Retention = applicable candidates retained / applicable total.
- Candidate rejection rate and abstention/missing-feature rates.
- Case-cluster bootstrap 95% confidence intervals.

The Fresh-30 applicable denominator is three adjudicated memories. Thus 3/3 is
a must-not-fail sanity check only; it is not a precise estimate (the exact 95%
interval is very wide).

## Locked pilot decisions

- `PILOT_GO`: Fresh exposure reduction >= 15 percentage points AND retain 3/3
  Fresh applicable memories AND the chosen rule met the Discovery >=85%
  retention constraint.
- `PILOT_NO_GO`: Fresh exposure reduction < 15pp OR any Fresh applicable memory
  is rejected.
- `NO_DEVELOPMENT_SOLUTION`: no Discovery budget meets >=85% retention.

No outcome/pass-rate claim follows from this offline candidate-level pilot.
`PILOT_GO` only authorizes preregistration and collection of a new confirmatory
evaluation cohort; it does not authorize online deployment.
