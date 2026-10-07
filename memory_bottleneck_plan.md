# Memory Bottleneck Diagnosis: Retrieval vs. Adoption

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan -> run -> validate
- Origin Date: 2026-07-17
- Verification Status: VERIFIED
- Version Label: memory_bottleneck_v2

## Objective

Determine whether current Memory failures are primarily caused by missing
gold-file-relevant commits, failure to adopt a relevant retrieved commit, or
failure after a relevant commit has been adopted.

The primary sample is the frozen Random pilot: 20 cases, four Memory repeats
per case. The outcome-selected V4 15-case set is sensitivity-only and is never
pooled with or allowed to override the Random-pilot direction.

## Locked Metrics

- A candidate is relevant when its complete indexed changed-files intersect
  normalized Python GoldFiles.
- Hit@5 keeps rank 1-5 within every executed query, unions queries within a
  trajectory, and deduplicates by `(case, repeat, SHA)`.
- Primary Hit@5 is the mean of four repeat-hit indicators per case, averaged
  equally across 20 cases.
- A candidate is primary-adopted only if it was examined and one of its changed
  files appears in the final localization top-5.
- Trust Precision is relevant primary-adopted candidates divided by all primary-
  adopted candidates, with case-cluster bootstrap uncertainty.
- Conditional Trust Recall is the case-balanced probability of adopting at
  least one relevant candidate among repeat trajectories with Hit@5=1.

Each Memory Acc@5 failure is assigned once: retrieval miss, adoption miss, or
post-adoption failure.

## Decision Rule

Point estimates always select a Paper 2 direction. Confidence intervals assign
`SUPPORTED_DIRECTION` only when all inequalities defining the selected matrix
cell are supported; otherwise the result is `POINT_ESTIMATE_DIRECTION` and
requires at least 30 additional random cases before a confirmatory claim.

| Hit@5 | Trust Precision | Direction |
|---:|---:|---|
| <50% | >=50% | RETRIEVAL_LIMITED |
| <50% | <50% | BOTH_LIMITED |
| >70% | <50% | TRUST_LIMITED |
| >70% | >70% | BOTH_GOOD |
| otherwise | any | MIXED_DIRECTION, resolved by the largest failure-funnel share |

For a mixed tie, compare deficits from 70%; an exact remaining tie becomes
`JOINT_RETRIEVAL_FIRST`. V4 is a contradiction caveat only when it selects a
different intervention direction and differs by at least 20 percentage points
on Hit@5 or Trust Precision.

## Execution

Run in `/data/wjl/RepoMem` after sourcing `openaigpt4o_env`:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=. \
  /data/conda_envs/locagent/bin/python \
  scripts/analyze_memory_bottleneck.py
```

Outputs are written under
`results/paper2_pilot/analysis/memory_bottleneck/`. No API call or raw-log
mutation is permitted. Two full reruns must produce identical hashes for all
structured and rendered artifacts.
