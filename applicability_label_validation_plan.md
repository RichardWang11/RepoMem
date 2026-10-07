# P0-1 Applicability Label Lock and Two-Axis Validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: revised plan
- Origin Date: 2026-07-17
- Protocol Version: applicability_label_validation_v2
- Verification: UNVERIFIED — annotation must not begin before preregistration freeze

## Objective

Validate whether a locked structural applicability label agrees with blinded
human judgment before investing in a deployable Memory verifier. This protocol
separates task applicability, current-code validity, and downstream outcome
impact. It does not call structurally irrelevant memories harmful.

## Sample

The unit is `(case_id, repeat_id, memory_commit_sha)`, deduplicated across
retrieval queries in the Random pilot. Draw 30 candidates with seed `20260717`:

| Exclusive stratum | Definition | n |
|---|---|---:|
| exact_file | MemoryFiles intersects GoldFiles | 8 |
| parent_only | no file match; immediate parent module matches | 7 |
| symbol_only | no file/parent match; patch symbol matches | 7 |
| none | no structural component matches | 8 |

The sample must cover at least 15 cases, contain at most two candidates per
case, and target at most 16 Django candidates. Sampling cannot use adoption,
Acc@5, paired harm, or human labels.

## Locked Labels

Machine weak baseline:

```text
WeakRelevant = FileOverlap
```

Machine primary label:

```text
StructuralRelevant = FileOverlap OR ParentModuleMatch OR SymbolOverlap
StructurallyIrrelevant = NOT StructuralRelevant
```

Human and GPT independently label task applicability as `Direct-help`,
`Indirect-help`, or `Irrelevant`. The binary reference is Direct/Indirect versus
Irrelevant. They separately label current-code validity as `Valid`,
`Partially-valid`, or `Invalid-obsolete`.

The packet includes the issue, Memory commit and patch, bounded current-code
context, and Gold patch. It excludes the machine label/stratum, Agent adoption,
localization outcome, and semantic Memory summary.

## Analysis and Gate

Primary statistic: Cohen's kappa between `StructuralRelevant` and the blinded
human binary applicability label. Report its point estimate and a 10,000-repeat
case-cluster percentile bootstrap CI. Track undefined bootstrap replicates;
more than 10% triggers `BOOTSTRAP_UNSTABLE`.

| Verdict | Point kappa | CI lower | Action |
|---|---:|---:|---|
| PROCEED | >= 0.50 | >= 0.30 | Freeze Level 2; proceed to P1-5 |
| REFINE_AND_REPLICATE | >= 0.30 | >= 0.00, but not PROCEED | Refine and validate on a fresh 30 |
| SERIOUS_PIVOT | otherwise | otherwise | Stop this applicability-filtering direction |

Also report Level-1 kappa, agreement, confusion matrix, sensitivity,
specificity, balanced accuracy, MCC, PABAK, Gwet AC1, Human-GPT weighted and
binary kappa, machine-validity agreement, component precision, stratum rates,
and every false-positive/false-negative case.

Temporal validity is three-state: `before_base`, `at_or_after_base`, or
`unknown_base_index`. Unknown items are not treated as obsolete; they are
excluded from machine-validity agreement and its coverage is reported.

Human-GPT binary kappa below 0.40 triggers `ANNOTATION_UNSTABLE`; this blocks
P2-9 even if the primary gate passes. Outcome/adoption analysis is deferred and
must not modify this gate.

## Structural-to-Impact Boundary

P0-1 does not define `HarmfulMemory`. A later preregistered analysis may define
off-target propagation only when a structurally irrelevant candidate also
propagates into observable Agent decisions. Structural irrelevance, propagation,
and outcome impact must remain separate layers.
