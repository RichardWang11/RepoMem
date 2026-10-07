## Material Passport

- Origin skill: academic-research-suite / experiment-agent
- Protocol: `memory_transfer_taxonomy_v1`
- Study type: oracle-informed measurement and taxonomy validation
- Repository root: `/data/wjl/RepoMem`
- Runtime: `/data/conda_envs/locagent/bin/python`
- Discovery source: `human_annotation_labels.csv` and the frozen P0-1 artifacts
- Candidate population: Random-pilot Memory top-5 exposures
- Holdout seed: `20260718`
- External upload: prohibited until the user separately authorizes the new GPT batch

# Memory Transfer Taxonomy Preregistration

## Research boundary

The study asks how retrieved software-engineering memories transfer to a current
task and why apparently similar memories fail to transfer. Retrieval remains an
important bottleneck; this study tests whether applicability judgment is an
additional, independent bottleneck. It does not compare which bottleneck is
largest.

The applicability label is an **oracle-informed measurement upper bound**. Human
and GPT annotators see the Gold patch. This label is not a training target for a
deployable classifier. A future deployable method may use only the issue, Memory,
and current repository; any non-oracle training label requires a separate
preregistration. Here Gold-visible labels are used only to validate the codebook
and measure what similarity heuristics could theoretically miss.

## Taxonomy

Applicability is `Direct-help`, `Indirect-help`, `Irrelevant`, or `Unclear`.
Applicable candidates receive one or more transfer bridges: T1 exact-component,
T2 structural-relational, T3 behavioral-analogical, and T4
evolutionary/historical. T4 is provisional until observed. Contribution roles
start as R1 repair template, R2 localization cue, R3 diagnostic clue, and R4
constraint/context. Irrelevant candidates receive one or more non-transfer tags:
S1 lexical/symbol collision, S2 coarse structural proximity, S3 exact-component
mechanism mismatch, S4 behavioral/thematic near-miss, and S5 obsolete/context
incompatible. `U` means insufficient evidence and is not an irrelevance mechanism.

R1 versus T1 and R3 versus T3 are tested on discovery data. A role is removed
from headline holdout reliability when absolute phi exceeds 0.80 and the smaller
directional conditional overlap is at least 0.80. R2 and R4 are always retained.

Annotators assign multi-label non-transfer tags while blinded to machine bands.
The analysis derives the primary tag with these locked priorities:

| Machine band | Priority |
|---|---|
| symbol_only | S1, S3, S4, S5, S2 |
| parent_only | S2, S3, S4, S5, S1 |
| exact_file | S3, S4, S5, S2, S1 |
| none | S4, S5, S1, S2, S3 |

Headline tables report multi-label prevalence and band-stratified primary tags;
they do not interpret an aggregate derived-primary frequency as natural prevalence.

## Samples and blinding

All current 30 items form the discovery cohort. Duplicate `(case, full SHA)`
items remain annotation audits but are collapsed for inferential counts. Two
humans independently code discovery, then adjudicate and freeze the codebook.

The confirmatory holdout contains 30 unique `(case, full SHA)` pairs from the
Random-pilot top-5 pool, excluding discovery identities. Cross-query/repeat
duplicates are collapsed. Sampling uses seed 20260718, at most two pairs per
case, at least 15 cases, and no applicability or structural-stratum balancing.
For repository `r`, its cap is `ceil(30 * max(0.25, p_r))`, where `p_r` is its
eligible-pool share. A holdout with any repository above 50% is
`REPO_DOMINATED`; cross-repository claims then remain exploratory.

Packets expose issue, Memory commit/patch/files, current-base context, and Gold
patch/files. They hide BM25 rank/score, structural bands, Agent actions, final
files, outcomes, and all other labels. Two humans and the GPT judge work
independently. Adoption and outcomes are unsealed only after label and codebook
hashes are frozen.

## Locked analyses and gates

Reliability reports raw agreement, Cohen's kappa, Gwet AC1, balanced accuracy,
per-tag positive agreement/F1/Jaccard, and 10,000 ordinary case-cluster bootstrap
replicates. Tags with fewer than three positives are descriptive only.

- `PROCEED`: coverage >= 85%, primary-code agreement >= 70%, kappa >= 0.50,
  and bootstrap lower bound > 0.
- `REFINE`: coverage 70-85% or kappa 0.30-0.50; changes require a new holdout.
- `PIVOT`: coverage < 70% or kappa < 0.30.

The applicability bottleneck receives preliminary support only when the blind
irrelevant estimate is at least 30% across at least five cases, at least 20% of
irrelevant candidates propagate into one Agent channel, taxonomy reliability is
`PROCEED`, at least one structural-positive/irrelevant counterexample exists,
and the holdout is not repo-dominated. Propagation is reported separately as
selection (`examine_commit`), exploration (exact changed-file tool match), and
final (changed file in final top-5).

Full bidirectional structural evidence additionally requires at least one
structural-negative/applicable item. If that low-frequency direction is absent
but the first direction exists, the verdict is `REFINE_SECOND_N30`, not
`PIVOT`. Outcome associations are descriptive, case-clustered, and non-causal.

## Reproducibility and stopping points

Preparation must assert identity uniqueness, discovery exclusion, case/repo
caps, forbidden-field absence, deterministic sample hashes, and provenance.
The experiment stops at `AWAITING_DISCOVERY_CODING`,
`AWAITING_CODEBOOK_FREEZE`, `AWAITING_HOLDOUT_HUMANS`, or
`AWAITING_GPT_AUTHORIZATION` rather than silently weakening a gate.
