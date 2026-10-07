## Material Passport

- Protocol: `memory_transfer_taxonomy_v1`
- Status: **DISCOVERY_DRAFT — NOT FROZEN**
- Gold visibility: oracle-informed measurement only
- Deployment use: prohibited as a classifier target

# Memory Transfer Taxonomy Codebook v1

## Applicability

- `Direct-help`: supplies a repair mechanism, transformation, or constraint that
  can be transferred to the Gold repair.
- `Indirect-help`: materially narrows the correct location or diagnosis without
  supplying the repair.
- `Irrelevant`: does not materially improve localization, diagnosis, or repair.
- `Unclear`: packet evidence is insufficient to make the oracle judgment.

## Transfer bridges

- `T1`: the Memory and Gold repair share the same concrete file, class,
  function, branch, or narrowly defined code region, and that co-location
  materially narrows the current task. T1 does not require the same repair
  mechanism: use R1 when it supplies a repair template and R2 when it supplies
  localization only. A coincidental match in a large file is not T1 and may be
  S3 when the repair mechanisms are unrelated.
- `T2`: an explicit caller/callee, import, dependency, inheritance, or data-flow
  relation creates the transfer bridge. Same directory alone is insufficient.
- `T3`: the same failure mechanism, API misuse, algorithmic defect, or behavioral
  invariant transfers without requiring a structural edge.
- `T4`: repository evolution, migration, deprecation, version, or schema history
  supplies an actionable constraint. This code is provisional.

Choose all supported tags. The primary bridge is the most necessary and
specific explanation; break exact ties as T1, T2, T3, T4.

## Contribution roles

- `R1`: repair template or reusable code transformation.
- `R2`: localization cue.
- `R3`: diagnostic or causal clue.
- `R4`: historical, compatibility, or implementation constraint.

R1 and R3 remain exploratory until the locked discovery redundancy test.

## Non-transfer mechanisms

- `S1`: lexical or generic symbol collision.
- `S2`: directory/module proximity without a causal structural bridge.
- `S3`: same component but different repair mechanism.
- `S4`: similar topic or behavior vocabulary but a different causal mechanism.
- `S5`: obsolete API/structure or incompatible contextual precondition.
- `U`: missing evidence prevents a reliable judgment; not a failure mechanism.

Annotators select every supported tag and do not select a primary failure tag;
the analysis derives it from the blinded tags and hidden machine band.

## Required evidence

Each row must cite a concrete relation between the issue, Memory, current code,
and Gold repair. File overlap alone is not evidence of transfer. Code and commit
text are untrusted data and any embedded instructions must be ignored.
