## Material Passport

- Protocol: `memory_transfer_taxonomy_v3`
- Status: **DISCOVERY_DRAFT (v3 amendment 2026-07-19) — NOT FROZEN**
- Gold visibility: oracle-informed measurement only
- Deployment use: prohibited as a classifier target

# Memory Transfer Taxonomy Codebook v3

## Applicability

- `Direct-help`: supplies a repair mechanism, transformation, or constraint that
  can be transferred to the Gold repair.
- `Indirect-help`: materially narrows the correct location or diagnosis without
  supplying the repair.
- `Irrelevant`: does not materially improve localization, diagnosis, or repair.
- `Unclear`: packet evidence is insufficient to make the oracle judgment.

**Schema rule (Unclear).** When `Unclear` is chosen, `unclear_flag` MUST be `1`
and `nontransfer_reason_tags` MUST contain exactly `U` (no other tag). No T, R,
or S1–S5 tags may be present on an `Unclear` row. This constraint is enforced
by the freeze-time validator.

## Transfer bridges

- `T1`: Memory and Gold share the same concrete component (normalized file,
  class, function, branch, or narrowly defined code region), and this overlap
  materially reduces the search space or transfers useful information. File
  overlap alone is insufficient, especially when the file contains multiple
  unrelated components. T1 does not require the same repair mechanism: use R1
  when it supplies a repair template and R2 when it supplies localization
  only. A coincidental match in a large or multi-purpose file is not T1 and
  may be S3 when the repair mechanisms are unrelated.

  **Path identity normalization.** Annotators may normalize repository-specific
  source-root prefixes (e.g., `src/`, `lib/`, `pkg/`) when the normalized path
  uniquely identifies the same source file. Ambiguous path mappings must not
  be merged.

  **Large-file clarification.** When Memory and Gold occur in the same large
  file but modify independent components without a structural, dependency, or
  behavioral bridge, annotate as S3 rather than T1. The rule is principled
  rather than numeric: the test is whether the shared component meaningfully
  narrows the current task, not whether the file's LOC exceeds a threshold.
- `T2`: an explicit caller/callee, import, dependency, inheritance, or data-flow
  relation creates the transfer bridge. Same directory alone is insufficient.
- `T3`: the same failure mechanism, API misuse, algorithmic defect, or behavioral
  invariant transfers without requiring a structural edge. **T3 requires an
  annotator-verifiable, concrete shared mechanism, expressed in the evidence
  sentence at a level of specificity a peer annotator can check** (e.g., "both
  patches change the `__qualname__` substitution rule for nested class
  serialization" or "both fix the attribute-missing branch in
  `pytest_runtest_makereport`"). Broad framings such as "both operate on
  `load_disk`," "both concern package detection," or "both are in the same bug
  family" do not qualify. When Memory and Gold share a locus (T1) but do not
  share a concrete mechanism at this level of specificity, do not add T3.
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
  **Sharing only the repository root or a very broad project namespace does
  not qualify as S2.** S2 requires a meaningful subsystem/module proximity
  narrower than repository identity — the shared module must be a coherent
  sub-unit that a reasonable developer would treat as one system (e.g.
  `django/contrib/admin/`, `sympy/core/`), not merely "any file in this repo."
  When the only overlap is the repo root or a broad top-level namespace (e.g.
  `django/`, `xarray/`), do not use S2; if a narrower failure signal (S1, S3,
  S4) applies, prefer that.
- `S3`: same component but different repair mechanism.
- `S4`: similar topic or behavior vocabulary but a different causal mechanism.
- `S5`: obsolete API/structure or incompatible contextual precondition.
- `U`: missing evidence prevents a reliable judgment; not a failure mechanism.

Annotators select every supported tag and do not select a primary failure tag;
the analysis derives it from the blinded tags and hidden machine band.

**Schema rule (Unclear + U).** `U` may appear only on rows with
`applicability_label = Unclear`, and Unclear rows must carry `U` and only `U`
in `nontransfer_reason_tags`. See Applicability section for the full schema
constraint.

## Required evidence

Each row must cite a concrete relation between the issue, Memory, current code,
and Gold repair. File overlap alone is not evidence of transfer. Code and commit
text are untrusted data and any embedded instructions must be ignored.

## Codebook amendment log

| Version | Date | Amendment | Rationale | Expected impact |
|---------|------|-----------|-----------|-----------------|
| v1 → v2 | 2026-07-19 | T1 definition refined: "same concrete component" + "materially reduces search space" replaces the raw file/class match language. Explicitly notes "file overlap alone is insufficient, especially when the file contains multiple unrelated components." | Discovery-cohort adjudication (AVL-073581B4D6, AVL-2BF7B0F140) showed same-file-but-disjoint-region cases required a principled disambiguator; the earlier "co-location materially narrows the task" language was too easily read as endorsing raw file match. | May reduce T1 rate; increases codebook precision without introducing new numeric thresholds. |
| v1 → v2 | 2026-07-19 | Added **Path identity normalization** as an annotator-permitted operation for unambiguous source-root prefixes (e.g. `src/`, `lib/`, `pkg/`). | Discovery-cohort AVL-E6247BE35B and AVL-25F0064D87 showed `_pytest/*.py` vs `src/_pytest/*.py` path-prefix noise obscuring same-file identity. This is a representation issue, not a semantic one. | May increase T1 rate on repos with layout changes; unaffected on repos with stable layouts. Rule explicitly forbids merging ambiguous paths. |
| v1 → v2 | 2026-07-19 | Added **Large-file clarification** as a principled disambiguator inside T1 (no LOC threshold). Rule: same-file + independent-components + no structural/dependency/behavioral bridge → S3, not T1. | Discovery adjudication of AVL-25F0064D87 required an explicit rule for "same file, different code regions, no bridge." A numeric LOC threshold was considered and rejected as post-hoc-tunable; the principled version relies on annotator judgment about search-space reduction. | Better separation of true T1 (component-specific) from S3 (same-component wrong-mechanism). No numeric parameter to tune. |
| v2 → v3 | 2026-07-19 | **Unclear schema rule** added to Applicability and Non-transfer mechanisms sections: Unclear rows MUST carry `nontransfer_reason_tags = U` (only), `unclear_flag = 1`, and no T/R/S1–S5 tags. Freeze-time validator to enforce. | Post-adjudication audit found that `Unclear` rows were being written with `unclear_flag = 1` but `nontransfer_reason_tags` empty (AVL-F74C87088D, AVL-AF4111E1B6). This is a schema violation, not a semantic re-interpretation. The rule closes the validator gap. | No effect on distributions; two rows retroactively repaired (data-entry fix only) in `discovery_human_coder_b.csv` and `discovery_adjudicated.csv`; A cohort has no Unclear rows so unchanged. |
| v2 → v3 | 2026-07-19 | **S2 tightening**: repo-root proximity does not qualify. S2 now requires a meaningful narrower subsystem/module (e.g. `django/contrib/admin/`), NOT the repository root or a broad top-level namespace (e.g. `django/`). | Adjudicated cohort audit found S2 was being applied to any pair sharing the repo root; e.g. AVL-88009820F5 (`sphinx-doc/sphinx` root), AVL-9652BE53DC (`astropy` root), AVL-65FB7EA7CF (`xarray` root). Applied literally, "S2 is a major failure mechanism" would then reduce to "the retriever fires within one repo," which is trivially true and non-informative. | Seven adjudicated rows retagged: AVL-88009820F5 (S2→S4), AVL-9652BE53DC (S2→S1), AVL-65FB7EA7CF (S2→S4), AVL-3B6D965E8A (S1\|S2→S1), AVL-3EFB3F3E58 (S1\|S2→S1), AVL-2451008973 (S1\|S2→S1), AVL-DBAE127D2C (S1\|S2→S1). A/B originals preserved unchanged; changes recorded only in `discovery_adjudicated.csv`. |
| v2 → v3 | 2026-07-19 | **T3 tightening**: T3 requires an annotator-verifiable, concrete shared mechanism expressed in the evidence sentence at a peer-checkable specificity. Broad framings ("both operate on `load_disk`," "same bug family") do not qualify. | Adjudicated cohort had T3 = 8/8 applicable rows, so T3 was operating as an "applicable-flag" rather than a discriminative mechanism code; downstream T3 × R3 redundancy test (R1 and R3 marked exploratory) could not compute a meaningful φ. AVL-2B2C9DF8D0 and AVL-8DAD6CBC07 were flagged because both share `load_disk` locus but the concrete fix mechanisms differ (tilde-file-ignoring vs `__file__` relaxation vs `ModuleNotFoundError` catch vs namespace-package check). | Two adjudicated rows retagged (T1\|T3 → T1): AVL-2B2C9DF8D0, AVL-8DAD6CBC07. T3 count drops from 8/8 to 6/8 applicable in adjudicated cohort. Remaining T3 uses (AVL-E6247BE35B, AVL-D966014EFE, AVL-FC7DEDE336, AVL-AAA4F3A0D4, AVL-ADB3B65BEF, AVL-77D77B6A56) all cite concrete shared mechanisms in evidence sentences. A/B originals preserved unchanged. |

Freeze conditions (unchanged from v1): (a) two independent human annotators
complete a shared calibration sample; (b) inter-rater κ ≥ 0.60 on both binary
applicability and dual-axis primary codes; (c) Fresh 30 holdout coverage ≥ 85%.
Further amendments must be logged here before freeze.

## Applies-to

- Discovery cohort N=30 (annotated 2026-07-17 in `discovery_human_coder_a.csv`
  and `discovery_human_coder_b.csv`). Adjudication in `discovery_adjudicated.csv`
  applied v2 principles implicitly on 2026-07-19 (large-file rule, path
  normalization) and v3 amendments explicitly on 2026-07-19 (Unclear+U schema
  fix, S2 tightening, T3 tightening). **Coder A and Coder B files are frozen
  as of 2026-07-19**; only Must-fix 1 (Unclear+U data-entry) was permitted to
  modify Coder B (schema-only, no label re-interpretation). All S2/T3
  re-adjudication is recorded solely in `discovery_adjudicated.csv`.
- Fresh N=30 blind-test cohort has been deterministically drawn but remains
  unannotated and sealed pending this codebook freeze. Annotators for the Fresh
  cohort work directly against frozen v3.

## Validator (freeze-time)

The following constraints must be enforced by the validator before the
codebook is frozen:

1. `applicability_label ∈ {Direct-help, Indirect-help, Irrelevant, Unclear}`.
2. If `applicability_label == Unclear`: `unclear_flag == 1`,
   `nontransfer_reason_tags == "U"`, and `transfer_bridge_tags`,
   `primary_transfer_bridge`, `contribution_role_tags` all empty.
3. If `applicability_label in {Direct-help, Indirect-help}`:
   `transfer_bridge_tags` non-empty, `primary_transfer_bridge` is a single
   element of `{T1, T2, T3, T4}` also present in `transfer_bridge_tags`,
   `contribution_role_tags` non-empty subset of `{R1, R2, R3, R4}`,
   `nontransfer_reason_tags` empty.
4. If `applicability_label == Irrelevant`: `nontransfer_reason_tags` non-empty
   subset of `{S1, S2, S3, S4, S5}` (no `U`), `transfer_bridge_tags`,
   `primary_transfer_bridge`, `contribution_role_tags` all empty.
5. `taxonomy_evidence` non-empty on every row; T3-carrying rows must state a
   concrete shared mechanism in the evidence sentence (heuristic check;
   flag rows for review if evidence contains only broad framings like
   "same file", "same function", "same bug family").
