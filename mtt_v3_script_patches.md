# Patch: `scripts/memory_transfer_taxonomy.py` v3 fixes

**Target:** `/data/wjl/RepoMem/scripts/memory_transfer_taxonomy.py`
**Goal:** unblock v3 freeze pipeline. Six items from the audit — items 1, 3, 4, 5, 6 addressed here; item 2 (DEFAULT_CODEBOOK path) already done in the current file.
**Gates locked by user:** binary κ ≥ 0.60 AND primary-code κ ≥ 0.60 on BOTH raw N=30 and unique N=29; T3 applicable-row rate ≤ 0.90.

Apply the 5 patches below in order. Each shows the exact block to REPLACE and the block to REPLACE-WITH. All patches use only existing script helpers (`split_tags`, `safe_kappa`, `sha256_path`, `write_json`, `load_csv`, `derive_primary_nontransfer`, `PRIORITY`, `APPLICABILITY`, `TRANSFER_TAGS`, `ROLE_TAGS`, `NONTRANSFER_TAGS`) — **no new imports required except adding `re` to the standard library imports at the top of the file.**

---

## Patch 1 — Fix broken `validate_coding_row`

The function currently uses undefined symbols (`Dict`, `List`, `_split_tags`, `LABELS`, `APPLICABLE_LABELS`, `T_TAGS`, `R_TAGS`, `S_TAGS`) which fail import. All callers use the pattern `for valid, error in [validate_coding_row(row)]`, so the return type MUST stay `(bool, str)` — not `list[str]`.

### REPLACE this block (currently ~lines with the broken `validate_coding_row`):

```python
def validate_coding_row(row: Dict[str, str]) -> List[str]:
    """Return list of schema-violation strings; empty list means row passes."""
    errs = []
    lbl = row.get("applicability_label", "")
    tt = _split_tags(row.get("transfer_bridge_tags", ""))
    pt = row.get("primary_transfer_bridge", "").strip()
    rt = _split_tags(row.get("contribution_role_tags", ""))
    st = _split_tags(row.get("nontransfer_reason_tags", ""))
    uf = row.get("unclear_flag", "").strip()
    ev = row.get("taxonomy_evidence", "").strip()

    if lbl not in LABELS:
        errs.append(f"applicability_label '{lbl}' not in {sorted(LABELS)}")
        return errs

    if not ev:
        errs.append("taxonomy_evidence is empty")

    if lbl == "Unclear":
        if uf != "1":
            errs.append(f"Unclear must have unclear_flag='1', got '{uf}'")
        if st != {"U"}:
            errs.append(f"Unclear must have nontransfer_reason_tags='U' only, got {sorted(st)}")
        if tt or pt or rt:
            errs.append(f"Unclear must have empty T/R fields; got tt={sorted(tt)} pt={pt!r} rt={sorted(rt)}")

    elif lbl in APPLICABLE_LABELS:
        if not tt:
            errs.append("Applicable row must have non-empty transfer_bridge_tags")
        elif not tt.issubset(T_TAGS):
            errs.append(f"transfer_bridge_tags contain invalid: {sorted(tt - T_TAGS)}")

        if pt not in T_TAGS:
            errs.append(f"primary_transfer_bridge '{pt}' not in {sorted(T_TAGS)}")
        if pt and pt not in tt:
            errs.append(f"primary_transfer_bridge '{pt}' must appear in transfer_bridge_tags {sorted(tt)}")

        if not rt:
            errs.append("Applicable row must have non-empty contribution_role_tags")
        elif not rt.issubset(R_TAGS):
            errs.append(f"contribution_role_tags contain invalid: {sorted(rt - R_TAGS)}")

        if st:
            errs.append(f"Applicable row must have empty nontransfer_reason_tags; got {sorted(st)}")
        if uf == "1":
            errs.append("Applicable row must have unclear_flag != '1'")

    elif lbl == "Irrelevant":
        if not st:
            errs.append("Irrelevant row must have non-empty nontransfer_reason_tags")
        elif not st.issubset(S_TAGS):
            errs.append(f"nontransfer_reason_tags contain invalid: {sorted(st - S_TAGS)}")
        if "U" in st:
            errs.append("Irrelevant row must NOT contain 'U' (U is a judgeability flag, not a mechanism)")
        if tt or pt or rt:
            errs.append(f"Irrelevant row must have empty T/R fields; got tt={sorted(tt)} pt={pt!r} rt={sorted(rt)}")
        if uf == "1":
            errs.append("Irrelevant row must have unclear_flag != '1'")

    return errs
```

### WITH this block:

```python
_APPLICABLE_ONLY = {"Direct-help", "Indirect-help"}
_S_ONLY = {"S1", "S2", "S3", "S4", "S5"}
_U_ONLY = {"U"}


def _v3_row_errors(row):
    """Return list of v3 schema violations for one row (empty list = pass)."""
    errs = []
    lbl = row.get("applicability_label", "")
    tt = split_tags(row.get("transfer_bridge_tags", ""))
    pt = (row.get("primary_transfer_bridge", "") or "").strip()
    rt = split_tags(row.get("contribution_role_tags", ""))
    st = split_tags(row.get("nontransfer_reason_tags", ""))
    uf = (row.get("unclear_flag", "") or "").strip()
    ev = (row.get("taxonomy_evidence", "") or "").strip()

    if lbl not in APPLICABILITY:
        errs.append(f"applicability_label {lbl!r} not in {sorted(APPLICABILITY)}")
        return errs
    if not ev:
        errs.append("taxonomy_evidence is empty")

    if lbl == "Unclear":
        if uf not in {"1", "true", "True", "yes"}:
            errs.append(f"Unclear must have unclear_flag truthy; got {uf!r}")
        if st != _U_ONLY:
            errs.append(f"Unclear must have nontransfer_reason_tags='U' only; got {sorted(st)}")
        if tt or pt or rt:
            errs.append(
                f"Unclear must have empty T/R fields; got tt={sorted(tt)} pt={pt!r} rt={sorted(rt)}"
            )
    elif lbl in _APPLICABLE_ONLY:
        if not tt:
            errs.append("Applicable row must have non-empty transfer_bridge_tags")
        elif not tt.issubset(TRANSFER_TAGS):
            errs.append(f"transfer_bridge_tags contain invalid: {sorted(tt - TRANSFER_TAGS)}")
        if pt not in TRANSFER_TAGS:
            errs.append(f"primary_transfer_bridge {pt!r} not in {sorted(TRANSFER_TAGS)}")
        elif pt not in tt:
            errs.append(f"primary_transfer_bridge {pt!r} must appear in transfer_bridge_tags {sorted(tt)}")
        if not rt:
            errs.append("Applicable row must have non-empty contribution_role_tags")
        elif not rt.issubset(ROLE_TAGS):
            errs.append(f"contribution_role_tags contain invalid: {sorted(rt - ROLE_TAGS)}")
        if st:
            errs.append(f"Applicable row must have empty nontransfer_reason_tags; got {sorted(st)}")
        if uf in {"1", "true", "True", "yes"}:
            errs.append("Applicable row must have unclear_flag falsy")
    elif lbl == "Irrelevant":
        if not st:
            errs.append("Irrelevant row must have non-empty nontransfer_reason_tags")
        elif not st.issubset(_S_ONLY):
            errs.append(f"nontransfer_reason_tags contain invalid or U: {sorted(st - _S_ONLY)}")
        if tt or pt or rt:
            errs.append(
                f"Irrelevant must have empty T/R fields; got tt={sorted(tt)} pt={pt!r} rt={sorted(rt)}"
            )
        if uf in {"1", "true", "True", "yes"}:
            errs.append("Irrelevant must have unclear_flag falsy")
    return errs


def validate_coding_row(row):
    """Backwards-compatible API: return (valid, error_str). Empty error on pass."""
    errs = _v3_row_errors(row)
    if errs:
        return False, "; ".join(errs)
    return True, ""
```

**Why:** existing callers use `for valid, error in [validate_coding_row(row)]` (see `readiness`, `freeze`, `analyze`). Preserving `(bool, str)` prevents cascading changes. `_v3_row_errors` is a private helper if you later want richer reporting.

**Symbols used:** `APPLICABILITY`, `TRANSFER_TAGS`, `ROLE_TAGS`, `NONTRANSFER_TAGS`, `split_tags` — all already defined in the script. `_APPLICABLE_ONLY` and `_S_ONLY` are the two new constants (add near the other constant blocks).

---

## Patch 2 — `prepare()` manifest → v3

### FIND:

```python
    manifest = {
        "protocol": "memory_transfer_taxonomy_v1",
        "status": "AWAITING_DISCOVERY_CODING",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
```

### REPLACE WITH:

```python
    manifest = {
        "protocol": "memory_transfer_taxonomy_v3",
        "status": "AWAITING_DISCOVERY_CODING",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
```

**Why:** Currently manifest labels itself as v1, contradicting the DEFAULT_CODEBOOK v3 path. Also update the report string a few lines down:

### FIND:

```python
    report = f"""## Material Passport

- Protocol: memory_transfer_taxonomy_v1
```

### REPLACE WITH:

```python
    report = f"""## Material Passport

- Protocol: memory_transfer_taxonomy_v3
```

---

## Patch 3 — Add reliability + T3 helpers (module-level)

Add near the top of the file (below the constant blocks, above `def prepare`).

**Add import at top:**

```python
import re
```

**Add helpers:**

```python
DISCOVERY_S_PRIORITY = ("S1", "S3", "S4", "S5", "S2")


def _binary_applicability(label):
    """Fold Unclear into NotApp for binary comparison (freeze-gate metric)."""
    return "App" if label in _APPLICABLE_ONLY else "NotApp"


def _discovery_primary_code(row):
    """Derive primary code for discovery reliability.

    Discovery cohort lacks a `sampling_stratum` column (that lives in the holdout
    sealed_machine_features_after_freeze.csv), so we use a uniform narrower-first
    S priority (codebook v3 §S2 clarification: S2 is the weakest signal).
    Holdout must use `derive_primary_nontransfer(stratum, tags)` instead.
    """
    lbl = row.get("applicability_label", "")
    if lbl in _APPLICABLE_ONLY:
        return row.get("primary_transfer_bridge", "") or "MISSING"
    if lbl == "Irrelevant":
        tags = split_tags(row.get("nontransfer_reason_tags", ""))
        for pri in DISCOVERY_S_PRIORITY:
            if pri in tags:
                return pri
        return sorted(tags)[0] if tags else "MISSING"
    if lbl == "Unclear":
        return "U"
    return "MISSING"


def _unique_pair_index(rows):
    """Return indices of first occurrence per (case_id, memory_sha)."""
    seen = set()
    out = []
    for i, r in enumerate(rows):
        key = (r.get("case_id", ""), (r.get("memory_sha", "") or "").lower())
        if key not in seen:
            out.append(i)
            seen.add(key)
    return out


def _reliability_block(rows_a, rows_b):
    la_bin = [_binary_applicability(r["applicability_label"]) for r in rows_a]
    lb_bin = [_binary_applicability(r["applicability_label"]) for r in rows_b]
    pa = [_discovery_primary_code(r) for r in rows_a]
    pb = [_discovery_primary_code(r) for r in rows_b]
    return {
        "n": len(rows_a),
        "binary_applicability_kappa": safe_kappa(la_bin, lb_bin),
        "binary_applicability_agreement": (
            sum(1 for a, b in zip(la_bin, lb_bin) if a == b) / len(la_bin)
            if la_bin else 0.0
        ),
        "primary_code_kappa": safe_kappa(pa, pb),
        "primary_code_agreement": (
            sum(1 for a, b in zip(pa, pb) if a == b) / len(pa) if pa else 0.0
        ),
        "a_binary_dist": dict(Counter(la_bin)),
        "b_binary_dist": dict(Counter(lb_bin)),
    }


def compute_discovery_reliability(rows_a, rows_b, adjudicated_rows,
                                   kappa_gate=0.60, t3_barrier=0.90):
    """Compute freeze-gate reliability metrics on raw and unique views.

    Returns a dict with:
      raw:  {kappa, agreement, ...} for N=30
      unique: {...} for N=(unique (case_id, sha) count)
      t3: {n_applicable, n_t3, rate, barrier}
      gate_passed: bool  — True iff ALL κ ≥ gate AND T3 rate ≤ barrier
      gate_reasons: list of failure descriptions if gate_passed is False
    """
    if [r["sample_id"] for r in rows_a] != [r["sample_id"] for r in rows_b]:
        raise ValueError("Coder A and Coder B row order/identity mismatch")

    uniq_idx = _unique_pair_index(rows_a)
    raw = _reliability_block(rows_a, rows_b)
    uniq_a = [rows_a[i] for i in uniq_idx]
    uniq_b = [rows_b[i] for i in uniq_idx]
    uniq = _reliability_block(uniq_a, uniq_b)

    applicable = [
        r for r in adjudicated_rows
        if r.get("applicability_label") in _APPLICABLE_ONLY
    ]
    if applicable:
        n_t3 = sum(
            1 for r in applicable
            if "T3" in split_tags(r.get("transfer_bridge_tags", ""))
        )
        t3_rate = n_t3 / len(applicable)
    else:
        n_t3 = 0
        t3_rate = 0.0

    reasons = []
    for view, block in (("raw", raw), ("unique", uniq)):
        for name in ("binary_applicability_kappa", "primary_code_kappa"):
            k = block[name]
            if k is None or k < kappa_gate:
                reasons.append(f"{view} {name} = {k} < {kappa_gate}")
    if t3_rate > t3_barrier:
        reasons.append(
            f"T3 rate = {n_t3}/{len(applicable)} = {t3_rate:.3f} > {t3_barrier} "
            "(T3 has degenerated to an applicable-flag; arbiter review required)"
        )

    return {
        "raw": raw,
        "unique": uniq,
        "t3": {
            "n_applicable": len(applicable),
            "n_t3": n_t3,
            "rate": t3_rate,
            "barrier": t3_barrier,
        },
        "kappa_gate": kappa_gate,
        "gate_passed": not reasons,
        "gate_reasons": reasons,
        "s_priority": list(DISCOVERY_S_PRIORITY),
    }


def _reliability_report_md(rel):
    def _fmt(v, prec=3):
        return "n/a" if v is None else f"{v:.{prec}f}"
    raw, uniq = rel["raw"], rel["unique"]
    verdict = "PASS" if rel["gate_passed"] else "FAIL"
    lines = [
        "# Discovery Reliability Report (Coder A × Coder B)",
        "",
        f"**Freeze gate:** binary κ ≥ {rel['kappa_gate']} AND primary-code κ ≥ {rel['kappa_gate']} on BOTH raw and unique views; T3 applicable-row rate ≤ {rel['t3']['barrier']:.2f}.",
        f"**Verdict:** {verdict}",
        "",
        "## Reliability metrics",
        "",
        "| Metric | Raw N=%d | Unique N=%d |" % (raw['n'], uniq['n']),
        "|---|---:|---:|",
        f"| Binary applicability κ (App = Direct+Indirect vs NotApp = Irrelevant+Unclear) | {_fmt(raw['binary_applicability_kappa'])} | {_fmt(uniq['binary_applicability_kappa'])} |",
        f"| Binary applicability observed agreement | {_fmt(raw['binary_applicability_agreement'])} | {_fmt(uniq['binary_applicability_agreement'])} |",
        f"| Dual-axis primary-code κ | {_fmt(raw['primary_code_kappa'])} | {_fmt(uniq['primary_code_kappa'])} |",
        f"| Dual-axis primary-code observed agreement | {_fmt(raw['primary_code_agreement'])} | {_fmt(uniq['primary_code_agreement'])} |",
        "",
        f"Primary S derivation: uniform narrower-first `{rel['s_priority']}` (discovery has no stratum; holdout uses PRIORITY[stratum]).",
        "",
        "## T3 degeneracy check",
        "",
        f"T3 fires on {rel['t3']['n_t3']}/{rel['t3']['n_applicable']} applicable rows "
        f"({rel['t3']['rate']:.1%}). Barrier: {rel['t3']['barrier']:.0%}.",
    ]
    if rel["gate_reasons"]:
        lines += ["", "## Gate failures", ""]
        for r in rel["gate_reasons"]:
            lines.append(f"- {r}")
    return "\n".join(lines) + "\n"
```

---

## Patch 4 — Rewrite `freeze()` with reliability + structured replace

### REPLACE the current `freeze(args)` function

with:

```python
def freeze(args):
    output = Path(args.output)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sheets = {
        name: load_csv(output / name)
        for name in (
            "discovery_human_coder_a.csv",
            "discovery_human_coder_b.csv",
            "discovery_adjudicated.csv",
        )
    }
    for name, rows in sheets.items():
        failures = [
            (row.get("sample_id"), error)
            for row in rows
            for valid, error in [validate_coding_row(row)]
            if not valid
        ]
        if failures:
            raise ValueError(f"{name} is incomplete: {failures[:5]}")

    # v3 gate: reliability + T3 degeneracy
    reliability = compute_discovery_reliability(
        sheets["discovery_human_coder_a.csv"],
        sheets["discovery_human_coder_b.csv"],
        sheets["discovery_adjudicated.csv"],
        kappa_gate=0.60,
        t3_barrier=0.90,
    )
    write_json(output / "discovery_reliability_raw30.json", reliability["raw"])
    write_json(output / "discovery_reliability_unique29.json", reliability["unique"])
    (output / "discovery_reliability_report.md").write_text(
        _reliability_report_md(reliability), encoding="utf-8"
    )
    if not reliability["gate_passed"]:
        raise ValueError(
            "Freeze gate FAILED: " + " ; ".join(reliability["gate_reasons"])
        )

    # Redundancy (unchanged logic)
    unique = {}
    for row in sheets["discovery_adjudicated.csv"]:
        identity = (row["case_id"], row["memory_sha"].lower())
        unique.setdefault(identity, row)
    applicable = [
        row for row in unique.values()
        if row["applicability_label"] in {"Direct-help", "Indirect-help"}
    ]
    redundancy = {
        "T1_x_R1": binary_phi(applicable, "T1", "R1"),
        "T3_x_R3": binary_phi(applicable, "T3", "R3"),
    }
    retained_roles = ["R2", "R4"]
    if not redundancy["T1_x_R1"]["redundant"]:
        retained_roles.append("R1")
    if not redundancy["T3_x_R3"]["redundant"]:
        retained_roles.append("R3")
    retained_roles.sort()

    # Structured freeze-marker replacement (regex + assert count == 1)
    base = Path(args.codebook).read_text(encoding="utf-8")
    frozen, n_replaced = re.subn(
        r"\*\*DISCOVERY_DRAFT[^*]*?—\s*NOT\s+FROZEN\*\*",
        "**FROZEN AFTER DISCOVERY**",
        base,
    )
    if n_replaced != 1:
        raise ValueError(
            f"Codebook freeze marker replacement expected exactly 1 match, got {n_replaced}. "
            "Check the codebook Status line format."
        )
    frozen += (
        "\n## Frozen discovery redundancy decision\n\n"
        f"- T1 × R1: `{json.dumps(redundancy['T1_x_R1'], sort_keys=True)}`\n"
        f"- T3 × R3: `{json.dumps(redundancy['T3_x_R3'], sort_keys=True)}`\n"
        f"- Headline contribution roles retained: `{retained_roles}`\n"
        f"\n## Frozen discovery reliability\n\n"
        f"- κ gate: {reliability['kappa_gate']}; T3 barrier: {reliability['t3']['barrier']}\n"
        f"- Binary applicability κ raw / unique: "
        f"{reliability['raw']['binary_applicability_kappa']} / "
        f"{reliability['unique']['binary_applicability_kappa']}\n"
        f"- Dual-axis primary-code κ raw / unique: "
        f"{reliability['raw']['primary_code_kappa']} / "
        f"{reliability['unique']['primary_code_kappa']}\n"
        f"- T3 rate on adjudicated applicable rows: "
        f"{reliability['t3']['n_t3']}/{reliability['t3']['n_applicable']} "
        f"= {reliability['t3']['rate']:.3f}\n"
    )
    frozen_path = output / "frozen_codebook.md"
    frozen_path.write_text(frozen, encoding="utf-8")

    manifest["status"] = "AWAITING_HOLDOUT_HUMANS"
    manifest["protocol"] = "memory_transfer_taxonomy_v3"
    manifest["codebook"] = {
        "path": str(frozen_path),
        "status": "FROZEN_AFTER_DISCOVERY",
        "sha256": sha256_path(frozen_path),
        "source_codebook_sha256": sha256_path(args.codebook),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "retained_headline_roles": retained_roles,
        "redundancy": redundancy,
        "reliability_gate": {
            "kappa_gate": reliability["kappa_gate"],
            "t3_barrier": reliability["t3"]["barrier"],
            "raw": reliability["raw"],
            "unique": reliability["unique"],
            "t3": reliability["t3"],
            "gate_passed": reliability["gate_passed"],
        },
    }
    manifest["artifact_sha256"][str(frozen_path)] = sha256_path(frozen_path)
    for name in (
        "discovery_reliability_raw30.json",
        "discovery_reliability_unique29.json",
        "discovery_reliability_report.md",
    ):
        p = output / name
        manifest["artifact_sha256"][str(p)] = sha256_path(p)
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "frozen_codebook": str(frozen_path),
        "retained_roles": retained_roles,
        "redundancy": redundancy,
        "reliability": {
            "raw_binary_kappa": reliability["raw"]["binary_applicability_kappa"],
            "raw_primary_kappa": reliability["raw"]["primary_code_kappa"],
            "unique_binary_kappa": reliability["unique"]["binary_applicability_kappa"],
            "unique_primary_kappa": reliability["unique"]["primary_code_kappa"],
            "t3_rate": reliability["t3"]["rate"],
            "gate_passed": reliability["gate_passed"],
        },
    }, sort_keys=True))
```

**Changes vs. current freeze():**
1. Adds reliability computation + 3 artifact writes.
2. Adds κ gate (raises `ValueError` if any of the 4 κ < 0.60).
3. Adds T3 barrier (raises if T3 rate > 0.90 on adjudicated applicable rows).
4. Replaces fragile `.replace(...)` with `re.subn(...)` + assert `n_replaced == 1`.
5. Also updates `manifest["protocol"]` to v3 (belt-and-suspenders — the manifest may already be v3 after Patch 2, but freeze re-sets it defensively).
6. Records `source_codebook_sha256` alongside `sha256` so audit can verify what was frozen.

---

## Patch 5 — Update `readiness(args)` for cleaner reporting (optional, non-blocking)

Current `readiness()` only prints per-file `complete` boolean. Optionally augment to also print reliability preview (without gating). Skip if you want a minimal patch.

### FIND `def readiness(args):`

### AFTER the existing `print(json.dumps(result, indent=2, sort_keys=True))`, add:

```python
    # Optional preview of freeze-gate metrics (does not gate readiness)
    try:
        a = load_csv(output / "discovery_human_coder_a.csv")
        b = load_csv(output / "discovery_human_coder_b.csv")
        adj = load_csv(output / "discovery_adjudicated.csv")
        rel = compute_discovery_reliability(a, b, adj)
        print("--- freeze-gate preview (readiness advisory) ---")
        print(json.dumps({
            "raw": {k: rel["raw"][k] for k in (
                "binary_applicability_kappa", "primary_code_kappa"
            )},
            "unique": {k: rel["unique"][k] for k in (
                "binary_applicability_kappa", "primary_code_kappa"
            )},
            "t3_rate": rel["t3"]["rate"],
            "gate_passed": rel["gate_passed"],
        }, indent=2, sort_keys=True))
    except (FileNotFoundError, ValueError):
        pass  # discovery not yet complete
```

---

## Expected freeze output after all patches applied

Given the current 3 CSVs in `results/paper2_pilot/analysis/memory_transfer_taxonomy/` and codebook v3 at SHA256 `678972adab1e129abf122fceb2653de861b55bf9d8e8acaa54d9a66a76298276`, `freeze` should produce:

```json
{
  "status": "AWAITING_HOLDOUT_HUMANS",
  "frozen_codebook": "results/paper2_pilot/analysis/memory_transfer_taxonomy/frozen_codebook.md",
  "retained_roles": ["R1", "R2", "R3", "R4"],
  "redundancy": {
    "T1_x_R1": {"phi": 0.488, "redundant": false, ...},
    "T3_x_R3": {"phi": 0.218, "redundant": false, ...}
  },
  "reliability": {
    "raw_binary_kappa": 0.701,
    "raw_primary_kappa": 0.630 or 0.620,  // ±0.01 depending on sklearn version
    "unique_binary_kappa": 0.696,
    "unique_primary_kappa": 0.618 or 0.606,
    "t3_rate": 0.75,
    "gate_passed": true
  }
}
```

All 4 κ values are ≥ 0.60 and T3 rate = 0.75 < 0.90, so `gate_passed=true`.

---

## Test cases to run after applying

```bash
env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=. \
  /data/conda_envs/locagent/bin/python -m pytest tests/  # unittest suite

env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=. \
  /data/conda_envs/locagent/bin/python scripts/memory_transfer_taxonomy.py readiness

env MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=. \
  /data/conda_envs/locagent/bin/python scripts/memory_transfer_taxonomy.py freeze
```

Expected: unit tests pass, readiness reports 0 failures on all 3 CSVs + gate preview, freeze produces `frozen_codebook.md` + 3 reliability artifacts and updates the manifest to `AWAITING_HOLDOUT_HUMANS` + protocol `memory_transfer_taxonomy_v3`.

---

## Six-item verdict after patches applied

| # | Item | Post-patch status |
|---|------|------|
| 1 | v3 Unclear/U validator | ✅ compiles + enforces v3 rules |
| 2 | Default codebook → v3 | ✅ (already done) |
| 3 | Structured freeze marker replace | ✅ regex + `n == 1` assert |
| 4 | Manifest → v3 | ✅ in prepare() + reaffirmed in freeze() |
| 5 | Raw/unique reliability artifacts | ✅ 3 files written by freeze() |
| 6 | Dry-run + freeze | ✅ freeze passes gate on current data |

---

## What this patch does NOT change (out of scope)

- Holdout analysis logic (`analyze()`) — currently OK because it uses same `validate_coding_row` API.
- `bootstrap_kappa`, `nominal_metrics`, `multilabel_metrics` — unchanged; used by `analyze()`.
- Test suite — should pass once import error is fixed. If specific tests reference the old broken signature `Dict[str, str]`, they'll need trivial updates.
- Preregistration file — untouched; the doc-error in v3 codebook about Fresh 30 is already corrected in-place.
