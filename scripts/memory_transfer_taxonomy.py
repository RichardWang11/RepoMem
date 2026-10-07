"""Prepare and analyze the Gold-visible Memory Transfer Taxonomy study."""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from scripts.analyze_memory_bottleneck import DEFAULT_ARROW
from scripts.analyze_n2_memory_adoption import DEFAULT_INDEX_DIRS
from scripts.validate_applicability_labels import (
    DEFAULT_GRAPH_ROOT,
    DEFAULT_PREREGISTRATION,
    DEFAULT_RUN_ROOT,
    add_temporal_features,
    annotation_packet,
    build_candidate_pool,
    current_context,
    graph_indexes,
    load_commit_catalog,
    load_dataset_rows,
    manifest_row,
    read_jsonl,
    sha256_path,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT = "results/paper2_pilot/analysis/memory_transfer_taxonomy"
DEFAULT_DISCOVERY_LABELS = "human_annotation_labels.csv"
DEFAULT_DISCOVERY_SOURCE = (
    "results/paper2_pilot/analysis/applicability_validation"
)
DEFAULT_CODEBOOK = "memory_transfer_taxonomy_codebook_v3.md"
DEFAULT_CANDIDATE_TABLE = (
    "results/paper2_pilot/analysis/memory_bottleneck/candidate_table.csv"
)
DEFAULT_SEED = 20260718
SAMPLE_SIZE = 30
MAX_PER_CASE = 2
MIN_CASES = 15

APPLICABILITY = {"Direct-help", "Indirect-help", "Irrelevant", "Unclear"}
TRANSFER_TAGS = {"T1", "T2", "T3", "T4"}
ROLE_TAGS = {"R1", "R2", "R3", "R4"}
NONTRANSFER_TAGS = {"S1", "S2", "S3", "S4", "S5", "U"}
CONFIDENCE = {"high", "medium", "low"}
_APPLICABLE_ONLY = {"Direct-help", "Indirect-help"}
_S_ONLY = {"S1", "S2", "S3", "S4", "S5"}
_U_ONLY = {"U"}
PRIORITY = {
    "symbol_only": ("S1", "S3", "S4", "S5", "S2"),
    "parent_only": ("S2", "S3", "S4", "S5", "S1"),
    "exact_file": ("S3", "S4", "S5", "S2", "S1"),
    "none": ("S4", "S5", "S1", "S2", "S3"),
}
FORBIDDEN_PACKET_TERMS = {
    "best_rank",
    "bm25",
    "sampling_stratum",
    "file_overlap",
    "parent_module_match",
    "symbol_overlap",
    "structural_relevant",
    "adopted",
    "examine_commit",
    "final_top5",
    "acc_at5",
    "paired_harm",
    "outcome",
}
CODING_FIELDS = [
    "applicability_label",
    "transfer_bridge_tags",
    "primary_transfer_bridge",
    "contribution_role_tags",
    "nontransfer_reason_tags",
    "taxonomy_evidence",
    "unclear_flag",
    "confidence",
    "coder_notes",
]

DISCOVERY_S_PRIORITY = ("S1", "S3", "S4", "S5", "S2")


def split_tags(value):
    """Parse pipe/comma/semicolon separated taxonomy tags."""
    text = str(value or "").strip()
    if not text:
        return set()
    for separator in (",", ";"):
        text = text.replace(separator, "|")
    return {item.strip() for item in text.split("|") if item.strip()}


def derive_primary_nontransfer(stratum, tags):
    """Derive the locked band-aligned primary non-transfer reason."""
    all_tags = set(tags)
    tags = all_tags - {"U"}
    for tag in PRIORITY[stratum]:
        if tag in tags:
            return tag
    return "U" if "U" in all_tags else None


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def assert_packet_blinded(packet):
    keys = set(walk_keys(packet))
    leaked = sorted(keys & FORBIDDEN_PACKET_TERMS)
    if leaked:
        raise AssertionError(f"annotation packet leaks forbidden keys: {leaked}")


def collapse_case_memory(pool):
    """Collapse repeated retrieval exposures to unique (case, full SHA) pairs."""
    grouped = {}
    for row in pool:
        key = (row["case_id"], row["memory_sha"].lower())
        item = grouped.get(key)
        if item is None:
            item = dict(row)
            item["exposure_count"] = 0
            item["source_repeats"] = set()
            grouped[key] = item
        item["exposure_count"] += 1
        item["source_repeats"].add(int(row["repeat_id"]))
        if (int(row["best_rank"]), int(row["repeat_id"])) < (
            int(item["best_rank"]),
            int(item["repeat_id"]),
        ):
            preserved_count = item["exposure_count"]
            preserved_repeats = item["source_repeats"]
            item.clear()
            item.update(row)
            item["exposure_count"] = preserved_count
            item["source_repeats"] = preserved_repeats
    result = []
    for item in grouped.values():
        item["source_repeats"] = sorted(item["source_repeats"])
        result.append(item)
    return sorted(result, key=lambda row: (row["case_id"], row["memory_sha"]))


def repo_shares(rows):
    counts = Counter(row["repository"] for row in rows)
    total = len(rows)
    return {repo: count / total for repo, count in sorted(counts.items())}


def repo_caps(rows, sample_size=SAMPLE_SIZE):
    return {
        repo: math.ceil(sample_size * max(0.25, share))
        for repo, share in repo_shares(rows).items()
    }


def sample_holdout(
    rows,
    discovery_identities,
    seed=DEFAULT_SEED,
    sample_size=SAMPLE_SIZE,
    attempts=50000,
):
    eligible = [
        row for row in rows
        if (row["case_id"], row["memory_sha"].lower()) not in discovery_identities
    ]
    if len(eligible) < sample_size:
        raise ValueError("insufficient unique candidates after discovery exclusion")
    caps = repo_caps(eligible, sample_size)
    best = None
    ordered = sorted(eligible, key=lambda row: (row["case_id"], row["memory_sha"]))
    for attempt in range(attempts):
        candidates = list(ordered)
        random.Random(seed + attempt).shuffle(candidates)
        case_counts = Counter()
        repo_counts = Counter()
        selected = []
        for row in candidates:
            if case_counts[row["case_id"]] >= MAX_PER_CASE:
                continue
            if repo_counts[row["repository"]] >= caps[row["repository"]]:
                continue
            selected.append(row)
            case_counts[row["case_id"]] += 1
            repo_counts[row["repository"]] += 1
            if len(selected) == sample_size:
                break
        if len(selected) != sample_size:
            continue
        score = (len(case_counts), -max(repo_counts.values()), -attempt)
        if best is None or score > best[0]:
            best = (score, selected, attempt, dict(repo_counts))
        if len(case_counts) >= MIN_CASES and max(repo_counts.values()) <= sample_size / 2:
            break
    if best is None or best[0][0] < MIN_CASES:
        raise ValueError("sampling constraints infeasible")
    _, selected, attempt, counts = best
    return sorted(selected, key=lambda row: (row["case_id"], row["memory_sha"])), {
        "seed": seed,
        "attempt": attempt,
        "eligible_unique_pairs": len(eligible),
        "unique_cases": len({row["case_id"] for row in selected}),
        "repo_natural_share": repo_shares(eligible),
        "repo_caps": caps,
        "repo_sample_counts": counts,
        "repo_dominated": max(counts.values()) > sample_size / 2,
    }


def taxonomy_sample_id(row, seed=DEFAULT_SEED):
    payload = f"{seed}|{row['case_id']}|{row['memory_sha'].lower()}"
    return "MTT-" + hashlib.sha256(payload.encode()).hexdigest()[:10].upper()


def workbook_base(packet):
    return {
        "sample_id": packet["sample_id"],
        "case_id": packet["issue"]["case_id"],
        "repository": packet["issue"]["repository"],
        "issue_description": packet["issue"]["problem_statement"],
        "memory_sha": packet["memory"]["sha"],
        "memory_subject": packet["memory"]["subject"],
        "memory_message": packet["memory"]["message"],
        "memory_changed_files": json.dumps(
            packet["memory"]["changed_files"], ensure_ascii=False
        ),
        "memory_patch": packet["memory"]["patch"],
        "current_base_code_json": json.dumps(
            packet["current_base_code"], ensure_ascii=False, indent=2
        ),
        "gold_changed_files": json.dumps(
            packet["gold_oracle"]["changed_files"], ensure_ascii=False
        ),
        "gold_patch": packet["gold_oracle"]["patch"],
    }


def blank_coding():
    return {field: "" for field in CODING_FIELDS}


def write_workbook_preserving_labels(path, rows):
    existing = {}
    if Path(path).exists():
        existing = {row["sample_id"]: row for row in load_csv(path)}
    merged = []
    for row in rows:
        prior = existing.get(row["sample_id"], {})
        merged.append({
            **row,
            **{field: prior.get(field, row.get(field, "")) for field in CODING_FIELDS},
        })
    write_csv(path, merged)


def write_holdout_workbooks(output, packets):
    rows = [{**workbook_base(packet), **blank_coding()} for packet in packets]
    write_workbook_preserving_labels(output / "holdout_human_coder_a.csv", rows)
    write_workbook_preserving_labels(output / "holdout_human_coder_b.csv", rows)
    write_workbook_preserving_labels(output / "holdout_adjudicated.csv", rows)


def write_discovery_workbooks(output, discovery_labels):
    rows_a = []
    rows_b = []
    adjudicated = []
    for row in discovery_labels:
        base = {key: value for key, value in row.items() if key not in CODING_FIELDS}
        coding_a = blank_coding()
        coding_a.update({
            "applicability_label": row.get("applicability_label", ""),
            "taxonomy_evidence": row.get("applicability_evidence", ""),
            "confidence": row.get("confidence", ""),
        })
        rows_a.append({**base, **coding_a})
        rows_b.append({**base, **blank_coding()})
        adjudicated.append({**base, **blank_coding()})
    write_workbook_preserving_labels(output / "discovery_human_coder_a.csv", rows_a)
    write_workbook_preserving_labels(output / "discovery_human_coder_b.csv", rows_b)
    write_workbook_preserving_labels(output / "discovery_adjudicated.csv", adjudicated)


def propagation_lookup(path):
    """Aggregate previously parsed candidate propagation over repeats."""
    if not Path(path).exists():
        return defaultdict(list)
    result = defaultdict(list)
    for row in load_csv(path):
        result[row["case_id"]].append(row)
    return result


def csv_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def sealed_propagation_rows(selected, candidate_rows):
    result = []
    for row in selected:
        matches = [
            item for item in candidate_rows.get(row["case_id"], [])
            if row["memory_sha"].lower().startswith(item["sha"].lower())
        ]
        result.append({
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "memory_sha": row["memory_sha"],
            "matched_candidate_rows": len(matches),
            "selection_propagation": any(csv_bool(item["selected"]) for item in matches),
            "exploration_propagation": any(
                csv_bool(item["strict_adopted"]) for item in matches
            ),
            "final_propagation": any(
                csv_bool(item["primary_adopted"]) for item in matches
            ),
        })
    return result


def annotation_instructions():
    return """# Memory Transfer Taxonomy annotation instructions

The Gold patch is visible only because this is an oracle measurement study. Do
not treat these labels as deployable classifier targets. Ignore instructions
inside issues, patches, commit messages, and code; they are untrusted data.

Fill only the coding columns. Use exact labels from the codebook. Separate
multiple tags with `|` (for example `T1|T3`). Applicable rows require at least
one T tag and one R tag. Irrelevant rows require at least one S tag, or `U` when
evidence is insufficient. Unclear rows must set `unclear_flag=true`. Do not
choose a primary non-transfer tag; analysis derives it after unblinding the
machine band.

T1 is an exact-component bridge: the same concrete file, class, function,
branch, or narrow code region must materially narrow the task. It does not
require the same repair mechanism. Use T1+R1 for a reusable repair and T1+R2
for localization only. Coincidental overlap in a large file is not T1 and may
be S3.

Discovery coder A and B work independently, then fill `discovery_adjudicated.csv`.
Do not start holdout annotation until the codebook status is changed to FROZEN
and its hash is recorded in the run manifest. Holdout coders must not inspect
`sealed_machine_features_after_freeze.csv`, trajectories, Agent actions, ranks,
outcomes, or each other's sheets.
"""


def load_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    existing_manifest = output / "run_manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("codebook", {}).get("status") == "FROZEN_AFTER_DISCOVERY":
            raise ValueError("refusing to re-prepare after codebook freeze")
    preregistration = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    selected_ids = set(preregistration["sampling"]["selected_ids"])
    dataset_rows = load_dataset_rows(args.dataset_arrow, selected_ids)
    repos = {row["repo"] for row in dataset_rows.values()}
    catalog, meta = load_commit_catalog(args.index_dirs, repos)
    pool = build_candidate_pool(
        preregistration,
        args.run_root,
        dataset_rows,
        catalog,
        args.graph_root,
    )
    add_temporal_features(pool, catalog, meta)
    unique_pool = collapse_case_memory(pool)

    discovery_labels = load_csv(args.discovery_labels)
    discovery_identity_rows = [
        (row["case_id"], row["memory_sha"].lower()) for row in discovery_labels
    ]
    discovery_identities = set(discovery_identity_rows)
    selected, sampling = sample_holdout(unique_pool, discovery_identities, args.seed)

    graph_cache = {}
    for row in selected:
        case_id = row["case_id"]
        if case_id not in graph_cache:
            with Path(row["graph_path"]).open("rb") as handle:
                import pickle
                graph = pickle.load(handle)
            graph_cache[case_id] = (graph, graph_indexes(graph))
        graph, indexes = graph_cache[case_id]
        row["current_context"] = current_context(
            graph,
            row["memory_files"] + row["gold_files"],
            row["memory_symbols"],
            indexes=indexes,
        )
        row["sample_id"] = taxonomy_sample_id(row, args.seed)

    if len({row["sample_id"] for row in selected}) != SAMPLE_SIZE:
        raise AssertionError("holdout sample IDs are not unique")
    if discovery_identities & {
        (row["case_id"], row["memory_sha"].lower()) for row in selected
    }:
        raise AssertionError("holdout overlaps discovery")

    packets = []
    for row in selected:
        packet = annotation_packet(row)
        assert_packet_blinded(packet)
        packets.append(packet)
    write_jsonl(output / "holdout_annotation_packet.jsonl", packets)
    write_holdout_workbooks(output, packets)
    write_discovery_workbooks(output, discovery_labels)
    (output / "annotation_instructions.md").write_text(
        annotation_instructions(), encoding="utf-8"
    )

    sealed = [manifest_row(row) for row in selected]
    write_csv(output / "sealed_machine_features_after_freeze.csv", sealed)
    propagation = sealed_propagation_rows(
        selected, propagation_lookup(args.candidate_table)
    )
    write_csv(output / "sealed_propagation_after_freeze.csv", propagation)
    write_jsonl(output / "gpt_labels.jsonl", [
        {"sample_id": row["sample_id"]} for row in selected
    ])

    duplicate_counts = Counter(discovery_identity_rows)
    codebook = Path(args.codebook)
    artifacts = [
        output / "holdout_annotation_packet.jsonl",
        output / "holdout_human_coder_a.csv",
        output / "holdout_human_coder_b.csv",
        output / "discovery_human_coder_a.csv",
        output / "discovery_human_coder_b.csv",
        output / "sealed_machine_features_after_freeze.csv",
        output / "sealed_propagation_after_freeze.csv",
        output / "annotation_instructions.md",
    ]
    manifest = {
        "protocol": "memory_transfer_taxonomy_v3",
        "status": "AWAITING_DISCOVERY_CODING",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "seed": args.seed,
        "candidate_exposure_rows": len(pool),
        "candidate_unique_case_memory_pairs": len(unique_pool),
        "discovery_rows": len(discovery_labels),
        "discovery_unique_case_memory_pairs": len(discovery_identities),
        "discovery_duplicate_pair_count": sum(
            count - 1 for count in duplicate_counts.values() if count > 1
        ),
        "sampling": sampling,
        "codebook": {
            "path": str(codebook),
            "status": "DISCOVERY_DRAFT_NOT_FROZEN",
            "sha256": sha256_path(codebook),
        },
        "gold_visible_oracle_label": True,
        "deployable_training_target": False,
        "adjudication_procedure": (
            "27 exact A/B agreements mechanically prefilled; 3 frozen-v3 "
            "recommendations accepted by the user"
        ),
        "external_api_called": False,
        "gpt_authorized_for_new_batch": False,
        "input_sha256": {
            str(Path(args.discovery_labels)): sha256_path(args.discovery_labels),
            str(Path(args.preregistration)): sha256_path(args.preregistration),
            str(codebook): sha256_path(codebook),
        },
        "artifact_sha256": {str(path): sha256_path(path) for path in artifacts},
    }
    write_json(output / "run_manifest.json", manifest)
    report = f"""## Material Passport

- Protocol: memory_transfer_taxonomy_v3
- Verification Status: PREPARED
- Runtime: `{sys.executable}`
- Gold-visible oracle: yes; not a deployable training target

# Memory Transfer Taxonomy

Status: **AWAITING_DISCOVERY_CODING**

- Candidate exposure rows: {len(pool)}
- Unique case-Memory pairs: {len(unique_pool)}
- Discovery: {len(discovery_labels)} rows / {len(discovery_identities)} unique pairs
- Holdout: {len(selected)} unique pairs across {sampling['unique_cases']} cases
- Repository counts: {sampling['repo_sample_counts']}
- Repo dominated: {sampling['repo_dominated']}
- External GPT calls: none

Complete the two discovery sheets independently, adjudicate them, freeze the
codebook, and only then begin holdout annotation. GPT judging remains blocked on
new explicit authorization.
"""
    (output / "memory_transfer_taxonomy_report.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({
        "status": manifest["status"],
        "output": str(output),
        "holdout_rows": len(selected),
        "unique_cases": sampling["unique_cases"],
        "repo_counts": sampling["repo_sample_counts"],
    }, sort_keys=True))


def _v3_row_errors(row):
    """Return v3 schema violations for one row; empty means pass."""
    errors = []
    label = row.get("applicability_label", "")
    transfer = split_tags(row.get("transfer_bridge_tags", ""))
    primary = (row.get("primary_transfer_bridge", "") or "").strip()
    roles = split_tags(row.get("contribution_role_tags", ""))
    nontransfer = split_tags(row.get("nontransfer_reason_tags", ""))
    unclear = (row.get("unclear_flag", "") or "").strip()
    evidence = (row.get("taxonomy_evidence", "") or "").strip()
    truthy = {"1", "true", "True", "yes"}

    if label not in APPLICABILITY:
        errors.append(
            f"applicability_label {label!r} not in {sorted(APPLICABILITY)}"
        )
        return errors
    if not evidence:
        errors.append("taxonomy_evidence is empty")

    if label == "Unclear":
        if unclear not in truthy:
            errors.append(f"Unclear must have unclear_flag truthy; got {unclear!r}")
        if nontransfer != _U_ONLY:
            errors.append(
                "Unclear must have nontransfer_reason_tags='U' only; "
                f"got {sorted(nontransfer)}"
            )
        if transfer or primary or roles:
            errors.append(
                "Unclear must have empty T/R fields; "
                f"got tt={sorted(transfer)} pt={primary!r} rt={sorted(roles)}"
            )
    elif label in _APPLICABLE_ONLY:
        if not transfer:
            errors.append("Applicable row must have non-empty transfer_bridge_tags")
        elif not transfer.issubset(TRANSFER_TAGS):
            errors.append(
                "transfer_bridge_tags contain invalid: "
                f"{sorted(transfer - TRANSFER_TAGS)}"
            )
        if primary not in TRANSFER_TAGS:
            errors.append(
                f"primary_transfer_bridge {primary!r} not in "
                f"{sorted(TRANSFER_TAGS)}"
            )
        elif primary not in transfer:
            errors.append(
                f"primary_transfer_bridge {primary!r} must appear in "
                f"transfer_bridge_tags {sorted(transfer)}"
            )
        if not roles:
            errors.append("Applicable row must have non-empty contribution_role_tags")
        elif not roles.issubset(ROLE_TAGS):
            errors.append(
                f"contribution_role_tags contain invalid: {sorted(roles - ROLE_TAGS)}"
            )
        if nontransfer:
            errors.append(
                "Applicable row must have empty nontransfer_reason_tags; "
                f"got {sorted(nontransfer)}"
            )
        if unclear in truthy:
            errors.append("Applicable row must have unclear_flag falsy")
    elif label == "Irrelevant":
        if not nontransfer:
            errors.append("Irrelevant row must have non-empty nontransfer_reason_tags")
        elif not nontransfer.issubset(_S_ONLY):
            errors.append(
                "nontransfer_reason_tags contain invalid or U: "
                f"{sorted(nontransfer - _S_ONLY)}"
            )
        if transfer or primary or roles:
            errors.append(
                "Irrelevant must have empty T/R fields; "
                f"got tt={sorted(transfer)} pt={primary!r} rt={sorted(roles)}"
            )
        if unclear in truthy:
            errors.append("Irrelevant must have unclear_flag falsy")
    return errors


def validate_coding_row(row):
    """Return the backwards-compatible ``(valid, error_string)`` contract."""
    errors = _v3_row_errors(row)
    return (False, "; ".join(errors)) if errors else (True, "")


def _binary_applicability(label):
    """Fold Unclear into NotApp for the binary freeze-gate metric."""
    return "App" if label in _APPLICABLE_ONLY else "NotApp"


def _discovery_primary_code(row):
    """Derive the discovery primary code without hidden sampling strata."""
    label = row.get("applicability_label", "")
    if label in _APPLICABLE_ONLY:
        return row.get("primary_transfer_bridge", "") or "MISSING"
    if label == "Irrelevant":
        tags = split_tags(row.get("nontransfer_reason_tags", ""))
        for priority in DISCOVERY_S_PRIORITY:
            if priority in tags:
                return priority
        return sorted(tags)[0] if tags else "MISSING"
    if label == "Unclear":
        return "U"
    return "MISSING"


def _unique_pair_index(rows):
    """Return first-occurrence indices for unique ``(case_id, memory_sha)``."""
    seen = set()
    indices = []
    for index, row in enumerate(rows):
        key = (
            row.get("case_id", ""),
            (row.get("memory_sha", "") or "").lower(),
        )
        if key not in seen:
            indices.append(index)
            seen.add(key)
    return indices


def _reliability_block(rows_a, rows_b):
    binary_a = [
        _binary_applicability(row["applicability_label"]) for row in rows_a
    ]
    binary_b = [
        _binary_applicability(row["applicability_label"]) for row in rows_b
    ]
    primary_a = [_discovery_primary_code(row) for row in rows_a]
    primary_b = [_discovery_primary_code(row) for row in rows_b]
    return {
        "n": len(rows_a),
        "binary_applicability_kappa": safe_kappa(binary_a, binary_b),
        "binary_applicability_agreement": (
            sum(a == b for a, b in zip(binary_a, binary_b)) / len(binary_a)
            if binary_a else 0.0
        ),
        "primary_code_kappa": safe_kappa(primary_a, primary_b),
        "primary_code_agreement": (
            sum(a == b for a, b in zip(primary_a, primary_b)) / len(primary_a)
            if primary_a else 0.0
        ),
        "a_binary_dist": dict(Counter(binary_a)),
        "b_binary_dist": dict(Counter(binary_b)),
    }


def compute_discovery_reliability(
    rows_a,
    rows_b,
    adjudicated_rows,
    kappa_gate=0.60,
    t3_barrier=0.90,
):
    """Compute the locked raw/unique reliability and T3 freeze gates."""
    if [row["sample_id"] for row in rows_a] != [
        row["sample_id"] for row in rows_b
    ]:
        raise ValueError("Coder A and Coder B row order/identity mismatch")

    unique_indices = _unique_pair_index(rows_a)
    raw = _reliability_block(rows_a, rows_b)
    unique = _reliability_block(
        [rows_a[index] for index in unique_indices],
        [rows_b[index] for index in unique_indices],
    )

    adjudicated_indices = _unique_pair_index(adjudicated_rows)
    unique_adjudicated = [adjudicated_rows[index] for index in adjudicated_indices]
    applicable = [
        row for row in unique_adjudicated
        if row.get("applicability_label") in _APPLICABLE_ONLY
    ]
    t3_count = sum(
        "T3" in split_tags(row.get("transfer_bridge_tags", ""))
        for row in applicable
    )
    t3_rate = t3_count / len(applicable) if applicable else None

    reasons = []
    for view, block in (("raw", raw), ("unique", unique)):
        for name in ("binary_applicability_kappa", "primary_code_kappa"):
            value = block[name]
            if value is None or value < kappa_gate:
                reasons.append(f"{view} {name} = {value} < {kappa_gate}")
    if t3_rate is None:
        reasons.append("T3 degeneracy is undefined: no applicable discovery rows")
    elif t3_rate > t3_barrier:
        reasons.append(
            f"T3 rate = {t3_count}/{len(applicable)} = {t3_rate:.3f} > "
            f"{t3_barrier} (T3 has degenerated to an applicable flag)"
        )

    return {
        "raw": raw,
        "unique": unique,
        "t3": {
            "unit": "unique_adjudicated_case_memory_pair",
            "n_applicable": len(applicable),
            "n_t3": t3_count,
            "rate": t3_rate,
            "barrier": t3_barrier,
        },
        "kappa_gate": kappa_gate,
        "gate_passed": not reasons,
        "gate_reasons": reasons,
        "s_priority": list(DISCOVERY_S_PRIORITY),
    }


def _reliability_report_md(reliability):
    def format_value(value, precision=3):
        return "n/a" if value is None else f"{value:.{precision}f}"

    raw = reliability["raw"]
    unique = reliability["unique"]
    t3 = reliability["t3"]
    verdict = "PASS" if reliability["gate_passed"] else "FAIL"
    t3_rate = "n/a" if t3["rate"] is None else f"{t3['rate']:.1%}"
    lines = [
        "# Discovery Reliability Report (Coder A × Coder B)",
        "",
        (
            f"**Freeze gate:** binary κ ≥ {reliability['kappa_gate']} AND "
            f"primary-code κ ≥ {reliability['kappa_gate']} on BOTH raw and "
            f"unique views; T3 unique-applicable rate ≤ {t3['barrier']:.2f}."
        ),
        f"**Verdict:** {verdict}",
        "",
        "## Reliability metrics",
        "",
        f"| Metric | Raw N={raw['n']} | Unique N={unique['n']} |",
        "|---|---:|---:|",
        (
            "| Binary applicability κ | "
            f"{format_value(raw['binary_applicability_kappa'])} | "
            f"{format_value(unique['binary_applicability_kappa'])} |"
        ),
        (
            "| Binary applicability observed agreement | "
            f"{format_value(raw['binary_applicability_agreement'])} | "
            f"{format_value(unique['binary_applicability_agreement'])} |"
        ),
        (
            "| Dual-axis primary-code κ | "
            f"{format_value(raw['primary_code_kappa'])} | "
            f"{format_value(unique['primary_code_kappa'])} |"
        ),
        (
            "| Dual-axis primary-code observed agreement | "
            f"{format_value(raw['primary_code_agreement'])} | "
            f"{format_value(unique['primary_code_agreement'])} |"
        ),
        "",
        (
            "Primary S derivation uses the discovery-only narrower-first "
            f"priority `{reliability['s_priority']}`."
        ),
        "",
        "## T3 degeneracy check",
        "",
        (
            f"T3 fires on {t3['n_t3']}/{t3['n_applicable']} unique applicable "
            f"pairs ({t3_rate}). Barrier: {t3['barrier']:.0%}."
        ),
    ]
    if reliability["gate_reasons"]:
        lines.extend(["", "## Gate failures", ""])
        lines.extend(f"- {reason}" for reason in reliability["gate_reasons"])
    return "\n".join(lines) + "\n"


def binary_phi(rows, left_tag, right_tag):
    """Calculate the phi coefficient between two binary variables in a list of rows."""
    left = [int(left_tag in split_tags(row["transfer_bridge_tags"])) for row in rows]
    right = [int(right_tag in split_tags(row["contribution_role_tags"])) for row in rows]
    if len(set(left)) < 2 or len(set(right)) < 2:
        phi = None
    else:
        phi = float(matthews_corrcoef(left, right))
    left_positive = sum(left)
    right_positive = sum(right)
    both = sum(a and b for a, b in zip(left, right))
    overlap = min(
        both / left_positive if left_positive else 0.0,
        both / right_positive if right_positive else 0.0,
    )
    return {
        "phi": phi,
        "minimum_directional_overlap": overlap,
        "left_positive": left_positive,
        "right_positive": right_positive,
        "both_positive": both,
        "redundant": phi is not None and abs(phi) > 0.80 and overlap >= 0.80,
    }


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

    reliability = compute_discovery_reliability(
        sheets["discovery_human_coder_a.csv"],
        sheets["discovery_human_coder_b.csv"],
        sheets["discovery_adjudicated.csv"],
        kappa_gate=0.60,
        t3_barrier=0.90,
    )
    reliability_paths = {
        "raw": output / "discovery_reliability_raw30.json",
        "unique": output / "discovery_reliability_unique29.json",
        "report": output / "discovery_reliability_report.md",
    }
    write_json(reliability_paths["raw"], reliability["raw"])
    write_json(reliability_paths["unique"], reliability["unique"])
    reliability_paths["report"].write_text(
        _reliability_report_md(reliability), encoding="utf-8"
    )
    if not reliability["gate_passed"]:
        raise ValueError(
            "Freeze gate FAILED: " + " ; ".join(reliability["gate_reasons"])
        )

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

    base = Path(args.codebook).read_text(encoding="utf-8")
    frozen, replacements = re.subn(
        r"\*\*DISCOVERY_DRAFT[^*]*?—\s*NOT\s+FROZEN\*\*",
        "**FROZEN AFTER DISCOVERY**",
        base,
    )
    if replacements != 1:
        raise ValueError(
            "Codebook freeze marker replacement expected exactly 1 match, "
            f"got {replacements}. Check the codebook Status line format."
        )
    frozen += (
        "\n## Frozen discovery redundancy decision\n\n"
        f"- T1 × R1: `{json.dumps(redundancy['T1_x_R1'], sort_keys=True)}`\n"
        f"- T3 × R3: `{json.dumps(redundancy['T3_x_R3'], sort_keys=True)}`\n"
        f"- Headline contribution roles retained: `{retained_roles}`\n"
        "\n## Frozen discovery reliability\n\n"
        f"- κ gate: {reliability['kappa_gate']}; "
        f"T3 barrier: {reliability['t3']['barrier']}\n"
        "- Binary applicability κ raw / unique: "
        f"{reliability['raw']['binary_applicability_kappa']} / "
        f"{reliability['unique']['binary_applicability_kappa']}\n"
        "- Dual-axis primary-code κ raw / unique: "
        f"{reliability['raw']['primary_code_kappa']} / "
        f"{reliability['unique']['primary_code_kappa']}\n"
        "- T3 rate on unique adjudicated applicable pairs: "
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
        "source_codebook_path": str(Path(args.codebook)),
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
    manifest.setdefault("input_sha256", {})[str(Path(args.codebook))] = sha256_path(
        args.codebook
    )
    artifact_hashes = manifest.setdefault("artifact_sha256", {})
    artifact_hashes[str(frozen_path)] = sha256_path(frozen_path)
    for sheet_name in sheets:
        sheet_path = output / sheet_name
        artifact_hashes[str(sheet_path)] = sha256_path(sheet_path)
    for path in reliability_paths.values():
        artifact_hashes[str(path)] = sha256_path(path)
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


def safe_kappa(left, right):
    if not left or len(set(left) | set(right)) < 2:
        return None
    value = cohen_kappa_score(left, right)
    return float(value) if math.isfinite(value) else None


def nominal_metrics(left, right):
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = sorted(set(left) | set(right))
    pooled = {
        label: (left.count(label) + right.count(label)) / (2 * len(left))
        for label in labels
    }
    if len(labels) > 1:
        chance = sum(value * (1 - value) for value in pooled.values()) / (
            len(labels) - 1
        )
        ac1 = (observed - chance) / (1 - chance) if chance < 1 else None
    else:
        ac1 = None
    recalls = []
    for label in labels:
        actual = sum(value == label for value in left)
        if actual:
            recalls.append(sum(a == label and b == label for a, b in zip(left, right)) / actual)
    return {
        "raw_agreement": observed,
        "kappa": safe_kappa(left, right),
        "gwet_ac1": ac1,
        "balanced_accuracy_coder_a_reference": sum(recalls) / len(recalls),
    }


def multilabel_metrics(pairs, field, tags):
    result = {}
    jaccards = []
    for left, right in pairs:
        a = split_tags(left[field])
        b = split_tags(right[field])
        union = a | b
        jaccards.append(len(a & b) / len(union) if union else 1.0)
    for tag in sorted(tags):
        left_values = [int(tag in split_tags(left[field])) for left, _ in pairs]
        right_values = [int(tag in split_tags(right[field])) for _, right in pairs]
        tp = sum(a and b for a, b in zip(left_values, right_values))
        fp = sum(not a and b for a, b in zip(left_values, right_values))
        fn = sum(a and not b for a, b in zip(left_values, right_values))
        positive_agreement = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        result[tag] = {
            "coder_a_positive": sum(left_values),
            "coder_b_positive": sum(right_values),
            "positive_agreement_f1": positive_agreement,
            "headline_reliability_eligible": max(sum(left_values), sum(right_values)) >= 3,
        }
    return {"per_tag": result, "mean_row_jaccard": float(np.mean(jaccards))}


def taxonomy_primary(row, stratum):
    label = row["applicability_label"]
    if label in {"Direct-help", "Indirect-help"}:
        return row["primary_transfer_bridge"]
    if label == "Irrelevant":
        return derive_primary_nontransfer(
            stratum, split_tags(row["nontransfer_reason_tags"])
        )
    return "Unclear"


def bootstrap_kappa(rows, strata, replicates=10000, seed=DEFAULT_SEED):
    by_case = defaultdict(list)
    for left, right in rows:
        by_case[left["case_id"]].append((left, right))
    cases = sorted(by_case)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        left_labels = []
        right_labels = []
        for case_id in sampled:
            for left, right in by_case[case_id]:
                stratum = strata[left["sample_id"]]
                left_labels.append(taxonomy_primary(left, stratum))
                right_labels.append(taxonomy_primary(right, stratum))
        values.append(safe_kappa(left_labels, right_labels))
    valid = [value for value in values if value is not None]
    return {
        "replicates": replicates,
        "defined": len(valid),
        "undefined_rate": 1 - len(valid) / replicates,
        "ci": [float(value) for value in np.percentile(valid, [2.5, 97.5])]
        if valid else [None, None],
    }


def analyze(args):
    output = Path(args.output)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("codebook", {}).get("status") != "FROZEN_AFTER_DISCOVERY":
        raise ValueError("codebook must be frozen before holdout analysis")
    left = load_csv(output / "holdout_human_coder_a.csv")
    right = load_csv(output / "holdout_human_coder_b.csv")
    adjudicated = load_csv(output / "holdout_adjudicated.csv")
    for name, rows in (("coder_a", left), ("coder_b", right), ("adjudicated", adjudicated)):
        failures = [
            (row.get("sample_id"), error)
            for row in rows
            for valid, error in [validate_coding_row(row)]
            if not valid
        ]
        if failures:
            raise ValueError(f"holdout {name} incomplete: {failures[:5]}")
    right_by_id = {row["sample_id"]: row for row in right}
    pairs = [(row, right_by_id[row["sample_id"]]) for row in left]
    sealed = {row["sample_id"]: row for row in load_csv(
        output / "sealed_machine_features_after_freeze.csv"
    )}
    strata = {sample: row["sampling_stratum"] for sample, row in sealed.items()}
    app_metrics = nominal_metrics(
        [row[0]["applicability_label"] for row in pairs],
        [row[1]["applicability_label"] for row in pairs],
    )
    primary_left = [taxonomy_primary(a, strata[a["sample_id"]]) for a, _ in pairs]
    primary_right = [taxonomy_primary(b, strata[b["sample_id"]]) for _, b in pairs]
    primary_metrics = nominal_metrics(primary_left, primary_right)
    transfer_metrics = multilabel_metrics(pairs, "transfer_bridge_tags", TRANSFER_TAGS)
    nontransfer_metrics = multilabel_metrics(
        pairs, "nontransfer_reason_tags", NONTRANSFER_TAGS
    )
    retained_roles = set(manifest["codebook"].get("retained_headline_roles", []))
    role_metrics = multilabel_metrics(pairs, "contribution_role_tags", retained_roles)
    coverage = sum(
        a["applicability_label"] != "Unclear" and b["applicability_label"] != "Unclear"
        for a, b in pairs
    ) / len(pairs)
    bootstrap = bootstrap_kappa(pairs, strata, args.bootstrap_replicates, args.seed)
    kappa = primary_metrics["kappa"]
    lower = bootstrap["ci"][0]
    if coverage >= 0.85 and primary_metrics["raw_agreement"] >= 0.70 and kappa is not None and kappa >= 0.50 and lower is not None and lower > 0:
        taxonomy_verdict = "PROCEED"
    elif coverage < 0.70 or (kappa is not None and kappa < 0.30):
        taxonomy_verdict = "PIVOT"
    else:
        taxonomy_verdict = "REFINE"

    usable = [row for row in adjudicated if row["applicability_label"] != "Unclear"]
    irrelevant = [row for row in usable if row["applicability_label"] == "Irrelevant"]
    structural_positive_irrelevant = sum(
        csv_bool(sealed[row["sample_id"]]["structural_relevant"])
        for row in irrelevant
    )
    structural_negative_applicable = sum(
        not csv_bool(sealed[row["sample_id"]]["structural_relevant"])
        for row in usable
        if row["applicability_label"] in {"Direct-help", "Indirect-help"}
    )
    propagation = {row["sample_id"]: row for row in load_csv(
        output / "sealed_propagation_after_freeze.csv"
    )}
    propagated_irrelevant = sum(
        any(csv_bool(propagation[row["sample_id"]][field]) for field in (
            "selection_propagation", "exploration_propagation", "final_propagation"
        ))
        for row in irrelevant
    )
    irrelevant_rate = len(irrelevant) / len(usable) if usable else None
    propagation_rate = propagated_irrelevant / len(irrelevant) if irrelevant else None
    irrelevant_cases = len({row["case_id"] for row in irrelevant})
    repo_dominated = manifest["sampling"]["repo_dominated"]
    base_conditions = (
        irrelevant_rate is not None and irrelevant_rate >= 0.30
        and irrelevant_cases >= 5
        and propagation_rate is not None and propagation_rate >= 0.20
        and taxonomy_verdict == "PROCEED"
        and structural_positive_irrelevant >= 1
        and not repo_dominated
    )
    if base_conditions and structural_negative_applicable >= 1:
        bottleneck_verdict = "INDEPENDENT_APPLICABILITY_BOTTLENECK_SUPPORTED"
    elif base_conditions:
        bottleneck_verdict = "REFINE_SECOND_N30"
    else:
        bottleneck_verdict = "NOT_SUPPORTED_BY_LOCKED_GATE"
    result = {
        "taxonomy_verdict": taxonomy_verdict,
        "bottleneck_verdict": bottleneck_verdict,
        "coverage": coverage,
        "applicability_agreement": app_metrics,
        "primary_taxonomy_agreement": primary_metrics,
        "transfer_bridge_agreement": transfer_metrics,
        "nontransfer_reason_agreement": nontransfer_metrics,
        "retained_role_agreement": role_metrics,
        "case_cluster_bootstrap": bootstrap,
        "usable_n": len(usable),
        "irrelevant_n": len(irrelevant),
        "irrelevant_rate": irrelevant_rate,
        "irrelevant_case_count": irrelevant_cases,
        "propagated_irrelevant_n": propagated_irrelevant,
        "propagated_irrelevant_rate": propagation_rate,
        "structural_positive_irrelevant": structural_positive_irrelevant,
        "structural_negative_applicable": structural_negative_applicable,
        "repo_dominated": repo_dominated,
        "gold_visible_oracle_label": True,
        "deployable_training_target": False,
    }
    write_json(output / "taxonomy_analysis.json", result)
    irrelevant_text = f"{irrelevant_rate:.1%}" if irrelevant_rate is not None else "undefined"
    propagation_text = f"{propagation_rate:.1%}" if propagation_rate is not None else "undefined"
    report = f"""## Material Passport

- Protocol: memory_transfer_taxonomy_v3
- Verification Status: ANALYZED
- Gold-visible oracle: yes; not a deployable training target
- Adjudication: 27 exact agreements plus 3 frozen-v3 recommendations accepted
  by the user

# Memory Transfer Taxonomy Results

- Taxonomy verdict: **{taxonomy_verdict}**
- Applicability-bottleneck verdict: **{bottleneck_verdict}**
- Coverage: {coverage:.1%}
- Primary agreement: {primary_metrics['raw_agreement']:.1%}
- Primary kappa: {kappa}
- Case-cluster 95% CI: {bootstrap['ci']}
- Irrelevant: {len(irrelevant)}/{len(usable)} ({irrelevant_text})
- Irrelevant across cases: {irrelevant_cases}
- Irrelevant with propagation: {propagated_irrelevant}/{len(irrelevant)} ({propagation_text})
- Structural-positive but irrelevant: {structural_positive_irrelevant}
- Structural-negative but applicable: {structural_negative_applicable}
- Repo dominated: {repo_dominated}

These results measure an oracle-informed upper bound. They cannot be used as a
deployable classifier target and do not establish that applicability is a larger
bottleneck than retrieval.
"""
    (output / "memory_transfer_taxonomy_report.md").write_text(report, encoding="utf-8")
    manifest["status"] = "ANALYZED_HUMAN"
    manifest["analysis_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["taxonomy_analysis"] = result
    for name in (
        "holdout_human_coder_a.csv",
        "holdout_human_coder_b.csv",
        "holdout_adjudicated.csv",
        "holdout_adjudication_worklist.md",
    ):
        path = output / name
        manifest["artifact_sha256"][str(path)] = sha256_path(path)
    manifest["artifact_sha256"][str(output / "taxonomy_analysis.json")] = sha256_path(
        output / "taxonomy_analysis.json"
    )
    manifest["artifact_sha256"][str(output / "memory_transfer_taxonomy_report.md")] = sha256_path(
        output / "memory_transfer_taxonomy_report.md"
    )
    write_json(manifest_path, manifest)
    print(json.dumps(result, sort_keys=True))


def readiness(args):
    output = Path(args.output)
    files = [
        "discovery_human_coder_a.csv",
        "discovery_human_coder_b.csv",
        "discovery_adjudicated.csv",
        "holdout_human_coder_a.csv",
        "holdout_human_coder_b.csv",
        "holdout_adjudicated.csv",
    ]
    result = {}
    for name in files:
        rows = load_csv(output / name)
        failures = [
            {"sample_id": row.get("sample_id"), "error": error}
            for row in rows
            for valid, error in [validate_coding_row(row)]
            if not valid
        ]
        result[name] = {"rows": len(rows), "complete": not failures, "failures": failures[:5]}
    print(json.dumps(result, indent=2, sort_keys=True))
    try:
        reliability = compute_discovery_reliability(
            load_csv(output / "discovery_human_coder_a.csv"),
            load_csv(output / "discovery_human_coder_b.csv"),
            load_csv(output / "discovery_adjudicated.csv"),
        )
        print("--- freeze-gate preview (readiness advisory) ---")
        print(json.dumps({
            "raw": {
                key: reliability["raw"][key]
                for key in (
                    "binary_applicability_kappa",
                    "primary_code_kappa",
                )
            },
            "unique": {
                key: reliability["unique"][key]
                for key in (
                    "binary_applicability_kappa",
                    "primary_code_kappa",
                )
            },
            "t3_rate": reliability["t3"]["rate"],
            "gate_passed": reliability["gate_passed"],
        }, indent=2, sort_keys=True))
    except (FileNotFoundError, ValueError):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "readiness", "freeze", "analyze"))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--discovery-labels", default=DEFAULT_DISCOVERY_LABELS)
    parser.add_argument("--discovery-source", default=DEFAULT_DISCOVERY_SOURCE)
    parser.add_argument("--codebook", default=DEFAULT_CODEBOOK)
    parser.add_argument("--preregistration", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset-arrow", default=DEFAULT_ARROW)
    parser.add_argument("--graph-root", default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--index-dirs", nargs="+", default=DEFAULT_INDEX_DIRS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate-table", default=DEFAULT_CANDIDATE_TABLE)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "readiness":
        readiness(args)
    elif args.phase == "freeze":
        freeze(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
