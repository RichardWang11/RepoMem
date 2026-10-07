"""Audit non-oracle Verifier feature extractability on three blind samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT = Path(
    "results/paper2_pilot/analysis/memory_transfer_taxonomy"
)
SELECTION_SEED = 20260720
SAFE_MACHINE_FIELDS = (
    "temporal_status",
    "temporal_visible",
    "age_commits",
    "age_days",
    "surviving_files",
    "missing_files",
    "file_survival_rate",
    "same_file_symbol_count",
    "moved_symbol_count",
    "symbol_survival_same_file_rate",
    "symbol_survival_moved_rate",
    "symbol_resolution",
    "current_relation_resolvable",
    "current_relation_types",
    "no_extractable_symbol",
    "any_changed_file_survives",
    "any_symbol_resolves_same_file",
    "any_symbol_resolves_uniquely_elsewhere",
    "machine_currently_usable",
)
FORBIDDEN_FEATURE_TERMS = {
    "gold",
    "applicability_label",
    "structural_relevant",
    "sampling_stratum",
    "file_overlap",
    "parent_module_match",
    "symbol_overlap",
    "adopted",
    "outcome",
}


def read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_json_field(value: str):
    return json.loads(value) if value else None


def as_bool(value: str):
    if value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes"}


def safe_machine_row(row: dict[str, str]) -> dict:
    result = {}
    for field in SAFE_MACHINE_FIELDS:
        value = row.get(field, "")
        if field in {
            "surviving_files",
            "missing_files",
            "symbol_resolution",
            "current_relation_types",
        }:
            result[field] = parse_json_field(value)
        elif field in {
            "temporal_visible",
            "current_relation_resolvable",
            "no_extractable_symbol",
            "any_changed_file_survives",
            "any_symbol_resolves_same_file",
            "any_symbol_resolves_uniquely_elsewhere",
            "machine_currently_usable",
        }:
            result[field] = as_bool(value)
        elif field in {
            "age_commits",
            "same_file_symbol_count",
            "moved_symbol_count",
        }:
            result[field] = int(value) if value else None
        elif field in {
            "age_days",
            "file_survival_rate",
            "symbol_survival_same_file_rate",
            "symbol_survival_moved_rate",
        }:
            result[field] = float(value) if value else None
        else:
            result[field] = value or None
    return result


def extract_features(packet: dict, machine: dict) -> dict:
    memory_files = set(packet["memory"]["changed_files"])
    # Current contexts also contain Gold paths for oracle annotation. A
    # deployable Verifier must discard every context not named by the Memory.
    memory_contexts = [
        row
        for row in packet["current_base_code"]["contexts"]
        if row["path"] in memory_files
    ]
    context_by_path = {row["path"]: row for row in memory_contexts}
    feature = {
        "sample_id": packet["sample_id"],
        "input_contract": {
            "issue_fields": ["case_id", "repository", "problem_statement"],
            "memory_fields": [
                "sha",
                "subject",
                "message",
                "changed_files",
                "patch",
            ],
            "current_fields": ["base_commit", "memory_path_contexts_only"],
            "oracle_fields_used": [],
        },
        "temporal_validity": {
            "status": (
                "EXTRACTABLE"
                if machine["temporal_visible"] is not None
                else "NOT_EXTRACTABLE"
            ),
            "memory_precedes_base": machine["temporal_visible"],
            "age_commits": machine["age_commits"],
            "age_days": machine["age_days"],
        },
        "file_existence": {
            "status": "EXTRACTABLE" if memory_files else "NOT_EXTRACTABLE",
            "checked_files": sorted(memory_files),
            "context_coverage": sorted(context_by_path),
            "surviving_files": machine["surviving_files"],
            "missing_files": machine["missing_files"],
            "survival_rate": machine["file_survival_rate"],
        },
        "symbol_resolution": {
            "status": (
                "NOT_EXTRACTABLE_NO_PATCH_SYMBOLS"
                if machine["no_extractable_symbol"]
                else "EXTRACTABLE"
            ),
            "same_file_count": machine["same_file_symbol_count"],
            "moved_count": machine["moved_symbol_count"],
            "details": machine["symbol_resolution"],
        },
        "dependency_consistency": {
            "status": "PARTIAL_CURRENT_GRAPH_ONLY",
            "current_relation_resolvable": machine[
                "current_relation_resolvable"
            ],
            "current_relation_types": machine["current_relation_types"],
            "limitation": (
                "No historical dependency graph is available for comparison."
            ),
        },
        "repository_evolution": {
            "status": "PARTIAL_SURVIVAL_PROXY",
            "file_survival_rate": machine["file_survival_rate"],
            "same_file_symbol_survival_rate": machine[
                "symbol_survival_same_file_rate"
            ],
            "moved_symbol_survival_rate": machine[
                "symbol_survival_moved_rate"
            ],
            "limitation": (
                "File/symbol survival does not detect semantic API drift."
            ),
        },
    }
    serialized = json.dumps(feature, sort_keys=True).lower()
    leaked = sorted(
        term for term in FORBIDDEN_FEATURE_TERMS if term in serialized
    )
    if leaked:
        raise AssertionError(f"non-oracle feature contract leaked: {leaked}")
    return feature


def run(output: Path, sample_size: int = 3) -> dict:
    packets = read_jsonl(output / "holdout_annotation_packet.jsonl")
    machine = {
        row["sample_id"]: safe_machine_row(row)
        for row in read_csv(
            output / "sealed_machine_features_after_freeze.csv"
        )
    }
    ordered = sorted(
        packets,
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SEED}|{row['sample_id']}".encode()
        ).hexdigest(),
    )
    selected = ordered[:sample_size]
    features = [
        extract_features(row, machine[row["sample_id"]]) for row in selected
    ]

    # Oracle labels enter only after feature extraction and are kept in a
    # separate descriptive section.
    labels = {
        row["sample_id"]: row["applicability_label"]
        for row in read_csv(output / "holdout_adjudicated.csv")
    }
    core_extractable = sum(
        all(
            item[name]["status"] == "EXTRACTABLE"
            for name in (
                "temporal_validity",
                "file_existence",
                "symbol_resolution",
            )
        )
        for item in features
    )
    temporal_extractable = sum(
        item["temporal_validity"]["status"] == "EXTRACTABLE"
        for item in features
    )
    file_extractable = sum(
        item["file_existence"]["status"] == "EXTRACTABLE"
        for item in features
    )
    symbol_extractable = sum(
        item["symbol_resolution"]["status"] == "EXTRACTABLE"
        for item in features
    )
    max_symbol_candidates = max(
        len(item["symbol_resolution"]["details"] or []) for item in features
    )
    result = {
        "protocol": "p1_5_verifier_feature_extractability_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "seed": SELECTION_SEED,
            "sample_size": sample_size,
            "blind_to_gold_and_labels": True,
            "sample_ids": [item["sample_id"] for item in features],
        },
        "verifier_input_uses_gold": False,
        "features": features,
        "posthoc_oracle_evaluation": [
            {
                "sample_id": item["sample_id"],
                "applicability_label": labels[item["sample_id"]],
            }
            for item in features
        ],
        "summary": {
            "core_extractable_n": core_extractable,
            "core_extractable_denominator": sample_size,
            "temporal_extractable_n": temporal_extractable,
            "file_extractable_n": file_extractable,
            "symbol_extractable_n": symbol_extractable,
            "max_symbol_candidates_in_sample": max_symbol_candidates,
            "temporal": "PARTIAL" if temporal_extractable < sample_size
            else "EXTRACTABLE",
            "file_existence": "PARTIAL" if file_extractable < sample_size
            else "EXTRACTABLE",
            "symbol_resolution": "PARTIAL" if symbol_extractable < sample_size
            else "EXTRACTABLE_WITH_SCOPE_CONTROL_REQUIRED",
            "dependency_consistency": "PARTIAL_CURRENT_GRAPH_ONLY",
            "repository_evolution": "PARTIAL_SURVIVAL_PROXY",
            "verdict": "FEASIBLE_WITH_LIMITATIONS",
        },
    }
    json_path = output / "p1_5_verifier_feature_extractability.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = f"""# P1-5 Verifier Feature Extractability Audit

- Verdict: **{result['summary']['verdict']}**
- Selection: deterministic blind 3/30, seed {SELECTION_SEED}
- Verifier input uses Gold: **no**
- Temporal validity: extractable from commit-history order
- File existence: extractable from current graph
- Symbol resolution: conditional on symbols parsed from the Memory patch
- Dependency consistency: partial; current relations only
- Repository evolution: partial; file/symbol survival proxy only
- Core temporal/file/symbol features fully extractable: {core_extractable}/{sample_size}
- Temporal / file / symbol extractable: {temporal_extractable}/{sample_size}, {file_extractable}/{sample_size}, {symbol_extractable}/{sample_size}
- Maximum parsed symbol candidates in one Memory: {max_symbol_candidates}

This is a feasibility audit, not a predictive benchmark. Gold-visible labels
are attached only after feature extraction for descriptive inspection.
"""
    (output / "p1_5_verifier_feature_extractability.md").write_text(
        report, encoding="utf-8"
    )
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["p1_5_feature_audit"] = result["summary"]
    manifest["p1_5_feature_audit"]["sample_ids"] = result["selection"][
        "sample_ids"
    ]
    for path in (
        json_path,
        output / "p1_5_verifier_feature_extractability.md",
    ):
        manifest["artifact_sha256"][str(path)] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.sample_size), sort_keys=True))


if __name__ == "__main__":
    main()
