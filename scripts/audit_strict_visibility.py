#!/usr/bin/env python3
"""Audit legacy vs. strict episodic windows and freeze a sensitivity cohort."""

import argparse
import hashlib
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from plugins.location_tools.repo_ops import repo_ops


def load_instances(dataset, split):
    path = Path(dataset)
    if path.is_file():
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return load_dataset(dataset, split=split)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_ids(values):
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def freeze_preregistration(path, preregistration):
    """Create the preregistration once and reject later configuration drift."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = {
            key: value for key, value in existing.items() if key != "created_at_utc"
        }
        comparable_new = {
            key: value
            for key, value in preregistration.items()
            if key != "created_at_utc"
        }
        if comparable_existing != comparable_new:
            raise RuntimeError(f"frozen preregistration would change: {path}")
        return existing
    write_json(path, preregistration)
    return preregistration


def select_sensitivity(rows, excluded, limit):
    affected = [
        row for row in rows
        if row["instance_id"] not in excluded and row["symmetric_difference_count"] > 0
    ]
    affected.sort(
        key=lambda row: (
            -row["legacy_nonancestor_count"],
            -row["symmetric_difference_count"],
            row["instance_id"],
        )
    )
    queues = defaultdict(deque)
    for row in affected:
        queues[row["repo"]].append(row)
    selected = []
    while len(selected) < limit and any(queues.values()):
        for repo in sorted(queues):
            if queues[repo] and len(selected) < limit:
                selected.append(queues[repo].popleft())
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--index_dir",
        default="repomem_index/verified_full500_episodic_with_issues_combined",
    )
    parser.add_argument(
        "--random_ids", default="reproduce_lists/verified_pilot_random_20.json"
    )
    parser.add_argument("--supplement_limit", type=int, default=10)
    parser.add_argument("--output_dir", default="results/strict_visibility_pilot/preflight")
    args = parser.parse_args()

    os.environ["REPOMEM_INDEX_DIR"] = str(Path(args.index_dir).resolve())
    os.environ["REPOMEM_BM25_PROFILE"] = "bm25-code-v2"
    os.environ["REPOMEM_LEAKAGE_FILTER"] = "1"
    random_ids = json.loads(Path(args.random_ids).read_text(encoding="utf-8"))
    rows = []
    failures = []
    repo_ops.REPOMEM_CACHE.clear()
    for instance in load_instances(args.dataset, args.split):
        memory = repo_ops._load_repomem_repo(instance["repo"])
        if not memory:
            failures.append({"instance_id": instance["instance_id"], "error": "missing index"})
            continue
        try:
            strict, strict_audit = repo_ops._strict_visible_commits(
                memory, instance["base_commit"]
            )
        except Exception as exc:
            failures.append({"instance_id": instance["instance_id"], "error": str(exc)})
            continue
        legacy = repo_ops._legacy_visible_commits(memory, instance["base_commit"])
        strict_shas = [row["sha"] for row in strict]
        legacy_shas = [row["sha"] for row in legacy]
        strict_set, legacy_set = set(strict_shas), set(legacy_shas)
        if strict_audit["canonical_base_sha"] in strict_set:
            raise AssertionError(f"base leaked into strict window for {instance['instance_id']}")
        union = strict_set | legacy_set
        rows.append({
            "instance_id": instance["instance_id"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "canonical_base_sha": strict_audit["canonical_base_sha"],
            "legacy_count": len(legacy_shas),
            "strict_count": len(strict_shas),
            "intersection_count": len(strict_set & legacy_set),
            "legacy_nonancestor_count": len(legacy_set - strict_set),
            "strict_missing_from_legacy_count": len(strict_set - legacy_set),
            "symmetric_difference_count": len(strict_set ^ legacy_set),
            "jaccard": len(strict_set & legacy_set) / len(union) if union else 1.0,
            "strict_manifest_sha256": strict_audit["manifest_sha256"],
            "strict_window_sha256": strict_audit["window_sha256"],
        })

    if failures:
        write_jsonl(Path(args.output_dir) / "visibility_failures.jsonl", failures)
        raise RuntimeError(f"{len(failures)} tasks failed strict visibility audit")
    selected_rows = select_sensitivity(rows, set(random_ids), args.supplement_limit)
    supplement_ids = [row["instance_id"] for row in selected_rows]
    write_jsonl(Path(args.output_dir) / "window_comparison.jsonl", rows)
    write_json(Path(args.output_dir) / "sensitivity_ids.json", supplement_ids)
    prereg = {
        "experiment": "legacy-vs-strict episodic visibility pilot",
        "status": "FROZEN_PRE_API",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_sha256": (
            hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
            if Path(args.dataset).is_file() else None
        ),
        "random_ids": random_ids,
        "random_ids_sha256": sha256_ids(random_ids),
        "sensitivity_ids": supplement_ids,
        "sensitivity_ids_sha256": sha256_ids(supplement_ids),
        "sensitivity_selection": (
            "round-robin by repository after sorting descending by legacy nonancestor "
            "count, symmetric difference, then instance_id; random20 excluded"
        ),
        "arms": ["legacy", "strict"],
        "samples_per_case_arm": 2,
        "model": "openai/gpt-4o-2024-05-13",
        "required_response_model": "gpt-4o-2024-05-13",
        "temperature": 1.0,
        "sampling_seed": 44,
        "tool_profile": "current",
        "search_commit_top_k_cap": 5,
        "bm25_profile": "bm25-code-v2",
        "budget_cap_usd": 50.0,
        "primary_metrics": ["complete_file_acc_at1", "complete_file_acc_at3", "complete_file_acc_at5"],
    }
    freeze_preregistration(Path(args.output_dir).parent / "preregistration.json", prereg)
    print(json.dumps({
        "audited": len(rows),
        "affected": sum(row["symmetric_difference_count"] > 0 for row in rows),
        "supplement": supplement_ids,
    }, indent=2))


if __name__ == "__main__":
    main()
