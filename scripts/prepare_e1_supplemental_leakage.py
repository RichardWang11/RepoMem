#!/usr/bin/env python3
"""Screen Verified for E1-affected cases and freeze the paired cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict, deque
from pathlib import Path

from datasets import load_dataset

from plugins.location_tools.repo_ops import repo_ops


def load_rows(dataset: str, split: str) -> list[dict]:
    path = Path(dataset)
    if path.is_file():
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [dict(row) for row in load_dataset(dataset, split=split)]


def delta_bin(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    return "6+"


def deterministic_cohort(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Round-robin by repo and pre-treatment |D| bin."""
    if len(rows) <= limit:
        return sorted(rows, key=lambda row: row["instance_id"])
    strata = defaultdict(list)
    for row in rows:
        key = (row["repo"], delta_bin(row["delta_count"]))
        tie = hashlib.sha256(
            f"{seed}:{row['instance_id']}".encode()
        ).hexdigest()
        strata[key].append((tie, row))
    queues = {
        key: deque(row for _, row in sorted(values))
        for key, values in strata.items()
    }
    selected = []
    keys = sorted(queues)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if queues[key] and len(selected) < limit:
                selected.append(queues[key].popleft())
                progressed = True
        if not progressed:
            break
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")
    os.environ["REPOMEM_INDEX_DIR"] = str(Path(args.index_dir).resolve())
    os.environ["REPOMEM_VISIBILITY_POLICY"] = "strict"
    os.environ["REPOMEM_LEAKAGE_FILTER"] = "1"
    os.environ["REPOMEM_SUPPLEMENTAL_LEAKAGE"] = "on"
    repo_ops.REPOMEM_CACHE.clear()

    audits = []
    failures = []
    for instance in load_rows(args.dataset, args.split):
        try:
            memory = repo_ops._load_repomem_repo(instance["repo"])
            if not memory:
                raise repo_ops.EpisodicMemoryUnavailable("index missing")
            row = repo_ops.build_e1_supplemental_leakage_case(memory, instance)
            row["repo"] = instance["repo"]
            audits.append(row)
        except Exception as exc:
            failures.append({
                "instance_id": instance.get("instance_id"),
                "repo": instance.get("repo"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "static_audit.jsonl", audits)
    write_jsonl(output_dir / "failures.jsonl", failures)
    if failures:
        raise RuntimeError(
            f"strict E1 screening failed closed for {len(failures)} cases"
        )

    affected = [row for row in audits if row["affected"]]
    selected = (
        deterministic_cohort(affected, args.limit, args.seed)
        if affected else []
    )
    instance_ids = [row["instance_id"] for row in selected]
    (output_dir / "selected_ids.json").write_text(
        json.dumps(instance_ids, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment": "E1 supplemental leakage propagation",
        "selection_is_outcome_independent": True,
        "strict_visibility": True,
        "control": "supplemental leakage propagation OFF",
        "treatment": "supplemental leakage propagation ON",
        "screened_count": len(audits),
        "affected_count": len(affected),
        "selected_count": len(selected),
        "status": "ready" if selected else "stopped_no_affected_cases",
        "selection_rule": (
            "all if affected<=limit; otherwise deterministic round-robin "
            "over (repo, |D| bin: 1/2-5/6+)"
        ),
        "selection_seed": args.seed,
        "selected_ids": instance_ids,
        "static_audit_sha256": hashlib.sha256(
            (output_dir / "static_audit.jsonl").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
