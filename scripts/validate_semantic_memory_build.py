#!/usr/bin/env python3
"""Fail-closed validation for a schema-v3 semantic-memory build."""

import argparse
import json
import statistics
from pathlib import Path

from scripts.build_repomem_semantic_index import inspected_target, sha256_file
from util.cost_analysis import calc_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir")
    parser.add_argument("--expected_count", type=int)
    parser.add_argument("--required_model", default="gpt-4o-2024-11-20")
    parser.add_argument("--max_median_cost_per_file", type=float)
    parser.add_argument("--max_single_cost", type=float)
    args = parser.parse_args()

    root = Path(args.build_dir)
    spec = json.loads((root / "build_spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "build_manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (root / "summaries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = args.expected_count or manifest["expected_count"]
    assert spec["schema_version"] == manifest["schema_version"] == 3
    assert spec.get("summary_truncation") is None
    assert spec.get("summary_normalization") == "strip_surrounding_whitespace_only"
    assert spec["build_id"] == manifest["build_id"]
    assert manifest["status"] == "complete"
    assert manifest["expected_count"] == expected
    assert manifest["num_rows"] == manifest["num_ok"] == len(rows) == expected
    assert manifest["build_spec_sha256"] == sha256_file(root / "build_spec.json")
    assert manifest["summaries_sha256"] == sha256_file(root / "summaries.jsonl")
    assert set(manifest["response_models"]) == {args.required_model}

    for row in rows:
        assert row["status"] == "ok"
        assert row["build_id"] == spec["build_id"]
        assert row["source_commit"] == spec["source_commit"]
        assert row["source_tree"] == spec["source_tree"]
        assert row["source_blob_sha"] and row["prompt_instance_sha256"]
        assert row["summary_sha256"] and row["summary_text"].strip()
        assert row["summary_text"] == row["summary_text"].strip()
        if spec["builder_mode"] == "agent":
            trajectory_path = Path(row["trajectory_path"])
            if not trajectory_path.is_absolute():
                trajectory_path = root / trajectory_path
            assert trajectory_path.is_file()
            assert row["trajectory_sha256"] == sha256_file(trajectory_path)
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            assert inspected_target(trajectory["messages"], row["file"])

    for corpus in manifest["corpora"].values():
        path = Path(corpus["path"])
        if not path.is_absolute():
            path = root / path
        assert path.is_file()
        assert corpus["sha256"] == sha256_file(path)
    costs = [
        calc_cost(spec["requested_model"], row.get("prompt_tokens", 0), row.get("completion_tokens", 0))
        for row in rows
    ]
    median_cost = statistics.median(costs) if costs else 0.0
    max_cost = max(costs, default=0.0)
    if args.max_median_cost_per_file is not None:
        assert median_cost <= args.max_median_cost_per_file, (
            median_cost, args.max_median_cost_per_file
        )
    if args.max_single_cost is not None:
        assert max_cost <= args.max_single_cost, (max_cost, args.max_single_cost)
    print(json.dumps({
        "valid": True,
        "build_id": spec["build_id"],
        "rows": len(rows),
        "corpora": sorted(manifest["corpora"]),
        "median_cost_per_file_usd": median_cost,
        "max_cost_per_file_usd": max_cost,
    }, indent=2))


if __name__ == "__main__":
    main()
