#!/usr/bin/env python3
"""Report the preregistered paired Django Acc@5 endpoint."""

import argparse
import json
import random
from pathlib import Path

from scripts.eval_verified_file_acc import files_from_patch, get_pred_files


def load(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["instance_id"]] = row
    return rows


def prediction(row):
    return get_pred_files(row)[:5]


def acc5(row):
    metadata = row.get("meta_data", {})
    gold = set(files_from_patch(metadata.get("patch", ""), py_only=True))
    return bool(gold) and gold.issubset(set(prediction(row)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/repomem_semantic_django_reproduction/baseline/merged_loc_outputs_mrr.jsonl")
    parser.add_argument("--semantic", default="results/repomem_semantic_django_reproduction/semantic/merged_loc_outputs_mrr.jsonl")
    parser.add_argument("--output", default="results/repomem_semantic_django_reproduction/report.json")
    args = parser.parse_args()
    baseline, semantic = load(args.baseline), load(args.semantic)
    ids = sorted(
        key for key in set(baseline) & set(semantic)
        if key.startswith("django__django-")
    )
    if len(ids) != 231:
        raise ValueError(f"Expected 231 paired Django tasks, got {len(ids)}")
    b = {key: acc5(baseline[key]) for key in ids}
    s = {key: acc5(semantic[key]) for key in ids}
    differences = [int(s[key]) - int(b[key]) for key in ids]
    generator = random.Random(44)
    bootstrap = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(10000)
    )
    result = {
        "primary_metric": "complete GoldFile Acc@5",
        "paired_tasks": len(ids),
        "baseline_acc5": sum(b.values()) / len(ids),
        "semantic_acc5": sum(s.values()) / len(ids),
        "paired_delta": (sum(s.values()) - sum(b.values())) / len(ids),
        "paired_delta_task_bootstrap_95ci": [bootstrap[249], bootstrap[9749]],
        "rescued": sum(not b[key] and s[key] for key in ids),
        "regressed": sum(b[key] and not s[key] for key in ids),
        "go": sum(s.values()) > sum(b.values()),
        "note": "Acc@5 is the sole primary endpoint; paper-version exact reproduction is not claimed.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
