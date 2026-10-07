#!/usr/bin/env python3
"""Freeze E3's 10-case, two-seed, two-arm execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CASE_IDS = [
    "astropy__astropy-8872",
    "django__django-11885",
    "django__django-13786",
    "django__django-17084",
    "psf__requests-1142",
    "pydata__xarray-3305",
    "pylint-dev__pylint-8898",
    "pytest-dev__pytest-5809",
    "scikit-learn__scikit-learn-26323",
    "sympy__sympy-18763",
]
SEEDS = [44, 45]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cells() -> list[dict]:
    cells = []
    for case_index, instance_id in enumerate(CASE_IDS):
        for seed_index, seed in enumerate(SEEDS):
            control_first = (case_index + seed_index) % 2 == 0
            arms = ["control", "treatment"] if control_first else [
                "treatment", "control"
            ]
            for order_in_pair, arm in enumerate(arms):
                cells.append({
                    "cell_id": f"{instance_id}__seed-{seed}__{arm}",
                    "instance_id": instance_id,
                    "case_index": case_index,
                    "seed": seed,
                    "seed_index": seed_index,
                    "arm": arm,
                    "task_prompt": (
                        "forced-early" if arm == "control" else "original"
                    ),
                    "order_in_pair": order_in_pair,
                })
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    available = {
        json.loads(line)["instance_id"]
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    missing = sorted(set(CASE_IDS) - available)
    if missing:
        raise ValueError(f"E3 cases missing from dataset: {missing}")

    cells = build_cells()
    manifest_path = output / "cells.json"
    manifest_path.write_text(
        json.dumps(cells, indent=2) + "\n", encoding="utf-8"
    )
    preregistration = {
        "experiment": "E3 main-task Memory instruction A/B pilot",
        "status": "FROZEN_PRE_API",
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "cases": CASE_IDS,
        "seeds": SEEDS,
        "arms": {
            "control": "current forced-early main-task Memory paragraph",
            "treatment": "original LocAgent task prompt with Memory tools",
        },
        "cell_count": len(cells),
        "cells_sha256": file_sha256(manifest_path),
        "effective_top_k": 20,
        "tool_schema": "exact E2 current profile; no E3 schema change",
        "primary": (
            "SearchCommit occurs before first current-code exploration"
        ),
        "secondary": ["successful ExamineCommit adoption", "complete-file Acc@5"],
        "cohort_provenance": "10 context-feasible valid E2 pilot cases",
    }
    (output / "preregistration.json").write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
