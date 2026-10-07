#!/usr/bin/env python3
"""Freeze E2's previously used 12-case pilot cohort and configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from plugins.location_tools.repo_ops import repo_ops


PILOT_IDS = [
    "django__django-11885",
    "django__django-13786",
    "astropy__astropy-8872",
    "django__django-17084",
    "matplotlib__matplotlib-14623",
    "psf__requests-1142",
    "pydata__xarray-3305",
    "pylint-dev__pylint-8898",
    "pytest-dev__pytest-5809",
    "scikit-learn__scikit-learn-26323",
    "sphinx-doc__sphinx-11510",
    "sympy__sympy-18763",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["instance_id"]: row for row in rows}
    available = set(by_id)
    missing = [instance_id for instance_id in PILOT_IDS if instance_id not in available]
    if missing:
        raise ValueError(f"E2 cases missing from dataset: {missing}")

    selected = output / "selected_ids.json"
    selected.write_text(
        json.dumps(PILOT_IDS, indent=2) + "\n", encoding="utf-8"
    )
    static_audit = []
    for instance_id in PILOT_IDS:
        instance = by_id[instance_id]
        memory = repo_ops._load_repomem_repo(instance["repo"])
        if not memory:
            raise ValueError(f"E2 index unavailable for {instance['repo']}")
        visible = repo_ops._visible_commits_for_instance(memory, instance)
        visibility = repo_ops.get_repomem_visibility_audit()
        shas = [row["sha"] for row in visible]
        canonical_base = visibility.get("canonical_base_sha")
        if canonical_base in set(shas):
            raise AssertionError(f"{instance_id}: base leaked into E2 corpus")
        static_audit.append({
            "instance_id": instance_id,
            "canonical_base_sha": canonical_base,
            "visible_count": len(shas),
            "visible_sha256": hashlib.sha256(
                "\n".join(shas).encode("ascii")
            ).hexdigest(),
            "visibility_policy": visibility.get("policy"),
            "supplemental_leakage_propagation": visibility.get(
                "supplemental_leakage_propagation"
            ),
            "base_not_visible": True,
        })
    audit_path = output / "static_audit.jsonl"
    audit_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in static_audit),
        encoding="utf-8",
    )
    preregistration = {
        "experiment": "E2 SearchCommit Top5 vs Top20 pilot",
        "status": "FROZEN_PRE_API",
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "selected_ids": PILOT_IDS,
        "selected_ids_sha256": sha256(selected),
        "static_audit_sha256": sha256(audit_path),
        "cohort_provenance": (
            "same 12 completed cases as strict-visibility pilot; descriptive, "
            "not a representative Verified-500 sample"
        ),
        "control_effective_top_k": 5,
        "treatment_effective_top_k": 20,
        "top_k_semantics": "per_query",
        "primary": "Memory-Adoption Change Rate",
        "secondary": ["Differential Exposure Rate", "complete-file Acc@5"],
    }
    (output / "preregistration.json").write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
