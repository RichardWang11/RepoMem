#!/usr/bin/env python3
"""Report E1's three frozen endpoints from paired lazy-fork trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from scripts.eval_verified_file_acc import files_from_patch


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_dataset_rows(dataset: str, split: str) -> dict[str, dict]:
    path = Path(dataset)
    rows = load_jsonl(path) if path.is_file() else load_dataset(dataset, split=split)
    return {row["instance_id"]: dict(row) for row in rows}


def complete_file_hit(predicted: list[str], gold: list[str], k: int) -> bool:
    return set(gold).issubset(set(predicted[:k]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    instances = load_dataset_rows(args.dataset, args.split)
    rows = []
    for result in load_jsonl(run_dir / "loc_trajs.jsonl"):
        trajectories = result.get("loc_trajs", {}).get("trajs", [])
        if len(trajectories) != 1:
            raise ValueError(
                f"{result.get('instance_id')}: E1 requires exactly one paired unit"
            )
        e1 = trajectories[0].get("e1_supplemental_leakage")
        if not e1:
            raise ValueError(f"{result.get('instance_id')}: missing E1 audit")
        case = e1["case"]
        if not case.get("affected") or not case.get("core_invariants_passed"):
            raise ValueError(f"{result.get('instance_id')}: invalid E1 unit")
        instance = instances[result["instance_id"]]
        gold = files_from_patch(instance["patch"])
        control = e1["control_found_files"]
        treatment = e1["treatment_found_files"]
        outcome = e1["outcome"]
        rows.append({
            "instance_id": result["instance_id"],
            "delta_count": case["delta_count"],
            "differential_exposure": outcome["differential_exposure"],
            "next_action_changed": outcome[
                "treatment_induced_next_action_change"
            ],
            "gold_files": gold,
            "control_top5": control[:5],
            "treatment_top5": treatment[:5],
            "control_acc5": complete_file_hit(control, gold, 5),
            "treatment_acc5": complete_file_hit(treatment, gold, 5),
        })

    if not rows:
        raise ValueError("no E1 paired units found")
    count = len(rows)
    control_correct = sum(row["control_acc5"] for row in rows)
    treatment_correct = sum(row["treatment_acc5"] for row in rows)
    summary = {
        "experiment": "E1 supplemental leakage propagation",
        "affected_cases": count,
        "primary": {
            "name": "Treatment-induced Next-Action Change Rate",
            "count": sum(row["next_action_changed"] for row in rows),
            "rate": sum(row["next_action_changed"] for row in rows) / count,
            "denominator": "all affected cases (D_i > 0)",
        },
        "secondary": {
            "differential_exposure": {
                "count": sum(row["differential_exposure"] for row in rows),
                "rate": sum(row["differential_exposure"] for row in rows) / count,
            },
            "complete_file_acc_at_5": {
                "control": control_correct / count,
                "treatment": treatment_correct / count,
                "treatment_minus_control": (
                    treatment_correct - control_correct
                ) / count,
                "control_only_wins": sum(
                    row["control_acc5"] and not row["treatment_acc5"]
                    for row in rows
                ),
                "treatment_only_wins": sum(
                    row["treatment_acc5"] and not row["control_acc5"]
                    for row in rows
                ),
            },
        },
        "cases": rows,
    }
    output = Path(args.output) if args.output else run_dir / "e1_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
