#!/usr/bin/env python3
"""Report E2's frozen adoption, exposure, and Acc@5 endpoints."""

from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument("--selected-ids", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    instances = load_dataset_rows(args.dataset, args.split)
    rows = []
    response_models = set()
    system_fingerprints = set()
    response_call_count = 0
    recorded_cost = 0.0
    for result in load_jsonl(run_dir / "loc_trajs.jsonl"):
        trajectories = result.get("loc_trajs", {}).get("trajs", [])
        if len(trajectories) != 1:
            raise ValueError(
                f"{result.get('instance_id')}: E2 requires one paired unit"
            )
        e2 = trajectories[0].get("e2_search_topk")
        if not e2:
            raise ValueError(f"{result.get('instance_id')}: missing E2 audit")
        if not e2["case"].get("base_not_visible"):
            raise ValueError(f"{result.get('instance_id')}: base leaked into corpus")
        differential = e2.get("differential")
        if differential and not differential["prefix_audit"].get(
            "ranking_prefix_invariant_passed"
        ):
            raise ValueError(
                f"{result.get('instance_id')}: ranking-prefix audit failed"
            )
        for branch in [
            trajectories[0],
            e2.get("treatment_continuation") or {},
        ]:
            for response in branch.get("provider_metadata", {}).get(
                "responses", []
            ):
                response_models.add(response.get("model"))
                system_fingerprints.add(response.get("system_fingerprint"))
                response_call_count += 1
        recorded_cost += float(result.get("usage", {}).get("cost($)", 0))
        instance = instances[result["instance_id"]]
        gold = files_from_patch(instance["patch"])
        control = e2["control_found_files"]
        treatment = e2["treatment_found_files"]
        outcome = e2["outcome"]
        added_short_shas = set()
        if differential:
            for section in differential["treatment_observation"].split(
                "##Searching for query "
            )[1:]:
                ranked = re.findall(r"^SHA: ([0-9a-f]+)", section, re.MULTILINE)
                added_short_shas.update(ranked[5:20])
        adopted_added_candidate = any(
            any(sha.startswith(short) for short in added_short_shas)
            for sha in outcome["treatment_examine_sequence"]
        )
        rows.append({
            "instance_id": result["instance_id"],
            "differential_exposure": outcome["differential_exposure"],
            "memory_adoption_changed": outcome["memory_adoption_changed"],
            "control_examine_sequence": outcome["control_examine_sequence"],
            "treatment_examine_sequence": outcome["treatment_examine_sequence"],
            "control_single_examine": (
                len(outcome["control_examine_sequence"]) == 1
            ),
            "treatment_single_examine": (
                len(outcome["treatment_examine_sequence"]) == 1
            ),
            "treatment_adopted_rank_6_20": adopted_added_candidate,
            "gold_files": gold,
            "control_top5": control[:5],
            "treatment_top5": treatment[:5],
            "control_acc5": complete_file_hit(control, gold, 5),
            "treatment_acc5": complete_file_hit(treatment, gold, 5),
        })

    if not rows:
        raise ValueError("no E2 paired units found")
    count = len(rows)
    valid_ids = {row["instance_id"] for row in rows}
    selected_ids = (
        json.loads(Path(args.selected_ids).read_text(encoding="utf-8"))
        if args.selected_ids else sorted(valid_ids)
    )
    missing_ids = [
        instance_id for instance_id in selected_ids
        if instance_id not in valid_ids
    ]
    control_correct = sum(row["control_acc5"] for row in rows)
    treatment_correct = sum(row["treatment_acc5"] for row in rows)
    summary = {
        "experiment": "E2 SearchCommit Top5 vs Top20 pilot",
        "pre_registered_cases": len(selected_ids),
        "paired_cases": count,
        "invalid_or_missing_cases": missing_ids,
        "paired_completion_rate": count / len(selected_ids),
        "provider_audit": {
            "response_models": sorted(str(value) for value in response_models),
            "system_fingerprints": sorted(
                str(value) for value in system_fingerprints
            ),
            "response_call_count": response_call_count,
        },
        "recorded_valid_pair_cost_usd": recorded_cost,
        "primary": {
            "name": "Memory-Adoption Change Rate",
            "count": sum(row["memory_adoption_changed"] for row in rows),
            "rate": sum(row["memory_adoption_changed"] for row in rows) / count,
            "denominator": (
                "valid paired cases; shared-prefix failures are listed "
                "separately and receive no outcome imputation"
            ),
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
        "diagnostics": {
            "control_single_examine_count": sum(
                row["control_single_examine"] for row in rows
            ),
            "treatment_single_examine_count": sum(
                row["treatment_single_examine"] for row in rows
            ),
            "treatment_adopted_rank_6_20_count": sum(
                row["treatment_adopted_rank_6_20"] for row in rows
            ),
        },
        "cases": rows,
    }
    output = Path(args.output) if args.output else run_dir / "e2_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
