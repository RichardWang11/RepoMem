#!/usr/bin/env python3
"""Analyze E3's prompt-only, two-seed paired pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.eval_verified_file_acc import files_from_patch


CURRENT_CODE_TOOLS = {
    "search_code_snippets",
    "get_entity_contents",
    "explore_graph_structure",
    "explore_tree_structure",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def tool_decisions(messages: list[dict]) -> list[list[dict]]:
    return [
        message.get("tool_calls") or []
        for message in messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]


def behavior_metrics(messages: list[dict]) -> dict[str, Any]:
    decisions = tool_decisions(messages)
    search_position = None
    code_position = None
    simultaneous = False
    for position, calls in enumerate(decisions):
        names = {
            (call.get("function") or {}).get("name")
            for call in calls
        }
        if search_position is None and "search_commit" in names:
            search_position = position
        if code_position is None and names & CURRENT_CODE_TOOLS:
            code_position = position
        if "search_commit" in names and names & CURRENT_CODE_TOOLS:
            simultaneous = True

    memory_first = bool(
        search_position is not None
        and (code_position is None or search_position < code_position)
    )
    call_names = {}
    examine_call_count = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name")
            call_names[call.get("id")] = name
            examine_call_count += name == "examine_commit"
    successful_examine = 0
    for message in messages:
        if message.get("role") != "tool":
            continue
        name = message.get("name") or call_names.get(message.get("tool_call_id"))
        if name == "examine_commit" and "##Commit for sha `" in (
            message.get("content") or ""
        ):
            successful_examine += 1
    return {
        "memory_first": memory_first,
        "search_position": search_position,
        "current_code_position": code_position,
        "simultaneous": simultaneous,
        "search_commit_ever": search_position is not None,
        "successful_examine_adoption": successful_examine > 0,
        "examine_call_count": examine_call_count,
        "successful_examine_observation_count": successful_examine,
        "single_examine": examine_call_count == 1,
    }


def initial_prompts(messages: list[dict]) -> tuple[str, str]:
    system = next(
        (message.get("content") or "" for message in messages
         if message.get("role") == "system"),
        "",
    )
    task = next(
        (message.get("content") or "" for message in messages
         if message.get("role") == "user"),
        "",
    )
    return system, task


def complete_file_hit(predicted: list[str], gold: list[str]) -> bool:
    return set(gold).issubset(set(predicted[:5]))


def mean(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--e2-run-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cells = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    instances = {
        row["instance_id"]: row
        for row in load_jsonl(Path(args.dataset))
    }
    rows = []
    for cell in cells:
        cell_dir = run_dir / "cells" / cell["cell_id"]
        trajectory_rows = load_jsonl(cell_dir / "loc_trajs.jsonl")
        if not trajectory_rows:
            rows.append({**cell, "valid_trajectory": False, "acc5": False})
            continue
        result = trajectory_rows[0]
        trajectories = result.get("loc_trajs", {}).get("trajs", [])
        if len(trajectories) != 1:
            raise ValueError(f"{cell['cell_id']}: expected one trajectory")
        trajectory = trajectories[0]
        audit = trajectory.get("e3_prompt")
        if not audit or audit.get("arm") != cell["arm"]:
            raise ValueError(f"{cell['cell_id']}: missing or wrong E3 audit")
        system_prompt, task_prompt = initial_prompts(trajectory["messages"])
        behavior = behavior_metrics(trajectory["messages"])
        gold = files_from_patch(instances[cell["instance_id"]]["patch"])
        predicted = (result.get("found_files") or [[]])[0]
        responses = trajectory.get("provider_metadata", {}).get("responses", [])
        rows.append({
            **cell,
            **behavior,
            "valid_trajectory": True,
            "system_prompt": system_prompt,
            "task_prompt": task_prompt,
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode()
            ).hexdigest(),
            "task_prompt_sha256": hashlib.sha256(
                task_prompt.encode()
            ).hexdigest(),
            "tool_schema_sha256": sha256_json(trajectory.get("tools")),
            "corpus_sha256": audit["case"]["visible_sha256"],
            "response_models": sorted({response.get("model") for response in responses}),
            "system_fingerprints": sorted({
                response.get("system_fingerprint") for response in responses
            }),
            "gold_files": gold,
            "predicted_top5": predicted[:5],
            "acc5": complete_file_hit(predicted, gold),
            "cost_usd": float(result.get("usage", {}).get("cost($)", 0)),
        })

    valid = [row for row in rows if row["valid_trajectory"]]
    by_pair = {}
    for row in valid:
        by_pair.setdefault((row["instance_id"], row["seed"]), {})[
            row["arm"]
        ] = row
    prompt_invariants = []
    for key, pair in by_pair.items():
        if set(pair) != {"control", "treatment"}:
            continue
        control = pair["control"]
        treatment = pair["treatment"]
        prompt_invariants.append({
            "instance_id": key[0],
            "seed": key[1],
            "system_equal": (
                control["system_prompt_sha256"]
                == treatment["system_prompt_sha256"]
            ),
            "tool_schema_equal": (
                control["tool_schema_sha256"]
                == treatment["tool_schema_sha256"]
            ),
            "corpus_equal": control["corpus_sha256"] == treatment["corpus_sha256"],
            "prompt_suffix_exact": (
                control["task_prompt"]
                == treatment["task_prompt"]
                + __import__("auto_search_main").EPISODIC_TASK_PROMPT
            ),
        })
    if any(
        not all(value for name, value in row.items()
                if name not in {"instance_id", "seed"})
        for row in prompt_invariants
    ):
        raise AssertionError("one or more E3 prompt-only invariants failed")

    e2_schema_hashes = []
    if args.e2_run_dir:
        for result in load_jsonl(Path(args.e2_run_dir) / "loc_trajs.jsonl"):
            for trajectory in result.get("loc_trajs", {}).get("trajs", []):
                e2_schema_hashes.append(sha256_json(trajectory.get("tools")))
    e3_schema_hashes = sorted({row["tool_schema_sha256"] for row in valid})
    if e2_schema_hashes and set(e3_schema_hashes) != set(e2_schema_hashes):
        raise AssertionError("E3 tool schema does not exactly match E2")

    complete_pair_keys = {
        key for key, pair in by_pair.items()
        if set(pair) == {"control", "treatment"}
    }
    paired_valid = [
        row for row in valid
        if (row["instance_id"], row["seed"]) in complete_pair_keys
    ]
    arm_summary = {}
    for arm in ("control", "treatment"):
        arm_rows = [row for row in paired_valid if row["arm"] == arm]
        arm_summary[arm] = {
            "complete_pair_trajectories": len(arm_rows),
            "memory_first_rate": mean([row["memory_first"] for row in arm_rows]),
            "memory_adoption_rate": mean([
                row["successful_examine_adoption"] for row in arm_rows
            ]),
            "acc5": mean([row["acc5"] for row in arm_rows]),
            "single_examine_rate": mean([
                row["single_examine"] for row in arm_rows
            ]),
        }
    summary = {
        "experiment": "E3 main-task Memory instruction A/B pilot",
        "pre_registered_cells": len(cells),
        "valid_trajectories": len(valid),
        "invalid_cells": [row["cell_id"] for row in rows if not row["valid_trajectory"]],
        "unpaired_valid_cells": [
            row["cell_id"] for row in valid
            if (row["instance_id"], row["seed"]) not in complete_pair_keys
        ],
        "complete_pairs": len(complete_pair_keys),
        "primary_analysis_population": "complete case/seed pairs",
        "arm_summary": arm_summary,
        "treatment_minus_control": {
            metric: arm_summary["treatment"][metric]
            - arm_summary["control"][metric]
            for metric in ("memory_first_rate", "memory_adoption_rate", "acc5")
            if arm_summary["control"][metric] is not None
            and arm_summary["treatment"][metric] is not None
        },
        "integrity": {
            "complete_pair_invariants": prompt_invariants,
            "e3_tool_schema_hashes": e3_schema_hashes,
            "e2_tool_schema_match": bool(e2_schema_hashes),
            "response_models": sorted({
                model for row in valid for model in row["response_models"]
                if model is not None
            }),
            "system_fingerprints": sorted({
                fingerprint
                for row in valid for fingerprint in row["system_fingerprints"]
                if fingerprint is not None
            }),
        },
        "recorded_cost_usd": sum(row.get("cost_usd", 0) for row in valid),
        "cells": [{key: value for key, value in row.items()
                   if key not in {"system_prompt", "task_prompt"}} for row in rows],
    }
    output = Path(args.output) if args.output else run_dir / "e3_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
