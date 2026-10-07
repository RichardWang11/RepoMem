#!/usr/bin/env python3
"""Audit, freeze, validate, and analyze the rank-signal confirmation."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_verified_file_acc import files_from_patch
from scripts.stale_memory_pilot import (
    MEMORY_TOOLS,
    preflight as provider_preflight,
    ranking_metrics,
    read_jsonl,
    unique_paths,
    write_csv,
    write_json,
)


ROOT = Path("results/stale_memory_rank_confirmation")
OLD_MANIFEST = Path("results/stale_memory_causal_pilot/pilot_manifest.json")
OLD_TASK_IDS = Path("reproduce_lists/verified_astropy22_django231.json")
INDEX_DIR = Path("repomem_index/verified_full500_episodic_with_issues_combined")
REPO_DIR = Path(
    "/data/wjl/addressingmemory/results/orcaloca_experiment3/repos/django__django"
)
GRAPH_DIR = Path("index_data/SWE-bench_Verified/graph_index_v2.3")
BM25_DIR = Path("index_data/SWE-bench_Verified/BM25_index")
SEEDS = (101, 202, 303)
SELECTION_SEED = 20260810
EXPECTED_RESPONSE_MODEL = "gpt-4o-2024-11-20"
TARGET_MEMORIES = (
    "33d453b2a713ad2557b6ba31a903c142263641ed",
    "ed0ff913c648b16c4471fc9a9441d1ee48cb5420",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_lines(repo_dir, *args):
    process = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.splitlines()


def load_dataset_rows(path):
    rows = read_jsonl(path)
    required = {"repo", "instance_id", "base_commit", "problem_statement", "patch"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError(f"{path}: invalid SWE-bench JSONL")
    return rows


def audit_and_prepare(
    dataset_path,
    output_dir=ROOT,
    old_manifest_path=OLD_MANIFEST,
    old_task_ids_path=OLD_TASK_IDS,
    index_dir=INDEX_DIR,
    repo_dir=REPO_DIR,
    selection_seed=SELECTION_SEED,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads(Path(old_manifest_path).read_text())
    old_memories = {row["memory_sha"] for row in old_manifest["cells"]}
    old_tasks = set(json.loads(Path(old_task_ids_path).read_text()))
    tasks = [
        row for row in load_dataset_rows(dataset_path)
        if row["repo"] == "django/django" and row["instance_id"] not in old_tasks
    ]
    tasks.sort(key=lambda row: row["instance_id"])
    if not tasks:
        raise ValueError("no unseen Full SWE-bench Django tasks")

    os.environ["REPOMEM_INDEX_DIR"] = str(Path(index_dir).resolve())
    os.environ["REPOMEM_TOKENIZER"] = "code"
    from plugins.location_tools.repo_ops import repo_ops

    memory_data = repo_ops._load_repomem_repo("django/django")
    if not memory_data:
        raise RuntimeError("Django episodic index could not be loaded")

    checkpoint = output_dir / "audit_checkpoint.jsonl"
    events = read_jsonl(checkpoint) if checkpoint.exists() else []
    completed = {row["instance_id"] for row in events}
    if completed - {row["instance_id"] for row in tasks}:
        raise ValueError("audit checkpoint does not match the current dataset")
    for number, task in enumerate(tasks, 1):
        if task["instance_id"] in completed:
            continue
        repo_ops.CURRENT_ISSUE_ID = task["instance_id"]
        repo_ops.CURRENT_INSTANCE = task
        candidates = list(repo_ops._iter_visible_commits(memory_data))
        ancestors = set(git_lines(repo_dir, "rev-list", task["base_commit"]))
        visible = [row for row in candidates if row["sha"] in ancestors]
        ranked = repo_ops._rank_repomem_rows(
            task["problem_statement"], visible, 1
        )
        event = {
            "instance_id": task["instance_id"],
            "base_commit": task["base_commit"],
            "candidate_count": len(candidates),
            "visible_ancestor_count": len(visible),
            "non_ancestor_excluded": len(candidates) - len(visible),
            "memory_sha": "",
            "bm25_score": "",
            "retrieval_status": "no_retrieval",
        }
        if ranked:
            score, memory = ranked[0]
            # Full SWE-bench may reference objects omitted by a partial clone.
            # Issue creation time is only used for the descriptive age field.
            base_time = int(datetime.fromisoformat(
                task["created_at"].replace("Z", "+00:00")
            ).timestamp())
            event.update({
                "memory_sha": memory["sha"],
                "bm25_score": score,
                "memory_commit_time": memory.get("commit_time", ""),
                "age_days": (
                    (base_time - int(memory["commit_time"])) / 86400
                    if memory.get("commit_time") else ""
                ),
                "age_time_source": "issue_created_at",
                "retrieval_status": "matched",
            })
        events.append(event)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        if number % 100 == 0 or number == len(tasks):
            print(f"audited {number}/{len(tasks)}", flush=True)

    repo_ops.CURRENT_ISSUE_ID = None
    repo_ops.CURRENT_INSTANCE = None
    events.sort(key=lambda row: row["instance_id"])
    write_csv(output_dir / "audit_events.csv", events)
    by_memory = defaultdict(list)
    for event in events:
        if event["memory_sha"]:
            by_memory[event["memory_sha"]].append(event)
    eligible = sorted(
        sha for sha, rows in by_memory.items()
        if sha not in old_memories and len(rows) >= 3
    )
    covered = sum(len(by_memory[sha]) for sha in eligible)
    density_passed = len(eligible) >= 12 and covered >= 36
    audit_summary = {
        "dataset_sha256": sha256_file(dataset_path),
        "unseen_django_tasks": len(tasks),
        "matched_tasks": sum(bool(row["memory_sha"]) for row in events),
        "eligible_new_memories_n_ge_3": len(eligible),
        "eligible_task_coverage": covered,
        "density_gate_passed": density_passed,
        "target_exposure_counts": {
            sha: len(by_memory.get(sha, [])) for sha in TARGET_MEMORIES
        },
    }
    write_json(output_dir / "audit_summary.json", audit_summary)
    if not density_passed:
        preregistration = {
            "version": "rank-signal-confirmation-v1",
            "status": "AUDIT_NO_GO_CLOSE_TO_SEMANTIC",
            "audit": audit_summary,
        }
        write_json(output_dir / "preregistration.json", preregistration)
        print(json.dumps(preregistration, indent=2))
        return preregistration

    rng = random.Random(selection_seed)
    primary_memories = sorted(rng.sample(eligible, 12))
    selected_events = []
    for sha in primary_memories:
        selected_events.extend(rng.sample(sorted(
            by_memory[sha], key=lambda row: row["instance_id"]
        ), 3))

    cells = []
    for event in selected_events:
        cells.append({
            "cell_id": (
                f"primary__{event['instance_id']}__{event['memory_sha'][:12]}"
            ),
            "instance_id": event["instance_id"],
            "memory_sha": event["memory_sha"],
            "base_commit": event["base_commit"],
            "cohort": "primary_new_memory",
            "stratum": "primary",
            "bm25_score": float(event["bm25_score"]),
            "age_days": float(event["age_days"]),
        })
    targeted_counts = {}
    for sha in TARGET_MEMORIES:
        candidates = sorted(by_memory.get(sha, []), key=lambda row: row["instance_id"])
        chosen = rng.sample(candidates, min(6, len(candidates)))
        targeted_counts[sha] = len(chosen)
        for event in chosen:
            cells.append({
                "cell_id": (
                    f"targeted__{event['instance_id']}__{event['memory_sha'][:12]}"
                ),
                "instance_id": event["instance_id"],
                "memory_sha": event["memory_sha"],
                "base_commit": event["base_commit"],
                "cohort": "targeted_old_harm",
                "stratum": "targeted",
                "bm25_score": float(event["bm25_score"]),
                "age_days": float(event["age_days"]),
            })
    cells.sort(key=lambda row: row["cell_id"])
    task_ids = sorted({row["instance_id"] for row in cells})
    task_by_id = {row["instance_id"]: row for row in tasks}
    selected_dataset = output_dir / "selected_dataset.jsonl"
    with selected_dataset.open("w", encoding="utf-8") as handle:
        for instance_id in task_ids:
            handle.write(json.dumps(task_by_id[instance_id]) + "\n")
    write_json(output_dir / "task_ids.json", task_ids)

    manifest = {
        "version": "rank-signal-confirmation-v1",
        "status": "FROZEN_READY",
        "selection_seed": selection_seed,
        "repeat_seeds": list(SEEDS),
        "primary_metric": "MRR@5",
        "primary_memories": primary_memories,
        "target_memories": list(TARGET_MEMORIES),
        "targeted_task_counts": targeted_counts,
        "cells": cells,
        "expected": {
            "primary_memories": 12,
            "primary_cells": 36,
            "targeted_cells": len(cells) - 36,
            "fixed_cells_per_repeat": len(cells),
            "control_tasks_per_repeat": len(task_ids),
            "successful_trajectories": 3 * (len(cells) + len(task_ids)),
        },
    }
    if manifest["expected"]["successful_trajectories"] > 288:
        raise ValueError("frozen design exceeds 288 successful trajectories")
    write_json(output_dir / "manifest.json", manifest)
    preregistration = {
        **manifest,
        "dataset_sha256": audit_summary["dataset_sha256"],
        "selected_dataset_sha256": sha256_file(selected_dataset),
        "audit": audit_summary,
        "model": {
            "requested": "gpt-4o",
            "required_response_model": EXPECTED_RESPONSE_MODEL,
            "temperature": 0,
            "seeds": list(SEEDS),
        },
        "estimand": "MRR_memory_at5 - MRR_no_memory_at5",
        "overall_benefit_gate": {
            "mean_at_least": 0.05,
            "hierarchical_bootstrap_ci_low_above": 0,
            "exact_memory_cluster_sign_flip_p_below": 0.05,
            "all_seed_means_positive": True,
            "all_leave_one_memory_out_means_positive": True,
        },
        "harmful_tail_gate": {
            "candidate_mean_at_most": -0.05,
            "all_seed_means_negative": True,
            "minimum_candidates": 3,
            "exact_memory_cluster_assignments": 4096,
            "p_below": 0.05,
        },
        "targeted_replication_gate": {
            "tasks_required": 6,
            "candidate_mean_at_most": -0.05,
            "all_seed_means_negative": True,
            "holm_adjusted_p_below": 0.05,
        },
    }
    write_json(output_dir / "preregistration.json", preregistration)
    missing = [
        instance_id for instance_id in task_ids
        if not (GRAPH_DIR / f"{instance_id}.pkl").exists()
        or not (BM25_DIR / instance_id).exists()
    ]
    write_json(output_dir / "missing_index_task_ids.json", missing)
    print(json.dumps({
        "status": manifest["status"],
        **manifest["expected"],
        "missing_indexes": len(missing),
        "targeted_task_counts": targeted_counts,
    }, indent=2))
    return preregistration


def preflight(lock_path, model, seed):
    current = provider_preflight(lock_path, model, seed)
    if current["response_model"] != EXPECTED_RESPONSE_MODEL:
        raise ValueError(
            f"confirmation requires {EXPECTED_RESPONSE_MODEL}, got "
            f"{current['response_model']}"
        )
    return current


def expected_runtime(run_dir, arm, repeat, lock_path):
    args = json.loads((Path(run_dir) / "args.json").read_text())
    lock = json.loads(Path(lock_path).read_text())
    expected = {
        "model": lock["requested_model"],
        "temperature": 0.0,
        "top_p": None,
        "sampling_seed": SEEDS[repeat - 1],
        "num_samples": 1,
        "max_attempt_num": 1,
        "max_iteration_num": 20,
        "context_token_budget": 100000,
        "tool_observation_token_cap": 20000,
        "fail_on_context_error": True,
        "bounded_graph_output": True,
        "bounded_entity_output": True,
        "bounded_search_output": True,
        "use_repomem": False,
    }
    mismatches = {
        key: {"expected": value, "actual": args.get(key)}
        for key, value in expected.items() if args.get(key) != value
    }
    has_manifest = bool(args.get("fixed_episodic_memory_manifest"))
    if has_manifest != (arm == "fixed"):
        mismatches["fixed_episodic_memory_manifest"] = {
            "expected": arm == "fixed", "actual": has_manifest,
        }
    if mismatches:
        raise ValueError(f"{run_dir}: runtime mismatch: {mismatches}")
    if lock["response_model"] != EXPECTED_RESPONSE_MODEL:
        raise ValueError("provider lock has the wrong response model")
    return lock


def validate_run(run_dir, arm, repeat, expected, lock_path, allow_retry=False):
    lock = expected_runtime(run_dir, arm, repeat, lock_path)
    run_dir = Path(run_dir)
    outputs = read_jsonl(run_dir / "loc_outputs.jsonl")
    records = read_jsonl(run_dir / "loc_trajs.jsonl")
    key = lambda row: row.get("experiment_cell_id", row["instance_id"])
    output_ids = {key(row) for row in outputs}
    record_ids = {key(row) for row in records}
    if len(output_ids) != len(outputs) or len(record_ids) != len(records):
        raise ValueError(f"{run_dir}: duplicate experiment cell")
    if len(records) != expected:
        if not allow_retry or len(records) >= expected or not record_ids <= output_ids:
            raise ValueError(
                f"{run_dir}: expected {expected} trajectories, got {len(records)}"
            )
        nonempty_missing = [
            key(row) for row in outputs
            if key(row) not in record_ids and row.get("found_files") != [[]]
        ]
        if nonempty_missing:
            raise ValueError(f"nonempty outputs missing trajectories: {nonempty_missing}")
        summary = {
            "run_dir": str(run_dir), "retryable": True,
            "trajectory_count": len(records), "expected": expected,
        }
        write_json(run_dir / "recovery_validation.json", summary)
        print(json.dumps(summary, indent=2))
        return summary

    fingerprints = Counter()
    for record in records:
        cell_id = key(record)
        trajectories = (record.get("loc_trajs") or {}).get("trajs") or []
        if len(trajectories) != 1:
            raise ValueError(f"{cell_id}: expected one trajectory")
        trajectory = trajectories[0]
        provider = trajectory.get("provider_metadata") or {}
        responses = provider.get("responses") or []
        if provider.get("temperature") != 0 or provider.get("seed") != SEEDS[repeat - 1]:
            raise ValueError(f"{cell_id}: sampling metadata mismatch")
        if not responses:
            raise ValueError(f"{cell_id}: missing provider responses")
        for response in responses:
            if response.get("model") != lock["response_model"]:
                raise ValueError(f"{cell_id}: response model drift")
            fingerprint = response.get("system_fingerprint")
            if not fingerprint:
                raise ValueError(f"{cell_id}: missing fingerprint")
            fingerprints[fingerprint] += 1
        tools = {
            (call.get("function") or {}).get("name")
            for message in trajectory.get("messages") or []
            for call in message.get("tool_calls") or []
        }
        if tools & MEMORY_TOOLS:
            raise ValueError(f"{cell_id}: memory tool exposed")
        fixed = (
            record.get("meta_data", {}).get("fixed_memory")
            or trajectory.get("fixed_memory")
        )
        if arm == "fixed" and (not fixed or not fixed.get("payload_sha256")):
            raise ValueError(f"{cell_id}: fixed payload metadata missing")
        if arm == "no_memory" and fixed:
            raise ValueError(f"{cell_id}: control contains fixed memory")
    summary = {
        "run_dir": str(run_dir), "valid": True, "arm": arm,
        "repeat": repeat, "trajectory_count": len(records),
        "fingerprint_counts": dict(sorted(fingerprints.items())),
    }
    write_json(run_dir / "protocol_validation.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def run_metrics(record):
    predicted = unique_paths(record.get("found_files") or [])
    gold = files_from_patch(record["meta_data"]["patch"], py_only=True)
    if not gold:
        raise ValueError(f"{record['instance_id']}: empty gold files")
    metrics = ranking_metrics(predicted, gold)
    return {
        "instance_id": record["instance_id"],
        "cell_id": record.get("experiment_cell_id", record["instance_id"]),
        "predicted_files": predicted,
        "gold_files": gold,
        **metrics,
    }


def exact_sign_flip(values, alternative="two-sided"):
    observed = statistics.mean(values)
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        value = statistics.mean(sign * item for sign, item in zip(signs, values))
        if alternative == "less":
            extreme += value <= observed + 1e-15
        elif alternative == "greater":
            extreme += value >= observed - 1e-15
        else:
            extreme += abs(value) >= abs(observed) - 1e-15
    return extreme / (2 ** len(values))


def hierarchical_ci(by_memory, replicates=10000, seed=20260810):
    rng = random.Random(seed)
    memories = sorted(by_memory)
    estimates = []
    for _ in range(replicates):
        sampled_memories = [rng.choice(memories) for _ in memories]
        values = []
        for memory in sampled_memories:
            tasks = by_memory[memory]
            sampled_tasks = [rng.choice(tasks) for _ in tasks]
            values.append(statistics.mean(sampled_tasks))
        estimates.append(statistics.mean(values))
    estimates.sort()
    return [estimates[int(0.025 * replicates)], estimates[int(0.975 * replicates)]]


def harmful_candidate(rows):
    overall = statistics.mean(row["benefit"] for row in rows)
    seed_means = {
        seed: statistics.mean(
            row["benefit"] for row in rows if row["seed"] == seed
        ) for seed in SEEDS
    }
    return overall <= -0.05 and all(value < 0 for value in seed_means.values())


def exact_harmful_tail(primary_rows):
    by_memory = defaultdict(list)
    for row in primary_rows:
        by_memory[row["memory_sha"]].append(row)
    memories = sorted(by_memory)
    if len(memories) != 12:
        raise ValueError(f"expected 12 primary memories, got {len(memories)}")
    observed = sum(harmful_candidate(by_memory[memory]) for memory in memories)
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=12):
        count = 0
        for sign, memory in zip(signs, memories):
            flipped = [
                {**row, "benefit": sign * row["benefit"]}
                for row in by_memory[memory]
            ]
            count += harmful_candidate(flipped)
        extreme += count >= observed
    return {
        "observed_harmful_memories": observed,
        "assignments": 4096,
        "exact_p": extreme / 4096,
        "passed": observed >= 3 and extreme / 4096 < 0.05,
    }


def holm_adjust(items):
    ordered = sorted(items, key=lambda row: row["raw_p"])
    running = 0
    for index, row in enumerate(ordered):
        running = max(running, min(1.0, row["raw_p"] * (len(items) - index)))
        row["holm_p"] = running


def analyze(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    fixed = []
    controls = {}
    payloads = defaultdict(set)
    for repeat, seed in enumerate(SEEDS, 1):
        for arm in ("fixed", "no_memory"):
            records = read_jsonl(root / "runs" / f"{arm}_r{repeat}" / "loc_trajs.jsonl")
            for record in records:
                row = {**run_metrics(record), "repeat": repeat, "seed": seed}
                if arm == "fixed":
                    metadata = record.get("meta_data", {}).get("fixed_memory") or {}
                    row.update({
                        "memory_sha": metadata["memory_sha"],
                        "cohort": metadata.get("cohort"),
                    })
                    payloads[row["cell_id"]].add(metadata["payload_sha256"])
                    fixed.append(row)
                else:
                    controls[(row["instance_id"], repeat)] = row
    expected = manifest["expected"]
    if len(fixed) != expected["fixed_cells_per_repeat"] * 3:
        raise ValueError("fixed trajectory count mismatch")
    if len(controls) != expected["control_tasks_per_repeat"] * 3:
        raise ValueError("control trajectory count mismatch")
    if len(payloads) != expected["fixed_cells_per_repeat"] or any(
        len(values) != 1 for values in payloads.values()
    ):
        raise ValueError("fixed payload changed across repeats")

    cell_manifest = {row["cell_id"]: row for row in manifest["cells"]}
    paired = []
    for row in fixed:
        control = controls[(row["instance_id"], row["repeat"])]
        cell = cell_manifest[row["cell_id"]]
        paired.append({
            **cell,
            "repeat": row["repeat"],
            "seed": row["seed"],
            "memory_mrr_at5": row["mrr_at5"],
            "control_mrr_at5": control["mrr_at5"],
            "benefit": row["mrr_at5"] - control["mrr_at5"],
            "first_gold_rank_benefit": (
                control["first_gold_rank_at5"] - row["first_gold_rank_at5"]
            ),
            "ndcg_benefit": row["ndcg_at5"] - control["ndcg_at5"],
        })
    primary = [row for row in paired if row["stratum"] == "primary"]
    by_memory_tasks = defaultdict(lambda: defaultdict(list))
    for row in primary:
        by_memory_tasks[row["memory_sha"]][row["instance_id"]].append(row["benefit"])
    task_means = {
        memory: [statistics.mean(values) for values in tasks.values()]
        for memory, tasks in by_memory_tasks.items()
    }
    memory_means = {
        memory: statistics.mean(values) for memory, values in task_means.items()
    }
    overall_mean = statistics.mean(memory_means.values())
    seed_means = {
        seed: statistics.mean(row["benefit"] for row in primary if row["seed"] == seed)
        for seed in SEEDS
    }
    leave_one_out = {
        memory: statistics.mean(
            value for other, value in memory_means.items() if other != memory
        ) for memory in memory_means
    }
    overall = {
        "mean_benefit": overall_mean,
        "hierarchical_bootstrap_ci": hierarchical_ci(task_means),
        "exact_memory_cluster_sign_flip_p": exact_sign_flip(
            list(memory_means.values())
        ),
        "seed_means": seed_means,
        "minimum_leave_one_memory_out_mean": min(leave_one_out.values()),
    }
    overall["passed"] = bool(
        overall_mean >= 0.05
        and overall["hierarchical_bootstrap_ci"][0] > 0
        and overall["exact_memory_cluster_sign_flip_p"] < 0.05
        and all(value > 0 for value in seed_means.values())
        and min(leave_one_out.values()) > 0
    )
    harmful_tail = exact_harmful_tail(primary)

    targeted = []
    for memory in TARGET_MEMORIES:
        rows = [row for row in paired if row["memory_sha"] == memory]
        tasks = defaultdict(list)
        for row in rows:
            tasks[row["instance_id"]].append(row["benefit"])
        task_values = [statistics.mean(values) for values in tasks.values()]
        result = {
            "memory_sha": memory,
            "task_count": len(task_values),
            "status": "NOT_TESTABLE" if len(task_values) < 3 else "DESCRIPTIVE",
        }
        if task_values:
            result.update({
                "mean_benefit": statistics.mean(task_values),
                "seed_means": {
                    seed: statistics.mean(
                        row["benefit"] for row in rows if row["seed"] == seed
                    ) for seed in SEEDS
                },
            })
        if len(task_values) == 6:
            result["raw_p"] = exact_sign_flip(task_values, "less")
            result["status"] = "FORMAL_TEST"
        targeted.append(result)
    formal = [row for row in targeted if row["status"] == "FORMAL_TEST"]
    if formal:
        holm_adjust(formal)
    for result in formal:
        result["passed"] = bool(
            result["mean_benefit"] <= -0.05
            and all(value < 0 for value in result["seed_means"].values())
            and result["holm_p"] < 0.05
        )
    targeted_passed = any(row.get("passed") for row in targeted)
    continue_episodic = harmful_tail["passed"] or targeted_passed
    decision = (
        "CONTINUE_EPISODIC_VALIDITY_STUDY" if continue_episodic
        else "CLOSE_EPISODIC_STALENESS_MOVE_TO_SEMANTIC"
    )
    summary = {
        "overall_top1_rank_benefit": overall,
        "new_harmful_tail": harmful_tail,
        "targeted_replication": targeted,
        "confirmed_top1_rank_benefit": overall["passed"],
        "decision": decision,
    }
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "per_pair.csv", paired)
    write_json(analysis_dir / "summary.json", summary)
    report = (
        "# Independent Rank-Signal Confirmation\n\n"
        "## Material Passport\n\n"
        "- Verification Status: ANALYZED\n"
        "- Primary metric: MRR@5\n"
        "- Statistical unit: memory cluster\n\n"
        f"## Decision\n\n**{decision}**\n\n"
        f"- Overall benefit: `{overall}`\n"
        f"- New harmful tail: `{harmful_tail}`\n"
        f"- Targeted replication: `{targeted}`\n"
    )
    (analysis_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--output", default=ROOT)
    flight = sub.add_parser("preflight")
    flight.add_argument("--lock", default=ROOT / "provider_lock.json")
    flight.add_argument("--model", default=os.environ.get("EVAL_MODEL", "gpt-4o"))
    flight.add_argument("--seed", type=int, required=True)
    check = sub.add_parser("validate-run")
    check.add_argument("--run-dir", required=True)
    check.add_argument("--arm", choices=("fixed", "no_memory"), required=True)
    check.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    check.add_argument("--expected", type=int, required=True)
    check.add_argument("--lock", default=ROOT / "provider_lock.json")
    check.add_argument("--allow-retry", action="store_true")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--root", default=ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        audit_and_prepare(args.dataset, args.output)
    elif args.command == "preflight":
        preflight(args.lock, args.model, args.seed)
    elif args.command == "validate-run":
        validate_run(
            args.run_dir, args.arm, args.repeat, args.expected,
            args.lock, args.allow_retry,
        )
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
