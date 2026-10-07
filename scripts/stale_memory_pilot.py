#!/usr/bin/env python3
"""Prepare and analyze the frozen stale-memory causal-harm pilot."""

import argparse
import csv
import itertools
import json
import math
import pickle
import random
import re
import statistics
import subprocess
import os
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import litellm

from scripts.audit_future_reuse import percentile, ranks, spearman
from scripts.eval_verified_file_acc import files_from_patch, normalize_path
from scripts.validate_applicability_labels import (
    graph_indexes,
    non_dunder_symbols,
    resolve_current_structure,
)


DEFAULT_AUDIT = Path(
    "results/future_reuse_audit/django231_bm25_code_top3/retrieval_events.csv"
)
DEFAULT_ROOT = Path("results/stale_memory_causal_pilot")
DEFAULT_INDEX = Path(
    "repomem_index/verified_full500_episodic_with_issues_combined/django__django"
)
DEFAULT_GRAPH = Path("index_data/SWE-bench_Verified/graph_index_v2.3")
DEFAULT_REPO = Path(
    "/data/wjl/addressingmemory/results/orcaloca_experiment3/repos/django__django"
)
SEARCH_TOOLS = {"search_code_snippets", "explore_tree_structure"}
MEMORY_TOOLS = {"search_commit", "examine_commit", "search_summary", "view_summary"}
SEEDS = (42, 43, 44)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ranking_metrics(predicted, gold, k=5):
    gold = set(gold)
    ranks_found = [
        rank for rank, path in enumerate(predicted[:k], 1) if path in gold
    ]
    first_gold_rank = min(ranks_found, default=k + 1)
    reciprocal_rank = 0 if first_gold_rank > k else 1 / first_gold_rank
    dcg = sum(1 / math.log2(rank + 1) for rank in ranks_found)
    ideal_dcg = sum(
        1 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1)
    )
    return {
        "first_gold_rank_at5": first_gold_rank,
        "mrr_at5": reciprocal_rank,
        "ndcg_at5": dcg / ideal_dcg,
    }


def endpoint_hash():
    value = next((os.environ.get(name) for name in (
        "OPENAI_API_BASE", "AZURE_API_BASE", "OPENAI_BASE_URL"
    ) if os.environ.get(name)), "default")
    return hashlib.sha256(value.encode()).hexdigest()


def preflight(lock_path, model, seed):
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Reply with OK."}],
        temperature=0,
        seed=seed,
        max_tokens=2,
        request_timeout=60,
    )
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else getattr(details, "cached_tokens", 0)
    current = {
        "requested_model": model,
        "response_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "temperature": 0,
        "seed": seed,
        "endpoint_hash": endpoint_hash(),
        "cached_prompt_tokens": int(cached or 0),
    }
    if not current["response_model"] or not current["system_fingerprint"]:
        raise ValueError(f"provider did not return model/fingerprint: {current}")
    lock_path = Path(lock_path)
    if lock_path.exists():
        locked = json.loads(lock_path.read_text())
        for key in ("requested_model", "response_model", "temperature", "endpoint_hash"):
            if current[key] != locked[key]:
                raise ValueError(f"provider lock mismatch for {key}: {current[key]} != {locked[key]}")
        observed = set(locked.get("observed_fingerprints") or [])
        if locked.get("system_fingerprint"):
            observed.add(locked["system_fingerprint"])
        observed.add(current["system_fingerprint"])
        locked["observed_fingerprints"] = sorted(observed)
        write_json(lock_path, locked)
    else:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        current["observed_fingerprints"] = [current["system_fingerprint"]]
        write_json(lock_path, current)
    print(json.dumps(current, indent=2))
    return current


def prepare(audit_path, output_dir, seed=42):
    with Path(audit_path).open(encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    for row in events:
        row["rank"] = int(row["rank"])

    top1_counts = Counter(row["memory_sha"] for row in events if row["rank"] == 1)
    primary_memories = {
        sha for sha, count in top1_counts.items() if count >= 3
    }
    top3_counts = Counter(row["memory_sha"] for row in events)
    remaining = sorted(
        sha for sha, count in top3_counts.items()
        if count >= 3 and sha not in primary_memories
    )
    if len(primary_memories) != 10 or len(remaining) != 25:
        raise ValueError(
            f"expected 10 primary and 25 remaining memories, got "
            f"{len(primary_memories)} and {len(remaining)}"
        )

    rng = random.Random(seed)
    extension_memories = sorted(rng.sample(remaining, 10))
    selected = [
        row for row in events
        if row["rank"] == 1 and row["memory_sha"] in primary_memories
    ]
    for sha in extension_memories:
        candidates = sorted(
            (row for row in events if row["memory_sha"] == sha),
            key=lambda row: (row["instance_id"], row["rank"]),
        )
        selected.extend(rng.sample(candidates, 3))

    cells = []
    for row in selected:
        cohort = "top1_primary" if row["memory_sha"] in primary_memories else "top3_extension"
        cells.append({
            "cell_id": f"{cohort}__{row['instance_id']}__{row['memory_sha'][:12]}",
            "instance_id": row["instance_id"],
            "memory_sha": row["memory_sha"],
            "cohort": cohort,
            "audit_rank": row["rank"],
            "bm25_score": float(row["bm25_score"]),
            "age_days": float(row["age_days"]),
            "base_commit": row["base_commit"],
        })
    cells.sort(key=lambda row: row["cell_id"])
    task_ids = sorted({row["instance_id"] for row in cells})
    primary_cells = sum(row["cohort"] == "top1_primary" for row in cells)
    if len(cells) != 65 or primary_cells != 35 or len(task_ids) != 58:
        raise ValueError(
            f"frozen design mismatch: cells={len(cells)}, "
            f"primary={primary_cells}, tasks={len(task_ids)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "stale-memory-causal-pilot-v1",
        "audit_source": str(audit_path),
        "selection_seed": seed,
        "retrieval_method": "BM25-code",
        "repeat_seeds": list(SEEDS),
        "expected": {
            "memories": 20,
            "fixed_cells": 65,
            "primary_cells": 35,
            "extension_cells": 30,
            "unique_tasks": 58,
            "fixed_runs": 195,
            "no_memory_runs": 174,
            "total_runs": 369,
        },
        "primary_memories": sorted(primary_memories),
        "extension_memories": extension_memories,
        "cells": cells,
    }
    write_json(output_dir / "pilot_manifest.json", manifest)
    write_json(output_dir / "fixed_task_ids.json", task_ids)
    write_json(output_dir / "no_memory_task_ids.json", task_ids)
    print(json.dumps(manifest["expected"], indent=2))
    return manifest


def flatten_files(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [path for item in value for path in flatten_files(item)]
    return []


def unique_paths(value):
    result = []
    for path in flatten_files(value):
        path = normalize_path(path)
        if path not in result:
            result.append(path)
    return result


def valid_graph_files(graph_root, instance_id, cache):
    if instance_id not in cache:
        with (Path(graph_root) / f"{instance_id}.pkl").open("rb") as handle:
            graph = pickle.load(handle)
        cache[instance_id] = {
            normalize_path(node)
            for node, data in graph.nodes(data=True)
            if data.get("type") == "file"
        }
    return cache[instance_id]


def surfaced_files(messages, valid_files):
    surfaced = set()
    rounds = set()
    calls = Counter()
    assistant_iteration = 0
    pending = {}
    memory_calls = []
    for message in messages:
        if message.get("role") == "assistant":
            assistant_iteration += 1
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name")
                pending[call.get("id")] = (name, assistant_iteration)
                if name in SEARCH_TOOLS:
                    rounds.add(assistant_iteration)
                    calls[name] += 1
                if name in MEMORY_TOOLS:
                    memory_calls.append(name)
        if message.get("role") != "tool":
            continue
        name = message.get("name")
        if not name and message.get("tool_call_id") in pending:
            name = pending[message["tool_call_id"]][0]
        if name not in SEARCH_TOOLS:
            continue
        content = str(message.get("content") or "")
        if name == "search_code_snippets":
            candidates = []
            for line in content.splitlines():
                if line.startswith("Found "):
                    candidates.extend(re.findall(r"`([^`]+\.py(?:[:][^`]*)?)`", line))
        else:
            candidates = re.findall(r"([\w./-]+\.py)(?::[\w.]+)?", content)
        surfaced.update(
            normalize_path(item.split(":", 1)[0])
            for item in candidates
            if normalize_path(item.split(":", 1)[0]) in valid_files
        )
    return {
        "candidate_count": len(surfaced),
        "candidate_files": sorted(surfaced),
        "search_rounds": len(rounds),
        "search_calls": sum(calls.values()),
        "search_code_snippets_calls": calls["search_code_snippets"],
        "explore_tree_structure_calls": calls["explore_tree_structure"],
        "memory_tool_calls": memory_calls,
    }


def trajectory_metrics(record, arm, repeat, graph_root, graph_cache):
    trajectories = (record.get("loc_trajs") or {}).get("trajs") or []
    if len(trajectories) != 1:
        raise ValueError(
            f"{record.get('experiment_cell_id', record.get('instance_id'))}: "
            f"expected one trajectory, got {len(trajectories)}"
        )
    trajectory = trajectories[0]
    instance_id = record["instance_id"]
    predicted = unique_paths(record.get("found_files") or [])
    gold = files_from_patch(record["meta_data"]["patch"], py_only=True)
    if not gold:
        raise ValueError(f"{instance_id}: empty gold file set")
    gold_set = set(gold)
    top5 = set(predicted[:5])
    ranking = ranking_metrics(predicted, gold)
    search = surfaced_files(
        trajectory.get("messages") or [],
        valid_graph_files(graph_root, instance_id, graph_cache),
    )
    provider = trajectory.get("provider_metadata") or {}
    responses = provider.get("responses") or []
    context_events = (trajectory.get("context_guard") or {}).get("events") or []
    prompt_values = [
        int(event.get("prompt_tokens") or 0)
        for event in context_events if event.get("event") == "request"
    ]
    fixed = record.get("meta_data", {}).get("fixed_memory") or trajectory.get("fixed_memory")
    return {
        "arm": arm,
        "repeat": repeat,
        "seed": provider.get("seed"),
        "temperature": provider.get("temperature"),
        "instance_id": instance_id,
        "cell_id": record.get("experiment_cell_id", instance_id),
        "memory_sha": (fixed or {}).get("memory_sha", ""),
        "cohort": (fixed or {}).get("cohort", ""),
        "payload_sha256": (fixed or {}).get("payload_sha256", ""),
        "payload_chars": (fixed or {}).get("payload_chars", 0),
        "payload_tokens": (fixed or {}).get("payload_tokens", 0),
        "gold_files": gold,
        "predicted_files": predicted,
        **ranking,
        "recall_at5": len(gold_set & top5) / len(gold_set),
        "localization_error": 1 - len(gold_set & top5) / len(gold_set),
        "acc_at5": int(gold_set.issubset(top5)),
        "predicted_file_count": len(predicted),
        "prompt_tokens": (trajectory.get("usage") or {}).get("total_prompt_tokens", 0),
        "completion_tokens": (trajectory.get("usage") or {}).get("total_completion_tokens", 0),
        "cached_prompt_tokens": sum(int(row.get("cached_prompt_tokens") or 0) for row in responses),
        "peak_context_tokens": max(prompt_values, default=0),
        "context_utilization": max(prompt_values, default=0) / 100000,
        "response_models": sorted({row.get("model") for row in responses if row.get("model")}),
        "system_fingerprints": sorted({row.get("system_fingerprint") for row in responses if row.get("system_fingerprint")}),
        "fingerprint_counts": dict(Counter(
            row.get("system_fingerprint") for row in responses
            if row.get("system_fingerprint")
        )),
        "response_count": len(responses),
        "missing_response_model_count": sum(not row.get("model") for row in responses),
        "missing_fingerprint_count": sum(not row.get("system_fingerprint") for row in responses),
        **search,
    }


def load_runs(root, graph_root):
    graph_cache = {}
    rows = []
    for repeat in range(1, 4):
        for arm in ("fixed", "no_memory"):
            path = Path(root) / "runs" / f"{arm}_r{repeat}" / "loc_trajs.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.extend(
                trajectory_metrics(record, arm, repeat, graph_root, graph_cache)
                for record in read_jsonl(path)
            )
    return rows


def validate_runtime(run_dir, arm, repeat, lock_path):
    run_args = json.loads((Path(run_dir) / "args.json").read_text())
    lock = json.loads(Path(lock_path).read_text())
    expected_seed = SEEDS[repeat - 1]
    expected_args = {
        "model": lock["requested_model"],
        "temperature": 0.0,
        "top_p": None,
        "sampling_seed": expected_seed,
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
        key: {"expected": value, "actual": run_args.get(key)}
        for key, value in expected_args.items() if run_args.get(key) != value
    }
    has_fixed_manifest = bool(run_args.get("fixed_episodic_memory_manifest"))
    if has_fixed_manifest != (arm == "fixed"):
        mismatches["fixed_episodic_memory_manifest"] = {
            "expected": arm == "fixed", "actual": has_fixed_manifest,
        }
    if mismatches:
        raise ValueError(f"{run_dir}: runtime argument mismatch: {mismatches}")
    return expected_seed, lock


def validate_retryable_run(run_dir, arm, repeat, expected, lock_path):
    validate_runtime(run_dir, arm, repeat, lock_path)
    run_dir = Path(run_dir)
    outputs = read_jsonl(run_dir / "loc_outputs.jsonl")
    trajectories = read_jsonl(run_dir / "loc_trajs.jsonl")
    if len(outputs) > expected or len(trajectories) >= expected:
        raise ValueError(
            f"{run_dir}: not an incomplete run: "
            f"{len(outputs)} outputs, {len(trajectories)} trajectories"
        )

    def row_key(row):
        return row.get("experiment_cell_id", row["instance_id"])

    output_by_id = {row_key(row): row for row in outputs}
    trajectory_ids = {row_key(row) for row in trajectories}
    if len(output_by_id) != len(outputs) or len(trajectory_ids) != len(trajectories):
        raise ValueError(f"{run_dir}: duplicate experiment cells")
    if not trajectory_ids <= set(output_by_id):
        raise ValueError(f"{run_dir}: trajectory without localization output")
    nonempty_missing = [
        cell_id for cell_id, row in output_by_id.items()
        if cell_id not in trajectory_ids and row.get("found_files") != [[]]
    ]
    if nonempty_missing:
        raise ValueError(
            f"{run_dir}: nonempty outputs missing trajectories: {nonempty_missing}"
        )
    summary = {
        "run_dir": str(run_dir),
        "trajectory_count": len(trajectories),
        "expected": expected,
        "missing_count": expected - len(trajectories),
        "retryable": True,
    }
    write_json(run_dir / "recovery_validation.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def validate_run(run_dir, arm, repeat, expected, lock_path):
    expected_seed, lock = validate_runtime(run_dir, arm, repeat, lock_path)
    records = read_jsonl(Path(run_dir) / "loc_trajs.jsonl")
    if len(records) != expected:
        raise ValueError(f"{run_dir}: expected {expected} trajectories, got {len(records)}")
    cell_ids = set()
    response_count = 0
    fingerprint_counts = Counter()
    for record in records:
        cell_id = record.get("experiment_cell_id", record["instance_id"])
        if cell_id in cell_ids:
            raise ValueError(f"{run_dir}: duplicate cell {cell_id}")
        cell_ids.add(cell_id)
        trajectories = (record.get("loc_trajs") or {}).get("trajs") or []
        if len(trajectories) != 1:
            raise ValueError(f"{run_dir}/{cell_id}: expected one trajectory")
        trajectory = trajectories[0]
        provider = trajectory.get("provider_metadata") or {}
        if provider.get("temperature") != 0 or provider.get("seed") != expected_seed:
            raise ValueError(f"{run_dir}/{cell_id}: sampling metadata mismatch")
        responses = provider.get("responses") or []
        if not responses:
            raise ValueError(f"{run_dir}/{cell_id}: no provider responses")
        response_count += len(responses)
        for response in responses:
            if response.get("model") != lock["response_model"]:
                raise ValueError(f"{run_dir}/{cell_id}: response model drift")
            if not response.get("system_fingerprint"):
                raise ValueError(f"{run_dir}/{cell_id}: fingerprint missing")
            fingerprint_counts[response["system_fingerprint"]] += 1
            if response.get("temperature") != 0 or response.get("seed") != expected_seed:
                raise ValueError(f"{run_dir}/{cell_id}: response sampling metadata mismatch")
        tool_names = [
            (call.get("function") or {}).get("name")
            for message in trajectory.get("messages") or []
            for call in message.get("tool_calls") or []
        ]
        if MEMORY_TOOLS & set(tool_names):
            raise ValueError(f"{run_dir}/{cell_id}: memory tool was exposed")
        fixed = record.get("meta_data", {}).get("fixed_memory") or trajectory.get("fixed_memory")
        if arm == "fixed" and (not fixed or not fixed.get("payload_sha256")):
            raise ValueError(f"{run_dir}/{cell_id}: fixed payload metadata missing")
        if arm == "no_memory" and fixed:
            raise ValueError(f"{run_dir}/{cell_id}: control contains fixed memory")
    summary = {
        "run_dir": str(run_dir),
        "arm": arm,
        "repeat": repeat,
        "seed": expected_seed,
        "trajectory_count": len(records),
        "response_count": response_count,
        "fingerprint_counts": dict(sorted(fingerprint_counts.items())),
        "fingerprint_mixture": len(fingerprint_counts) > 1,
        "valid": True,
    }
    write_json(Path(run_dir) / "protocol_validation.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def cluster_bootstrap_mean(rows, field, replicates=10000, seed=42):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"]].append(row[field])
    keys = sorted(grouped)
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled = [rng.choice(keys) for _ in keys]
        values.append(mean([value for key in sampled for value in grouped[key]]))
    return [percentile(values, 0.025), percentile(values, 0.975)]


def permutation_dispersion(rows, field="harm", replicates=10000, seed=42):
    observed_by_memory = defaultdict(list)
    for row in rows:
        observed_by_memory[row["memory_sha"]].append(row[field])
    observed = iqr([mean(values) for values in observed_by_memory.values()])
    rng = random.Random(seed)
    extreme = 0
    by_repeat = defaultdict(list)
    for row in rows:
        by_repeat[row["repeat"]].append(row)
    for _ in range(replicates):
        permuted = defaultdict(list)
        for repeat_rows in by_repeat.values():
            values = [row[field] for row in repeat_rows]
            rng.shuffle(values)
            for row, value in zip(repeat_rows, values):
                permuted[row["memory_sha"]].append(value)
        extreme += iqr([mean(values) for values in permuted.values()]) >= observed
    return (extreme + 1) / (replicates + 1)


def rank_dispersion(memory_rows, paired, cohort, field):
    selected = (
        memory_rows if cohort == "pooled"
        else [row for row in memory_rows if row["cohort"] == cohort]
    )
    selected_pairs = (
        paired if cohort == "pooled"
        else [row for row in paired if row["cohort"] == cohort]
    )
    values = [row[field] for row in selected]
    repeat_rhos = []
    for left, right in itertools.combinations(SEEDS, 2):
        ordered = sorted(selected, key=lambda row: row["memory_sha"])
        rho = spearman(
            [row[f"{field}_seed_{left}"] for row in ordered],
            [row[f"{field}_seed_{right}"] for row in ordered],
        )
        if rho is not None:
            repeat_rhos.append(rho)
    return {
        "memory_count": len(selected),
        "mean_harm": mean(values),
        "harm_iqr": iqr(values),
        "median_repeat_spearman": (
            statistics.median(repeat_rhos) if repeat_rhos else None
        ),
        "harmful_memory_count": sum(value > 0 for value in values),
        "zero_memory_count": sum(value == 0 for value in values),
        "helpful_memory_count": sum(value < 0 for value in values),
        "dispersion_permutation_p": permutation_dispersion(
            selected_pairs, field
        ),
    }


def iqr(values):
    return percentile(values, 0.75) - percentile(values, 0.25)


def partial_spearman(xs, ys, controls):
    if len(xs) < 4:
        return None
    xr, yr, cr = map(np.asarray, (ranks(xs), ranks(ys), ranks(controls)))
    design = np.column_stack([np.ones(len(cr)), cr])
    x_resid = xr - design @ (np.linalg.pinv(design) @ xr)
    y_resid = yr - design @ (np.linalg.pinv(design) @ yr)
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def commit_lookup(index_dir):
    rows = read_jsonl(Path(index_dir) / "commits.jsonl")
    result = {row["sha"]: row for row in rows}
    for row in result.values():
        patch_path = row.get("patch_path")
        if patch_path:
            path = Path(index_dir) / patch_path
            row["patch"] = path.read_text(errors="replace") if path.exists() else ""
    return result


def file_changed(repo_dir, memory_sha, base_sha, path):
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "--format=%H", f"{memory_sha}..{base_sha}", "--", path],
        check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip()), len(result.stdout.splitlines())


def evolution_rows(cells, commits, repo_dir, graph_root):
    graph_cache = {}
    rows = []
    for cell in cells:
        commit = commits[cell["memory_sha"]]
        memory_files = [normalize_path(path) for path in commit.get("changed_files") or []]
        gold = set(cell["gold_files"])
        changes = {path: file_changed(repo_dir, cell["memory_sha"], cell["base_commit"], path) for path in memory_files}
        nongold = [path for path in memory_files if path not in gold]
        production = [path for path in memory_files if path.endswith(".py") and not path.startswith(("tests/", "docs/")) and "/tests/" not in path]
        if cell["instance_id"] not in graph_cache:
            graph_path = Path(graph_root) / f"{cell['instance_id']}.pkl"
            with graph_path.open("rb") as handle:
                graph = pickle.load(handle)
            graph_cache[cell["instance_id"]] = (graph, graph_indexes(graph))
        graph, indexes = graph_cache[cell["instance_id"]]
        structure = resolve_current_structure(
            graph, [path for path in memory_files if path.endswith(".py")],
            non_dunder_symbols(commit.get("patch", "")), indexes,
        )
        fraction = lambda paths: mean([int(changes[path][0]) for path in paths])
        rows.append({
            "cell_id": cell["cell_id"],
            "instance_id": cell["instance_id"],
            "memory_sha": cell["memory_sha"],
            "cohort": cell["cohort"],
            "age_days": cell["age_days"],
            "change_fraction_all": fraction(memory_files),
            "change_fraction_nongold": fraction(nongold),
            "change_fraction_production": fraction(production),
            "intervening_file_touch_commits": sum(changes[path][1] for path in memory_files),
            "file_survival_rate": structure["file_survival_rate"],
            "symbol_survival_same_file_rate": structure["symbol_survival_same_file_rate"],
            "symbol_survival_moved_rate": structure["symbol_survival_moved_rate"],
        })
    return rows


def cohort_gate(memory_rows, cohort):
    selected = memory_rows if cohort == "pooled" else [row for row in memory_rows if row["cohort"] == cohort]
    harms = [row["harm"] for row in selected]
    repeat_rhos = []
    for left, right in itertools.combinations(SEEDS, 2):
        ordered = sorted(selected, key=lambda row: row["memory_sha"])
        rho = spearman([row[f"harm_seed_{left}"] for row in ordered], [row[f"harm_seed_{right}"] for row in ordered])
        if rho is not None:
            repeat_rhos.append(rho)
    required_harmful = 5 if cohort == "pooled" else 3
    result = {
        "memory_count": len(selected),
        "harm_iqr": iqr(harms),
        "median_repeat_spearman": statistics.median(repeat_rhos) if repeat_rhos else None,
        "memories_harm_ge_0_20": sum(value >= 0.20 for value in harms),
        "required_harmful": required_harmful,
    }
    result["passed"] = (
        result["harm_iqr"] >= 0.20
        and result["median_repeat_spearman"] is not None
        and result["median_repeat_spearman"] >= 0.50
        and result["memories_harm_ge_0_20"] >= required_harmful
    )
    return result


def final_verdict(q1_passed, q2, crowding):
    if not q1_passed or not q2["passed"]:
        return "TARGET_DRIFT_ONLY" if q1_passed and q2["target_drift_only"] else "STOP"
    return "GO_PLACEBO_CONTROL_REQUIRED" if crowding else "GO_VALIDITY_UTILITY_STUDY"


def fingerprint_balance(run_rows):
    grouped = defaultdict(Counter)
    for row in run_rows:
        grouped[(row["repeat"], row["arm"])].update(row["fingerprint_counts"])
    rows = []
    for repeat in range(1, 4):
        left = grouped[(repeat, "fixed")]
        right = grouped[(repeat, "no_memory")]
        keys = set(left) | set(right)
        left_total, right_total = sum(left.values()), sum(right.values())
        tv = 0.5 * sum(
            abs(left[key] / left_total - right[key] / right_total)
            for key in keys
        )
        rows.append({
            "repeat": repeat,
            "fixed": dict(sorted(left.items())),
            "no_memory": dict(sorted(right.items())),
            "total_variation_distance": tv,
        })
    return {
        "mixed_backend_observed": len({key for counts in grouped.values() for key in counts}) > 1,
        "per_repeat": rows,
        "max_total_variation_distance": max(row["total_variation_distance"] for row in rows),
        "imbalance_flag": any(row["total_variation_distance"] > 0.15 for row in rows),
    }


def analyze(root, manifest_path, index_dir, graph_root, repo_dir):
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text())
    run_rows = load_runs(root, graph_root)
    fixed = [row for row in run_rows if row["arm"] == "fixed"]
    controls = {(row["instance_id"], row["repeat"]): row for row in run_rows if row["arm"] == "no_memory"}
    if len(fixed) != 195 or len(controls) != 174:
        raise ValueError(f"expected 195 fixed and 174 controls, got {len(fixed)} and {len(controls)}")
    if any(row["memory_tool_calls"] for row in run_rows):
        raise ValueError("memory/semantic tool call found in a frozen pilot trajectory")
    expected_seed = {index: seed for index, seed in enumerate(SEEDS, 1)}
    bad_sampling = [
        row["cell_id"] for row in run_rows
        if row["temperature"] != 0 or row["seed"] != expected_seed[row["repeat"]]
    ]
    if bad_sampling:
        raise ValueError(f"sampling metadata mismatch: {bad_sampling[:5]}")
    fingerprints = {
        item for row in run_rows for item in row["system_fingerprints"]
    }
    models = {item for row in run_rows for item in row["response_models"]}
    if len(models) != 1:
        raise ValueError(f"provider model mismatch: models={models}")
    if any(not row["response_count"] or row["missing_response_model_count"] or row["missing_fingerprint_count"] for row in run_rows):
        raise ValueError("provider response model/fingerprint metadata is missing")
    lock_path = root / "fingerprint_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text())
    if models != {lock["response_model"]}:
        raise ValueError("trajectory response model does not match provider lock")
    payloads = defaultdict(set)
    for row in fixed:
        if not row["memory_sha"] or not row["payload_sha256"]:
            raise ValueError(f"missing fixed memory metadata for {row['cell_id']}")
        payloads[row["cell_id"]].add(row["payload_sha256"])
    if any(len(values) != 1 for values in payloads.values()) or len(payloads) != 65:
        raise ValueError("fixed memory payload hash changed across repeats")

    manifest_cells = {row["cell_id"]: row for row in manifest["cells"]}
    paired = []
    for row in fixed:
        control = controls[(row["instance_id"], row["repeat"])]
        cell = manifest_cells[row["cell_id"]]
        paired.append({
            **cell,
            "repeat": row["repeat"],
            "seed": row["seed"],
            "memory_error": row["localization_error"],
            "control_error": control["localization_error"],
            "harm": row["localization_error"] - control["localization_error"],
            "first_gold_rank_harm_at5": (
                row["first_gold_rank_at5"] - control["first_gold_rank_at5"]
            ),
            "mrr_harm_at5": control["mrr_at5"] - row["mrr_at5"],
            "ndcg_harm_at5": control["ndcg_at5"] - row["ndcg_at5"],
            "delta_acc_at5": row["acc_at5"] - control["acc_at5"],
            "delta_search_rounds": row["search_rounds"] - control["search_rounds"],
            "delta_candidate_count": row["candidate_count"] - control["candidate_count"],
            "delta_context_utilization": row["context_utilization"] - control["context_utilization"],
            "gold_files": row["gold_files"],
        })

    by_memory = defaultdict(list)
    for row in paired:
        by_memory[row["memory_sha"]].append(row)
    memory_rows = []
    for sha, rows in sorted(by_memory.items()):
        item = {
            "memory_sha": sha,
            "cohort": rows[0]["cohort"],
            "harm": mean([row["harm"] for row in rows]),
            "first_gold_rank_harm_at5": mean([
                row["first_gold_rank_harm_at5"] for row in rows
            ]),
            "mrr_harm_at5": mean([row["mrr_harm_at5"] for row in rows]),
            "ndcg_harm_at5": mean([row["ndcg_harm_at5"] for row in rows]),
            "delta_acc_at5": mean([row["delta_acc_at5"] for row in rows]),
            "task_count": len({row["instance_id"] for row in rows}),
            "age_days": mean([row["age_days"] for row in rows]),
        }
        item["harm_ci_low"], item["harm_ci_high"] = cluster_bootstrap_mean(rows, "harm")
        for repeat, seed in enumerate(SEEDS, 1):
            item[f"harm_seed_{seed}"] = mean([row["harm"] for row in rows if row["repeat"] == repeat])
            for field in (
                "first_gold_rank_harm_at5", "mrr_harm_at5", "ndcg_harm_at5"
            ):
                item[f"{field}_seed_{seed}"] = mean([
                    row[field] for row in rows if row["repeat"] == repeat
                ])
        memory_rows.append(item)

    evolution = evolution_rows(
        [{**manifest_cells[row["cell_id"]], "gold_files": row["gold_files"]} for row in paired if row["repeat"] == 1],
        commit_lookup(index_dir), repo_dir, graph_root,
    )
    evo_by_memory = defaultdict(list)
    for row in evolution:
        evo_by_memory[row["memory_sha"]].append(row)
    for row in memory_rows:
        evo = evo_by_memory[row["memory_sha"]]
        for field in ("change_fraction_all", "change_fraction_nongold", "change_fraction_production", "file_survival_rate", "symbol_survival_same_file_rate", "symbol_survival_moved_rate"):
            row[field] = mean([item[field] for item in evo])

    valid_q2 = [row for row in memory_rows if row["change_fraction_nongold"] is not None]
    q2_rho = spearman([row["harm"] for row in valid_q2], [row["change_fraction_nongold"] for row in valid_q2]) if len(valid_q2) >= 2 else None
    q2_partial = partial_spearman([row["harm"] for row in valid_q2], [row["change_fraction_nongold"] for row in valid_q2], [row["age_days"] for row in valid_q2]) if len(valid_q2) >= 4 else None
    q2_all = spearman([row["harm"] for row in memory_rows], [row["change_fraction_all"] for row in memory_rows])
    q2 = {
        "defined_memory_count": len(valid_q2),
        "spearman_harm_vs_change_nongold": q2_rho,
        "partial_spearman_controlling_age": q2_partial,
        "spearman_harm_vs_change_all": q2_all,
        "passed": len(valid_q2) >= 15 and q2_rho is not None and q2_rho >= 0.40 and q2_partial is not None and q2_partial >= 0.30,
    }
    q2["target_drift_only"] = bool(q2_all is not None and q2_all >= 0.40 and not q2["passed"])

    primary = cohort_gate(memory_rows, "top1_primary")
    pooled = cohort_gate(memory_rows, "pooled")
    primary["dispersion_permutation_p"] = permutation_dispersion(
        [row for row in paired if row["cohort"] == "top1_primary"]
    )
    pooled["dispersion_permutation_p"] = permutation_dispersion(paired)
    q1_passed = primary["passed"] and pooled["passed"]
    rank_sensitivity = {}
    rank_tests = []
    for field in (
        "first_gold_rank_harm_at5", "mrr_harm_at5", "ndcg_harm_at5"
    ):
        rank_sensitivity[field] = {}
        for cohort in ("top1_primary", "pooled"):
            result = rank_dispersion(memory_rows, paired, cohort, field)
            rank_sensitivity[field][cohort] = result
            rank_tests.append(result)
    for result in rank_tests:
        result["bonferroni_p_6"] = min(
            1.0, result["dispersion_permutation_p"] * len(rank_tests)
        )
        result["stable_dispersion"] = bool(
            result["median_repeat_spearman"] is not None
            and result["median_repeat_spearman"] >= 0.50
            and result["bonferroni_p_6"] < 0.05
        )
        result["exploratory_signal"] = bool(
            result["median_repeat_spearman"] is not None
            and result["median_repeat_spearman"] >= 0.50
            and result["dispersion_permutation_p"] < 0.05
        )
    if any(result["stable_dispersion"] for result in rank_tests):
        step0_verdict = "KEEP_EPISODIC_UTILITY_ROUTE"
    elif any(result["exploratory_signal"] for result in rank_tests):
        step0_verdict = "WEAK_TOP1_RANK_SIGNAL_CONFIRM_BEFORE_CONTINUING"
    else:
        step0_verdict = "CLOSE_EPISODIC_UTILITY_ROUTE"
    crowding = (
        mean([row["delta_context_utilization"] for row in paired]) > 0
        and statistics.median(row["delta_search_rounds"] for row in paired) <= -1
        and cluster_bootstrap_mean(paired, "delta_search_rounds")[1] < 0
    )
    verdict = final_verdict(q1_passed, q2, crowding)
    summary = {
        "provider": {
            "models": sorted(models),
            "system_fingerprints": sorted(fingerprints),
            "fingerprint_balance": fingerprint_balance(run_rows),
        },
        "q1_primary": primary,
        "q1_pooled": pooled,
        "q2": q2,
        "step0_rank_sensitivity": rank_sensitivity,
        "step0_verdict": step0_verdict,
        "crowding": {
            "flag": crowding,
            "delta_search_rounds_cluster_bootstrap_ci": cluster_bootstrap_mean(paired, "delta_search_rounds"),
        },
        "verdict": verdict,
    }

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "per_run.csv", run_rows)
    write_csv(analysis_dir / "per_task_memory.csv", paired)
    write_csv(analysis_dir / "per_memory.csv", memory_rows)
    write_csv(analysis_dir / "repository_evolution.csv", evolution)
    write_json(analysis_dir / "summary.json", summary)
    report = f"""# Stale-Memory Causal Harm Pilot\n\n## Material Passport\n\n- Origin: RepoMem stale-memory causal pilot\n- Verification Status: ANALYZED\n- Fixed runs: {len(fixed)}\n- No-memory runs: {len(controls)}\n\n## Decision\n\n**{verdict}**\n\n- Q1 Top-1: `{primary}`\n- Q1 pooled: `{pooled}`\n- Q2: `{q2}`\n- Context crowding flag: `{crowding}`\n\n## Step 0: Rank Sensitivity (Post-hoc)\n\nRanks are capped at 5; a miss is assigned rank 6. Positive values mean fixed memory is worse than no memory. Six dispersion tests are Bonferroni-corrected.\n\n**{step0_verdict}**\n\n- Rank sensitivity: `{rank_sensitivity}`\n"""
    (analysis_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--audit", default=DEFAULT_AUDIT)
    prep.add_argument("--output", default=DEFAULT_ROOT)
    prep.add_argument("--seed", type=int, default=42)
    flight = sub.add_parser("preflight")
    flight.add_argument("--lock", default=DEFAULT_ROOT / "fingerprint_lock.json")
    flight.add_argument("--model", default=os.environ.get("EVAL_MODEL", "gpt-4o"))
    flight.add_argument("--seed", type=int, default=42)
    check = sub.add_parser("validate-run")
    check.add_argument("--run-dir", required=True)
    check.add_argument("--arm", choices=("fixed", "no_memory"), required=True)
    check.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    check.add_argument("--expected", type=int, required=True)
    check.add_argument("--lock", default=DEFAULT_ROOT / "fingerprint_lock.json")
    retry = sub.add_parser("validate-retry")
    retry.add_argument("--run-dir", required=True)
    retry.add_argument("--arm", choices=("fixed", "no_memory"), required=True)
    retry.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    retry.add_argument("--expected", type=int, required=True)
    retry.add_argument("--lock", default=DEFAULT_ROOT / "fingerprint_lock.json")
    ana = sub.add_parser("analyze")
    ana.add_argument("--root", default=DEFAULT_ROOT)
    ana.add_argument("--manifest", default=DEFAULT_ROOT / "pilot_manifest.json")
    ana.add_argument("--index", default=DEFAULT_INDEX)
    ana.add_argument("--graph", default=DEFAULT_GRAPH)
    ana.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.audit, args.output, args.seed)
    elif args.command == "preflight":
        preflight(args.lock, args.model, args.seed)
    elif args.command == "validate-run":
        validate_run(args.run_dir, args.arm, args.repeat, args.expected, args.lock)
    elif args.command == "validate-retry":
        validate_retryable_run(
            args.run_dir, args.arm, args.repeat, args.expected, args.lock
        )
    else:
        analyze(args.root, args.manifest, args.index, args.graph, args.repo)


if __name__ == "__main__":
    main()
