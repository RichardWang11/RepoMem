#!/usr/bin/env python3
"""Analyze the preregistered legacy-vs-strict visibility pilot."""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from scripts.eval_verified_file_acc import files_from_patch, get_pred_files


KS = (1, 3, 5)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_instances(dataset, split):
    path = Path(dataset)
    if path.is_file():
        return load_jsonl(path)
    return load_dataset(dataset, split=split)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def exact_mcnemar(left, right):
    """Return the two-sided exact McNemar result for paired binary outcomes."""
    if len(left) != len(right):
        raise ValueError("paired outcomes must have equal lengths")
    legacy_only = sum(a and not b for a, b in zip(left, right))
    strict_only = sum(b and not a for a, b in zip(left, right))
    discordant = legacy_only + strict_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(legacy_only, strict_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "legacy_only": legacy_only,
        "strict_only": strict_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def bootstrap_delta(left, right, repetitions=10000, seed=42):
    """Bootstrap a percentile CI for strict-minus-legacy paired accuracy."""
    if len(left) != len(right):
        raise ValueError("paired outcomes must have equal lengths")
    differences = [float(b) - float(a) for a, b in zip(left, right)]
    if not differences:
        return [None, None]
    generator = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sample = [generator.choice(differences) for _ in differences]
        values.append(sum(sample) / len(sample))
    values.sort()
    low = values[int(0.025 * (repetitions - 1))]
    high = values[int(0.975 * (repetitions - 1))]
    return [low, high]


def parse_run_name(name):
    parts = name.split("_", 2)
    if len(parts) != 3 or parts[0] not in {"random", "sensitivity"}:
        return None
    cohort, position, remainder = parts
    for policy in ("legacy", "strict"):
        suffix = f"_{policy}"
        if remainder.endswith(suffix):
            return cohort, int(position), remainder[: -len(suffix)], policy
    return None


def load_run(run_dir):
    merged_path = run_dir / "merged_loc_outputs_mrr.jsonl"
    trajectory_path = run_dir / "loc_trajs.jsonl"
    if not merged_path.exists() or not trajectory_path.exists():
        return None
    merged = load_jsonl(merged_path)
    trajectories = load_jsonl(trajectory_path)
    if len(merged) != 1 or len(trajectories) != 1:
        raise ValueError(f"expected one task in {run_dir}")
    response_models = set()
    fingerprints = set()
    visibility = []
    for trajectory in trajectories[0].get("loc_trajs", {}).get("trajs", []):
        visibility.append(trajectory.get("episodic_visibility") or {})
        provider = trajectory.get("provider_metadata") or {}
        for response in provider.get("responses", []):
            if response.get("model"):
                response_models.add(response["model"])
            if response.get("system_fingerprint"):
                fingerprints.add(response["system_fingerprint"])
    usage = trajectories[0].get("usage") or {}
    cost_usd = float(usage.get("cost($)", 0) or 0)
    if cost_usd == 0:
        cost_usd = (
            int(usage.get("prompt_tokens", 0) or 0) * 0.000005
            + int(usage.get("completion_tokens", 0) or 0) * 0.000015
        )
    return {
        "instance_id": merged[0]["instance_id"],
        "predictions": get_pred_files(merged[0]),
        "cost_usd": cost_usd,
        "response_models": sorted(response_models),
        "system_fingerprints": sorted(fingerprints),
        "visibility": visibility,
        "samples": len(trajectories[0].get("loc_trajs", {}).get("trajs", [])),
    }


def summarize(rows, cohort):
    selected = [row for row in rows if row["cohort"] == cohort]
    result = {"cases": len(selected), "metrics": {}}
    for k in KS:
        legacy = [row["legacy"][f"ok@{k}"] for row in selected]
        strict = [row["strict"][f"ok@{k}"] for row in selected]
        legacy_acc = sum(legacy) / len(legacy) if legacy else None
        strict_acc = sum(strict) / len(strict) if strict else None
        result["metrics"][f"complete_file_acc_at{k}"] = {
            "legacy": legacy_acc,
            "strict": strict_acc,
            "strict_minus_legacy": (
                strict_acc - legacy_acc if legacy_acc is not None else None
            ),
            "bootstrap_95pct_ci": bootstrap_delta(legacy, strict),
            "mcnemar": exact_mcnemar(legacy, strict),
        }
    return result


def format_number(value, signed=False):
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/strict_visibility_pilot")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Analyze completed paired cases as a clearly labeled preliminary result.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    prereg = json.loads((root / "preregistration.json").read_text(encoding="utf-8"))
    amendment_path = root / "preregistration_amendment.json"
    amendment = (
        json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment_path.exists() else {}
    )
    required_response_model = amendment.get(
        "effective_required_response_model", prereg["required_response_model"]
    )
    dataset_path = Path(args.dataset)
    if prereg.get("dataset_sha256") and (
        not dataset_path.is_file()
        or hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        != prereg["dataset_sha256"]
    ):
        raise RuntimeError("dataset differs from the frozen preregistration")
    planned = {
        (cohort, instance_id, policy)
        for cohort, key in (("random", "random_ids"), ("sensitivity", "sensitivity_ids"))
        for instance_id in prereg[key]
        for policy in ("legacy", "strict")
    }
    runs = {}
    for run_dir in sorted((root / "runs").glob("*")):
        parsed = parse_run_name(run_dir.name)
        if not parsed:
            continue
        cohort, position, safe_id, policy = parsed
        run = load_run(run_dir)
        if run is None:
            continue
        key = (cohort, run["instance_id"], policy)
        if key in runs:
            raise ValueError(f"duplicate completed run: {key}")
        runs[key] = {**run, "position": position, "safe_id": safe_id}

    missing = sorted(planned - set(runs))
    unexpected = sorted(set(runs) - planned)
    incomplete_pairs = sorted({
        (cohort, instance_id)
        for cohort, instance_id, _ in planned
        if ((cohort, instance_id, "legacy") in runs)
        != ((cohort, instance_id, "strict") in runs)
    })
    if unexpected or incomplete_pairs or (missing and not args.allow_partial):
        raise RuntimeError(
            f"pilot is incomplete or contains unexpected runs: "
            f"missing={missing}, unexpected={unexpected}, "
            f"incomplete_pairs={incomplete_pairs}"
        )

    instances = {row["instance_id"]: row for row in load_instances(args.dataset, args.split)}
    per_case = []
    response_models = set()
    fingerprints = set()
    invariant_failures = []
    total_cost = 0.0
    for cohort, key in (("random", "random_ids"), ("sensitivity", "sensitivity_ids")):
        for instance_id in prereg[key]:
            if (cohort, instance_id, "legacy") not in runs:
                continue
            gold = files_from_patch(instances[instance_id]["patch"], py_only=True)
            if not gold:
                raise ValueError(f"no Python gold files for {instance_id}")
            record = {"cohort": cohort, "instance_id": instance_id, "gold_files": gold}
            for policy in ("legacy", "strict"):
                run = runs[(cohort, instance_id, policy)]
                if run["samples"] != prereg["samples_per_case_arm"]:
                    invariant_failures.append(
                        f"{instance_id}/{policy}: samples={run['samples']}"
                    )
                response_models.update(run["response_models"])
                fingerprints.update(run["system_fingerprints"])
                total_cost += run["cost_usd"]
                for sample_visibility in run["visibility"]:
                    if sample_visibility.get("policy") != policy:
                        invariant_failures.append(
                            f"{instance_id}/{policy}: visibility policy mismatch"
                        )
                    if policy == "strict" and sample_visibility.get("status") != "available":
                        invariant_failures.append(
                            f"{instance_id}/strict: visibility unavailable"
                        )
                predictions = run["predictions"]
                arm = {
                    "predicted_files": predictions[:5],
                    "cost_usd": run["cost_usd"],
                    "response_models": run["response_models"],
                    "system_fingerprints": run["system_fingerprints"],
                }
                for k in KS:
                    arm[f"ok@{k}"] = set(gold).issubset(predictions[:k])
                record[policy] = arm
            record["flips"] = {
                f"at{k}": (
                    "legacy_to_strict_win"
                    if not record["legacy"][f"ok@{k}"] and record["strict"][f"ok@{k}"]
                    else "strict_to_legacy_win"
                    if record["legacy"][f"ok@{k}"] and not record["strict"][f"ok@{k}"]
                    else "unchanged"
                )
                for k in KS
            }
            per_case.append(record)

    if response_models != {required_response_model}:
        invariant_failures.append(f"response_models={sorted(response_models)}")
    excluded_recovery_cost = 0.0
    rejected_run_count = 0
    rejected_root = root / "rejected_runs"
    if rejected_root.is_dir():
        for run_dir in sorted(rejected_root.iterdir()):
            if not run_dir.is_dir():
                continue
            rejected_run = load_run(run_dir)
            if rejected_run is not None:
                excluded_recovery_cost += rejected_run["cost_usd"]
                rejected_run_count += 1
    unpriced_context_errors = sum(
        len(load_jsonl(path))
        for path in rejected_root.glob("*/context_errors.jsonl")
    ) if rejected_root.is_dir() else 0
    known_recorded_cost = total_cost + excluded_recovery_cost
    if known_recorded_cost > float(prereg["budget_cap_usd"]):
        invariant_failures.append(
            f"known recorded cost ${known_recorded_cost:.5f} exceeds cap "
            f"${prereg['budget_cap_usd']:.2f}"
        )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PRELIMINARY_PARTIAL" if missing else "COMPLETE",
        "planned_arm_runs": len(planned),
        "completed_arm_runs": len(runs),
        "missing_arm_runs": len(missing),
        "metric_definition": "all Python gold patch files occur in merged prediction top-k",
        "random": summarize(per_case, "random"),
        "sensitivity": summarize(per_case, "sensitivity"),
        "provider": {
            "required_response_model": required_response_model,
            "observed_response_models": sorted(response_models),
            "system_fingerprints": sorted(fingerprints),
            "preregistration_amendment": amendment or None,
        },
        "cost_usd": total_cost,
        "recorded_excluded_recovery_cost_usd": excluded_recovery_cost,
        "known_recorded_api_cost_usd": known_recorded_cost,
        "rejected_run_count": rejected_run_count,
        "unpriced_context_error_count": unpriced_context_errors,
        "budget_cap_usd": float(prereg["budget_cap_usd"]),
        "invariant_failures": invariant_failures,
        "paper_reference_non_comparable": {
            "LocAgent_file_acc_at1": 0.678,
            "LocAgent_file_acc_at3": 0.724,
            "LocAgent_file_acc_at5": 0.743,
            "reason": "paper values are full-benchmark context, not this paired pilot",
        },
        "known_deviations_from_paper": [
            "Model is gpt-4o-2024-11-20; paper used gpt-4o-2024-05-13.",
            "bm25-code-v2 is a regex/code-token splitter; paper used an LLM-based tokenizer.",
            "Current tool profile caps SearchCommit at top-5; the paper interface uses top_k=20.",
            "This is a small paired pilot, not the paper's full-benchmark run.",
        ],
    }
    with (root / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_case:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(root / "summary.json", summary)
    lines = [
        "# Strict visibility pilot\n",
        f"- Completed paired cases: {len(per_case)}",
        f"- Included-arm API cost: ${total_cost:.5f}",
        f"- Known recorded API cost including rejected recovery runs: "
        f"${known_recorded_cost:.5f} / ${prereg['budget_cap_usd']:.2f}",
        f"- Rejected recovery runs: {rejected_run_count}; unpriced context errors: "
        f"{unpriced_context_errors}",
        f"- Response model(s): {', '.join(sorted(response_models))}",
        f"- Invariant failures: {len(invariant_failures)}\n",
    ]
    for cohort in ("random", "sensitivity"):
        lines.append(f"## {cohort.capitalize()} cohort\n")
        lines.append("| Metric | Legacy | Strict | Strict − legacy | 95% bootstrap CI | McNemar p |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for metric, value in summary[cohort]["metrics"].items():
            ci = value["bootstrap_95pct_ci"]
            lines.append(
                f"| {metric} | {format_number(value['legacy'])} | "
                f"{format_number(value['strict'])} | "
                f"{format_number(value['strict_minus_legacy'], signed=True)} | "
                f"[{format_number(ci[0], signed=True)}, "
                f"{format_number(ci[1], signed=True)}] | "
                f"{value['mcnemar']['two_sided_exact_p']:.4f} |"
            )
        lines.append("")
    lines.append(
        "Paper LocAgent values (67.8/72.4/74.3 at k=1/3/5) are context only and "
        "are not directly comparable to this small paired pilot.\n"
    )
    lines.append("## Known deviations from the paper\n")
    for deviation in summary["known_deviations_from_paper"]:
        lines.append(f"- {deviation}")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if invariant_failures:
        raise RuntimeError(f"pilot invariants failed: {invariant_failures}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
