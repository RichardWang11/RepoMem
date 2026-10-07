import argparse
import json
from pathlib import Path

from scripts.sample_random_verified_pilot import load_ids
from scripts.validate_context_safe_repeat import load_jsonl
from scripts.validate_replication_v4 import (
    EXPECTED_FINGERPRINT,
    EXPECTED_MODEL,
    audit_trajectory,
    write_jsonl,
)


EXPECTED_ARGS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "split": "test",
    "model": "gpt-4o",
    "use_function_calling": True,
    "simple_desc": True,
    "max_attempt_num": 1,
    "num_samples": 1,
    "num_processes": 4,
    "ranking_method": "mrr",
    "timeout": 240,
    "request_timeout": 60,
    "max_iteration_num": 8,
    "temperature": 1.0,
    "top_p": None,
    "context_token_budget": 100000,
    "tool_observation_token_cap": 20000,
    "fail_on_context_error": True,
    "bounded_graph_output": True,
    "bounded_entity_output": True,
    "bounded_search_output": True,
}


def validate_batch(run_dir, arm, repeat, expected_ids):
    run_dir = Path(run_dir)
    run_args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    failures = {
        key: {"expected": expected, "actual": run_args.get(key)}
        for key, expected in EXPECTED_ARGS.items()
        if run_args.get(key) != expected
    }
    if run_args.get("used_list_file") is None:
        failures["used_list_file"] = "missing"
    expected_memory = arm == "memory"
    if bool(run_args.get("use_repomem")) != expected_memory:
        failures["use_repomem"] = {
            "expected": expected_memory,
            "actual": run_args.get("use_repomem"),
        }
    if expected_memory and run_args.get("repomem_mode") != "episodic":
        failures["repomem_mode"] = {
            "expected": "episodic",
            "actual": run_args.get("repomem_mode"),
        }

    records = load_jsonl(run_dir / "loc_trajs.jsonl")
    actual_ids = [row.get("instance_id") for row in records]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        failures["trajectory_ids"] = {
            "expected": sorted(expected_ids),
            "actual": sorted(actual_ids),
        }
    if len(actual_ids) != len(set(actual_ids)):
        failures["duplicate_trajectories"] = actual_ids

    repaired_path = run_dir / "loc_trajs.repaired.jsonl"
    repaired = load_jsonl(repaired_path) if repaired_path.exists() else []
    repaired_ids = [row.get("instance_id") for row in repaired]
    if set(repaired_ids) != set(expected_ids) or len(repaired_ids) != len(expected_ids):
        failures["repaired_trajectory_ids"] = {
            "expected": sorted(expected_ids),
            "actual": sorted(repaired_ids),
        }

    context_errors_path = run_dir / "context_errors.jsonl"
    context_errors = (
        load_jsonl(context_errors_path) if context_errors_path.exists() else []
    )
    if context_errors:
        failures["context_errors"] = len(context_errors)

    audits = load_jsonl(run_dir / "context_guard_audit.jsonl")
    if len(audits) != len(expected_ids):
        failures["context_audit_count"] = {
            "expected": len(expected_ids),
            "actual": len(audits),
        }
    truncated = [
        row.get("instance_id")
        for row in audits
        if row.get("summary", {}).get("truncated_observation_count")
    ]
    if truncated:
        failures["fallback_context_truncations"] = truncated
    bounded_missing = [
        row.get("instance_id")
        for row in audits
        if not all(
            row.get(key)
            for key in (
                "bounded_graph_output",
                "bounded_entity_output",
                "bounded_search_output",
            )
        )
    ]
    if bounded_missing:
        failures["bounded_policy_missing"] = bounded_missing

    response_rows = []
    no_responses = []
    for record in records:
        trajectories = record.get("loc_trajs", {}).get("trajs", [])
        if len(trajectories) != 1:
            failures[f"trajectory_samples:{record.get('instance_id')}"] = {
                "expected": 1,
                "actual": len(trajectories),
            }
            continue
        rows = audit_trajectory(record["instance_id"], trajectories[0], arm, repeat)
        if not rows:
            no_responses.append(record["instance_id"])
        response_rows.extend(rows)
    if no_responses:
        failures["trajectories_without_responses"] = no_responses

    mismatch_count = sum(
        row["system_fingerprint"] != EXPECTED_FINGERPRINT
        for row in response_rows
    )
    model_mismatch_count = sum(
        row["response_model"] != EXPECTED_MODEL for row in response_rows
    )
    if mismatch_count:
        failures["fingerprint_mismatch_count"] = mismatch_count
    if model_mismatch_count:
        failures["model_mismatch_count"] = model_mismatch_count

    summary = {
        "run_dir": str(run_dir),
        "arm": arm,
        "repeat": repeat,
        "expected_cases": len(expected_ids),
        "trajectory_count": len(records),
        "response_count": len(response_rows),
        "expected_model": EXPECTED_MODEL,
        "expected_fingerprint": EXPECTED_FINGERPRINT,
        "response_models": sorted({row["response_model"] for row in response_rows}),
        "system_fingerprints": sorted({
            row["system_fingerprint"] for row in response_rows
        }),
        "fingerprint_mismatch_count": mismatch_count,
        "model_mismatch_count": model_mismatch_count,
        "failures": failures,
        "valid": not failures,
    }
    write_jsonl(run_dir / "fingerprint_audit.jsonl", response_rows)
    (run_dir / "fingerprint_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--arm", choices=("baseline", "memory"), required=True)
    parser.add_argument("--repeat", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--expected_list", required=True)
    args = parser.parse_args()
    summary = validate_batch(
        args.run_dir, args.arm, args.repeat, load_ids(args.expected_list)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid"]:
        fingerprint_failure = any(
            key in summary["failures"]
            for key in ("fingerprint_mismatch_count", "model_mismatch_count")
        )
        raise SystemExit(2 if fingerprint_failure else 1)


if __name__ == "__main__":
    main()
