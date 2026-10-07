import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from scripts.validate_context_safe_repeat import load_jsonl
from scripts.validate_replication_v3_repeat import EXPECTED_ARGS


EXPECTED_MODEL = "gpt-4o-2024-11-20"
EXPECTED_FINGERPRINT = "fp_40854b0bd2"


def parse_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_preflight(path, max_age_seconds=1800, now=None):
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    now = now or datetime.now(timezone.utc)
    age = (now - parse_timestamp(row["timestamp_utc"])).total_seconds()
    failures = []
    if row.get("verdict") != "CURRENT_MATCH":
        failures.append(f"verdict={row.get('verdict')}")
    response = row.get("response") or {}
    if response.get("response_model") != EXPECTED_MODEL:
        failures.append(f"response_model={response.get('response_model')}")
    if response.get("system_fingerprint") != EXPECTED_FINGERPRINT:
        failures.append(
            f"system_fingerprint={response.get('system_fingerprint')}"
        )
    if not row.get("endpoint_hash_matches_v3"):
        failures.append("endpoint hash does not match V3")
    if age < 0 or age > max_age_seconds:
        failures.append(f"preflight age={age:.1f}s")
    result = {
        "valid": not failures,
        "age_seconds": age,
        "max_age_seconds": max_age_seconds,
        "response_model": response.get("response_model"),
        "system_fingerprint": response.get("system_fingerprint"),
        "failures": failures,
    }
    if failures:
        raise ValueError(f"invalid V4 preflight: {failures}")
    return result


def audit_trajectory(instance_id, trajectory, arm, repeat):
    provider = trajectory.get("provider_metadata") or {}
    responses = provider.get("responses") or []
    rows = []
    for response_index, response in enumerate(responses, 1):
        model = response.get("model")
        fingerprint = response.get("system_fingerprint")
        rows.append({
            "arm": arm,
            "repeat": repeat,
            "instance_id": instance_id,
            "response_index": response_index,
            "response_model": model,
            "system_fingerprint": fingerprint,
            "model_matches": model == EXPECTED_MODEL,
            "fingerprint_present": bool(fingerprint),
            "fingerprint_matches": fingerprint == EXPECTED_FINGERPRINT,
        })
    return rows


def validate_run(run_dir, arm, repeat, expected_cases=15):
    run_dir = Path(run_dir)
    run_args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    hard_failures = {
        key: {"expected": expected, "actual": run_args.get(key)}
        for key, expected in EXPECTED_ARGS.items()
        if run_args.get(key) != expected
    }
    expected_memory = arm == "memory"
    if bool(run_args.get("use_repomem")) != expected_memory:
        hard_failures["use_repomem"] = {
            "expected": expected_memory,
            "actual": run_args.get("use_repomem"),
        }
    if expected_memory and run_args.get("repomem_mode") != "episodic":
        hard_failures["repomem_mode"] = {
            "expected": "episodic",
            "actual": run_args.get("repomem_mode"),
        }

    records = load_jsonl(run_dir / "loc_trajs.jsonl")
    if len(records) != expected_cases:
        hard_failures["trajectory_count"] = {
            "expected": expected_cases,
            "actual": len(records),
        }
    error_path = run_dir / "context_errors.jsonl"
    errors = load_jsonl(error_path) if error_path.exists() else []
    if errors:
        hard_failures["context_errors"] = len(errors)

    response_rows = []
    trajectories_without_responses = []
    for record in records:
        trajectories = record.get("loc_trajs", {}).get("trajs", [])
        if len(trajectories) != 1:
            hard_failures[f"trajectory_samples:{record.get('instance_id')}"] = {
                "expected": 1,
                "actual": len(trajectories),
            }
            continue
        rows = audit_trajectory(
            record["instance_id"], trajectories[0], arm, repeat
        )
        if not rows:
            trajectories_without_responses.append(record["instance_id"])
        response_rows.extend(rows)

    fingerprint_values = sorted({
        row["system_fingerprint"]
        for row in response_rows
        if row["system_fingerprint"]
    })
    model_values = sorted({
        row["response_model"] for row in response_rows if row["response_model"]
    })
    mismatch_count = sum(not row["fingerprint_matches"] for row in response_rows)
    missing_count = sum(not row["fingerprint_present"] for row in response_rows)
    model_mismatch_count = sum(not row["model_matches"] for row in response_rows)
    if trajectories_without_responses:
        hard_failures["trajectories_without_responses"] = (
            trajectories_without_responses
        )
    same_fingerprint_eligible = bool(response_rows) and not (
        mismatch_count
        or missing_count
        or model_mismatch_count
        or trajectories_without_responses
    )
    summary = {
        "run_dir": str(run_dir),
        "arm": arm,
        "repeat": repeat,
        "trajectory_count": len(records),
        "response_count": len(response_rows),
        "response_models": model_values,
        "system_fingerprints": fingerprint_values,
        "fingerprint_mismatch_count": mismatch_count,
        "fingerprint_missing_count": missing_count,
        "model_mismatch_count": model_mismatch_count,
        "trajectories_without_responses": trajectories_without_responses,
        "same_fingerprint_eligible": same_fingerprint_eligible,
        "soft_status": (
            "EXPECTED_FINGERPRINT"
            if same_fingerprint_eligible
            else "FINGERPRINT_DRIFT_OR_MISSING"
        ),
        "hard_failures": hard_failures,
        "runtime_valid": not hard_failures,
    }
    return summary, response_rows


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight")
    mode.add_argument("--run_dir")
    parser.add_argument("--max_age_seconds", type=int, default=1800)
    parser.add_argument("--arm", choices=("baseline", "memory"))
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--expected_cases", type=int, default=15)
    args = parser.parse_args()

    if args.preflight:
        result = validate_preflight(args.preflight, args.max_age_seconds)
        print(json.dumps(result, indent=2))
        return
    if args.arm is None or args.repeat is None:
        parser.error("--arm and --repeat are required with --run_dir")
    summary, response_rows = validate_run(
        args.run_dir, args.arm, args.repeat, args.expected_cases
    )
    write_jsonl(Path(args.run_dir) / "fingerprint_audit.jsonl", response_rows)
    (Path(args.run_dir) / "fingerprint_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if summary["hard_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
