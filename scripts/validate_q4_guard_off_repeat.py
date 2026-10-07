import argparse
import json
import os


GUARD_ARGUMENTS = {
    "context_token_budget": 0,
    "tool_observation_token_cap": 0,
    "fail_on_context_error": False,
    "bounded_graph_output": False,
    "bounded_entity_output": False,
    "bounded_search_output": False,
}
MATCHED_ARGUMENTS = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "split": "test",
    "used_list_file": "reproduce_lists/verified_m3a_matched_15.json",
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
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_run(run_dir, expected_cases=15):
    args_path = os.path.join(run_dir, "args.json")
    with open(args_path, encoding="utf-8") as f:
        run_args = json.load(f)

    expected_arguments = {**MATCHED_ARGUMENTS, **GUARD_ARGUMENTS}
    mismatches = {
        key: {"expected": expected, "actual": run_args.get(key)}
        for key, expected in expected_arguments.items()
        if run_args.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{run_dir}: guard-off mismatch: {mismatches}")
    if run_args.get("use_repomem"):
        raise ValueError(f"{run_dir}: RepoMem must be disabled")

    rows = load_jsonl(os.path.join(run_dir, "loc_trajs.jsonl"))
    if len(rows) != expected_cases:
        raise ValueError(
            f"{run_dir}: expected {expected_cases} trajectories, got {len(rows)}"
        )
    case_ids = [row.get("instance_id") for row in rows]
    if len(set(case_ids)) != expected_cases:
        raise ValueError(f"{run_dir}: duplicate instance IDs")

    context_errors_path = os.path.join(run_dir, "context_errors.jsonl")
    context_errors = (
        load_jsonl(context_errors_path) if os.path.exists(context_errors_path) else []
    )
    if context_errors:
        raise ValueError(f"{run_dir}: found {len(context_errors)} context errors")

    for row in rows:
        trajectories = row.get("loc_trajs", {}).get("trajs", [])
        if len(trajectories) != 1:
            raise ValueError(
                f"{run_dir}/{row.get('instance_id')}: expected one trajectory, "
                f"got {len(trajectories)}"
            )
        guard = trajectories[0].get("context_guard", {})
        if guard.get("enabled"):
            raise ValueError(f"{run_dir}/{row.get('instance_id')}: guard enabled")
        for key, expected in GUARD_ARGUMENTS.items():
            if guard.get(key, expected) != expected:
                raise ValueError(
                    f"{run_dir}/{row.get('instance_id')}: {key}={guard.get(key)}"
                )
        if guard.get("events"):
            raise ValueError(
                f"{run_dir}/{row.get('instance_id')}: unexpected guard audit events"
            )

    audit_path = os.path.join(run_dir, "context_guard_audit.jsonl")
    if os.path.exists(audit_path) and load_jsonl(audit_path):
        raise ValueError(f"{run_dir}: guard audit rows exist in guard-off run")

    return {
        "run_dir": run_dir,
        "trajectories": len(rows),
        "context_errors": 0,
        "guard_enabled": False,
        "bounded_tool_policy": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--expected_cases", type=int, default=15)
    args = parser.parse_args()
    print(json.dumps(validate_run(args.run_dir, args.expected_cases), indent=2))


if __name__ == "__main__":
    main()
