import argparse
import json
import os

from scripts.validate_context_safe_repeat import load_jsonl


EXPECTED_ARGS = {
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
    "temperature": 1.0,
    "top_p": None,
    "context_token_budget": 100000,
    "tool_observation_token_cap": 20000,
    "fail_on_context_error": True,
    "bounded_graph_output": True,
    "bounded_entity_output": True,
    "bounded_search_output": True,
}


def validate_repeat(run_dir, arm, expected_cases=15):
    with open(os.path.join(run_dir, "args.json"), encoding="utf-8") as f:
        run_args = json.load(f)
    mismatches = {
        key: {"expected": expected, "actual": run_args.get(key)}
        for key, expected in EXPECTED_ARGS.items()
        if run_args.get(key) != expected
    }
    expected_memory = arm == "memory"
    if bool(run_args.get("use_repomem")) != expected_memory:
        mismatches["use_repomem"] = {
            "expected": expected_memory,
            "actual": run_args.get("use_repomem"),
        }
    if expected_memory and run_args.get("repomem_mode") != "episodic":
        mismatches["repomem_mode"] = {
            "expected": "episodic",
            "actual": run_args.get("repomem_mode"),
        }
    if mismatches:
        raise ValueError(f"{run_dir}: argument mismatch: {mismatches}")

    rows = load_jsonl(os.path.join(run_dir, "loc_trajs.jsonl"))
    if len(rows) != expected_cases:
        raise ValueError(
            f"{run_dir}: expected {expected_cases} trajectories, got {len(rows)}"
        )
    error_path = os.path.join(run_dir, "context_errors.jsonl")
    errors = load_jsonl(error_path) if os.path.exists(error_path) else []
    if errors:
        raise ValueError(f"{run_dir}: found {len(errors)} context errors")

    response_models = set()
    fingerprints = set()
    for row in rows:
        trajectories = row.get("loc_trajs", {}).get("trajs", [])
        if len(trajectories) != 1:
            raise ValueError(
                f"{run_dir}/{row['instance_id']}: expected one trajectory"
            )
        trajectory = trajectories[0]
        provider = trajectory.get("provider_metadata", {})
        if provider.get("requested_model") != "gpt-4o":
            raise ValueError(
                f"{run_dir}/{row['instance_id']}: missing requested model metadata"
            )
        if provider.get("temperature") != 1.0 or provider.get("top_p") is not None:
            raise ValueError(
                f"{run_dir}/{row['instance_id']}: sampling metadata mismatch"
            )
        for response in provider.get("responses", []):
            if response.get("model"):
                response_models.add(response["model"])
            if response.get("system_fingerprint"):
                fingerprints.add(response["system_fingerprint"])

    return {
        "run_dir": run_dir,
        "arm": arm,
        "trajectories": len(rows),
        "response_models": sorted(response_models),
        "system_fingerprints": sorted(fingerprints),
        "valid": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--arm", choices=("baseline", "memory"), required=True)
    parser.add_argument("--expected_cases", type=int, default=15)
    args = parser.parse_args()
    result = validate_repeat(args.run_dir, args.arm, args.expected_cases)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
