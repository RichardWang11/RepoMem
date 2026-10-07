import argparse
import json
import os
from datetime import datetime, timezone


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def inspect_repeat(run_dir, expected_cases):
    rows = load_jsonl(os.path.join(run_dir, "loc_trajs.jsonl"))
    errors = load_jsonl(os.path.join(run_dir, "context_errors.jsonl"))
    repair_path = os.path.join(run_dir, "parser_repair.log")
    repair_log = ""
    if os.path.exists(repair_path):
        with open(repair_path, encoding="utf-8") as f:
            repair_log = f.read().strip()
    return {
        "run_dir": run_dir,
        "exists": os.path.isdir(run_dir),
        "trajectory_count": len(rows),
        "expected_trajectories": expected_cases,
        "context_error_count": len(errors),
        "context_errors": [
            {
                "instance_id": row.get("instance_id"),
                "type": row.get("type"),
                "error": row.get("error"),
            }
            for row in errors
        ],
        "parser_repair_log": repair_log,
        "valid": len(rows) == expected_cases and not errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="results/context_safe_m3a/q4_guard_off_baseline"
    )
    parser.add_argument("--expected_cases", type=int, default=15)
    args = parser.parse_args()

    repeats = []
    for repeat in range(1, 5):
        repeats.append(
            inspect_repeat(
                os.path.join(args.root, "runs", f"baseline_r{repeat}"),
                args.expected_cases,
            )
        )
    complete = all(row["valid"] for row in repeats)
    status = {
        "experiment": "Q4 guard-off baseline sequential replication",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_trajectories": 60,
        "completed_trajectories": sum(row["trajectory_count"] for row in repeats),
        "valid_repeats": sum(row["valid"] for row in repeats),
        "complete": complete,
        "verdict": "READY_FOR_ANALYSIS" if complete else "INFRASTRUCTURE_INVALID",
        "automatic_replacement_performed": False,
        "repeats": repeats,
    }

    os.makedirs(args.root, exist_ok=True)
    with open(
        os.path.join(args.root, "experiment_status.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(status, f, indent=2)
    with open(
        os.path.join(args.root, "experiment_status.md"), "w", encoding="utf-8"
    ) as f:
        f.write("# Q4 Guard-Off Baseline Replication Status\n\n")
        f.write(f"- Planned trajectories: 60\n")
        f.write(f"- Completed trajectories: {status['completed_trajectories']}\n")
        f.write(f"- Valid repeats: {status['valid_repeats']}/4\n")
        f.write(f"- Verdict: **{status['verdict']}**\n")
        f.write("- Automatic replacement performed: no\n\n")
        f.write("| Repeat | Trajectories | Context errors | Valid |\n")
        f.write("|---:|---:|---:|---:|\n")
        for index, row in enumerate(repeats, 1):
            f.write(
                f"| {index} | {row['trajectory_count']}/{args.expected_cases} | "
                f"{row['context_error_count']} | {str(row['valid']).lower()} |\n"
            )
        errors = [error for row in repeats for error in row["context_errors"]]
        if errors:
            f.write("\n## Blocking Infrastructure Events\n\n")
            for error in errors:
                f.write(
                    f"- `{error['instance_id']}`: `{error['type']}`; "
                    f"{error['error']}\n"
                )
        f.write("\nNo Q4 stability estimate is produced unless all four repeats pass validation.\n")
    print(json.dumps(status, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
