import argparse
import json
import os


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--expected_cases", type=int, default=15)
    args = parser.parse_args()

    audit_path = os.path.join(args.run_dir, "context_guard_audit.jsonl")
    rows = load_jsonl(audit_path)
    if len(rows) != args.expected_cases:
        raise ValueError(
            f"{args.run_dir}: expected {args.expected_cases} audits, got {len(rows)}"
        )
    error_path = os.path.join(args.run_dir, "context_errors.jsonl")
    errors = load_jsonl(error_path) if os.path.exists(error_path) else []
    if errors:
        raise ValueError(f"{args.run_dir}: found {len(errors)} context errors")
    truncated = [
        row for row in rows if row["summary"]["truncated_observation_count"]
    ]
    if truncated:
        details = [
            (row["instance_id"], row["summary"]["truncated_tools"])
            for row in truncated
        ]
        raise ValueError(f"{args.run_dir}: fallback guard truncations: {details}")
    if any(
        not row.get("bounded_graph_output")
        or not row.get("bounded_entity_output")
        or not row.get("bounded_search_output")
        for row in rows
    ):
        raise ValueError(f"{args.run_dir}: bounded tool policy missing")
    print(
        json.dumps(
            {
                "run_dir": args.run_dir,
                "trajectories": len(rows),
                "context_errors": 0,
                "fallback_truncations": 0,
                "max_prompt_tokens": max(
                    row["summary"]["max_prompt_tokens"] for row in rows
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
