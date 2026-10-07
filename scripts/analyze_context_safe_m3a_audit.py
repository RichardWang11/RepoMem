import argparse
import json
import os


BOUND_MARKERS = {
    "entity_line_bound": "[... entity output",
    "graph_line_bound": "[... graph output",
    "graph_depth_bound": "[Context-safe graph bound applied:",
    "search_line_bound": "[... code search output",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_cases", type=int, default=15)
    args = parser.parse_args()

    rows = []
    failures = []
    context_error_count = 0
    canonical_args = None
    parameter_rows = []
    ignored_arg_keys = {
        "output_folder",
        "output_file",
        "merge_file",
        "use_repomem",
    }
    for arm in ("baseline", "memory"):
        for repeat in range(1, 5):
            run_dir = os.path.join(args.run_root, f"{arm}_r{repeat}")
            with open(os.path.join(run_dir, "args.json"), encoding="utf-8") as f:
                run_args = json.load(f)
            normalized_args = {
                key: value
                for key, value in run_args.items()
                if key not in ignored_arg_keys
            }
            if canonical_args is None:
                canonical_args = normalized_args
            elif normalized_args != canonical_args:
                failures.append(f"{arm}_r{repeat}: runtime argument mismatch")
            expected_memory = arm == "memory"
            if bool(run_args.get("use_repomem")) != expected_memory:
                failures.append(
                    f"{arm}_r{repeat}: use_repomem={run_args.get('use_repomem')}"
                )
            parameter_rows.append(
                {
                    "arm": arm,
                    "repeat": repeat,
                    "use_repomem": bool(run_args.get("use_repomem")),
                    "normalized_args": normalized_args,
                }
            )
            audit_path = os.path.join(run_dir, "context_guard_audit.jsonl")
            audit_rows = load_jsonl(audit_path)
            trajectory_rows = {
                row["instance_id"]: row
                for row in load_jsonl(os.path.join(run_dir, "loc_trajs.jsonl"))
            }
            if len(audit_rows) != args.expected_cases:
                failures.append(
                    f"{arm}_r{repeat}: expected {args.expected_cases} audits, "
                    f"got {len(audit_rows)}"
                )
            error_path = os.path.join(run_dir, "context_errors.jsonl")
            errors = load_jsonl(error_path) if os.path.exists(error_path) else []
            context_error_count += len(errors)
            if errors:
                failures.append(f"{arm}_r{repeat}: {len(errors)} context errors")
            for row in audit_rows:
                summary = row["summary"]
                trajectory = trajectory_rows[row["instance_id"]]["loc_trajs"]["trajs"][0]
                message_text = "\n".join(
                    str(message.get("content") or "")
                    for message in trajectory.get("messages", [])
                )
                bound_counts = {
                    key: message_text.count(marker)
                    for key, marker in BOUND_MARKERS.items()
                }
                item = {
                    "arm": arm,
                    "repeat": repeat,
                    "instance_id": row["instance_id"],
                    "bounded_graph_output": row.get("bounded_graph_output"),
                    "bounded_entity_output": row.get("bounded_entity_output"),
                    "bounded_search_output": row.get("bounded_search_output"),
                    **bound_counts,
                    **summary,
                }
                rows.append(item)
                if summary["truncated_observation_count"]:
                    failures.append(
                        f"{arm}_r{repeat}/{row['instance_id']}: fallback guard truncated "
                        f"{summary['truncated_observation_count']} observations"
                    )
                if (
                    not row.get("bounded_graph_output")
                    or not row.get("bounded_entity_output")
                    or not row.get("bounded_search_output")
                ):
                    failures.append(
                        f"{arm}_r{repeat}/{row['instance_id']}: bounded tool policy missing"
                    )

    expected_trajectories = args.expected_cases * 8
    if len(rows) != expected_trajectories:
        failures.append(
            f"expected {expected_trajectories} trajectory audits, got {len(rows)}"
        )

    summary = {
        "trajectory_count": len(rows),
        "expected_trajectory_count": expected_trajectories,
        "context_error_count": context_error_count,
        "fallback_truncation_count": sum(
            row["truncated_observation_count"] for row in rows
        ),
        "bounded_tool_interaction_trajectory_count": sum(
            any(row[key] for key in BOUND_MARKERS) for row in rows
        ),
        "bounded_tool_interaction_event_count": sum(
            row[key] for row in rows for key in BOUND_MARKERS
        ),
        "bounded_tool_event_breakdown": {
            key: sum(row[key] for row in rows) for key in BOUND_MARKERS
        },
        "max_prompt_tokens": max(
            (row["max_prompt_tokens"] for row in rows), default=0
        ),
        "failures": failures,
        "primary_analysis_valid": not failures,
        "parameter_parity_verified": not any(
            "argument mismatch" in item or "use_repomem=" in item
            for item in failures
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "context_audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "context_audit_per_trajectory.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(args.output_dir, "matched_run_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(parameter_rows, f, indent=2)
    with open(os.path.join(args.output_dir, "context_audit_report.md"), "w", encoding="utf-8") as f:
        f.write("# Context-Safe M3a Runtime Audit\n\n")
        f.write(f"- trajectories: {len(rows)}/{expected_trajectories}\n")
        f.write(f"- context errors: {summary['context_error_count']}\n")
        f.write(f"- fallback truncations: {summary['fallback_truncation_count']}\n")
        f.write(
            f"- trajectories with bounded-tool interaction: "
            f"{summary['bounded_tool_interaction_trajectory_count']}\n"
        )
        f.write(
            f"- bounded-tool interaction events: "
            f"{summary['bounded_tool_interaction_event_count']}\n"
        )
        f.write(f"- maximum prompt tokens: {summary['max_prompt_tokens']}\n")
        f.write(
            f"- parameter parity verified: **{summary['parameter_parity_verified']}**\n"
        )
        f.write(
            f"- primary analysis valid: **{summary['primary_analysis_valid']}**\n"
        )
        if failures:
            f.write("\n## Failures\n\n")
            for failure in failures:
                f.write(f"- {failure}\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
