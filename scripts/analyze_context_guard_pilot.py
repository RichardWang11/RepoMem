import argparse
import json
import os


MEMORY_TOOLS = {"search_commit", "examine_commit", "search_summary", "view_summary"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def decide(summary, context_errors, bounded_tool_output=False):
    truncated_tools = set(summary.get("truncated_tools", []))
    memory_tools_truncated = sorted(truncated_tools & MEMORY_TOOLS)
    sparse = (
        summary.get("truncated_observation_count", 0) <= 2
        and summary.get("observation_token_removal_rate", 0.0) <= 0.15
        and truncated_tools <= {"explore_tree_structure", "explore_graph_structure"}
        and not memory_tools_truncated
        and not context_errors
    )
    if bounded_tool_output and sparse:
        decision = "TOOL_C_VALIDATED"
    else:
        decision = "GUARD_A" if sparse else "TOOL_C"
    return {
        "decision": decision,
        "criteria": {
            "at_most_two_truncations": summary.get(
                "truncated_observation_count", 0
            ) <= 2,
            "at_most_15_percent_tokens_removed": summary.get(
                "observation_token_removal_rate", 0.0
            ) <= 0.15,
            "only_graph_tools_truncated": truncated_tools
            <= {"explore_tree_structure", "explore_graph_structure"},
            "no_memory_tools_truncated": not memory_tools_truncated,
            "no_context_errors": not context_errors,
        },
        "summary": summary,
        "context_error_count": len(context_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    audit_path = os.path.join(args.run_dir, "context_guard_audit.jsonl")
    audit_rows = load_jsonl(audit_path)
    if len(audit_rows) != 1:
        raise ValueError(f"expected one pilot trajectory, got {len(audit_rows)}")
    errors_path = os.path.join(args.run_dir, "context_errors.jsonl")
    context_errors = load_jsonl(errors_path) if os.path.exists(errors_path) else []
    bounded_tool_output = bool(
        audit_rows[0].get("bounded_graph_output")
        or audit_rows[0].get("bounded_entity_output")
        or audit_rows[0].get("bounded_search_output")
    )
    result = decide(
        audit_rows[0]["summary"],
        context_errors,
        bounded_tool_output=bounded_tool_output,
    )
    result["pilot"] = {
        "instance_id": audit_rows[0]["instance_id"],
        "use_repomem": audit_rows[0]["use_repomem"],
        "context_token_budget": audit_rows[0]["context_token_budget"],
        "tool_observation_token_cap": audit_rows[0]["tool_observation_token_cap"],
        "bounded_graph_output": audit_rows[0].get("bounded_graph_output", False),
        "bounded_entity_output": audit_rows[0].get("bounded_entity_output", False),
        "bounded_search_output": audit_rows[0].get("bounded_search_output", False),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "pilot_decision.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.output_dir, "pilot_report.md"), "w", encoding="utf-8") as f:
        f.write("# Context Guard Pilot\n\n")
        f.write(f"- decision: **{result['decision']}**\n")
        f.write(f"- instance: `{result['pilot']['instance_id']}`\n")
        f.write(
            f"- truncated observations: {result['summary']['truncated_observation_count']}\n"
        )
        f.write(
            f"- observation tokens removed: "
            f"{result['summary']['observation_token_removal_rate']:.2%}\n"
        )
        f.write(
            f"- truncated tools: {result['summary']['truncated_tools']}\n"
        )
        f.write(f"- context errors: {result['context_error_count']}\n\n")
        f.write("The decision follows the pre-specified sparse-truncation criteria.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
