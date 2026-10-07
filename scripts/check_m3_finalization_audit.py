import argparse
import csv
import itertools
import json
import math
import os
from collections import Counter

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


DEFAULT_FLIP_FILE = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "flip_decomposition/flip_decomposition.csv"
)
DEFAULT_ORIGINAL_BASELINE = [
    "results/locagent_verified_astropy22_django231_azure_gpt4o/loc_trajs.jsonl",
    "results/locagent_verified_remaining247_azure_gpt4o/loc_trajs.jsonl",
]
DEFAULT_RERUN_BASELINE = (
    "results/check9_label_stability_gpt4o_baseline_harm30/loc_trajs.jsonl"
)
DEFAULT_ORIGINAL_MEMORY = [
    "results/repomem_verified_astropy22_django231_azure_gpt4o_episodic_with_issues/loc_trajs.jsonl",
    "results/repomem_verified_remaining247_azure_gpt4o_episodic_with_issues/loc_trajs.jsonl",
]
DEFAULT_RERUN_MEMORY = (
    "results/check9_label_stability_gpt4o_episodic_harm30/loc_trajs.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "m3_memory_variance/m3b_finalization_audit"
)

CONTINUOUS_METRICS = [
    "total_tool_calls",
    "assistant_iterations",
    "effective_finish_iteration",
    "code_verify_call_count",
    "active_code_verify_count",
    "commit_examine_ratio",
    "predicted_file_count",
]
BINARY_METRICS = [
    "finished_before_max",
    "forced_finalization_prompt_seen",
    "finish_called",
    "final_output_parseable",
]
PRIMARY_METRICS = [
    "code_verify_call_count",
    "active_code_verify_count",
    "finished_before_max",
    "forced_finalization_prompt_seen",
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_flip_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in (
            "orig_baseline_at5",
            "orig_memory_at5",
            "rerun_baseline_at5",
            "rerun_memory_at5",
        ):
            row[key] = int(row[key])
    return rows


def flatten_files(value):
    out = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        for item in value:
            out.extend(flatten_files(item))
    return out


def extract_trajectory(row, max_iterations):
    trajectories = (row.get("loc_trajs") or {}).get("trajs") or []
    if len(trajectories) != 1:
        raise ValueError(
            f"expected one trajectory for {row.get('instance_id')}, got {len(trajectories)}"
        )
    messages = trajectories[0].get("messages") or []
    assistant_iteration = 0
    finish_at_iteration = None
    tool_names = []
    forced_prompt_seen = False
    final_text = ""

    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user" and "Maximum number of interation has been reached" in content:
            forced_prompt_seen = True
        if role != "assistant":
            continue
        assistant_iteration += 1
        if content.strip():
            final_text = content
        for call in message.get("tool_calls") or []:
            name = ((call.get("function") or {}).get("name") or "<missing_tool_name>")
            tool_names.append(name)
            if name == "finish" and finish_at_iteration is None:
                finish_at_iteration = assistant_iteration

    counts = Counter(tool_names)
    search_count = counts.get("search_commit", 0)
    examine_count = counts.get("examine_commit", 0)
    predicted_files = [
        path for path in flatten_files(row.get("found_files") or []) if path.endswith(".py")
    ]
    finish_called = finish_at_iteration is not None
    return {
        "total_tool_calls": len(tool_names),
        "assistant_iterations": assistant_iteration,
        "finish_at_iteration": finish_at_iteration,
        "effective_finish_iteration": (
            finish_at_iteration if finish_called else max_iterations
        ),
        "finished_before_max": int(
            finish_called
            and finish_at_iteration < max_iterations
            and not forced_prompt_seen
        ),
        "forced_finalization_prompt_seen": int(forced_prompt_seen),
        "finish_called": int(finish_called),
        "code_verify_call_count": (
            counts.get("search_code_snippets", 0)
            + counts.get("get_entity_contents", 0)
        ),
        "active_code_verify_count": (
            counts.get("search_code_snippets", 0)
            + counts.get("get_entity_contents", 0)
            + counts.get("explore_tree_structure", 0)
        ),
        "search_commit_count": search_count,
        "examine_commit_count": examine_count,
        "commit_examine_ratio": (
            examine_count / search_count if search_count else None
        ),
        "final_output_parseable": int(bool(predicted_files)),
        "predicted_file_count": len(dict.fromkeys(predicted_files)),
        "final_text_chars": len(final_text),
        "tool_counts": dict(counts),
    }


def load_run(paths, selected_ids, max_iterations):
    if isinstance(paths, str):
        paths = [paths]
    result = {}
    for path in paths:
        for row in load_jsonl(path):
            case_id = row.get("instance_id")
            if case_id not in selected_ids:
                continue
            if case_id in result:
                raise ValueError(f"duplicate case {case_id} across {paths}")
            result[case_id] = {
                "source_file": path,
                **extract_trajectory(row, max_iterations),
            }
    missing = selected_ids - set(result)
    if missing:
        raise ValueError(f"missing {len(missing)} trajectories: {sorted(missing)}")
    return result


def finite_values(rows, field):
    return [float(row[field]) for row in rows if row.get(field) is not None]


def cliffs_delta(left, right):
    if not left or not right:
        return None
    greater = sum(a > b for a in left for b in right)
    less = sum(a < b for a in left for b in right)
    return (greater - less) / (len(left) * len(right))


def exact_label_permutation(left, right):
    values = np.asarray(left + right, dtype=float)
    n_left = len(left)
    observed = float(np.mean(left) - np.mean(right))
    extreme = 0
    total = 0
    for indexes in itertools.combinations(range(len(values)), n_left):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(indexes)] = True
        diff = float(values[mask].mean() - values[~mask].mean())
        extreme += abs(diff) >= abs(observed) - 1e-12
        total += 1
    return observed, extreme / total


def bootstrap_mean_gap(left, right, n_boot=10000, seed=42):
    if not left or not right:
        return [None, None]
    rng = np.random.default_rng(seed)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    gaps = np.empty(n_boot)
    for index in range(n_boot):
        lsample = rng.choice(left, size=len(left), replace=True)
        rsample = rng.choice(right, size=len(right), replace=True)
        gaps[index] = lsample.mean() - rsample.mean()
    return np.percentile(gaps, [2.5, 97.5]).tolist()


def holm_adjust(pvalues):
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [None] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - rank)))
        adjusted[index] = running
    return adjusted


def compare_continuous(left_rows, right_rows, field, comparison):
    left = finite_values(left_rows, field)
    right = finite_values(right_rows, field)
    if not left or not right:
        return {
            "comparison": comparison,
            "metric": field,
            "kind": "continuous",
            "n_flipped": len(left),
            "n_stable": len(right),
            "insufficient_data": True,
        }
    gap, permutation_p = exact_label_permutation(left, right)
    mann_p = mannwhitneyu(left, right, alternative="two-sided").pvalue
    ci = bootstrap_mean_gap(left, right)
    return {
        "comparison": comparison,
        "metric": field,
        "kind": "continuous",
        "n_flipped": len(left),
        "n_stable": len(right),
        "flipped_mean": float(np.mean(left)),
        "stable_mean": float(np.mean(right)),
        "flipped_median": float(np.median(left)),
        "stable_median": float(np.median(right)),
        "mean_gap": gap,
        "bootstrap_ci_low": ci[0],
        "bootstrap_ci_high": ci[1],
        "cliffs_delta": cliffs_delta(left, right),
        "permutation_p": permutation_p,
        "mannwhitney_p": float(mann_p),
    }


def compare_binary(left_rows, right_rows, field, comparison):
    left_true = sum(int(row[field]) for row in left_rows)
    right_true = sum(int(row[field]) for row in right_rows)
    table = [
        [left_true, len(left_rows) - left_true],
        [right_true, len(right_rows) - right_true],
    ]
    odds_ratio, fisher_p = fisher_exact(table)
    return {
        "comparison": comparison,
        "metric": field,
        "kind": "binary",
        "n_flipped": len(left_rows),
        "n_stable": len(right_rows),
        "flipped_rate": left_true / len(left_rows),
        "stable_rate": right_true / len(right_rows),
        "rate_gap": left_true / len(left_rows) - right_true / len(right_rows),
        "odds_ratio": None if math.isnan(odds_ratio) else float(odds_ratio),
        "fisher_p": float(fisher_p),
    }


def comparison_table(rows, comparison, difference_view=False):
    flipped = [row for row in rows if row["flip_class"] == "memory_flipped"]
    stable = [row for row in rows if row["flip_class"] == "stable_harm"]
    results = [
        compare_continuous(flipped, stable, field, comparison)
        for field in CONTINUOUS_METRICS
    ]
    if difference_view:
        results.extend(
            compare_continuous(flipped, stable, field, comparison)
            for field in BINARY_METRICS
        )
    else:
        results.extend(
            compare_binary(flipped, stable, field, comparison)
            for field in BINARY_METRICS
        )
    primary_indexes = [
        index for index, row in enumerate(results) if row["metric"] in PRIMARY_METRICS
    ]
    raw = [
        results[index].get("permutation_p", results[index].get("fisher_p"))
        for index in primary_indexes
    ]
    for index, adjusted in zip(primary_indexes, holm_adjust(raw)):
        results[index]["holm_primary_p"] = adjusted
    return results


def subtract_metrics(left, right):
    out = {}
    for field in CONTINUOUS_METRICS:
        if left.get(field) is None or right.get(field) is None:
            out[field] = None
        else:
            out[field] = left[field] - right[field]
    for field in BINARY_METRICS:
        out[field] = int(left.get(field, 0)) - int(right.get(field, 0))
    return out


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None if math.isnan(value) else ("Inf" if value > 0 else "-Inf")
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_report(path, summary, stats):
    lines = [
        "# M3b Finalization Behavior Audit",
        "",
        "## Design",
        "",
        "The primary analysis is rerun-only. Both baseline and memory reruns use the same "
        "function-calling configuration with max_iteration_num=8. Cross-configuration "
        "rerun-minus-original deltas are exploratory because the original runs used a "
        "different model string, CodeAct/function-calling mode, iteration budget, and sample count.",
        "",
        f"- all selected harm cases: {summary['all_cases']}",
        f"- clean memory_flipped cases: {summary['memory_flipped_cases']}",
        f"- clean stable_harm controls: {summary['stable_harm_cases']}",
        "",
        "## Primary And Negative-Control Comparisons",
        "",
        "| comparison | metric | flipped | stable | gap | p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        if row["metric"] not in PRIMARY_METRICS:
            continue
        if row["kind"] == "continuous":
            left = row["flipped_mean"]
            right = row["stable_mean"]
            gap = row["mean_gap"]
            pvalue = row["permutation_p"]
        else:
            left = row["flipped_rate"]
            right = row["stable_rate"]
            gap = row["rate_gap"]
            pvalue = row["fisher_p"]
        holm = row.get("holm_primary_p")
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {left:.4f} | {right:.4f} | "
            f"{gap:+.4f} | {pvalue:.4f} | {holm:.4f} |"
        )

    corr = summary["rerun_correctness_association"]
    lines.extend(
        [
            "",
            "## All-30 Rerun Correctness Association",
            "",
            "| metric | Spearman rho | p | N |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in corr:
        lines.append(
            f"| {row['metric']} | {row['rho']:.4f} | {row['p']:.4f} | {row['n']} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            summary["verdict"],
            "",
            "## Limitations",
            "",
            "- Flip classes were selected from original-harm outcomes and include rerun correctness in their definition; associations are diagnostic, not causal.",
            "- Original-only comparisons are difficulty negative controls, not matched repeats of the rerun configuration.",
            "- Cross-configuration deltas are retained only for auditability and must not be used as mechanism evidence.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit finalization behavior without cross-config confounding.")
    parser.add_argument("--flip_file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--original_baseline", nargs="+", default=DEFAULT_ORIGINAL_BASELINE)
    parser.add_argument("--rerun_baseline", default=DEFAULT_RERUN_BASELINE)
    parser.add_argument("--original_memory", nargs="+", default=DEFAULT_ORIGINAL_MEMORY)
    parser.add_argument("--rerun_memory", default=DEFAULT_RERUN_MEMORY)
    parser.add_argument("--original_max_iterations", type=int, default=20)
    parser.add_argument("--rerun_max_iterations", type=int, default=8)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    flip_rows = load_flip_rows(args.flip_file)
    selected_ids = {row["case_id"] for row in flip_rows}
    original_baseline = load_run(args.original_baseline, selected_ids, args.original_max_iterations)
    rerun_baseline = load_run(args.rerun_baseline, selected_ids, args.rerun_max_iterations)
    original_memory = load_run(args.original_memory, selected_ids, args.original_max_iterations)
    rerun_memory = load_run(args.rerun_memory, selected_ids, args.rerun_max_iterations)

    cases = []
    for source in flip_rows:
        case_id = source["case_id"]
        row = {
            "case_id": case_id,
            "repo": source["repo"],
            "flip_class": source["flip_class"],
            "original_baseline": original_baseline[case_id],
            "rerun_baseline": rerun_baseline[case_id],
            "original_memory": original_memory[case_id],
            "rerun_memory": rerun_memory[case_id],
            "rerun_memory_minus_baseline": subtract_metrics(
                rerun_memory[case_id], rerun_baseline[case_id]
            ),
            "exploratory_memory_rerun_minus_original": subtract_metrics(
                rerun_memory[case_id], original_memory[case_id]
            ),
            "rerun_memory_at5": source["rerun_memory_at5"],
        }
        cases.append(row)

    clean = [
        row for row in cases if row["flip_class"] in {"memory_flipped", "stable_harm"}
    ]
    views = {
        "rerun_memory_absolute": [
            {"flip_class": row["flip_class"], **row["rerun_memory"]} for row in clean
        ],
        "rerun_memory_minus_baseline": [
            {"flip_class": row["flip_class"], **row["rerun_memory_minus_baseline"]}
            for row in clean
        ],
        "original_memory_negative_control": [
            {"flip_class": row["flip_class"], **row["original_memory"]} for row in clean
        ],
        "rerun_baseline_negative_control": [
            {"flip_class": row["flip_class"], **row["rerun_baseline"]} for row in clean
        ],
        "exploratory_cross_config_delta": [
            {
                "flip_class": row["flip_class"],
                **row["exploratory_memory_rerun_minus_original"],
            }
            for row in clean
        ],
    }
    stats = []
    difference_views = {
        "rerun_memory_minus_baseline",
        "exploratory_cross_config_delta",
    }
    for name, rows in views.items():
        stats.extend(comparison_table(rows, name, name in difference_views))

    associations = []
    for field in CONTINUOUS_METRICS:
        values = [
            (row["rerun_memory"].get(field), row["rerun_memory_at5"])
            for row in cases
            if row["rerun_memory"].get(field) is not None
        ]
        rho, pvalue = spearmanr(
            [value for value, _ in values], [correct for _, correct in values]
        )
        associations.append(
            {"metric": field, "rho": float(rho), "p": float(pvalue), "n": len(values)}
        )

    primary = {
        (row["comparison"], row["metric"]): row
        for row in stats
        if row["metric"] in PRIMARY_METRICS
    }
    support = []
    for metric in ("code_verify_call_count", "active_code_verify_count"):
        absolute = primary[("rerun_memory_absolute", metric)]
        differential = primary[("rerun_memory_minus_baseline", metric)]
        original = primary[("original_memory_negative_control", metric)]
        support.append(
            absolute["mean_gap"] > 0
            and differential["mean_gap"] > 0
            and min(absolute.get("holm_primary_p", 1), differential.get("holm_primary_p", 1)) < 0.10
            and original.get("holm_primary_p", 1) >= 0.10
        )
    verdict = (
        "SUPPORTIVE DIAGNOSTIC: rerun-only verification behavior differs in the predicted direction without an equivalent original-memory negative-control signal. Confirm with M3a matched repeats before making a mechanism claim."
        if any(support)
        else "INCONCLUSIVE/NULL DIAGNOSTIC: the pre-registered rerun-only and same-config differential criteria are not jointly met. Cross-configuration deltas are not mechanism evidence."
    )
    summary = {
        "all_cases": len(cases),
        "memory_flipped_cases": sum(row["flip_class"] == "memory_flipped" for row in cases),
        "stable_harm_cases": sum(row["flip_class"] == "stable_harm" for row in cases),
        "rerun_correctness_association": associations,
        "verdict": verdict,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "finalization_per_case.jsonl"), "w", encoding="utf-8") as f:
        for row in cases:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    write_csv(os.path.join(args.output_dir, "comparison_statistics.csv"), [json_safe(row) for row in stats])
    with open(os.path.join(args.output_dir, "m3b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False, allow_nan=False)
    write_report(os.path.join(args.output_dir, "finalization_behavior_audit.md"), summary, stats)
    print(verdict)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
