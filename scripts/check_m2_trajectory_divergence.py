import argparse
import csv
import json
import math
import os
import pickle
import re
import statistics
from collections import Counter


DEFAULT_FLIP_FILE = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "flip_decomposition/flip_decomposition.csv"
)
DEFAULT_ORIGINAL_BASELINE = [
    "results/locagent_verified_astropy22_django231_azure_gpt4o/loc_trajs.jsonl",
    "results/locagent_verified_remaining247_azure_gpt4o/loc_trajs.jsonl",
]
DEFAULT_RERUN_BASELINE = "results/check9_label_stability_gpt4o_baseline_harm30/loc_trajs.jsonl"
DEFAULT_ORIGINAL_MEMORY = [
    "results/repomem_verified_astropy22_django231_azure_gpt4o_episodic_with_issues/loc_trajs.jsonl",
    "results/repomem_verified_remaining247_azure_gpt4o_episodic_with_issues/loc_trajs.jsonl",
]
DEFAULT_RERUN_MEMORY = "results/check9_label_stability_gpt4o_episodic_harm30/loc_trajs.jsonl"
DEFAULT_GRAPH_DIR = "index_data/SWE-bench_Verified/graph_index_v2.3"
DEFAULT_OUTPUT_DIR = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "m2_trajectory_divergence"
)

MEMORY_TOOLS = {"search_commit", "examine_commit"}
ACTIVE_CODE_TOOLS = {
    "search_code_snippets",
    "get_entity_contents",
    "explore_tree_structure",
}
PRIMARY_METRIC = "tool_seq_edit_distance"
METRICS = [
    PRIMARY_METRIC,
    "length_delta",
    "tool_set_jaccard",
    "explored_files_jaccard",
]
FOUND_ENTITY_RE = re.compile(
    r"Found\s+(?:file|class|function)\s+`([^`]+)`", re.IGNORECASE
)
FOUND_SNIPPET_RE = re.compile(
    r"Found\s+code snippet in file\s+`([^`]+)`", re.IGNORECASE
)
GLOB_CHARS = set("*?[]{}")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(value), f, indent=2, ensure_ascii=False, allow_nan=False)


def json_safe(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        if math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def safe_json(raw):
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def normalize_candidate(value):
    value = str(value or "").strip().strip("`'\" ,:;")
    value = value.replace("\\", "/")
    for prefix in ("./", "a/", "b/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    if ":" in value:
        value = value.split(":", 1)[0]
    return value.strip("/")


def resolve_file(value, valid_files):
    candidate = normalize_candidate(value)
    if not candidate or any(char in candidate for char in GLOB_CHARS):
        return None
    if candidate in valid_files:
        return candidate
    matches = [
        path
        for path in valid_files
        if path.endswith("/" + candidate) or path.endswith(candidate)
    ]
    return matches[0] if len(matches) == 1 else None


def load_valid_files(instance_id, graph_dir):
    path = os.path.join(graph_dir, f"{instance_id}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing graph index: {path}")
    with open(path, "rb") as f:
        graph = pickle.load(f)
    return {str(node) for node, data in graph.nodes(data=True) if data.get("type") == "file"}


def files_from_tool_call(name, args, observation, valid_files):
    if name not in ACTIVE_CODE_TOOLS:
        return set()
    values = []
    if args:
        if name == "get_entity_contents":
            values.extend(args.get("entity_names") or [])
        elif name == "explore_tree_structure":
            values.extend(args.get("start_entities") or [])
        elif name == "search_code_snippets":
            pattern = args.get("file_path_or_pattern")
            if pattern and not any(char in str(pattern) for char in GLOB_CHARS):
                values.append(pattern)
    if name in {"get_entity_contents", "search_code_snippets"}:
        values.extend(FOUND_ENTITY_RE.findall(observation or ""))
        values.extend(FOUND_SNIPPET_RE.findall(observation or ""))
    return {
        resolved
        for value in values
        if (resolved := resolve_file(value, valid_files)) is not None
    }


def extract_trajectory(messages, valid_files):
    observations = {
        message.get("tool_call_id"): message.get("content") or ""
        for message in messages or []
        if message.get("role") == "tool"
    }
    tool_sequence = []
    explored_files = set()
    malformed_arguments = 0
    unknown_tools = Counter()
    final_text = ""
    known_tools = ACTIVE_CODE_TOOLS | MEMORY_TOOLS | {"finish"}

    for message in messages or []:
        if message.get("role") == "assistant" and message.get("content"):
            final_text = str(message["content"])
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = function.get("name") or "<missing_tool_name>"
            tool_sequence.append(name)
            if name not in known_tools:
                unknown_tools[name] += 1
            args = safe_json(function.get("arguments"))
            if args is None:
                malformed_arguments += 1
                args = {}
            explored_files.update(
                files_from_tool_call(
                    name,
                    args,
                    observations.get(tool_call.get("id"), ""),
                    valid_files,
                )
            )

    return {
        "tool_sequence": tool_sequence,
        "locagent_tool_sequence": [name for name in tool_sequence if name not in MEMORY_TOOLS],
        "tool_counts": dict(Counter(tool_sequence)),
        "explored_files": sorted(explored_files),
        "finish_called": "finish" in tool_sequence,
        "last_tool": tool_sequence[-1] if tool_sequence else None,
        "final_text_present": bool(final_text.strip()),
        "final_text_chars": len(final_text),
        "malformed_arguments": malformed_arguments,
        "unknown_tools": dict(unknown_tools),
    }


def load_run(paths, selected_ids, graph_dir):
    rows = {}
    valid_cache = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for row in load_jsonl(path):
            instance_id = row.get("instance_id")
            if instance_id not in selected_ids:
                continue
            trajectories = (row.get("loc_trajs") or {}).get("trajs") or []
            if len(trajectories) != 1:
                raise ValueError(
                    f"expected one trajectory for {instance_id} in {path}, got {len(trajectories)}"
                )
            if instance_id not in valid_cache:
                valid_cache[instance_id] = load_valid_files(instance_id, graph_dir)
            rows[instance_id] = {
                "source_file": path,
                **extract_trajectory(
                    trajectories[0].get("messages") or [], valid_cache[instance_id]
                ),
            }
    return rows


def levenshtein_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, start=1):
        current = [i]
        for j, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_distance(left, right):
    denominator = max(len(left), len(right))
    return levenshtein_distance(left, right) / denominator if denominator else 0.0


def relative_length_delta(left, right):
    mean_length = (len(left) + len(right)) / 2
    return abs(len(left) - len(right)) / mean_length if mean_length else 0.0


def jaccard_distance(left, right):
    left = set(left or [])
    right = set(right or [])
    if not left and not right:
        return 0.0
    return 1 - len(left & right) / len(left | right)


def trajectory_distances(original, rerun, sequence_key="tool_sequence"):
    left_sequence = original[sequence_key]
    right_sequence = rerun[sequence_key]
    return {
        "tool_seq_edit_distance": normalized_edit_distance(left_sequence, right_sequence),
        "length_delta": relative_length_delta(left_sequence, right_sequence),
        "tool_set_jaccard": jaccard_distance(left_sequence, right_sequence),
        "explored_files_jaccard": jaccard_distance(
            original["explored_files"], rerun["explored_files"]
        ),
    }


def load_selected_cases(flip_file):
    with open(flip_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        row["case_id"]: row
        for row in rows
        if row.get("flip_class") in {"memory_flipped", "stable_harm"}
    }


def require_complete_join(selected, run_maps):
    missing = {}
    for label, rows in run_maps.items():
        absent = sorted(set(selected) - set(rows))
        if absent:
            missing[label] = absent
    if missing:
        raise ValueError(f"four-way trajectory join failed: {missing}")


def build_case_rows(selected, original_baseline, rerun_baseline, original_memory, rerun_memory):
    rows = []
    for case_id in sorted(selected):
        ob = original_baseline[case_id]
        rb = rerun_baseline[case_id]
        om = original_memory[case_id]
        rm = rerun_memory[case_id]
        baseline = trajectory_distances(ob, rb)
        memory = trajectory_distances(om, rm)
        memory_locagent = trajectory_distances(om, rm, "locagent_tool_sequence")
        row = {
            "case_id": case_id,
            "repo": selected[case_id].get("repo"),
            "repo_full": selected[case_id].get("repo_full"),
            "flip_class": selected[case_id]["flip_class"],
            "original_baseline": ob,
            "rerun_baseline": rb,
            "original_memory": om,
            "rerun_memory": rm,
            "baseline_divergence": baseline,
            "memory_divergence": memory,
            "memory_locagent_only_divergence": memory_locagent,
            "divergence_delta": {
                metric: memory[metric] - baseline[metric] for metric in METRICS
            },
            "locagent_only_delta": {
                metric: memory_locagent[metric] - baseline[metric] for metric in METRICS
            },
        }
        rows.append(row)
    return rows


def percentile_interval(values, alpha=0.05):
    values = sorted(values)
    if not values:
        return None, None
    low = int((alpha / 2) * (len(values) - 1))
    high = int((1 - alpha / 2) * (len(values) - 1))
    return values[low], values[high]


def bootstrap_paired_median_delta(differences, n_boot=10000, seed=42):
    import random

    if not differences:
        return None, None
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        sample = [rng.choice(differences) for _ in differences]
        boot.append(statistics.median(sample))
    return percentile_interval(boot)


def bootstrap_group_median_gap(flipped, stable, n_boot=10000, seed=42):
    import random

    if not flipped or not stable:
        return None, None
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        left = [rng.choice(flipped) for _ in flipped]
        right = [rng.choice(stable) for _ in stable]
        boot.append(statistics.median(left) - statistics.median(right))
    return percentile_interval(boot)


def wilcoxon_p(differences):
    if not differences or all(abs(value) < 1e-15 for value in differences):
        return 1.0, 0.0
    from scipy.stats import wilcoxon

    result = wilcoxon(differences, zero_method="pratt", alternative="two-sided")
    return float(result.pvalue), float(result.statistic)


def median_ratio(baseline_median, memory_median):
    if baseline_median == 0:
        if memory_median == 0:
            return 1.0
        return math.inf
    return memory_median / baseline_median


def holm_adjust(pvalues):
    indexed = sorted(enumerate(pvalues), key=lambda pair: pair[1])
    adjusted = [None] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, (index, pvalue) in enumerate(indexed):
        candidate = min(1.0, (total - rank) * pvalue)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def summarize_metrics(case_rows, divergence_key="memory_divergence"):
    summaries = []
    raw_pvalues = []
    for metric in METRICS:
        baseline = [row["baseline_divergence"][metric] for row in case_rows]
        memory = [row[divergence_key][metric] for row in case_rows]
        differences = [right - left for left, right in zip(baseline, memory)]
        pvalue, statistic = wilcoxon_p(differences)
        ci_low, ci_high = bootstrap_paired_median_delta(differences)
        baseline_median = statistics.median(baseline)
        memory_median = statistics.median(memory)
        summaries.append(
            {
                "metric": metric,
                "divergence_variant": divergence_key,
                "n": len(case_rows),
                "baseline_median": baseline_median,
                "memory_median": memory_median,
                "median_ratio": median_ratio(baseline_median, memory_median),
                "paired_median_delta": statistics.median(differences),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "memory_gt_baseline": sum(value > 1e-15 for value in differences),
                "equal": sum(abs(value) <= 1e-15 for value in differences),
                "memory_lt_baseline": sum(value < -1e-15 for value in differences),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_raw": pvalue,
            }
        )
        raw_pvalues.append(pvalue)
    for summary, adjusted in zip(summaries, holm_adjust(raw_pvalues)):
        summary["wilcoxon_p_holm"] = adjusted
    return summaries


def summarize_flip_association(case_rows, delta_key="divergence_delta"):
    summaries = []
    for metric in METRICS:
        flipped = [
            row[delta_key][metric]
            for row in case_rows
            if row["flip_class"] == "memory_flipped"
        ]
        stable = [
            row[delta_key][metric]
            for row in case_rows
            if row["flip_class"] == "stable_harm"
        ]
        from scipy.stats import mannwhitneyu

        test = mannwhitneyu(flipped, stable, alternative="two-sided")
        ci_low, ci_high = bootstrap_group_median_gap(flipped, stable)
        summaries.append(
            {
                "metric": metric,
                "delta_variant": delta_key,
                "n_memory_flipped": len(flipped),
                "n_stable_harm": len(stable),
                "median_delta_memory_flipped": statistics.median(flipped),
                "median_delta_stable_harm": statistics.median(stable),
                "median_gap_flipped_minus_stable": statistics.median(flipped)
                - statistics.median(stable),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "mann_whitney_u": float(test.statistic),
                "mann_whitney_p": float(test.pvalue),
            }
        )
    return summaries


def tool_count_breakdown(case_rows, group):
    tools = sorted(
        {
            tool
            for row in case_rows
            for side in (row["original_memory"], row["rerun_memory"])
            for tool in side["tool_counts"]
        }
    )
    output = []
    for tool in tools:
        deltas = [
            row["rerun_memory"]["tool_counts"].get(tool, 0)
            - row["original_memory"]["tool_counts"].get(tool, 0)
            for row in case_rows
        ]
        output.append(
            {
                "group": group,
                "tool": tool,
                "total_rerun_minus_original": sum(deltas),
                "cases_increased": sum(value > 0 for value in deltas),
                "cases_decreased": sum(value < 0 for value in deltas),
                "cases_equal": sum(value == 0 for value in deltas),
                "median_delta": statistics.median(deltas),
            }
        )
    return output


def zscores(values):
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    return [(value - mean) / std if std else 0.0 for value in values]


def rank_illustrative_cases(case_rows):
    score_parts = {
        metric: zscores([row["divergence_delta"][metric] for row in case_rows])
        for metric in METRICS
    }
    ranked = []
    for index, row in enumerate(case_rows):
        ranked.append(
            {
                "case_id": row["case_id"],
                "flip_class": row["flip_class"],
                "score": statistics.mean(score_parts[metric][index] for metric in METRICS),
                "baseline_primary": row["baseline_divergence"][PRIMARY_METRIC],
                "row": row,
            }
        )
    candidates = [row for row in ranked if row["flip_class"] == "memory_flipped"]
    candidates.sort(key=lambda row: (-row["score"], row["case_id"]))
    baseline_median = statistics.median(row["baseline_primary"] for row in ranked)
    eligible = [row for row in candidates if row["baseline_primary"] <= baseline_median]
    primary = eligible[0] if eligible else candidates[0]
    top_three = candidates[:3]
    return primary, top_three


def classify_result(metric_summaries, association_summaries):
    primary = next(row for row in metric_summaries if row["metric"] == PRIMARY_METRIC)
    association = next(
        row for row in association_summaries if row["metric"] == PRIMARY_METRIC
    )
    secondary_support = sum(
        row["metric"] != PRIMARY_METRIC
        and row["memory_median"] > row["baseline_median"]
        and row["median_ratio"] >= 1.2
        for row in metric_summaries
    )
    primary_strong = (
        primary["median_ratio"] > 2
        and primary["wilcoxon_p_raw"] < 0.05
        and primary["bootstrap_ci_low"] > 0
        and secondary_support >= 1
    )
    associated = (
        association["median_gap_flipped_minus_stable"] > 0
        and association["mann_whitney_p"] < 0.10
    )
    if primary_strong and associated:
        return "M2_CONFIRMED", "Memory increases trajectory divergence and the increase is concentrated in memory-flipped cases."
    if primary_strong:
        return "GENERAL_VARIANCE_ONLY", "Memory increases trajectory divergence, but divergence does not distinguish memory-flipped from stable-harm cases."
    partial = (
        primary["median_ratio"] >= 1.2
        or primary["wilcoxon_p_raw"] < 0.15
        or secondary_support >= 2
    )
    if partial:
        return "PARTIAL_M2", "Trajectory-level evidence is mixed; proceed to M3 and retain finalization as a competing mechanism."
    return "M2_REJECTED", "Memory does not materially increase trajectory-structure divergence; inspect finalization and output parsing."


def fmt(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isinf(value):
        return "Inf"
    return f"{value:.{digits}f}"


def build_illustrative_report(primary, top_three):
    lines = [
        "# M2 Illustrative Cases",
        "",
        "Cases are ranked deterministically by the mean z-score of the four memory-minus-baseline divergence metrics.",
        "",
        "## Ranked Candidates",
        "",
        "| rank | case | score | baseline tool distance | memory tool distance |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, item in enumerate(top_three, start=1):
        row = item["row"]
        lines.append(
            f"| {rank} | {item['case_id']} | {item['score']:.3f} | "
            f"{row['baseline_divergence'][PRIMARY_METRIC]:.3f} | "
            f"{row['memory_divergence'][PRIMARY_METRIC]:.3f} |"
        )
    row = primary["row"]
    lines.extend(
        [
            "",
            f"## Primary Example: {primary['case_id']}",
            "",
            f"- original baseline tools: `{row['original_baseline']['tool_sequence']}`",
            f"- rerun baseline tools: `{row['rerun_baseline']['tool_sequence']}`",
            f"- original memory tools: `{row['original_memory']['tool_sequence']}`",
            f"- rerun memory tools: `{row['rerun_memory']['tool_sequence']}`",
            f"- original memory files: `{row['original_memory']['explored_files']}`",
            f"- rerun memory files: `{row['rerun_memory']['explored_files']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_main_report(
    summary,
    metric_summaries,
    association,
    locagent_summaries,
    locagent_association,
    tool_rows,
):
    lines = [
        "# M2 Trajectory Divergence Analysis",
        "",
        "## Data",
        "",
        f"- paired cases: {summary['n_cases']}",
        f"- memory_flipped: {summary['n_memory_flipped']}",
        f"- stable_harm controls: {summary['n_stable_harm']}",
        f"- malformed tool arguments: {summary['malformed_arguments']}",
        f"- unknown tool calls: {summary['unknown_tool_calls']}",
        "",
        "## Primary Paired Comparison",
        "",
        "| metric | baseline median | memory median | ratio | paired delta | 95% CI | Wilcoxon p | Holm p | M>B / = / M<B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_summaries:
        lines.append(
            f"| {row['metric']} | {fmt(row['baseline_median'])} | {fmt(row['memory_median'])} | "
            f"{fmt(row['median_ratio'])} | {fmt(row['paired_median_delta'])} | "
            f"[{fmt(row['bootstrap_ci_low'])}, {fmt(row['bootstrap_ci_high'])}] | "
            f"{fmt(row['wilcoxon_p_raw'], 4)} | {fmt(row['wilcoxon_p_holm'], 4)} | "
            f"{row['memory_gt_baseline']} / {row['equal']} / {row['memory_lt_baseline']} |"
        )
    lines.extend(
        [
            "",
            "## Flip Association",
            "",
            "| metric | flipped median delta | stable median delta | gap | 95% CI | Mann-Whitney p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in association:
        lines.append(
            f"| {row['metric']} | {fmt(row['median_delta_memory_flipped'])} | "
            f"{fmt(row['median_delta_stable_harm'])} | {fmt(row['median_gap_flipped_minus_stable'])} | "
            f"[{fmt(row['bootstrap_ci_low'])}, {fmt(row['bootstrap_ci_high'])}] | "
            f"{fmt(row['mann_whitney_p'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## LocAgent-Only Sensitivity",
            "",
            "Memory tools are removed from the memory trajectories before sequence comparison; explored active-code files remain unchanged.",
            "",
            "| metric | baseline median | memory median | ratio | Wilcoxon p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in locagent_summaries:
        lines.append(
            f"| {row['metric']} | {fmt(row['baseline_median'])} | {fmt(row['memory_median'])} | "
            f"{fmt(row['median_ratio'])} | {fmt(row['wilcoxon_p_raw'], 4)} |"
        )
    locagent_primary_association = next(
        row for row in locagent_association if row["metric"] == PRIMARY_METRIC
    )
    lines.extend(
        [
            "",
            "LocAgent-only tool-sequence flip association: "
            f"flipped-minus-stable median gap={fmt(locagent_primary_association['median_gap_flipped_minus_stable'])}, "
            f"Mann-Whitney p={fmt(locagent_primary_association['mann_whitney_p'], 4)}.",
        ]
    )
    lines.extend(
        [
            "",
            "## Memory Tool Count Changes in Memory-Flipped Cases (rerun - original)",
            "",
            "| tool | total delta | cases increased | decreased | equal |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in tool_rows:
        lines.append(
            f"| {row['tool']} | {row['total_rerun_minus_original']} | "
            f"{row['cases_increased']} | {row['cases_decreased']} | {row['cases_equal']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{summary['verdict']}**: {summary['interpretation']}",
            "",
            "## Limitations",
            "",
            "This is a paired observational diagnostic on cases selected from original harm outcomes. "
            "Original and rerun model strings and iteration budgets differ, so the baseline pair is used as the shared pipeline-noise control. "
            "The result does not estimate full-500 trajectory stability.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(args):
    selected = load_selected_cases(args.flip_file)
    selected_ids = set(selected)
    runs = {
        "original_baseline": load_run(args.original_baseline, selected_ids, args.graph_dir),
        "rerun_baseline": load_run([args.rerun_baseline], selected_ids, args.graph_dir),
        "original_memory": load_run(args.original_memory, selected_ids, args.graph_dir),
        "rerun_memory": load_run([args.rerun_memory], selected_ids, args.graph_dir),
    }
    require_complete_join(selected, runs)
    case_rows = build_case_rows(selected, **runs)
    metric_summaries = summarize_metrics(case_rows)
    locagent_summaries = summarize_metrics(
        case_rows, divergence_key="memory_locagent_only_divergence"
    )
    association = summarize_flip_association(case_rows)
    locagent_association = summarize_flip_association(
        case_rows, delta_key="locagent_only_delta"
    )
    flipped_cases = [row for row in case_rows if row["flip_class"] == "memory_flipped"]
    stable_cases = [row for row in case_rows if row["flip_class"] == "stable_harm"]
    tool_rows_all = tool_count_breakdown(case_rows, "all_23")
    tool_rows_flipped = tool_count_breakdown(flipped_cases, "memory_flipped_15")
    tool_rows_stable = tool_count_breakdown(stable_cases, "stable_harm_8")
    tool_rows = tool_rows_all + tool_rows_flipped + tool_rows_stable
    primary_example, top_three = rank_illustrative_cases(case_rows)
    verdict, interpretation = classify_result(metric_summaries, association)
    malformed = sum(
        side["malformed_arguments"]
        for row in case_rows
        for side in (
            row["original_baseline"],
            row["rerun_baseline"],
            row["original_memory"],
            row["rerun_memory"],
        )
    )
    unknown = sum(
        sum(side["unknown_tools"].values())
        for row in case_rows
        for side in (
            row["original_baseline"],
            row["rerun_baseline"],
            row["original_memory"],
            row["rerun_memory"],
        )
    )
    finish_changes = {
        "baseline_finish_presence_changed": sum(
            row["original_baseline"]["finish_called"]
            != row["rerun_baseline"]["finish_called"]
            for row in case_rows
        ),
        "memory_finish_presence_changed": sum(
            row["original_memory"]["finish_called"]
            != row["rerun_memory"]["finish_called"]
            for row in case_rows
        ),
    }
    summary = {
        "n_cases": len(case_rows),
        "n_memory_flipped": sum(row["flip_class"] == "memory_flipped" for row in case_rows),
        "n_stable_harm": sum(row["flip_class"] == "stable_harm" for row in case_rows),
        "malformed_arguments": malformed,
        "unknown_tool_calls": unknown,
        **finish_changes,
        "verdict": verdict,
        "interpretation": interpretation,
        "primary_example": primary_example["case_id"],
        "memory_flipped_tool_count_changes": tool_rows_flipped,
        "metric_summaries": metric_summaries,
        "flip_association": association,
        "locagent_only_sensitivity": locagent_summaries,
        "locagent_only_flip_association": locagent_association,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(os.path.join(args.output_dir, "trajectory_divergence_per_case.jsonl"), case_rows)
    write_csv(
        os.path.join(args.output_dir, "metric_summary.csv"),
        metric_summaries + locagent_summaries + association + locagent_association,
    )
    write_csv(os.path.join(args.output_dir, "tool_count_deltas.csv"), tool_rows)
    write_json(os.path.join(args.output_dir, "m2_summary.json"), summary)
    with open(os.path.join(args.output_dir, "illustrative_cases.md"), "w", encoding="utf-8") as f:
        f.write(build_illustrative_report(primary_example, top_three))
    with open(
        os.path.join(args.output_dir, "trajectory_divergence_analysis.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            build_main_report(
                summary,
                metric_summaries,
                association,
                locagent_summaries,
                locagent_association,
                tool_rows_flipped,
            )
        )
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip-file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--original-baseline", nargs="+", default=DEFAULT_ORIGINAL_BASELINE)
    parser.add_argument("--rerun-baseline", default=DEFAULT_RERUN_BASELINE)
    parser.add_argument("--original-memory", nargs="+", default=DEFAULT_ORIGINAL_MEMORY)
    parser.add_argument("--rerun-memory", default=DEFAULT_RERUN_MEMORY)
    parser.add_argument("--graph-dir", default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
