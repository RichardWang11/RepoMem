import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SELECTION = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "label_stability/label_stability_harm30_selection.jsonl"
)
DEFAULT_ORIGINAL_TRAJS = [
    "results/repomem_verified_astropy22_django231_azure_gpt4o_episodic_with_issues/loc_trajs.jsonl",
    "results/repomem_verified_remaining247_azure_gpt4o_episodic_with_issues/loc_trajs.jsonl",
]
DEFAULT_RERUN_TRAJ = "results/check9_label_stability_gpt4o_episodic_harm30/loc_trajs.jsonl"
DEFAULT_FLIP_FILE = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "flip_decomposition/flip_decomposition.csv"
)
DEFAULT_INDEX_DIR = "repomem_index/verified_full500_episodic_with_issues_combined"
DEFAULT_OUTPUT_DIR = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "m1_retrieval_determinism"
)

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
QUERY_SECTION_RE = re.compile(
    r"##Searching for query `(.*?)`\.\.\.\s*(.*?)(?=##Searching for query `|\Z)",
    re.DOTALL,
)
SHA_LINE_RE = re.compile(
    r"^SHA:\s+(?P<sha>[0-9a-fA-F]+).*?\|\s+score:\s+"
    r"(?P<score>[-+]?\d+(?:\.\d+)?)\s*$"
)
EXAMINED_SHA_RE = re.compile(r"^##Commit for sha `([0-9a-fA-F]+)`", re.MULTILINE)


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
        json.dump(value, f, indent=2, ensure_ascii=False)


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


def normalize_sha(value):
    return str(value or "").strip().lower()


def tokenize_query(value):
    return [token.lower() for token in TOKEN_RE.findall(value or "")]


def jaccard(left, right):
    left = set(left or [])
    right = set(right or [])
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def symmetric_best_query_jaccard(left_queries, right_queries):
    left = [set(tokenize_query(q)) for q in left_queries]
    right = [set(tokenize_query(q)) for q in right_queries]
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    def directed(source, target):
        return statistics.mean(max(jaccard(q, other) for other in target) for q in source)

    return (directed(left, right) + directed(right, left)) / 2


def parse_search_observation(content):
    sections = []
    for query, body in QUERY_SECTION_RE.findall(content or ""):
        shas = []
        scores = []
        for line in body.splitlines():
            match = SHA_LINE_RE.match(line.strip())
            if match:
                shas.append(normalize_sha(match.group("sha")))
                scores.append(float(match.group("score")))
        sections.append({"query": query, "commit_ids": shas, "scores": scores})
    return sections


def safe_arguments(raw):
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def extract_memory_calls(messages, instance_id, run_label, traj_index=0):
    pending = {}
    search_rows = []
    examine_rows = []
    parse_errors = []
    call_index = 0

    for message in messages or []:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = function.get("name")
            if name not in {"search_commit", "examine_commit"}:
                continue
            args = safe_arguments(function.get("arguments"))
            call_id = tool_call.get("id")
            pending[call_id] = {
                "name": name,
                "args": args,
                "call_index": call_index,
            }
            if args is None:
                parse_errors.append(f"malformed arguments for {name} call {call_id}")
            call_index += 1

        if message.get("role") != "tool" or message.get("name") not in {
            "search_commit",
            "examine_commit",
        }:
            continue
        call_id = message.get("tool_call_id")
        call = pending.get(call_id)
        if not call:
            parse_errors.append(f"unmatched tool observation {call_id}")
            continue
        args = call.get("args") or {}
        common = {
            "instance_id": instance_id,
            "run": run_label,
            "traj_index": traj_index,
            "call_index": call["call_index"],
            "tool_call_id": call_id,
        }
        if call["name"] == "search_commit":
            requested_top_k = args.get("top_k", 20)
            try:
                requested_top_k = int(requested_top_k)
            except (TypeError, ValueError):
                requested_top_k = 20
            requested_queries = args.get("query_list") or []
            if isinstance(requested_queries, str):
                requested_queries = [requested_queries]
            sections = parse_search_observation(message.get("content", ""))
            for query_index, query in enumerate(requested_queries[:3]):
                section = sections[query_index] if query_index < len(sections) else None
                if section and section["query"] != query:
                    matching = [s for s in sections if s["query"] == query]
                    section = matching[0] if matching else section
                search_rows.append(
                    {
                        **common,
                        "query_index": query_index,
                        "query": str(query),
                        "requested_top_k": requested_top_k,
                        "commit_ids": list(section["commit_ids"]) if section else [],
                        "scores": list(section["scores"]) if section else [],
                        "observation_query": section["query"] if section else None,
                        "missing_observation_section": section is None,
                    }
                )
            if len(sections) != min(len(requested_queries), 3):
                parse_errors.append(
                    f"search call {call_id}: {len(requested_queries[:3])} queries, "
                    f"{len(sections)} observation sections"
                )
        else:
            requested = args.get("sha_list") or []
            if isinstance(requested, str):
                requested = [requested]
            examine_rows.append(
                {
                    **common,
                    "requested_commit_ids": [normalize_sha(x) for x in requested],
                    "returned_commit_ids": [
                        normalize_sha(x)
                        for x in EXAMINED_SHA_RE.findall(message.get("content", ""))
                    ],
                }
            )
    return search_rows, examine_rows, parse_errors


def load_trajectories(paths, selected_ids, run_label):
    cases = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for row in load_jsonl(path):
            instance_id = row.get("instance_id")
            if instance_id not in selected_ids:
                continue
            all_search = []
            all_examine = []
            errors = []
            trajs = (row.get("loc_trajs") or {}).get("trajs") or []
            for traj_index, traj in enumerate(trajs):
                search, examine, parse_errors = extract_memory_calls(
                    traj.get("messages") or [], instance_id, run_label, traj_index
                )
                all_search.extend(search)
                all_examine.extend(examine)
                errors.extend(parse_errors)
            meta = dict(row.get("meta_data") or {})
            meta["instance_id"] = instance_id
            cases[instance_id] = {
                "instance_id": instance_id,
                "source_file": path,
                "trajectory_count": len(trajs),
                "search_rows": all_search,
                "examine_rows": all_examine,
                "parse_errors": errors,
                "meta_data": meta,
            }
    return cases


def commit_pool(search_rows, limit=None):
    ids = []
    for row in search_rows:
        values = row.get("commit_ids") or []
        ids.extend(values[:limit] if limit else values)
    return set(ids)


def examine_pool(examine_rows):
    return {
        sha
        for row in examine_rows
        for sha in row.get("returned_commit_ids") or row.get("requested_commit_ids") or []
    }


def load_flip_classes(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["case_id"]: row["flip_class"] for row in csv.DictReader(f)}


def build_case_rows(selection, original, rerun, flip_classes):
    rows = []
    for selected in selection:
        instance_id = selected.get("case_id") or selected.get("instance_id")
        left = original.get(instance_id)
        right = rerun.get(instance_id)
        if not left or not right:
            rows.append(
                {
                    "case_id": instance_id,
                    "missing_original": left is None,
                    "missing_rerun": right is None,
                    "flip_class": flip_classes.get(instance_id, "unknown"),
                }
            )
            continue
        left_queries = [row["query"] for row in left["search_rows"]]
        right_queries = [row["query"] for row in right["search_rows"]]
        left_commits = commit_pool(left["search_rows"])
        right_commits = commit_pool(right["search_rows"])
        left_top5 = commit_pool(left["search_rows"], limit=5)
        right_top5 = commit_pool(right["search_rows"], limit=5)
        flip_class = flip_classes.get(instance_id, "unknown")
        memory_flipped = flip_class in {"memory_flipped", "harm_to_rescue"}
        rows.append(
            {
                "case_id": instance_id,
                "repo": selected.get("repo"),
                "repo_full": selected.get("repo_full"),
                "flip_class": flip_class,
                "memory_side_flipped": memory_flipped,
                "original_trajectory_count": left["trajectory_count"],
                "rerun_trajectory_count": right["trajectory_count"],
                "original_search_calls": len({r["tool_call_id"] for r in left["search_rows"]}),
                "rerun_search_calls": len({r["tool_call_id"] for r in right["search_rows"]}),
                "original_query_count": len(left_queries),
                "rerun_query_count": len(right_queries),
                "exact_query_set_jaccard": jaccard(left_queries, right_queries),
                "query_token_symmetric_best_jaccard": symmetric_best_query_jaccard(
                    left_queries, right_queries
                ),
                "commit_id_jaccard_full": jaccard(left_commits, right_commits),
                "commit_id_jaccard_top5_per_query": jaccard(left_top5, right_top5),
                "original_commit_count_full": len(left_commits),
                "rerun_commit_count_full": len(right_commits),
                "commits_dropped_in_rerun": len(left_commits - right_commits),
                "commits_added_in_rerun": len(right_commits - left_commits),
                "examine_commit_jaccard": jaccard(
                    examine_pool(left["examine_rows"]), examine_pool(right["examine_rows"])
                ),
                "original_examine_calls": len(left["examine_rows"]),
                "rerun_examine_calls": len(right["examine_rows"]),
                "parse_error_count": len(left["parse_errors"]) + len(right["parse_errors"]),
            }
        )
    return rows


def replay_task_id(task):
    payload = json.dumps(
        [task["instance_id"], task["query"], task["requested_top_k"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_replay_tasks(original, rerun):
    tasks = {}
    for case_map in (original, rerun):
        for instance_id, case in case_map.items():
            for search in case["search_rows"]:
                task = {
                    "instance_id": instance_id,
                    "repo": case["meta_data"].get("repo"),
                    "meta_data": case["meta_data"],
                    "query": search["query"],
                    "requested_top_k": search["requested_top_k"],
                }
                task["task_id"] = replay_task_id(task)
                tasks[task["task_id"]] = task
    return sorted(tasks.values(), key=lambda row: row["task_id"])


def replay_one_task(task, index_dir):
    os.environ["REPOMEM_INDEX_DIR"] = os.path.abspath(index_dir)
    os.environ.setdefault("REPOMEM_COMMIT_WINDOW", "7000")
    os.environ.setdefault("REPOMEM_SEARCH_COMMIT_TOPK_CAP", "20")
    os.environ.setdefault("REPOMEM_LEAKAGE_FILTER", "1")
    os.environ.setdefault("REPOMEM_TOKENIZER", "code")
    from plugins.location_tools.repo_ops import repo_ops

    instance = dict(task["meta_data"])
    instance["instance_id"] = task["instance_id"]
    repo_ops.CURRENT_ISSUE_ID = task["instance_id"]
    repo_ops.CURRENT_INSTANCE = instance
    memory = repo_ops._load_repomem_repo(instance["repo"])
    if not memory:
        raise RuntimeError(f"missing episodic index for {instance['repo']}")
    visible = list(repo_ops._iter_visible_commits(memory))
    doc_freq, avg_doc_len = repo_ops._bm25_corpus_stats(visible)
    query_tokens = repo_ops._repomem_tokenize(task["query"])
    scored = []
    for history_order, row in enumerate(visible):
        score = repo_ops._bm25_score(
            query_tokens, row, doc_freq, len(visible), avg_doc_len
        )
        if score > 0:
            scored.append((score, history_order, row))
    requested = max(1, int(task.get("requested_top_k") or 20))
    effective_top_k = min(requested, 20)
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:effective_top_k]
    return {
        "task_id": task["task_id"],
        "instance_id": task["instance_id"],
        "query": task["query"],
        "requested_top_k": requested,
        "effective_top_k": effective_top_k,
        "visible_commit_count": len(visible),
        "commit_ids": [normalize_sha(row["short_sha"]) for _, _, row in ranked],
        "scores": [score for score, _, _ in ranked],
    }


def run_replay_worker(input_path, output_path, index_dir):
    os.environ["REPOMEM_COMMIT_WINDOW"] = "7000"
    os.environ["REPOMEM_SEARCH_COMMIT_TOPK_CAP"] = "20"
    os.environ["REPOMEM_LEAKAGE_FILTER"] = "1"
    os.environ["REPOMEM_TOKENIZER"] = "code"
    results = [replay_one_task(task, index_dir) for task in load_jsonl(input_path)]
    write_jsonl(output_path, results)


def run_fresh_replays(script_path, task_file, output_dir, index_dir):
    paths = []
    for pass_number in (1, 2):
        path = os.path.join(output_dir, f"offline_replay_pass{pass_number}.jsonl")
        command = [
            sys.executable,
            script_path,
            "--replay-worker",
            "--replay-input",
            task_file,
            "--replay-output",
            path,
            "--index-dir",
            index_dir,
        ]
        subprocess.run(command, check=True, cwd=os.getcwd())
        paths.append(path)
    return paths


def scores_equal(left, right, tolerance=1e-12):
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def compare_replay_passes(tasks, pass1, pass2):
    left = {row["task_id"]: row for row in pass1}
    right = {row["task_id"]: row for row in pass2}
    rows = []
    for task in tasks:
        a = left.get(task["task_id"])
        b = right.get(task["task_id"])
        rows.append(
            {
                "task_id": task["task_id"],
                "case_id": task["instance_id"],
                "query": task["query"],
                "requested_top_k": task["requested_top_k"],
                "pass1_present": a is not None,
                "pass2_present": b is not None,
                "sha_order_identical": bool(a and b and a["commit_ids"] == b["commit_ids"]),
                "score_identical": bool(a and b and scores_equal(a["scores"], b["scores"])),
                "pass1_visible_commit_count": a.get("visible_commit_count") if a else None,
                "pass2_visible_commit_count": b.get("visible_commit_count") if b else None,
            }
        )
    return rows


def compare_replay_to_observations(original, rerun, replay_rows):
    replay = {row["task_id"]: row for row in replay_rows}
    rows = []
    for run_label, case_map in (("original", original), ("rerun", rerun)):
        for instance_id, case in case_map.items():
            for observed in case["search_rows"]:
                task = {
                    "instance_id": instance_id,
                    "query": observed["query"],
                    "requested_top_k": observed["requested_top_k"],
                }
                task_id = replay_task_id(task)
                replayed = replay.get(task_id)
                observed_ids = observed["commit_ids"]
                replay_ids = replayed["commit_ids"] if replayed else []
                min_k = min(len(observed_ids), len(replay_ids), 5)
                rows.append(
                    {
                        "run": run_label,
                        "case_id": instance_id,
                        "query": observed["query"],
                        "requested_top_k": observed["requested_top_k"],
                        "observed_count": len(observed_ids),
                        "replay_count": len(replay_ids),
                        "full_order_identical": observed_ids == replay_ids,
                        "top5_prefix_identical": observed_ids[:min_k] == replay_ids[:min_k],
                        "commit_set_jaccard": jaccard(observed_ids, replay_ids),
                    }
                )
    return rows


def group_query_results(search_rows):
    grouped = defaultdict(list)
    for row in search_rows:
        grouped[row["query"]].append(row["commit_ids"])
    return grouped


def build_same_query_rows(original, rerun):
    rows = []
    for instance_id in sorted(set(original) & set(rerun)):
        left = group_query_results(original[instance_id]["search_rows"])
        right = group_query_results(rerun[instance_id]["search_rows"])
        for query in sorted(set(left) & set(right)):
            pair_values = [(a, b) for a in left[query] for b in right[query]]
            order_equal = [a == b for a, b in pair_values]
            set_equal = [set(a) == set(b) for a, b in pair_values]
            prefix_equal = [a[: min(5, len(a), len(b))] == b[: min(5, len(a), len(b))] for a, b in pair_values]
            rows.append(
                {
                    "case_id": instance_id,
                    "query": query,
                    "original_occurrences": len(left[query]),
                    "rerun_occurrences": len(right[query]),
                    "all_full_order_identical": all(order_equal),
                    "all_commit_sets_identical": all(set_equal),
                    "all_top5_prefix_identical": all(prefix_equal),
                    "min_commit_jaccard": min(jaccard(a, b) for a, b in pair_values),
                    "max_commit_jaccard": max(jaccard(a, b) for a, b in pair_values),
                    "original_result_sizes": json.dumps([len(x) for x in left[query]]),
                    "rerun_result_sizes": json.dumps([len(x) for x in right[query]]),
                }
            )
    return rows


def percentile_interval(values, alpha=0.05):
    values = sorted(values)
    if not values:
        return None, None
    low_index = int((alpha / 2) * (len(values) - 1))
    high_index = int((1 - alpha / 2) * (len(values) - 1))
    return values[low_index], values[high_index]


def group_comparison(case_rows, metric, n_boot=10000, seed=42):
    stable = [float(r[metric]) for r in case_rows if r.get("memory_side_flipped") is False]
    flipped = [float(r[metric]) for r in case_rows if r.get("memory_side_flipped") is True]
    result = {
        "metric": metric,
        "n_memory_stable": len(stable),
        "n_memory_flipped": len(flipped),
        "median_memory_stable": statistics.median(stable) if stable else None,
        "median_memory_flipped": statistics.median(flipped) if flipped else None,
    }
    if not stable or not flipped:
        return result
    result["median_gap_stable_minus_flipped"] = (
        result["median_memory_stable"] - result["median_memory_flipped"]
    )
    try:
        from scipy.stats import mannwhitneyu

        test = mannwhitneyu(stable, flipped, alternative="two-sided")
        result["mann_whitney_u"] = float(test.statistic)
        result["mann_whitney_p"] = float(test.pvalue)
    except ImportError:
        result["mann_whitney_u"] = None
        result["mann_whitney_p"] = None

    import random

    rng = random.Random(seed)
    gaps = []
    for _ in range(n_boot):
        stable_sample = [rng.choice(stable) for _ in stable]
        flipped_sample = [rng.choice(flipped) for _ in flipped]
        gaps.append(statistics.median(stable_sample) - statistics.median(flipped_sample))
    low, high = percentile_interval(gaps)
    result["bootstrap_ci_low"] = low
    result["bootstrap_ci_high"] = high
    return result


def rate(rows, key):
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else None


def fmt(value, digits=3):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def build_report(summary):
    group = summary["memory_flip_group_comparison"]
    lines = [
        "# M1 Retrieval Determinism Check",
        "",
        "## Data",
        "",
        f"- paired cases: {summary['n_paired_cases']}",
        f"- original observed queries: {summary['n_original_queries']}",
        f"- rerun observed queries: {summary['n_rerun_queries']}",
        f"- exact same-query case/query groups: {summary['same_query_groups']}",
        f"- offline replay tasks: {summary['offline_replay_tasks']}",
        "",
        "The original merge-time args files are stale; this analysis uses the actual saved trajectories. "
        "All paired cases contain one saved trajectory on each side.",
        "",
        "## M1a: Trajectory Retrieval Divergence",
        "",
        "| metric | all-case median | memory-stable median | memory-flipped median | stable-minus-flipped gap | Mann-Whitney p | bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, values in group.items():
        all_median = summary["all_case_medians"].get(metric)
        lines.append(
            f"| {metric} | {fmt(all_median)} | {fmt(values.get('median_memory_stable'))} | "
            f"{fmt(values.get('median_memory_flipped'))} | "
            f"{fmt(values.get('median_gap_stable_minus_flipped'))} | "
            f"{fmt(values.get('mann_whitney_p'), 4)} | "
            f"[{fmt(values.get('bootstrap_ci_low'))}, {fmt(values.get('bootstrap_ci_high'))}] |"
        )
    lines.extend(
        [
            "",
            "Full-pool Jaccard is sensitive to the model-requested top_k. "
            "The top5-per-query metric is the cap-normalized primary comparison.",
            "",
            "## M1b: Fresh-Process Offline Replay",
            "",
            f"- SHA order identical: {summary['m1b_sha_order_identical_count']}/"
            f"{summary['offline_replay_tasks']} ({fmt(summary['m1b_sha_order_identical_rate'])})",
            f"- scores identical within 1e-12: {summary['m1b_score_identical_count']}/"
            f"{summary['offline_replay_tasks']} ({fmt(summary['m1b_score_identical_rate'])})",
            f"- replay vs original observation top-5 prefix: {fmt(summary['replay_vs_original_top5_rate'])}",
            f"- replay vs rerun observation top-5 prefix: {fmt(summary['replay_vs_rerun_top5_rate'])}",
            "",
            "## M1c: Exact Same-Query Comparison",
            "",
            f"- full ordered results identical: {summary['m1c_full_order_identical_count']}/"
            f"{summary['same_query_groups']} ({fmt(summary['m1c_full_order_identical_rate'])})",
            f"- top-5 prefixes identical: {summary['m1c_top5_identical_count']}/"
            f"{summary['same_query_groups']} ({fmt(summary['m1c_top5_identical_rate'])})",
            "",
            "## Decision",
            "",
            summary["decision"],
            "",
            "## Limitations",
            "",
            "This is a trajectory-level diagnostic on 30 cases selected because the original run was harmful. "
            "It does not estimate full-500 stability or establish that memory is intrinsically 2.4x noisier. "
            "The original and stability runs also used different iteration budgets and model strings, so agent-level "
            "divergence remains observational until a parameter-matched repeat is run.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(args):
    os.makedirs(args.output_dir, exist_ok=True)
    selection = load_jsonl(args.selection)
    selected_ids = {row.get("case_id") or row.get("instance_id") for row in selection}
    original = load_trajectories(args.original_trajs, selected_ids, "original")
    rerun = load_trajectories([args.rerun_traj], selected_ids, "rerun")
    flip_classes = load_flip_classes(args.flip_file)
    case_rows = build_case_rows(selection, original, rerun, flip_classes)

    missing = [r for r in case_rows if r.get("missing_original") or r.get("missing_rerun")]
    if missing:
        raise RuntimeError(f"trajectory join failed for {len(missing)} cases")

    tasks = build_replay_tasks(original, rerun)
    task_file = os.path.join(args.output_dir, "offline_replay_tasks.jsonl")
    write_jsonl(task_file, tasks)
    existing_replays = [
        os.path.join(args.output_dir, "offline_replay_pass1.jsonl"),
        os.path.join(args.output_dir, "offline_replay_pass2.jsonl"),
    ]
    if args.reuse_replays and all(os.path.exists(path) for path in existing_replays):
        replay_paths = existing_replays
    else:
        replay_paths = run_fresh_replays(
            os.path.abspath(__file__), task_file, args.output_dir, args.index_dir
        )
    replay1 = load_jsonl(replay_paths[0])
    replay2 = load_jsonl(replay_paths[1])
    replay_compare = compare_replay_passes(tasks, replay1, replay2)
    observation_compare = compare_replay_to_observations(original, rerun, replay1)
    same_query_rows = build_same_query_rows(original, rerun)

    metrics = [
        "commit_id_jaccard_top5_per_query",
        "commit_id_jaccard_full",
        "query_token_symmetric_best_jaccard",
        "examine_commit_jaccard",
    ]
    group_results = {metric: group_comparison(case_rows, metric) for metric in metrics}
    all_case_medians = {
        metric: statistics.median(float(row[metric]) for row in case_rows)
        for metric in metrics
    }
    original_obs = [r for r in observation_compare if r["run"] == "original"]
    rerun_obs = [r for r in observation_compare if r["run"] == "rerun"]
    m1b_sha_rate = rate(replay_compare, "sha_order_identical")
    m1c_top5_rate = rate(same_query_rows, "all_top5_prefix_identical")
    median_top5 = all_case_medians["commit_id_jaccard_top5_per_query"]
    primary_group = group_results["commit_id_jaccard_top5_per_query"]
    flipped_is_more_divergent = (
        primary_group["median_memory_flipped"]
        < primary_group["median_memory_stable"]
    )
    divergence_association_supported = (
        flipped_is_more_divergent
        and primary_group.get("mann_whitney_p") is not None
        and primary_group["mann_whitney_p"] < 0.10
    )
    if m1b_sha_rate != 1.0:
        decision = "BM25 fresh-process replay is not 100% deterministic. Stop before M2 and diagnose retrieval ordering/configuration."
    elif m1c_top5_rate == 1.0 and median_top5 < 0.60 and divergence_association_supported:
        decision = "BM25 and same-query retrieval are deterministic. Memory-flipped cases have significantly more divergent retrieved pools, supporting query-driven trajectory divergence; proceed to M2."
    elif m1c_top5_rate == 1.0 and median_top5 < 0.60:
        decision = (
            "BM25 and same-query retrieval are deterministic, and agent-generated queries often produce different commit pools. "
            "However, memory-flipped cases are not more divergent than memory-stable cases, so retrieval-pool divergence does not explain the observed flips. "
            "Proceed to M2 to test post-retrieval tool use and reasoning divergence."
        )
    elif m1c_top5_rate == 1.0 and median_top5 > 0.90:
        decision = "Retrieval is highly stable. Inspect ExamineCommit usage and post-retrieval reasoning before broader M2 analysis."
    else:
        decision = "BM25 replay is deterministic, with partial trajectory-level retrieval divergence. Proceed to M2 while retaining retrieval divergence as a partial mechanism."

    summary = {
        "n_selected_cases": len(selection),
        "n_paired_cases": len(case_rows),
        "n_original_queries": sum(len(c["search_rows"]) for c in original.values()),
        "n_rerun_queries": sum(len(c["search_rows"]) for c in rerun.values()),
        "same_query_groups": len(same_query_rows),
        "offline_replay_tasks": len(tasks),
        "m1b_sha_order_identical_count": sum(r["sha_order_identical"] for r in replay_compare),
        "m1b_sha_order_identical_rate": m1b_sha_rate,
        "m1b_score_identical_count": sum(r["score_identical"] for r in replay_compare),
        "m1b_score_identical_rate": rate(replay_compare, "score_identical"),
        "replay_vs_original_top5_rate": rate(original_obs, "top5_prefix_identical"),
        "replay_vs_rerun_top5_rate": rate(rerun_obs, "top5_prefix_identical"),
        "m1c_full_order_identical_count": sum(r["all_full_order_identical"] for r in same_query_rows),
        "m1c_full_order_identical_rate": rate(same_query_rows, "all_full_order_identical"),
        "m1c_top5_identical_count": sum(r["all_top5_prefix_identical"] for r in same_query_rows),
        "m1c_top5_identical_rate": m1c_top5_rate,
        "all_case_medians": all_case_medians,
        "memory_flip_group_comparison": group_results,
        "retrieval_divergence_association_supported": divergence_association_supported,
        "decision": decision,
    }

    write_jsonl(os.path.join(args.output_dir, "retrieval_determinism_per_case.jsonl"), case_rows)
    write_csv(os.path.join(args.output_dir, "retrieval_calls.csv"), observation_compare)
    write_csv(os.path.join(args.output_dir, "offline_replay.csv"), replay_compare)
    write_csv(os.path.join(args.output_dir, "same_query_pairs.csv"), same_query_rows)
    write_json(os.path.join(args.output_dir, "m1_summary.json"), summary)
    with open(
        os.path.join(args.output_dir, "retrieval_determinism_check.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(build_report(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--original-trajs", nargs="+", default=DEFAULT_ORIGINAL_TRAJS)
    parser.add_argument("--rerun-traj", default=DEFAULT_RERUN_TRAJ)
    parser.add_argument("--flip-file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-worker", action="store_true")
    parser.add_argument("--replay-input")
    parser.add_argument("--replay-output")
    parser.add_argument("--reuse-replays", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replay_worker:
        if not args.replay_input or not args.replay_output:
            raise SystemExit("--replay-worker requires --replay-input and --replay-output")
        run_replay_worker(args.replay_input, args.replay_output, args.index_dir)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
