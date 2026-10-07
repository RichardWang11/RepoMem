#!/usr/bin/env python3
"""Audit future Top-1/Top-3 reuse of RepoMem episodic commits."""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


EVENT_FIELDS = [
    "instance_id", "base_commit", "base_time", "visibility_status",
    "visible_count", "excluded_count", "rank", "memory_sha",
    "memory_commit_time", "age_days", "bm25_score", "top1_margin",
    "tied_top1", "retrieval_status",
]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tasks(id_path, metadata_path, expected):
    ids = json.loads(Path(id_path).read_text(encoding="utf-8"))
    django_ids = [item for item in ids if item.startswith("django__django-")]
    if len(django_ids) != expected or len(set(django_ids)) != expected:
        raise ValueError(f"Expected {expected} unique Django task ids, got {len(set(django_ids))}")

    wanted = set(django_ids)
    tasks = {}
    for row in read_jsonl(metadata_path):
        instance_id = row.get("instance_id")
        if instance_id not in wanted:
            continue
        metadata = row.get("meta_data", row)
        task = {
            "instance_id": instance_id,
            "repo": metadata.get("repo"),
            "base_commit": metadata.get("base_commit"),
            "problem_statement": metadata.get("problem_statement"),
        }
        if instance_id in tasks and tasks[instance_id] != task:
            raise ValueError(f"Conflicting metadata for {instance_id}")
        tasks[instance_id] = task

    missing = sorted(wanted - set(tasks))
    if missing:
        raise ValueError(f"Missing task metadata for {missing[:5]} ({len(missing)} total)")
    for task in tasks.values():
        if task["repo"] != "django/django" or not task["base_commit"] or not task["problem_statement"]:
            raise ValueError(f"Incomplete Django metadata for {task['instance_id']}")
    return list(tasks.values())


def git_lines(repo_dir, *args):
    process = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.stdout.splitlines()


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def spearman(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    xr, yr = ranks(xs), ranks(ys)
    xmean, ymean = statistics.mean(xr), statistics.mean(yr)
    numerator = sum((x - xmean) * (y - ymean) for x, y in zip(xr, yr))
    denominator = math.sqrt(
        sum((x - xmean) ** 2 for x in xr) * sum((y - ymean) ** 2 for y in yr)
    )
    return numerator / denominator if denominator else None


def reuse_bucket(count):
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count < 5:
        return "3-4"
    return "5+"


def summarize_cutoff(memory_rows, events, cutoff):
    count_field = f"reuse_count_top{cutoff}"
    counts = [row[count_field] for row in memory_rows]
    qualifying = {row["memory_sha"] for row in memory_rows if row[count_field] >= 3}
    covered = {
        event["instance_id"] for event in events
        if event["rank"] and int(event["rank"]) <= cutoff and event["memory_sha"] in qualifying
    }
    repeated = len(qualifying)
    gate = repeated >= 10 and len(covered) >= 30
    classification = "GO" if gate else ("WEAK" if repeated else "NO-GO")

    ages = defaultdict(list)
    for event in events:
        if not event["rank"] or int(event["rank"]) > cutoff:
            continue
        memory_count = next(
            row[count_field] for row in memory_rows if row["memory_sha"] == event["memory_sha"]
        )
        ages[reuse_bucket(memory_count)].append(float(event["age_days"]))

    first_age_field = f"first_age_days_top{cutoff}"
    correlated = [row for row in memory_rows if row[count_field] > 0]
    correlation = spearman(
        [row[count_field] for row in correlated],
        [row[first_age_field] for row in correlated],
    )
    denominator = len(counts)
    query_count = len({event["instance_id"] for event in events})
    return {
        "eligible_memories": denominator,
        "unique_memories_retrieved": sum(value > 0 for value in counts),
        "counts": {
            "n_eq_0": counts.count(0),
            "n_eq_1": counts.count(1),
            "n_eq_2": counts.count(2),
            "n_ge_3": sum(value >= 3 for value in counts),
            "n_ge_5": sum(value >= 5 for value in counts),
        },
        "probabilities": {
            "p_n_eq_0": counts.count(0) / denominator if denominator else None,
            "p_n_eq_1": counts.count(1) / denominator if denominator else None,
            "p_n_eq_2": counts.count(2) / denominator if denominator else None,
            "p_n_ge_3": sum(value >= 3 for value in counts) / denominator if denominator else None,
            "p_n_ge_5": sum(value >= 5 for value in counts) / denominator if denominator else None,
        },
        "repeated_memory_task_coverage": len(covered),
        "repeated_memory_task_coverage_rate": len(covered) / query_count if query_count else None,
        "gate": {
            "classification": classification,
            "passed": gate,
            "requires_memories_n_ge_3": 10,
            "requires_distinct_tasks_covered": 30,
        },
        "age_days_by_reuse_bucket": {
            bucket: {
                "n_events": len(values),
                "median": statistics.median(values),
                "q1": percentile(values, 0.25),
                "q3": percentile(values, 0.75),
            }
            for bucket, values in sorted(ages.items())
        },
        "spearman_reuse_count_vs_first_reuse_age": correlation,
        "top_memories": [
            {
                "memory_sha": row["memory_sha"],
                "reuse_count": row[count_field],
                "subject": row["subject"],
            }
            for row in sorted(
                memory_rows,
                key=lambda row: (-row[count_field], row["memory_sha"]),
            )[:10]
        ],
    }


def build_memory_rows(commits, opportunities, events):
    by_sha = {row["sha"]: row for row in commits}
    reuse = {1: Counter(), 3: Counter()}
    reuse_events = {1: defaultdict(list), 3: defaultdict(list)}
    for event in events:
        if not event["rank"]:
            continue
        rank = int(event["rank"])
        for cutoff in (1, 3):
            if rank <= cutoff:
                reuse[cutoff][event["memory_sha"]] += 1
                reuse_events[cutoff][event["memory_sha"]].append(event)

    rows = []
    for sha in sorted(opportunities, key=lambda item: by_sha[item]["history_index"]):
        source = by_sha[sha]
        row = {
            "memory_sha": sha,
            "memory_commit_time": source.get("commit_time"),
            "history_index": source.get("history_index"),
            "subject": source.get("subject", ""),
            "opportunity_count": opportunities[sha],
        }
        for cutoff in (1, 3):
            matched = sorted(reuse_events[cutoff].get(sha, []), key=lambda event: event["base_time"])
            count = reuse[cutoff][sha]
            row.update({
                f"reuse_count_top{cutoff}": count,
                f"top{cutoff}_rate": count / opportunities[sha],
                f"first_reuse_top{cutoff}": matched[0]["instance_id"] if matched else "",
                f"last_reuse_top{cutoff}": matched[-1]["instance_id"] if matched else "",
                f"first_age_days_top{cutoff}": float(matched[0]["age_days"]) if matched else "",
                f"last_age_days_top{cutoff}": float(matched[-1]["age_days"]) if matched else "",
            })
        rows.append(row)
    return rows


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary):
    lines = [
        "# RepoMem Future-Reuse Audit v0",
        "",
        f"Retrieval method: **{summary['bm25_profile']}** "
        "(not the paper's unavailable BM25-LLM tokenizer).",
        "",
        "| Cutoff | P(N=0) | P(N=1) | P(N=2) | P(N≥3) | P(N≥5) | N≥3 memories | Covered tasks | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cutoff in (1, 3):
        item = summary[f"top{cutoff}"]
        p, counts = item["probabilities"], item["counts"]
        lines.append(
            f"| Top-{cutoff} | {p['p_n_eq_0']:.4f} | {p['p_n_eq_1']:.4f} | "
            f"{p['p_n_eq_2']:.4f} | {p['p_n_ge_3']:.4f} | {p['p_n_ge_5']:.4f} | "
            f"{counts['n_ge_3']} | {item['repeated_memory_task_coverage']} | "
            f"{item['gate']['classification']} |"
        )

    lines += ["", "## Age diagnostics", ""]
    for cutoff in (1, 3):
        lines += [f"### Top-{cutoff}", "", "| Reuse bucket | Events | Median days | IQR |", "|---|---:|---:|---:|"]
        for bucket, values in summary[f"top{cutoff}"]["age_days_by_reuse_bucket"].items():
            lines.append(
                f"| {bucket} | {values['n_events']} | {values['median']:.1f} | "
                f"{values['q1']:.1f}–{values['q3']:.1f} |"
            )
        rho = summary[f"top{cutoff}"]["spearman_reuse_count_vs_first_reuse_age"]
        lines += ["", f"Spearman reuse count vs. first-reuse age: {rho if rho is not None else 'undefined'}.", ""]

    lines += [
        "",
        "## Audit checks",
        "",
        f"- Queries: {summary['queries']} (no retrieval: {summary['no_retrieval_queries']}).",
        f"- Eligible memories: {summary['top1']['eligible_memories']}.",
        f"- `base_after_index` queries: {summary['base_after_index_queries']}.",
        f"- Query-specific leakage exclusions: {summary['leakage_exclusions']}.",
        f"- Top-1 ties: {summary['top1_ties']}.",
        f"- Ancestry violations: {summary['ancestry_violations']}.",
    ]
    if summary["top1"]["gate"]["classification"] != "GO" and summary["top3"]["gate"]["passed"]:
        lines += [
            "",
            "> Top-1 conservative gate 未过；Top-3（agent-realistic sensitivity）显示 repeated exposure "
            "显著更活跃，需要下一层实验决定 utility estimation 路径。",
        ]
    lines += [
        "",
        "Top-1 is the preregistered primary gate. Top-3 is sensitivity analysis only; "
        "it does not replace the Top-1 conclusion.",
        "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", default="reproduce_lists/verified_astropy22_django231.json")
    parser.add_argument(
        "--task-metadata",
        default="results/locagent_verified_astropy22_django231_tokenhub_glm51/merged_loc_outputs_mrr.jsonl",
    )
    parser.add_argument(
        "--index-dir",
        default="repomem_index/verified_full500_episodic_with_issues_combined",
    )
    parser.add_argument(
        "--repo-dir",
        default="/data/wjl/addressingmemory/results/orcaloca_experiment3/repos/django__django",
    )
    parser.add_argument(
        "--output-dir",
        default="results/future_reuse_audit/django231_bm25_code_v2_top3",
    )
    parser.add_argument(
        "--bm25-profile",
        choices=["bm25-code-v2", "bm25-experepair-v1"],
        default="bm25-code-v2",
    )
    parser.add_argument("--expected-tasks", type=int, default=231)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["REPOMEM_INDEX_DIR"] = str(Path(args.index_dir).resolve())
    os.environ["REPOMEM_BM25_PROFILE"] = args.bm25_profile
    os.environ.setdefault("GRAPH_INDEX_DIR", str(Path("index_data/SWE-bench_Verified/graph_index_v2.3").resolve()))
    os.environ.setdefault("BM25_INDEX_DIR", str(Path("index_data/SWE-bench_Verified/BM25_index").resolve()))

    from plugins.location_tools.repo_ops import repo_ops

    tasks = load_tasks(args.task_ids, args.task_metadata, args.expected_tasks)
    memory_data = repo_ops._load_repomem_repo("django/django")
    if not memory_data:
        raise RuntimeError("Django episodic index could not be loaded")
    commits = memory_data["commits"]
    sha_to_index = memory_data["meta"].get("sha_to_history_index", {})
    window = repo_ops._repomem_commit_window_size()

    for task in tasks:
        task["base_time"] = int(git_lines(args.repo_dir, "show", "-s", "--format=%ct", task["base_commit"])[0])
    tasks.sort(key=lambda task: (task["base_time"], task["instance_id"]))

    events = []
    opportunities = Counter()
    base_after_index = 0
    for number, task in enumerate(tasks, start=1):
        base_index = sha_to_index.get(task["base_commit"])
        end = base_index if base_index is not None else len(commits)
        raw_visible = commits[max(0, end - window):end]
        ancestors = set(git_lines(args.repo_dir, "rev-list", task["base_commit"]))
        visibility_status = "base_in_index"
        if base_index is None:
            visibility_status = "base_after_index"
            base_after_index += 1
            if not commits or commits[-1]["sha"] not in ancestors:
                raise RuntimeError(f"Unsafe base-after-index fallback for {task['instance_id']}")

        repo_ops.CURRENT_ISSUE_ID = task["instance_id"]
        repo_ops.CURRENT_INSTANCE = task
        visible = list(repo_ops._iter_visible_commits(memory_data))
        violations = [row["sha"] for row in visible if row["sha"] not in ancestors]
        if violations:
            raise RuntimeError(
                f"{task['instance_id']} exposes {len(violations)} non-ancestor commits: {violations[:3]}"
            )
        opportunities.update(row["sha"] for row in visible)
        ranked = repo_ops._rank_repomem_rows(task["problem_statement"], visible, 3)
        top1_margin = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else ""
        tied_top1 = bool(len(ranked) > 1 and math.isclose(ranked[0][0], ranked[1][0], abs_tol=1e-12))
        common = {
            "instance_id": task["instance_id"],
            "base_commit": task["base_commit"],
            "base_time": task["base_time"],
            "visibility_status": visibility_status,
            "visible_count": len(visible),
            "excluded_count": len(raw_visible) - len(visible),
            "top1_margin": top1_margin,
            "tied_top1": tied_top1,
        }
        if not ranked:
            events.append({
                **common, "rank": "", "memory_sha": "", "memory_commit_time": "",
                "age_days": "", "bm25_score": "", "retrieval_status": "no_retrieval",
            })
        for rank, (score, row) in enumerate(ranked, start=1):
            events.append({
                **common,
                "rank": rank,
                "memory_sha": row["sha"],
                "memory_commit_time": row.get("commit_time", ""),
                "age_days": round((task["base_time"] - row["commit_time"]) / 86400, 6),
                "bm25_score": score,
                "retrieval_status": "matched",
            })
        if number % 25 == 0 or number == len(tasks):
            print(f"audited {number}/{len(tasks)} tasks", flush=True)

    memory_rows = build_memory_rows(commits, opportunities, events)
    query_rows = {event["instance_id"]: event for event in events}
    summary = {
        "method": args.bm25_profile,
        "bm25_profile": args.bm25_profile,
        "paper_bm25_llm_equivalent": False,
        "query": "problem_statement",
        "corpus": "commit subject + message",
        "commit_window": window,
        "leakage_filter": repo_ops._repomem_leakage_filter_enabled(),
        "queries": len(tasks),
        "no_retrieval_queries": sum(event["retrieval_status"] == "no_retrieval" for event in events),
        "base_after_index_queries": base_after_index,
        "leakage_exclusions": sum(int(event["excluded_count"]) for event in query_rows.values()),
        "top1_ties": sum(bool(event["tied_top1"]) for event in events if event["rank"] == 1),
        "ancestry_violations": 0,
        "top1": summarize_cutoff(memory_rows, events, 1),
        "top3": summarize_cutoff(memory_rows, events, 3),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "retrieval_events.csv", events, EVENT_FIELDS)
    write_csv(output / "memory_reuse.csv", memory_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary))


if __name__ == "__main__":
    main()
