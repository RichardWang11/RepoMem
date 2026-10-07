import argparse
import ast
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import combinations


SHA_LINE_RE = re.compile(
    r"^SHA:\s+(?P<sha>[0-9a-fA-F]+)\s+\|\s+Commit message:\s+(?P<subject>.*?)\s+"
    r"\|\s+Edited files:\s+(?P<files>\[.*?\])(?:\s+\|\s+score:\s+(?P<score>[-+]?\d+(?:\.\d+)?))?\s*$"
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


DEFAULT_TRAJ_FILES = [
    "results/repomem_verified_astropy22_django231_azure_gpt4o_episodic_with_issues/loc_trajs.jsonl",
    "results/repomem_verified_astropy22_django231_azure_gpt4o_episodic_with_issues_rerun_newempty12/loc_trajs.jsonl",
    "results/repomem_verified_remaining247_azure_gpt4o_episodic_with_issues/loc_trajs.jsonl",
    "results/repomem_verified_remaining247_azure_gpt4o_episodic_with_issues_rerun_newempty10/loc_trajs.jsonl",
]

DEFAULT_INDEX_DIRS = [
    "repomem_index/verified_200_episodic_with_issues",
    "repomem_index/verified_remaining_247_episodic_with_issues_v2",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_repo(repo):
    return repo.replace("/", "__")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def parse_edited_files(raw):
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [str(x) for x in value]


def parse_search_commit_observation(content):
    commits = []
    seen = set()
    for line in (content or "").splitlines():
        match = SHA_LINE_RE.match(line.strip())
        if not match:
            continue
        short_sha = match.group("sha")
        if short_sha in seen:
            continue
        seen.add(short_sha)
        score = match.group("score")
        commits.append(
            {
                "sha": short_sha,
                "short_sha": short_sha,
                "subject": match.group("subject").strip(),
                "changed_files": parse_edited_files(match.group("files")),
                "score": float(score) if score is not None else None,
            }
        )
    return commits


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def top_level_dir(path):
    path = (path or "").strip("/")
    if not path:
        return "."
    if "/" not in path:
        return "."
    return path.split("/", 1)[0]


def bm25_stats(tokenized_docs):
    n_docs = len(tokenized_docs)
    doc_freq = Counter()
    for doc in tokenized_docs:
        doc_freq.update(set(doc))
    avg_len = mean(len(doc) for doc in tokenized_docs) or 0.0
    return doc_freq, avg_len, n_docs


def bm25_score(query_tokens, doc_tokens, doc_freq, avg_doc_len, n_docs, k1=1.5, b=0.75):
    if not query_tokens or not doc_tokens or not avg_doc_len or not n_docs:
        return 0.0
    tf = Counter(doc_tokens)
    score = 0.0
    doc_len = len(doc_tokens)
    for token in query_tokens:
        freq = tf[token]
        if not freq:
            continue
        df = doc_freq.get(token, 0)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        denom = freq + k1 * (1 - b + b * doc_len / avg_doc_len)
        score += idf * freq * (k1 + 1) / denom
    return score


def normalized_subject_bm25_self_similarity(subjects):
    tokenized = [tokenize(s) for s in subjects]
    if len(tokenized) < 2:
        return None
    doc_freq, avg_doc_len, n_docs = bm25_stats(tokenized)
    self_scores = [
        bm25_score(doc, doc, doc_freq, avg_doc_len, n_docs)
        for doc in tokenized
    ]
    pair_scores = []
    for i, j in combinations(range(len(tokenized)), 2):
        left = (
            bm25_score(tokenized[i], tokenized[j], doc_freq, avg_doc_len, n_docs)
            / self_scores[i]
            if self_scores[i] > 0
            else 0.0
        )
        right = (
            bm25_score(tokenized[j], tokenized[i], doc_freq, avg_doc_len, n_docs)
            / self_scores[j]
            if self_scores[j] > 0
            else 0.0
        )
        pair_scores.append((left + right) / 2)
    return mean(pair_scores)


def compute_fragmentation_metrics(commits):
    commits = list(commits)
    file_sets = [set(c.get("changed_files") or []) for c in commits]
    py_file_sets = [
        {f for f in c.get("changed_files") or [] if f.endswith(".py")}
        for c in commits
    ]
    subjects = [c.get("subject", "") for c in commits]
    all_files = [f for c in commits for f in c.get("changed_files") or []]
    dirs = [top_level_dir(f) for f in all_files]
    dir_counts = Counter(dirs)

    return {
        "num_unique_commits": len(commits),
        "num_changed_files_union": len(set(all_files)),
        "file_overlap_jaccard_mean": mean(
            jaccard(a, b) for a, b in combinations(file_sets, 2)
        ),
        "file_overlap_jaccard_py_mean": mean(
            jaccard(a, b) for a, b in combinations(py_file_sets, 2)
        ),
        "subject_bm25_selfsim_norm": normalized_subject_bm25_self_similarity(subjects),
        "distinct_top_level_dir_count": len(set(dirs)),
        "dominant_top_level_dir_share": (
            max(dir_counts.values()) / sum(dir_counts.values()) if dir_counts else None
        ),
    }


def assign_terciles(rows, metric_key, out_key, high_is_fragmented=True):
    valid = [r for r in rows if r.get(metric_key) is not None]
    valid.sort(key=lambda r: (r[metric_key], r.get("instance_id", r.get("id", ""))))
    n = len(valid)
    for i, row in enumerate(valid):
        rank_label = "low" if i < n / 3 else "mid" if i < 2 * n / 3 else "high"
        if high_is_fragmented:
            row[out_key] = rank_label
        else:
            row[out_key] = {"low": "high", "mid": "mid", "high": "low"}[rank_label]
    for row in rows:
        row.setdefault(out_key, "missing")


def load_commit_lookup(index_dirs):
    lookup = {}
    for root in index_dirs:
        if not os.path.isdir(root):
            continue
        for repo_dir in sorted(os.listdir(root)):
            commits_path = os.path.join(root, repo_dir, "commits.jsonl")
            if not os.path.exists(commits_path):
                continue
            for row in load_jsonl(commits_path):
                repo = row.get("repo") or repo_dir.replace("__", "/")
                sha = row.get("sha")
                if not sha:
                    continue
                keys = {
                    (repo, sha),
                    (repo, sha[:9]),
                    (normalize_repo(repo), sha),
                    (normalize_repo(repo), sha[:9]),
                }
                for key in keys:
                    lookup[key] = row
    return lookup


def enrich_commits_from_index(commits, repo, commit_lookup):
    enriched = []
    for commit in commits:
        row = (
            commit_lookup.get((repo, commit["sha"]))
            or commit_lookup.get((repo, commit["short_sha"]))
            or commit_lookup.get((normalize_repo(repo), commit["sha"]))
            or commit_lookup.get((normalize_repo(repo), commit["short_sha"]))
        )
        if row:
            merged = dict(commit)
            merged["sha"] = row.get("sha", commit["sha"])
            merged["short_sha"] = row.get("short_sha", commit["short_sha"])
            merged["subject"] = row.get("subject") or commit.get("subject", "")
            merged["changed_files"] = row.get("changed_files") or commit.get("changed_files", [])
            enriched.append(merged)
        else:
            enriched.append(commit)
    return enriched


def extract_search_commits_from_messages(messages, max_commits):
    commits = []
    search_calls = 0
    seen = set()
    for msg in messages:
        if msg.get("role") != "tool" or msg.get("name") != "search_commit":
            continue
        search_calls += 1
        for commit in parse_search_commit_observation(msg.get("content", "")):
            key = commit["sha"]
            if key in seen:
                continue
            seen.add(key)
            commits.append(commit)
            if len(commits) >= max_commits:
                return commits, search_calls
    return commits, search_calls


def load_observed_search_commits(traj_files, commit_lookup, max_commits):
    per_instance = {}
    for path in traj_files:
        if not os.path.exists(path):
            continue
        for row in load_jsonl(path):
            instance_id = row["instance_id"]
            repo = row.get("meta_data", {}).get("repo", "")
            best = {"repo": repo, "commits": [], "search_calls": 0, "traj_file": path}
            for traj in row.get("loc_trajs", {}).get("trajs", []):
                commits, search_calls = extract_search_commits_from_messages(
                    traj.get("messages", []),
                    max_commits=max_commits,
                )
                if len(commits) > len(best["commits"]):
                    best = {
                        "repo": repo,
                        "commits": commits,
                        "search_calls": search_calls,
                        "traj_file": path,
                    }
            best["commits"] = enrich_commits_from_index(best["commits"], repo, commit_lookup)
            per_instance[instance_id] = best
    return per_instance


def load_pair_cases(pair_files):
    cases = {}
    for k, path in pair_files.items():
        data = load_json(path)
        for row in data["cases"]:
            iid = row["instance_id"]
            case = cases.setdefault(
                iid,
                {
                    "instance_id": iid,
                    "repo_group": row.get("repo_group", "unknown"),
                    "gt": row.get("gt", []),
                    "baseline_pred_top5": row.get("baseline_pred_top5", []),
                    "repomem_pred_top5": row.get("repomem_pred_top5", []),
                },
            )
            case[f"bucket_k{k}"] = row["bucket"]
            case[f"baseline_ok_k{k}"] = bool(row["baseline_ok"])
            case[f"repomem_ok_k{k}"] = bool(row["repomem_ok"])
    return cases


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_terciles(rows, metric_key, tercile_key, k):
    out = []
    for label in ["low", "mid", "high", "missing"]:
        group = [r for r in rows if r.get(tercile_key) == label]
        if not group:
            continue
        n = len(group)
        bw_mc = sum(1 for r in group if r.get(f"bucket_k{k}") == "baseline_wrong__repomem_correct")
        bc_mw = sum(1 for r in group if r.get(f"bucket_k{k}") == "baseline_correct__repomem_wrong")
        out.append(
            {
                "metric": metric_key,
                "fragmentation_tercile": label,
                "k": k,
                "n": n,
                "metric_mean": mean(r.get(metric_key) for r in group if r.get(metric_key) is not None),
                "baseline_acc": mean(1.0 if r.get(f"baseline_ok_k{k}") else 0.0 for r in group),
                "repomem_acc": mean(1.0 if r.get(f"repomem_ok_k{k}") else 0.0 for r in group),
                "delta_acc": (
                    mean(1.0 if r.get(f"repomem_ok_k{k}") else 0.0 for r in group)
                    - mean(1.0 if r.get(f"baseline_ok_k{k}") else 0.0 for r in group)
                ),
                "rescue_count": bw_mc,
                "regression_count": bc_mw,
                "rescue_rate": bw_mc / n,
                "regression_rate": bc_mw / n,
            }
        )
    return out


def format_float(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            values.append(format_float(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def short_commit_preview(commits, limit=5):
    previews = []
    for commit in commits[:limit]:
        files = ", ".join((commit.get("changed_files") or [])[:3])
        previews.append(f"- `{commit.get('short_sha')}` {commit.get('subject', '')} [{files}]")
    return "\n".join(previews)


def write_report(path, rows, summaries, metric_specs):
    total = len(rows)
    with_search = sum(1 for r in rows if r.get("num_unique_commits", 0) > 0)
    k5_counts = Counter(r.get("bucket_k5") for r in rows)
    by_metric_tercile = {
        (row["metric"], row["k"], row["fragmentation_tercile"]): row
        for row in summaries
    }
    key_findings = []
    for metric_key, _, title in metric_specs:
        low = by_metric_tercile.get((metric_key, 5, "low"))
        high = by_metric_tercile.get((metric_key, 5, "high"))
        if not low or not high:
            continue
        key_findings.append(
            {
                "metric": title,
                "low_delta": low["delta_acc"],
                "high_delta": high["delta_acc"],
                "low_regression_rate": low["regression_rate"],
                "high_regression_rate": high["regression_rate"],
                "low_rescue_rate": low["rescue_rate"],
                "high_rescue_rate": high["rescue_rate"],
            }
        )

    lines = [
        "# Memory Fragmentation Analysis",
        "",
        "This is an offline analysis of the existing GPT-4o full500 repaired comparison. "
        "It uses the actual `search_commit` tool observations recorded in `loc_trajs.jsonl`; no LLM calls were made.",
        "",
        f"- cases: {total}",
        f"- cases with parsed SearchCommit commits: {with_search}",
        f"- k=5 regressions (`baseline_correct__repomem_wrong`): {k5_counts.get('baseline_correct__repomem_wrong', 0)}",
        f"- k=5 rescues (`baseline_wrong__repomem_correct`): {k5_counts.get('baseline_wrong__repomem_correct', 0)}",
        "",
        "## Key Findings",
        "",
        "At Acc@5, lower file-overlap coherence shows the clearest fragmentation signal: "
        "memory's gain falls as retrieval becomes more fragmented, and the regression rate rises. "
        "Subject BM25 self-similarity is mixed rather than monotonic. Distinct top-level directory count "
        "does not support the hypothesis in this run; high-dir-count cases have fewer regressions and more rescues.",
        "",
        markdown_table(
            key_findings,
            [
                "metric",
                "low_delta",
                "high_delta",
                "low_regression_rate",
                "high_regression_rate",
                "low_rescue_rate",
                "high_rescue_rate",
            ],
        ),
        "",
    ]

    for metric_key, _, title in metric_specs:
        lines.extend([f"## {title}", ""])
        for k in [1, 3, 5]:
            group = [r for r in summaries if r["metric"] == metric_key and r["k"] == k]
            lines.append(f"### Acc@{k}")
            lines.append(
                markdown_table(
                    group,
                    [
                        "fragmentation_tercile",
                        "n",
                        "metric_mean",
                        "baseline_acc",
                        "repomem_acc",
                        "delta_acc",
                        "rescue_count",
                        "regression_count",
                        "rescue_rate",
                        "regression_rate",
                    ],
                )
            )
            lines.append("")

    regressions = [
        r for r in rows
        if r.get("bucket_k5") == "baseline_correct__repomem_wrong"
        and r.get("num_unique_commits", 0) > 0
    ]
    regressions.sort(
        key=lambda r: (
            -(r.get("distinct_top_level_dir_count") or 0),
            r.get("file_overlap_jaccard_mean") if r.get("file_overlap_jaccard_mean") is not None else 1,
            r["instance_id"],
        )
    )
    lines.extend(["## Most Fragmented k=5 Regression Cases", ""])
    for row in regressions[:10]:
        lines.extend(
            [
                f"### `{row['instance_id']}`",
                "",
                f"- repo: {row.get('repo_group')}",
                f"- file Jaccard mean: {format_float(row.get('file_overlap_jaccard_mean'))}",
                f"- subject BM25 self-sim: {format_float(row.get('subject_bm25_selfsim_norm'))}",
                f"- distinct top-level dirs: {row.get('distinct_top_level_dir_count')}",
                f"- baseline top5: `{row.get('baseline_pred_top5')}`",
                f"- memory top5: `{row.get('repomem_pred_top5')}`",
                "",
                short_commit_preview(row.get("_commits", []), limit=5),
                "",
            ]
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_rows(pair_cases, observed):
    rows = []
    for iid in sorted(pair_cases):
        row = dict(pair_cases[iid])
        obs = observed.get(iid, {"commits": [], "search_calls": 0, "repo": ""})
        commits = obs.get("commits", [])
        row.update(compute_fragmentation_metrics(commits))
        row["repo"] = obs.get("repo", "")
        row["search_commit_calls"] = obs.get("search_calls", 0)
        row["traj_file"] = obs.get("traj_file", "")
        row["top_commit_shas"] = [c.get("short_sha") for c in commits]
        row["_commits"] = commits
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired_k1", default="results/full500_azure_gpt4o_repaired_compare/paired_episodic_vs_baseline_k1.json")
    parser.add_argument("--paired_k3", default="results/full500_azure_gpt4o_repaired_compare/paired_episodic_vs_baseline_k3.json")
    parser.add_argument("--paired_k5", default="results/full500_azure_gpt4o_repaired_compare/paired_episodic_vs_baseline_k5.json")
    parser.add_argument("--traj_file", action="append", dest="traj_files")
    parser.add_argument("--index_dir", action="append", dest="index_dirs")
    parser.add_argument("--output_dir", default="results/full500_azure_gpt4o_repaired_compare/fragmentation")
    parser.add_argument("--max_commits", type=int, default=20)
    args = parser.parse_args()

    traj_files = args.traj_files or DEFAULT_TRAJ_FILES
    index_dirs = args.index_dirs or DEFAULT_INDEX_DIRS
    os.makedirs(args.output_dir, exist_ok=True)

    pair_cases = load_pair_cases({1: args.paired_k1, 3: args.paired_k3, 5: args.paired_k5})
    commit_lookup = load_commit_lookup(index_dirs)
    observed = load_observed_search_commits(traj_files, commit_lookup, args.max_commits)
    rows = build_rows(pair_cases, observed)

    metric_specs = [
        ("file_overlap_jaccard_mean", False, "File-Overlap Jaccard Mean"),
        ("subject_bm25_selfsim_norm", False, "Subject BM25 Self-Similarity"),
        ("distinct_top_level_dir_count", True, "Distinct Top-Level Directory Count"),
    ]
    for metric_key, high_is_fragmented, _ in metric_specs:
        assign_terciles(
            rows,
            metric_key,
            f"{metric_key}_frag_tercile",
            high_is_fragmented=high_is_fragmented,
        )

    serializable_rows = []
    for row in rows:
        copy = dict(row)
        copy.pop("_commits", None)
        serializable_rows.append(copy)

    write_jsonl(os.path.join(args.output_dir, "fragmentation_per_case.jsonl"), serializable_rows)
    write_csv(os.path.join(args.output_dir, "fragmentation_per_case.csv"), serializable_rows)

    all_summaries = []
    for metric_key, _, _ in metric_specs:
        metric_summaries = []
        for k in [1, 3, 5]:
            metric_summaries.extend(
                summarize_terciles(rows, metric_key, f"{metric_key}_frag_tercile", k)
            )
        all_summaries.extend(metric_summaries)
        write_csv(
            os.path.join(args.output_dir, f"tercile_summary_{metric_key}.csv"),
            metric_summaries,
        )

    write_csv(os.path.join(args.output_dir, "tercile_summary_all_metrics.csv"), all_summaries)
    write_report(
        os.path.join(args.output_dir, "fragmentation_report.md"),
        rows,
        all_summaries,
        metric_specs,
    )

    print(json.dumps({
        "cases": len(rows),
        "cases_with_search_commit": sum(1 for r in rows if r.get("num_unique_commits", 0) > 0),
        "output_dir": args.output_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
