#!/usr/bin/env python3
"""Build reproducible static and task-aligned semantic-memory memberships."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

try:
    from scripts.build_repomem_semantic_index import is_test_path
except ModuleNotFoundError:
    from build_repomem_semantic_index import is_test_path


FIGURE8_FILES = [
    "django/db/migrations/operations/special.py",
    "django/db/migrations/serializer.py",
    "django/core/management/commands/makemigrations.py",
]


def git(repo_dir, *args, input_text=None):
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout


def sha256_lines(values):
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_tasks(id_path, metadata_path, repo_name, expected_tasks=None):
    ids = json.loads(Path(id_path).read_text(encoding="utf-8"))
    wanted = set(ids)
    tasks = {}
    with Path(metadata_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            if instance_id not in wanted:
                continue
            metadata = row.get("meta_data", row)
            if metadata.get("repo") != repo_name:
                continue
            base_commit = metadata.get("base_commit")
            if not base_commit:
                raise ValueError(f"Missing base_commit for {instance_id}")
            tasks[instance_id] = {
                "instance_id": instance_id,
                "repo": repo_name,
                "base_commit": base_commit,
            }
    if expected_tasks is not None and len(tasks) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} {repo_name} tasks, got {len(tasks)}")
    if not tasks:
        raise ValueError(f"No {repo_name} tasks found")
    return sorted(tasks.values(), key=lambda row: row["instance_id"])


def is_ancestor(repo_dir, ancestor, descendant):
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "merge-base", "--is-ancestor", ancestor, descendant],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip())
    return result.returncode == 0


def commit_time(repo_dir, commit):
    return int(git(repo_dir, "show", "-s", "--format=%ct", commit).strip())


def choose_static_cutoff(repo_dir, tasks):
    bases = [task["base_commit"] for task in tasks]
    earliest = min(bases, key=lambda sha: (commit_time(repo_dir, sha), sha))
    if all(is_ancestor(repo_dir, earliest, base) for base in bases):
        cutoff = earliest
        method = "chronologically_earliest_task_base"
    else:
        candidates = git(repo_dir, "merge-base", "--all", "--octopus", *bases).split()
        verified = [
            candidate
            for candidate in candidates
            if all(is_ancestor(repo_dir, candidate, base) for base in bases)
        ]
        if not verified:
            raise RuntimeError("No common ancestor found for all task base commits")
        cutoff = max(verified, key=lambda sha: (commit_time(repo_dir, sha), sha))
        method = "latest_verified_best_common_ancestor"
    if not all(is_ancestor(repo_dir, cutoff, base) for base in bases):
        raise RuntimeError(f"Selected cutoff {cutoff} is not an ancestor of every task base")
    return cutoff, method


def parent_commits(repo_dir, commit):
    fields = git(repo_dir, "rev-list", "--parents", "-n", "1", commit).split()
    return fields[1:]


def history_records(repo_dir, anchor, commit_window):
    parents = parent_commits(repo_dir, anchor)
    if not parents:
        return []
    output = git(
        repo_dir,
        "log",
        "--topo-order",
        "--no-renames",
        f"--max-count={commit_window}",
        "--format=%x1e%H",
        "--name-only",
        *parents,
    )
    records = []
    for raw in output.split("\x1e"):
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines:
            records.append((lines[0], lines[1:]))
    return records


def tree_blobs(repo_dir, commit):
    blobs = {}
    output = git(repo_dir, "ls-tree", "-r", commit)
    for line in output.splitlines():
        metadata, path = line.split("\t", 1)
        _, object_type, sha = metadata.split()
        if object_type == "blob":
            blobs[path] = sha
    return blobs


def build_membership(
    repo_dir,
    repo_name,
    anchor,
    commit_window,
    top_k,
    include_tests=False,
):
    records = history_records(repo_dir, anchor, commit_window)
    counts = Counter(
        path
        for _, paths in records
        for path in paths
        if path.endswith(".py") and (include_tests or not is_test_path(path))
    )
    blobs = tree_blobs(repo_dir, anchor)
    eligible = [
        (path, edit_count)
        for path, edit_count in counts.items()
        if path in blobs
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    selected = eligible[:top_k]
    files = [
        {
            "rank": rank,
            "file": path,
            "edit_count": edit_count,
            "source_blob_sha": blobs[path],
        }
        for rank, (path, edit_count) in enumerate(selected, start=1)
    ]
    commit_shas = [sha for sha, _ in records]
    manifest = {
        "schema_version": 2,
        "builder": "build_semantic_membership_manifests.py",
        "repo": repo_name,
        "source_commit_sha": anchor,
        "source_tree_sha": git(repo_dir, "rev-parse", f"{anchor}^{{tree}}").strip(),
        "exclude_anchor_commit": True,
        "commit_window": commit_window,
        "window_commit_count": len(commit_shas),
        "window_commits_sha256": sha256_lines(commit_shas),
        "requested_top_k": top_k,
        "eligible_file_count": len(eligible),
        "selected_count": len(files),
        "include_tests": include_tests,
        "selection_order": "edit_count_desc,file_path_asc",
        "files": files,
        "membership_sha256": sha256_json(files),
    }
    return manifest, blobs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def figure8_ranks(manifest):
    ranks = {row["file"]: row["rank"] for row in manifest["files"]}
    return {path: ranks.get(path) for path in FIGURE8_FILES}


def render_report(summary):
    static = summary["static"]
    lines = [
        "# Semantic Membership Integrity Audit",
        "",
        f"- Static cutoff: `{static['c0_commit_sha']}` ({static['c0_selection_method']}).",
        f"- Static tree: `{static['c0_tree_sha']}`.",
        f"- Window hash: `{static['window_commits_sha256']}`.",
        f"- Top-200 membership hash: `{static['membership_sha256']}`.",
        f"- Selected files: {static['selected_count']}/{static['requested_top_k']}.",
        "",
        "## Figure 8 files",
        "",
        "| File | Static rank | Fresh-control appearances |",
        "|---|---:|---:|",
    ]
    for path in FIGURE8_FILES:
        rank = static["figure8_ranks"].get(path)
        lines.append(
            f"| `{path}` | {rank if rank is not None else 'absent'} | "
            f"{summary['fresh']['figure8_appearances'].get(path, 0)} |"
        )
    if summary["fresh"]["task_count"]:
        lines += [
            "",
            "## Static versus fresh controls",
            "",
            f"- Tasks: {summary['fresh']['task_count']}.",
            f"- Mean Top-K Jaccard: {summary['fresh']['mean_membership_jaccard']:.4f}.",
            f"- Static-file snapshot survival: {summary['fresh']['static_file_survival_rate']:.4f}.",
            f"- Blob drift among surviving static files: {summary['fresh']['static_file_blob_drift_rate']:.4f}.",
        ]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", default="reproduce_lists/verified_astropy22_django231.json")
    parser.add_argument(
        "--task-metadata",
        default="results/locagent_verified_astropy22_django231_tokenhub_glm51/merged_loc_outputs_mrr.jsonl",
    )
    parser.add_argument(
        "--repo-dir",
        default="/data/wjl/addressingmemory/results/orcaloca_experiment3/repos/django__django",
    )
    parser.add_argument("--repo", default="django/django")
    parser.add_argument("--expected-tasks", type=int, default=231)
    parser.add_argument("--commit-window", type=int, default=7000)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--mode", choices=["static", "fresh", "both"], default="both")
    parser.add_argument(
        "--output-dir",
        default="repomem_index/verified_django231_semantic_membership_v2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks(
        args.task_ids,
        args.task_metadata,
        args.repo,
        expected_tasks=args.expected_tasks,
    )
    output = Path(args.output_dir) / args.repo.replace("/", "__")
    cutoff, cutoff_method = choose_static_cutoff(args.repo_dir, tasks)
    static, _ = build_membership(
        args.repo_dir,
        args.repo,
        cutoff,
        args.commit_window,
        args.top_k,
        include_tests=args.include_tests,
    )
    static.update({
        "bank_type": "static",
        "c0_commit_sha": cutoff,
        "c0_tree_sha": static["source_tree_sha"],
        "c0_selection_method": cutoff_method,
        "cohort_task_count": len(tasks),
        "cohort_base_commits_sha256": sha256_lines(sorted(task["base_commit"] for task in tasks)),
    })
    if static["selected_count"] > args.top_k:
        raise RuntimeError(
            f"Static membership exceeds Top-K: {static['selected_count']}/{args.top_k}"
        )
    if args.mode in ("static", "both"):
        write_json(output / "static" / "membership.json", static)

    static_files = {row["file"]: row["source_blob_sha"] for row in static["files"]}
    jaccards = []
    surviving = 0
    drifted = 0
    fresh_figure_counts = Counter()
    if args.mode in ("fresh", "both"):
        for number, task in enumerate(tasks, start=1):
            fresh, blobs = build_membership(
                args.repo_dir,
                args.repo,
                task["base_commit"],
                args.commit_window,
                args.top_k,
                include_tests=args.include_tests,
            )
            fresh.update({
                "bank_type": "fresh_control",
                "instance_id": task["instance_id"],
                "base_commit": task["base_commit"],
            })
            if fresh["selected_count"] > args.top_k:
                raise RuntimeError(
                    f"{task['instance_id']} fresh membership exceeds Top-K: "
                    f"{fresh['selected_count']}/{args.top_k}"
                )
            write_json(
                output / "fresh_controls" / task["instance_id"] / "membership.json",
                fresh,
            )
            fresh_files = {row["file"] for row in fresh["files"]}
            static_names = set(static_files)
            union = static_names | fresh_files
            jaccards.append(len(static_names & fresh_files) / len(union) if union else 1.0)
            for path, source_blob in static_files.items():
                current_blob = blobs.get(path)
                if current_blob:
                    surviving += 1
                    drifted += current_blob != source_blob
            for path, rank in figure8_ranks(fresh).items():
                if rank is not None:
                    fresh_figure_counts[path] += 1
            if number % 25 == 0 or number == len(tasks):
                print(f"built fresh controls {number}/{len(tasks)}", flush=True)

    comparisons = len(tasks) * len(static_files) if jaccards else 0
    summary = {
        "static": {
            key: static[key]
            for key in (
                "c0_commit_sha",
                "c0_tree_sha",
                "c0_selection_method",
                "window_commit_count",
                "window_commits_sha256",
                "requested_top_k",
                "selected_count",
                "membership_sha256",
            )
        },
        "fresh": {
            "task_count": len(jaccards),
            "mean_membership_jaccard": sum(jaccards) / len(jaccards) if jaccards else None,
            "static_file_survival_rate": surviving / comparisons if comparisons else None,
            "static_file_blob_drift_rate": drifted / surviving if surviving else None,
            "figure8_appearances": dict(fresh_figure_counts),
        },
    }
    summary["static"]["figure8_ranks"] = figure8_ranks(static)
    write_json(output / "audit_summary.json", summary)
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary))


if __name__ == "__main__":
    main()
