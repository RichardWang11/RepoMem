#!/usr/bin/env python3
"""Prepare and orchestrate leakage-safe static Semantic Memory banks for Verified."""

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from scripts.build_semantic_membership_manifests import (
    build_membership,
    choose_static_cutoff,
    sha256_json,
    sha256_lines,
    write_json,
)
from scripts.eval_verified_file_acc import files_from_patch


REPOS = (
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
)
COMMON_REPO_ROOT = Path(
    "/data/yjm/orcaloca-common93-disambiguation-ablation/work/h1/repos"
)
ADDRESSING_REPO_ROOT = Path(
    "/data/wjl/addressingmemory/results/orcaloca_experiment3/repos"
)
EXPERPAIR_TESTBEDS = Path(
    "/data/wjl/ExpeRepair/ExpeRepair-v1.0/data/verified_500/testbed"
)
DEFAULT_METADATA = (
    Path("results/locagent_verified_astropy22_django231_azure_gpt4o/merged_loc_outputs_mrr.jsonl"),
    Path("results/locagent_verified_remaining247_azure_gpt4o/merged_loc_outputs_mrr.jsonl"),
)


def repo_key(repo):
    return repo.replace("/", "__")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tasks(id_path, metadata_paths):
    ids = json.loads(Path(id_path).read_text(encoding="utf-8"))
    wanted = set(ids)
    rows = {}
    for path in metadata_paths:
        for row in read_jsonl(path):
            instance_id = row.get("instance_id")
            if instance_id in wanted:
                metadata = row.get("meta_data", row)
                rows[instance_id] = {
                    "instance_id": instance_id,
                    "repo": metadata["repo"],
                    "base_commit": metadata["base_commit"],
                    "patch": metadata.get("patch", ""),
                }
    missing = sorted(wanted - set(rows))
    if missing:
        raise ValueError(f"Missing metadata for {len(missing)} instances: {missing[:5]}")
    if len(rows) != 500:
        raise ValueError(f"Expected the Verified 500 cohort, got {len(rows)}")
    unknown = sorted({row["repo"] for row in rows.values()} - set(REPOS))
    if unknown:
        raise ValueError(f"Unexpected repositories: {unknown}")
    return [rows[instance_id] for instance_id in ids]


def repo_overrides():
    raw = os.environ.get("REPOMEM_VERIFIED_REPO_MAP", "").strip()
    if not raw:
        return {}
    path = Path(raw)
    value = json.loads(path.read_text(encoding="utf-8") if path.is_file() else raw)
    return {key: Path(item) for key, item in value.items()}


def repo_candidates(repo):
    key = repo_key(repo)
    # ExpeRepair testbeds are full local clones.  The Common93 clones are
    # promisor/partial clones and cannot guarantee the 7,000-commit window
    # without network access, so keep them only as fallbacks.
    values = list(sorted(EXPERPAIR_TESTBEDS.glob(f"{key}__*")))
    special = {
        "pallets/flask": EXPERPAIR_TESTBEDS / "pallets__flask__2.3",
        "mwaskom/seaborn": EXPERPAIR_TESTBEDS / "mwaskom__seaborn__0.12",
    }
    if repo in special:
        values.insert(0, special[repo])
    values.extend([COMMON_REPO_ROOT / key, ADDRESSING_REPO_ROOT / key])
    return values


def resolve_repo(repo, overrides):
    candidates = ([overrides[repo]] if repo in overrides else []) + repo_candidates(repo)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / ".git").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No git clone found for {repo}; set REPOMEM_VERIFIED_REPO_MAP to JSON "
        "mapping repo names to complete clones"
    )


def git(repo_dir, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require_git_objects(repo_dir, tasks):
    missing = []
    for task in tasks:
        result = git(
            repo_dir, "rev-list", "--max-count=7001", task["base_commit"], check=False
        )
        if result.returncode:
            missing.append(task["instance_id"])
    if missing:
        raise RuntimeError(
            f"{repo_dir} lacks {len(missing)} base-commit trees ({missing[:5]}). "
            "Use a complete clone or run the explicit hydrate action."
        )


def membership_coverage(manifest, tasks):
    members = {row["file"] for row in manifest["files"]}
    covered = 0
    gold_count = 0
    for task in tasks:
        gold = set(files_from_patch(task.get("patch", ""), py_only=True))
        gold_count += len(gold)
        covered += len(gold & members)
    return {
        "gold_file_occurrences": gold_count,
        "covered_gold_file_occurrences": covered,
        "gold_file_coverage": covered / gold_count if gold_count else None,
    }


def prepare(args):
    tasks = load_tasks(args.ids, args.metadata)
    by_repo = {repo: [] for repo in REPOS}
    for task in tasks:
        by_repo[task["repo"]].append(task)
    overrides = repo_overrides()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    configs = []
    for repo in REPOS:
        repo_tasks = sorted(by_repo[repo], key=lambda row: row["instance_id"])
        repo_dir = resolve_repo(repo, overrides)
        require_git_objects(repo_dir, repo_tasks)
        cutoff, method = choose_static_cutoff(repo_dir, repo_tasks)
        primary, _ = build_membership(
            repo_dir, repo, cutoff, args.commit_window, args.top_k, include_tests=False
        )
        sensitivity, _ = build_membership(
            repo_dir, repo, cutoff, args.commit_window, args.top_k, include_tests=True
        )
        common = {
            "bank_type": "static",
            "c0_commit_sha": cutoff,
            "c0_tree_sha": primary["source_tree_sha"],
            "c0_selection_method": method,
            "cohort_task_count": len(repo_tasks),
            "cohort_base_commits_sha256": sha256_lines(
                sorted(task["base_commit"] for task in repo_tasks)
            ),
        }
        primary.update(common)
        sensitivity.update(common)
        key = repo_key(repo)
        repo_root = root / "memberships" / key
        primary_path = repo_root / "exclude_tests.json"
        sensitivity_path = repo_root / "include_tests_sensitivity.json"
        write_json(primary_path, primary)
        write_json(sensitivity_path, sensitivity)
        task_ids_path = root / "cohorts" / f"{key}.json"
        task_ids_path.parent.mkdir(parents=True, exist_ok=True)
        task_ids_path.write_text(
            json.dumps([row["instance_id"] for row in repo_tasks], indent=2) + "\n",
            encoding="utf-8",
        )
        c0_tasks = [row for row in repo_tasks if row["base_commit"] == cutoff]
        c0_instance = c0_tasks[0]["instance_id"] if c0_tasks else f"repomem_semantic_{key}_c0"
        primary_files = {row["file"] for row in primary["files"]}
        sensitivity_files = {row["file"] for row in sensitivity["files"]}
        configs.append({
            "repo": repo,
            "repo_key": key,
            "repo_dir": str(repo_dir),
            "task_count": len(repo_tasks),
            "task_ids": str(task_ids_path),
            "source_commit": cutoff,
            "source_tree": primary["source_tree_sha"],
            "c0_selection_method": method,
            "c0_instance_id": c0_instance,
            "c0_matches_verified_instance": bool(c0_tasks),
            "primary_membership": str(primary_path),
            "sensitivity_membership": str(sensitivity_path),
            "primary_selected_count": primary["selected_count"],
            "sensitivity_selected_count": sensitivity["selected_count"],
            "membership_jaccard": (
                len(primary_files & sensitivity_files) / len(primary_files | sensitivity_files)
                if primary_files or sensitivity_files else 1.0
            ),
            "primary_coverage": membership_coverage(primary, repo_tasks),
            "sensitivity_coverage": membership_coverage(sensitivity, repo_tasks),
        })
    django_ids = [row["instance_id"] for row in tasks if row["repo"] == "django/django"]
    remaining_ids = [row["instance_id"] for row in tasks if row["repo"] != "django/django"]
    for name, values in (("django231", django_ids), ("remaining269", remaining_ids), ("full500", [row["instance_id"] for row in tasks])):
        path = root / "cohorts" / f"{name}.json"
        path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    config = {
        "schema_version": 1,
        "experiment": "SWE-bench Verified Static Semantic-only reproduction",
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "task_count": len(tasks),
        "commit_window": args.commit_window,
        "requested_top_k": args.top_k,
        "primary_membership": "exclude_tests",
        "sensitivity_membership": "include_tests_membership_only",
        "summary_policy": "full_model_response_strip_only_no_truncation",
        "summary_model": "gpt-4o-2024-11-20",
        "localization_model": "gpt-4o-2024-11-20",
        "retrieval": "BM25-code-v2 summary_only",
        "repos": configs,
        "config_sha256": None,
    }
    config["config_sha256"] = sha256_json({**config, "config_sha256": None})
    write_json(root / "experiment_manifest.json", config)
    print(json.dumps({
        "prepared": True,
        "task_count": len(tasks),
        "repo_count": len(configs),
        "output": str(root / "experiment_manifest.json"),
        "selected_counts": {row["repo"]: row["primary_selected_count"] for row in configs},
    }, indent=2))


def load_config(root):
    return json.loads((Path(root) / "experiment_manifest.json").read_text(encoding="utf-8"))


def selected_repos(config, scope):
    if scope == "django":
        return [row for row in config["repos"] if row["repo"] == "django/django"]
    if scope == "remaining":
        return [row for row in config["repos"] if row["repo"] != "django/django"]
    return config["repos"]


def hydrate(args):
    config = load_config(args.output_root)
    for row in selected_repos(config, args.scope):
        print(f"hydrating {row['repo']} from configured remotes", flush=True)
        subprocess.run(
            ["git", "-C", row["repo_dir"], "fetch", "--all", "--tags", "--prune"],
            check=True,
        )


def build_indexes(args):
    config = load_config(args.output_root)
    root = Path(args.output_root)
    for row in selected_repos(config, args.scope):
        index_root = root / "c0_indexes" / row["repo_key"]
        command = [
            args.python, "scripts/build_semantic_c0_indexes.py",
            "--repo_dir", row["repo_dir"],
            "--membership_manifest", row["primary_membership"],
            "--output_dir", str(index_root),
            "--instance_id", row["c0_instance_id"],
        ]
        source_graph = Path(args.graph_root) / f"{row['c0_instance_id']}.pkl"
        source_bm25 = Path(args.bm25_root) / row["c0_instance_id"]
        if row["c0_matches_verified_instance"] and source_graph.is_file() and source_bm25.is_dir():
            command += [
                "--reuse_graph_index", str(source_graph),
                "--reuse_code_bm25_index", str(source_bm25),
            ]
        print(f"building C0 indexes for {row['repo']}", flush=True)
        subprocess.run(command, check=True)


def build_banks(args):
    config = load_config(args.output_root)
    root = Path(args.output_root)
    for row in selected_repos(config, args.scope):
        index_root = root / "c0_indexes" / row["repo_key"]
        command = [
            args.python, "scripts/build_repomem_semantic_index.py",
            "--repo", f"{row['repo']}={row['repo_dir']}",
            "--output_dir", str(root / "banks"),
            "--membership_manifest", row["primary_membership"],
            "--builder_mode", "agent", "--model", "gpt-4o",
            "--temperature", "0", "--seed", "44", "--max_iterations", "4",
            "--required_response_model", "gpt-4o-2024-11-20",
            "--retrieval_profile", "bm25-code-v2", "--num_workers", "1",
            "--retry_errors",
            "--graph_index", str(index_root / "graph" / f"{row['c0_instance_id']}.pkl"),
            "--code_bm25_index", str(index_root / "code_bm25" / row["c0_instance_id"]),
        ]
        if args.pilot_limit:
            command += ["--pilot_limit", str(args.pilot_limit)]
        print(f"building Semantic Memory for {row['repo']}", flush=True)
        subprocess.run(command, check=True)


def validate_banks(args):
    config = load_config(args.output_root)
    root = Path(args.output_root)
    reports = []
    for row in selected_repos(config, args.scope):
        bank = root / "banks" / row["repo_key"]
        expected = args.expected_count or row["primary_selected_count"]
        subprocess.run([
            args.python, "scripts/validate_semantic_memory_build.py", str(bank),
            "--expected_count", str(expected),
            "--required_model", "gpt-4o-2024-11-20",
        ], check=True)
        build_rows = read_jsonl(bank / "summaries.jsonl")
        reports.append({
            "repo": row["repo"],
            "expected": expected,
            "rows": len(build_rows),
            "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in build_rows),
            "completion_tokens": sum(item.get("completion_tokens", 0) for item in build_rows),
        })
    print(json.dumps({"valid": True, "banks": reports}, indent=2))


def status(args):
    config = load_config(args.output_root)
    root = Path(args.output_root)
    rows = []
    for row in config["repos"]:
        bank = root / "banks" / row["repo_key"]
        summaries = read_jsonl(bank / "summaries.jsonl") if (bank / "summaries.jsonl").is_file() else []
        ok = sum(item.get("status") == "ok" for item in summaries)
        rows.append({
            "repo": row["repo"],
            "tasks": row["task_count"],
            "selected": row["primary_selected_count"],
            "summaries": len(summaries),
            "ok": ok,
            "complete": ok == row["primary_selected_count"],
        })
    print(json.dumps({"repos": rows, "complete": all(row["complete"] for row in rows)}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "hydrate", "indexes", "banks", "validate", "status"))
    parser.add_argument("--scope", choices=("django", "remaining", "all"), default="all")
    parser.add_argument("--output-root", default="repomem_index/verified_static_semantic_gpt4o_20241120_v1")
    parser.add_argument("--ids", default="reproduce_lists/verified_full_500.json")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--commit-window", type=int, default=7000)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--pilot-limit", type=int, default=0)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", "/data/conda_envs/locagent/bin/python"))
    parser.add_argument("--graph-root", default="index_data/SWE-bench_Verified/graph_index_v2.3")
    parser.add_argument("--bm25-root", default="index_data/SWE-bench_Verified/BM25_index")
    args = parser.parse_args()
    if not args.metadata:
        args.metadata = [str(path) for path in DEFAULT_METADATA]
    return args


def main():
    args = parse_args()
    actions = {
        "prepare": prepare,
        "hydrate": hydrate,
        "indexes": build_indexes,
        "banks": build_banks,
        "validate": validate_banks,
        "status": status,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
