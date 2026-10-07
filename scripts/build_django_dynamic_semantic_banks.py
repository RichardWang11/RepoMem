#!/usr/bin/env python3
"""Build content-addressed Django task-aligned Top-200 semantic banks.

The paper does not document content-addressed reuse. This replication reuses a
summary only for an identical (repo, file_path, blob_sha, summary_recipe) tuple
and records the canonical task context that generated it.
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def object_key(repo, file_path, blob_sha):
    return sha256_text(f"{repo}\0{file_path}\0{blob_sha}")


def load_memberships(root):
    paths = sorted(Path(root).glob("*/membership.json"))
    manifests = []
    for path in paths:
        manifest = read_json(path)
        if manifest.get("bank_type") != "fresh_control":
            raise ValueError(f"{path}: expected bank_type=fresh_control")
        if manifest.get("selected_count") != 200 or manifest.get("include_tests"):
            raise ValueError(f"{path}: expected exclude-tests Top-200 membership")
        manifests.append((path, manifest))
    if len(manifests) != 231:
        raise ValueError(f"Expected 231 Django task banks, got {len(manifests)}")
    return manifests


def make_recipe(args, builder):
    recipe = {
        "schema_version": 1,
        "builder_mode": "agent",
        "requested_model": args.model,
        "required_response_model": args.required_response_model,
        "temperature": 0,
        "seed": 44,
        "max_iterations": 4,
        "system_prompt": builder.SUMMARY_SYSTEM_PROMPT,
        "user_prompt_template": builder.AGENT_USER_PROMPT,
        "fake_user_prompt": builder.AGENT_FAKE_USER,
        "context_token_budget": builder.AGENT_CONTEXT_TOKEN_BUDGET,
        "tool_observation_token_cap": builder.AGENT_OBSERVATION_TOKEN_CAP,
        "summary_truncation": {"unit": "whitespace_word", "max_words": builder.SUMMARY_MAX_WORDS},
        "retrieval_profile": "bm25-code-v2",
        "content_reuse_key": ["repo", "file_path", "source_blob_sha", "recipe_sha256"],
        "dependency_context_reuse_caveat": (
            "An identical target blob can be summarized in different dependency snapshots. "
            "Reuse is an engineering approximation not specified by the RepoMem paper."
        ),
    }
    recipe["recipe_sha256"] = sha256_text(canonical_json(recipe))
    return recipe


def plan_objects(manifests, recipe):
    uses = defaultdict(list)
    for path, manifest in manifests:
        for item in manifest["files"]:
            key = object_key(manifest["repo"], item["file"], item["source_blob_sha"])
            uses[key].append({
                "instance_id": manifest["instance_id"],
                "base_commit": manifest["base_commit"],
                "source_tree": manifest["source_tree_sha"],
                "membership_path": str(path),
                "file": item["file"],
                "source_blob_sha": item["source_blob_sha"],
                "edit_count": item["edit_count"],
            })
    objects = []
    for key, rows in sorted(uses.items()):
        rows.sort(key=lambda row: row["instance_id"])
        canonical = rows[0]
        objects.append({
            "object_id": key,
            "repo": "django/django",
            "file": canonical["file"],
            "source_blob_sha": canonical["source_blob_sha"],
            "recipe_sha256": recipe["recipe_sha256"],
            "canonical_instance_id": canonical["instance_id"],
            "canonical_source_commit": canonical["base_commit"],
            "canonical_source_tree": canonical["source_tree"],
            "canonical_edit_count": canonical["edit_count"],
            "reuse_count": len(rows),
        })
    return objects


def validate_index_inputs(objects, graph_dir, bm25_dir):
    missing = []
    for item in objects:
        instance = item["canonical_instance_id"]
        if not (Path(graph_dir) / f"{instance}.pkl").is_file():
            missing.append(f"graph:{instance}")
        if not (Path(bm25_dir) / instance).is_dir():
            missing.append(f"bm25:{instance}")
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} canonical indexes; first={missing[:5]}")


def build_objects(args, manifests, objects, recipe, builder):
    from plugins.location_tools.repo_ops import repo_ops

    object_root = Path(args.output_root) / "object_store"
    trajectory_dir = object_root / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    object_path = object_root / "objects.jsonl"
    existing = {row["object_id"]: row for row in read_jsonl(object_path)}
    pending = [
        item for item in objects
        if item["object_id"] not in existing
        or (args.retry_errors and existing[item["object_id"]].get("status") != "ok")
    ]
    if not pending:
        print("All content-addressed summary objects already exist.", flush=True)
        return existing

    validate_index_inputs(pending, args.graph_index_dir, args.code_bm25_index_dir)
    builder.preflight_provider(args.model, args.required_response_model, 44, args.request_timeout)
    repo_ops.GRAPH_INDEX_DIR = args.graph_index_dir
    repo_ops.BM25_INDEX_DIR = args.code_bm25_index_dir
    object_build_id = recipe["recipe_sha256"]
    for number, item in enumerate(pending, 1):
        instance = {
            "instance_id": item["canonical_instance_id"],
            "repo": item["repo"],
            "base_commit": item["canonical_source_commit"],
            "problem_statement": "Build repository semantic memory.",
            "patch": "",
        }
        repo_ops.set_current_issue(instance_data=instance)
        try:
            row = builder.summarize_hotspot_file(
                item["repo"], args.repo_dir, item["file"], item["canonical_edit_count"],
                args.model, "agent", 24000, 0, args.request_timeout,
                item["canonical_source_commit"], item["source_blob_sha"], object_build_id,
                44, 4, trajectory_dir, args.required_response_model,
            )
        finally:
            repo_ops.reset_current_issue()
        row.update({
            "object_id": item["object_id"],
            "recipe_sha256": recipe["recipe_sha256"],
            "source_tree": item["canonical_source_tree"],
            "canonical_instance_id": item["canonical_instance_id"],
            "reuse_count": item["reuse_count"],
        })
        if row.get("trajectory_path"):
            row["trajectory_path"] = str((trajectory_dir / f"{row['memory_id']}.json").resolve())
        existing[item["object_id"]] = row
        write_jsonl(object_path, [existing[key] for key in sorted(existing)])
        ok_rows = [value for value in existing.values() if value.get("status") == "ok"]
        write_json(object_root / "manifest.json", {
            "schema_version": 1,
            "recipe_sha256": recipe["recipe_sha256"],
            "expected_objects": len(objects),
            "completed_objects": len(existing),
            "ok_objects": len(ok_rows),
            "prompt_tokens": sum(row.get("prompt_tokens", 0) for row in ok_rows),
            "completion_tokens": sum(row.get("completion_tokens", 0) for row in ok_rows),
            "response_models": sorted({model for row in ok_rows for model in row.get("response_models", [])}),
            "system_fingerprints": sorted({fp for row in ok_rows for fp in row.get("system_fingerprints", [])}),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"objects {number}/{len(pending)} {item['file']} [{row['status']}]", flush=True)
    return existing


def materialize(args, manifests, objects_by_id, recipe, builder):
    output_root = Path(args.output_root)
    failures = [row for row in objects_by_id.values() if row.get("status") != "ok"]
    if failures:
        raise RuntimeError(f"Cannot materialize with {len(failures)} failed summary objects")
    now = datetime.now(timezone.utc).isoformat()
    completed = 0
    for _, membership in manifests:
        instance_id = membership["instance_id"]
        bank_dir = output_root / "banks" / instance_id
        bank_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "schema_version": 3,
            "reproduction_label": "RepoMem-compatible task-aligned Semantic Memory replication",
            "bank_type": "task_aligned_dynamic",
            "instance_id": instance_id,
            "repo": membership["repo"],
            "source_commit": membership["source_commit_sha"],
            "source_tree": membership["source_tree_sha"],
            "commit_window": membership["commit_window"],
            "window_commit_count": membership["window_commit_count"],
            "window_commits_sha256": membership["window_commits_sha256"],
            "membership_sha256": membership["membership_sha256"],
            "selected_count": membership["selected_count"],
            "include_tests": False,
            "selection_order": membership["selection_order"],
            "summary_recipe": recipe,
            "object_store": str((output_root / "object_store").resolve()),
            "reuse_policy": "identical repo+file_path+source_blob_sha+recipe_sha256",
            "retrieval": {"profile": "bm25-code-v2", "primary_document": "summary_only"},
        }
        spec["build_id"] = sha256_text(canonical_json(spec))
        write_json(bank_dir / "build_spec.json", spec)
        rows = []
        for item in membership["files"]:
            key = object_key(membership["repo"], item["file"], item["source_blob_sha"])
            source = objects_by_id[key]
            row = dict(source)
            row.update({
                "memory_id": key,
                "edit_count": item["edit_count"],
                "source_commit": membership["source_commit_sha"],
                "source_tree": membership["source_tree_sha"],
                "source_blob_sha": item["source_blob_sha"],
                "build_id": spec["build_id"],
                "prompt_instance_sha256": builder.prompt_instance_hash(
                    spec["build_id"], item["file"], item["source_blob_sha"]
                ),
                "summary_source_commit": source["source_commit"],
                "summary_source_tree": source["source_tree"],
                "summary_source_instance_id": source["canonical_instance_id"],
                "bank_instance_id": instance_id,
            })
            rows.append(row)
        write_jsonl(bank_dir / "summaries.jsonl", rows)
        corpora = builder.write_corpora(bank_dir, rows, "bm25-code-v2", [membership])
        response_models = sorted({model for row in rows for model in row.get("response_models", [])})
        build_manifest = {
            "schema_version": 3,
            "build_id": spec["build_id"],
            "repo": membership["repo"],
            "instance_id": instance_id,
            "status": "complete",
            "expected_count": 200,
            "num_rows": 200,
            "num_ok": 200,
            "requested_model": args.model,
            "response_models": response_models,
            "system_fingerprints": sorted({fp for row in rows for fp in row.get("system_fingerprints", [])}),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0,
            "content_addressed_reuse": True,
            "build_spec_sha256": builder.sha256_file(bank_dir / "build_spec.json"),
            "summaries_sha256": builder.sha256_file(bank_dir / "summaries.jsonl"),
            "corpora": corpora,
            "completed_at": now,
        }
        write_json(bank_dir / "build_manifest.json", build_manifest)
        meta = {
            "schema_version": 3,
            "build_id": spec["build_id"],
            "repo": membership["repo"],
            "instance_id": instance_id,
            "top_k": 200,
            "selected_count": 200,
            "num_summaries": 200,
            "num_ok": 200,
            "summary_model": args.model,
            "builder_mode": "agent",
            "c0_commit_sha": membership["source_commit_sha"],
            "c0_tree_sha": membership["source_tree_sha"],
            "window_commits_sha256": membership["window_commits_sha256"],
            "membership_sha256": membership["membership_sha256"],
            "retrieval_profile": "bm25-code-v2",
            "primary_document": "summary_only",
            "build_manifest_sha256": builder.sha256_file(bank_dir / "build_manifest.json"),
        }
        write_json(bank_dir / "meta.json", meta)
        completed += 1
        if completed % 25 == 0 or completed == len(manifests):
            print(f"materialized banks {completed}/{len(manifests)}", flush=True)


def validate_banks(args, manifests, builder):
    from plugins.location_tools.repo_ops import repo_ops

    os.environ.update({
        "REPOMEM_SEMANTIC_INDEX_DIR": str((Path(args.output_root) / "banks").resolve()),
        "REPOMEM_SEMANTIC_LAYOUT": "instance",
        "REPOMEM_SEMANTIC_MEMBERSHIP": "exclude_tests",
        "REPOMEM_SEMANTIC_DOCUMENT": "summary_only",
        "REPOMEM_BM25_PROFILE": "bm25-code-v2",
    })
    repo_ops.REPOMEM_SEMANTIC_CACHE.clear()
    for _, membership in manifests:
        repo_ops.CURRENT_INSTANCE = {
            "instance_id": membership["instance_id"],
            "repo": membership["repo"],
            "base_commit": membership["base_commit"],
        }
        data = repo_ops._load_repomem_semantic_repo(membership["repo"])
        if not data or not data["search_ready"] or len(data["summaries"]) != 200:
            raise RuntimeError(f"Invalid runtime bank: {membership['instance_id']}")
        if data["meta"]["c0_commit_sha"] != membership["base_commit"]:
            raise RuntimeError(f"Snapshot mismatch: {membership['instance_id']}")
    repo_ops.CURRENT_INSTANCE = None
    print(json.dumps({"valid": True, "banks": len(manifests), "documents_per_bank": 200}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "build-objects", "materialize", "validate"])
    parser.add_argument(
        "--membership-root",
        default="repomem_index/verified_django231_semantic_membership_v2/django__django/fresh_controls",
    )
    parser.add_argument(
        "--repo-dir",
        default="/data/wjl/addressingmemory/results/orcaloca_experiment3/repos/django__django",
    )
    parser.add_argument("--graph-index-dir", default="index_data/SWE-bench_Verified/graph_index_v2.3")
    parser.add_argument("--code-bm25-index-dir", default="index_data/SWE-bench_Verified/BM25_index")
    parser.add_argument("--output-root", default="repomem_index/verified_django231_semantic_dynamic_gpt4o_v1")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--required-response-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    os.environ["GRAPH_INDEX_DIR"] = str(Path(args.graph_index_dir).resolve())
    os.environ["BM25_INDEX_DIR"] = str(Path(args.code_bm25_index_dir).resolve())
    from scripts import build_repomem_semantic_index as builder

    manifests = load_memberships(args.membership_root)
    recipe = make_recipe(args, builder)
    objects = plan_objects(manifests, recipe)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "bank_count": len(manifests),
        "entries": sum(item[1]["selected_count"] for item in manifests),
        "unique_summary_objects": len(objects),
        "reuse_fraction": 1 - len(objects) / sum(item[1]["selected_count"] for item in manifests),
        "recipe": recipe,
        "objects": objects,
    }
    write_json(output_root / "plan.json", plan)
    write_json(output_root / "django231_ids.json", [item[1]["instance_id"] for item in manifests])
    if args.action == "plan":
        print(json.dumps({key: value for key, value in plan.items() if key != "objects"}, indent=2))
        return
    object_rows = {
        row["object_id"]: row
        for row in read_jsonl(output_root / "object_store" / "objects.jsonl")
        if row.get("recipe_sha256") == recipe["recipe_sha256"]
    }
    if args.action == "build-objects":
        build_objects(args, manifests, objects, recipe, builder)
    elif args.action == "materialize":
        if len(object_rows) != len(objects):
            raise RuntimeError(f"Need {len(objects)} objects, found {len(object_rows)}")
        materialize(args, manifests, object_rows, recipe, builder)
    else:
        validate_banks(args, manifests, builder)


if __name__ == "__main__":
    main()
