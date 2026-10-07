#!/usr/bin/env python3
"""Freeze an existing Common93 agentic summary bank into runtime schema v3.

This is deliberately a materialization step: it never calls an LLM and it
refuses incomplete source banks. Existing summaries and construction traces
remain the evidence; the output adds deterministic truncation, git provenance,
BM25 corpora, and integrity manifests required by SearchSummary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from scripts import build_repomem_semantic_index as semantic


def git_value(repo_dir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo_dir, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def is_test_path(file_path: str) -> bool:
    return semantic.is_test_path(file_path)


def membership(rows: list[dict], include_tests: bool) -> dict:
    files = [
        {
            "rank": index,
            "file": row["file"],
            "edit_count": row.get("edit_count", 0),
            "source_blob_sha": row["source_blob_sha"],
        }
        for index, row in enumerate(rows, 1)
        if include_tests or not is_test_path(row["file"])
    ]
    payload = {
        "include_tests": include_tests,
        "selected_count": len(files),
        "files": files,
    }
    payload["membership_sha256"] = semantic.sha256_text(
        semantic.canonical_json(files)
    )
    return payload


def materialize(source_dir: Path, repo_dir: Path, output_root: Path) -> Path:
    source_manifest_path = source_dir / "manifest.json"
    source_summaries_path = source_dir / "summaries.jsonl"
    if not source_manifest_path.is_file() or not source_summaries_path.is_file():
        raise FileNotFoundError("source bank requires manifest.json and summaries.jsonl")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_rows = semantic.load_jsonl(source_summaries_path)
    expected = int(source_manifest.get("selected_count", 0))
    ok_rows = [row for row in source_rows if row.get("status") == "ok" and (row.get("summary") or "").strip()]
    if source_manifest.get("status") != "complete" or len(source_rows) != expected or len(ok_rows) != expected:
        raise ValueError(
            f"source bank is not complete: status={source_manifest.get('status')}, "
            f"rows={len(source_rows)}, ok={len(ok_rows)}, expected={expected}"
        )

    instance_id = source_manifest["instance_id"]
    repo_name = source_manifest["repo"]
    source_commit = source_manifest["base_commit"]
    source_tree = git_value(repo_dir, "rev-parse", f"{source_commit}^{{tree}}")
    if source_tree != source_manifest.get("source_tree_sha"):
        raise ValueError("repository tree does not match the source manifest")

    output_dir = output_root / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)

    provenance = []
    for row in source_rows:
        file_path = row["file"]
        blob_sha = git_value(repo_dir, "rev-parse", f"{source_commit}:{file_path}")
        provenance.append({
            "file": file_path,
            "edit_count": row.get("edit_count", 0),
            "source_blob_sha": blob_sha,
        })

    spec = {
        "schema_version": semantic.SCHEMA_VERSION,
        "reproduction_label": "Common93 agentic Semantic Memory materialization",
        "bank_type": "task-aligned",
        "instance_id": instance_id,
        "repo": repo_name,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "commit_window": source_manifest.get("commit_window"),
        "window_commit_count": source_manifest.get("visible_commits"),
        "window_commits_sha256": source_manifest.get("window_commits_sha256"),
        "membership_files": provenance,
        "membership_sha256": semantic.sha256_text(semantic.canonical_json(provenance)),
        "selected_count": expected,
        "builder_mode": "agent",
        "requested_model": source_manifest.get("requested_model"),
        "required_response_model": "gpt-4o-2024-11-20",
        "temperature": source_manifest.get("temperature"),
        "max_iterations": source_manifest.get("max_iterations"),
        "prompt_version": source_manifest.get("prompt_version"),
        "original_build_protocol": source_manifest.get("build_protocol"),
        "original_build_protocol_sha256": source_manifest.get("build_protocol_sha256"),
        "original_builder_sha256": source_manifest.get("builder_sha256"),
        "summary_truncation": {
            "unit": "whitespace_word",
            "max_words": semantic.SUMMARY_MAX_WORDS,
        },
        "retrieval": {
            "profile": "bm25-code-v2",
            "primary_document": "summary_only",
            "sensitivity_document": "path_plus_summary",
            "scorer": "RepoMem BM25",
        },
        "source_manifest_sha256": semantic.sha256_file(source_manifest_path),
        "source_summaries_sha256": semantic.sha256_file(source_summaries_path),
    }
    spec["build_id"] = semantic.sha256_text(semantic.canonical_json(spec))
    build_id = spec["build_id"]

    materialized_rows = []
    for source_row, source_info in zip(source_rows, provenance):
        summary_text = semantic.truncate_summary_words(source_row["summary"])
        memory_id = semantic.sha256_text(semantic.canonical_json([
            build_id, source_row["file"], source_info["source_blob_sha"],
        ]))
        source_trace = source_dir / source_row["trajectory_path"]
        if not source_trace.is_file():
            raise FileNotFoundError(f"missing construction trace: {source_trace}")
        target_trace = trajectory_dir / f"{memory_id}.json"
        shutil.copyfile(source_trace, target_trace)
        responses = source_row.get("provider_metadata", {}).get("responses", [])
        response_models = sorted({r.get("model") for r in responses if r.get("model")})
        fingerprints = sorted({
            r.get("system_fingerprint") for r in responses if r.get("system_fingerprint")
        })
        usage = source_row.get("usage", {})
        materialized_rows.append({
            "memory_id": memory_id,
            "repo": repo_name,
            "instance_id": instance_id,
            "file": source_row["file"],
            "edit_count": source_row.get("edit_count", 0),
            "summary": summary_text,
            "summary_text": summary_text,
            "summary_sha256": semantic.sha256_text(summary_text),
            "summary_model": source_row.get("summary_model"),
            "requested_model": source_manifest.get("requested_model"),
            "response_model": response_models[-1] if response_models else None,
            "response_models": response_models,
            "system_fingerprint": fingerprints[-1] if fingerprints else None,
            "system_fingerprints": fingerprints,
            "temperature": source_manifest.get("temperature"),
            "builder_mode": "agent",
            "status": "ok",
            "source_commit": source_commit,
            "source_tree": source_tree,
            "source_blob_sha": source_info["source_blob_sha"],
            "prompt_instance_sha256": semantic.prompt_instance_hash(
                build_id, source_row["file"], source_info["source_blob_sha"]
            ),
            "build_id": build_id,
            "trajectory_path": str(Path("trajectories") / target_trace.name),
            "trajectory_sha256": semantic.sha256_file(target_trace),
            "prompt_tokens": usage.get("total_prompt_tokens", usage.get("prompt_tokens", 0)),
            "completion_tokens": usage.get("total_completion_tokens", usage.get("completion_tokens", 0)),
            "source_row_sha256": semantic.sha256_text(semantic.canonical_json(source_row)),
        })

    include_manifest = membership(materialized_rows, True)
    exclude_manifest = membership(materialized_rows, False)
    manifests = [exclude_manifest, include_manifest]
    semantic.write_json(output_dir / "build_spec.json", spec)
    semantic.write_jsonl(output_dir / "summaries.jsonl", materialized_rows)
    corpora = semantic.write_corpora(
        output_dir, materialized_rows, "bm25-code-v2", manifests
    )
    response_models = sorted({m for row in materialized_rows for m in row["response_models"]})
    fingerprints = sorted({f for row in materialized_rows for f in row["system_fingerprints"]})
    build_manifest = {
        "schema_version": semantic.SCHEMA_VERSION,
        "build_id": build_id,
        "repo": repo_name,
        "status": "complete",
        "expected_count": expected,
        "num_rows": expected,
        "num_ok": expected,
        "requested_model": source_manifest.get("requested_model"),
        "response_models": response_models,
        "system_fingerprints": fingerprints,
        "prompt_tokens": sum(row["prompt_tokens"] for row in materialized_rows),
        "completion_tokens": sum(row["completion_tokens"] for row in materialized_rows),
        "build_spec_sha256": semantic.sha256_file(output_dir / "build_spec.json"),
        "summaries_sha256": semantic.sha256_file(output_dir / "summaries.jsonl"),
        "corpora": corpora,
        "source_manifest_sha256": semantic.sha256_file(source_manifest_path),
        "source_summaries_sha256": semantic.sha256_file(source_summaries_path),
    }
    semantic.write_json(output_dir / "build_manifest.json", build_manifest)
    semantic.write_json(output_dir / "meta.json", {
        "schema_version": semantic.SCHEMA_VERSION,
        "build_id": build_id,
        "repo": repo_name,
        "instance_id": instance_id,
        "top_k": expected,
        "selected_count": expected,
        "num_summaries": expected,
        "num_ok": expected,
        "summary_model": source_manifest.get("requested_model"),
        "builder_mode": "agent",
        "c0_commit_sha": source_commit,
        "c0_tree_sha": source_tree,
        "window_commits_sha256": source_manifest.get("window_commits_sha256"),
        "membership_variants": {
            "exclude_tests": {
                "membership_sha256": exclude_manifest["membership_sha256"],
                "selected_count": exclude_manifest["selected_count"],
                "include_tests": False,
            },
            "include_tests": {
                "membership_sha256": include_manifest["membership_sha256"],
                "selected_count": include_manifest["selected_count"],
                "include_tests": True,
            },
        },
        "membership_sha256": exclude_manifest["membership_sha256"],
        "retrieval_profile": "bm25-code-v2",
        "primary_document": "summary_only",
        "build_manifest_sha256": semantic.sha256_file(output_dir / "build_manifest.json"),
    })
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = materialize(args.source_dir, args.repo_dir, args.output_root)
    print(json.dumps({"output": str(output), "valid": True}, indent=2))


if __name__ == "__main__":
    main()
