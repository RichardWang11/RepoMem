#!/usr/bin/env python3
"""Build dedicated LocAgent graph/code indexes for a semantic-bank C0."""

import argparse
import hashlib
import io
import json
import os
import pickle
import subprocess
import tarfile
import tempfile
from pathlib import Path

from dependency_graph.build_graph import build_graph
from plugins.location_tools.retriever.bm25_retriever import build_code_retriever_from_repo
from scripts.build_repomem_semantic_index import sha256_file, sha256_path, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--membership_manifest", required=True)
    parser.add_argument("--additional_membership_manifest", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--instance_id", default="repomem_semantic_django_c0")
    parser.add_argument("--reuse_graph_index")
    parser.add_argument("--reuse_code_bm25_index")
    args = parser.parse_args()

    manifest = json.loads(Path(args.membership_manifest).read_text(encoding="utf-8"))
    commit = manifest["source_commit_sha"]
    output = Path(args.output_dir)
    graph_dir = output / "graph"
    bm25_dir = output / "code_bm25"
    graph_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / f"{args.instance_id}.pkl"
    code_path = bm25_dir / args.instance_id
    index_manifest_path = output / "index_manifest.json"
    if index_manifest_path.exists():
        frozen = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        if (
            frozen.get("source_commit") == commit
            and frozen.get("source_tree") == manifest["source_tree_sha"]
            and graph_path.is_file()
            and code_path.is_dir()
            and frozen.get("graph_index_sha256") == sha256_file(graph_path)
            and frozen.get("code_bm25_index_sha256") == sha256_path(code_path)
        ):
            print(graph_path)
            print(code_path)
            return
        raise ValueError("existing C0 indexes do not match their immutable manifest")
    if graph_path.exists() or code_path.exists():
        raise ValueError("partial C0 index exists without a valid index_manifest.json")

    reused = bool(args.reuse_graph_index or args.reuse_code_bm25_index)
    verified_blob_count = None
    if reused:
        if not args.reuse_graph_index or not args.reuse_code_bm25_index:
            raise ValueError("offline reuse requires both graph and code-BM25 indexes")
        source_graph = Path(args.reuse_graph_index).resolve()
        source_bm25 = Path(args.reuse_code_bm25_index).resolve()
        if not source_graph.is_file() or not source_bm25.is_dir():
            raise FileNotFoundError("reused indexes do not exist")
        memberships = [manifest] + [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in args.additional_membership_manifest
        ]
        members = {row["file"]: row for item in memberships for row in item["files"]}
        with source_graph.open("rb") as handle:
            graph = pickle.load(handle)
        for file_path, row in members.items():
            if not graph.has_node(file_path):
                raise ValueError(f"reused graph is missing {file_path}")
            content = graph.nodes[file_path].get("code", "").encode()
            blob_sha = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            if blob_sha != row["source_blob_sha"]:
                raise ValueError(f"reused graph blob mismatch for {file_path}")
        verified_blob_count = len(members)
        os.symlink(os.path.relpath(source_graph, graph_dir), graph_path)
        os.symlink(os.path.relpath(source_bm25, bm25_dir), code_path)

    else:
        with tempfile.TemporaryDirectory(prefix="repomem-c0-") as tmp:
            archive = subprocess.run(
                ["git", "-C", args.repo_dir, "archive", commit],
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(tmp, filter="data")
            graph = build_graph(tmp, global_import=True)
            with graph_path.open("wb") as handle:
                pickle.dump(graph, handle)
            build_code_retriever_from_repo(tmp, persist_path=str(code_path), similarity_top_k=10)

    write_json(index_manifest_path, {
        "repo": manifest["repo"],
        "source_commit": commit,
        "source_tree": manifest["source_tree_sha"],
        "instance_id": args.instance_id,
        "origin": "verified_existing_c0" if reused else "rebuilt_from_git_archive",
        "verified_membership_blob_count": verified_blob_count,
        "graph_index": str(graph_path),
        "graph_index_sha256": sha256_file(graph_path),
        "code_bm25_index": str(code_path),
        "code_bm25_index_sha256": sha256_path(code_path),
    })
    print(graph_path)
    print(code_path)


if __name__ == "__main__":
    main()
