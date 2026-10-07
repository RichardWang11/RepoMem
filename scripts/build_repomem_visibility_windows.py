#!/usr/bin/env python3
"""Build exact base-specific ancestor windows for an episodic-memory index."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from util.runtime.episodic_visibility import VisibilityBuildError, write_visibility_windows


def load_instances(dataset: str, split: str):
    path = Path(dataset)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("instances", data.get("rows"))
        if not isinstance(data, list):
            raise ValueError("local dataset must be a JSON/JSONL list of instance records")
        return data
    return load_dataset(dataset, split=split)


def parse_repo_map(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected OWNER/REPO=/path/to/full/clone, got {item!r}")
        repo, path = item.split("=", 1)
        result[repo] = path
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        help="OWNER/REPO=/path/to/full/non-shallow/non-partial/clone",
    )
    parser.add_argument("--selected_ids", help="Optional JSON list of instance IDs")
    args = parser.parse_args()

    selected = None
    if args.selected_ids:
        selected = set(json.loads(Path(args.selected_ids).read_text(encoding="utf-8")))
    grouped = defaultdict(list)
    for row in load_instances(args.dataset, args.split):
        if selected is not None and row["instance_id"] not in selected:
            continue
        grouped[row["repo"]].append(row["base_commit"])

    repo_map = parse_repo_map(args.repo)
    missing = sorted(set(grouped) - set(repo_map))
    if missing:
        raise VisibilityBuildError(f"missing full Git clones for repositories: {missing}")
    for repo_name, bases in sorted(grouped.items()):
        index_repo_dir = Path(args.index_dir) / repo_name.replace("/", "__")
        manifest = write_visibility_windows(
            repo_name, repo_map[repo_name], index_repo_dir, bases
        )
        print(f"{repo_name}: wrote {len(manifest['bases'])} strict windows")


if __name__ == "__main__":
    main()
