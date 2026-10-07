import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


DEFAULT_FLIP_FILE = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "flip_decomposition/flip_decomposition.csv"
)
DEFAULT_OUTPUT = "reproduce_lists/verified_m3a_matched_15.json"
DEFAULT_MANIFEST = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "m3_memory_variance/m3a_matched_noise_floor/selection_manifest.json"
)


def largest_remainder_allocation(rows, target, rng):
    by_repo = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row)
    total = len(rows)
    quotas = {repo: target * len(items) / total for repo, items in by_repo.items()}
    allocation = {repo: int(np.floor(quota)) for repo, quota in quotas.items()}
    remaining = target - sum(allocation.values())
    tie_break = {repo: rng.random() for repo in by_repo}
    order = sorted(
        by_repo,
        key=lambda repo: (quotas[repo] - allocation[repo], tie_break[repo]),
        reverse=True,
    )
    for repo in order[:remaining]:
        allocation[repo] += 1

    selected = []
    for repo in sorted(by_repo):
        candidates = sorted(by_repo[repo], key=lambda row: row["case_id"])
        indexes = rng.choice(len(candidates), size=allocation[repo], replace=False)
        selected.extend(candidates[int(index)] for index in sorted(indexes))
    return selected, allocation, quotas


def main():
    parser = argparse.ArgumentParser(description="Create the deterministic M3a matched-repeat subset.")
    parser.add_argument("--flip_file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.flip_file, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(args.seed)
    targets = {"memory_flipped": 10, "stable_harm": 5}
    selected = []
    details = {}
    for flip_class, target in targets.items():
        candidates = [row for row in rows if row["flip_class"] == flip_class]
        chosen, allocation, quotas = largest_remainder_allocation(candidates, target, rng)
        selected.extend(chosen)
        details[flip_class] = {
            "candidate_count": len(candidates),
            "target": target,
            "allocation": allocation,
            "quotas": quotas,
            "selected": [row["case_id"] for row in chosen],
        }

    selected_ids = sorted(row["case_id"] for row in selected)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(selected_ids, f, indent=2)
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "method": "repo-stratified largest-remainder allocation; random selection within repo",
                "source": args.flip_file,
                "selected_count": len(selected_ids),
                "classes": details,
            },
            f,
            indent=2,
        )
    print(f"selected {len(selected_ids)} cases")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
