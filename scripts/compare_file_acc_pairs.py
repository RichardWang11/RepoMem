import argparse
import json
from collections import Counter, defaultdict


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_debug(path):
    return {row["instance_id"]: row for row in load_jsonl(path)}


def load_empty_ids(path):
    empty = set()
    for row in load_jsonl(path):
        pred = row.get("found_files", [])
        if not pred or pred == [[]]:
            empty.add(row["instance_id"])
    return empty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_debug", required=True)
    parser.add_argument("--repomem_debug", required=True)
    parser.add_argument("--baseline_loc", required=True)
    parser.add_argument("--repomem_loc", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    key = f"ok@{args.k}"
    base = load_debug(args.baseline_debug)
    mem = load_debug(args.repomem_debug)
    ids = sorted(set(base) & set(mem))

    buckets = Counter()
    per_repo = defaultdict(lambda: Counter())
    rows = []

    for iid in ids:
        b = bool(base[iid][key])
        r = bool(mem[iid][key])
        if b and r:
            bucket = "baseline_correct__repomem_correct"
        elif b and not r:
            bucket = "baseline_correct__repomem_wrong"
        elif not b and r:
            bucket = "baseline_wrong__repomem_correct"
        else:
            bucket = "baseline_wrong__repomem_wrong"

        repo = base[iid].get("repo_group", "unknown")
        buckets[bucket] += 1
        per_repo[repo][bucket] += 1
        per_repo[repo]["total"] += 1
        rows.append(
            {
                "instance_id": iid,
                "repo_group": repo,
                "bucket": bucket,
                "baseline_ok": b,
                "repomem_ok": r,
                "gt": base[iid].get("gt", []),
                "baseline_pred_top5": base[iid].get("pred_top5", []),
                "repomem_pred_top5": mem[iid].get("pred_top5", []),
            }
        )

    base_empty = load_empty_ids(args.baseline_loc)
    mem_empty = load_empty_ids(args.repomem_loc)

    summary = {
        "k": args.k,
        "num_instances": len(ids),
        "buckets": dict(buckets),
        "baseline_acc": (
            buckets["baseline_correct__repomem_correct"]
            + buckets["baseline_correct__repomem_wrong"]
        ) / len(ids) if ids else 0.0,
        "repomem_acc": (
            buckets["baseline_correct__repomem_correct"]
            + buckets["baseline_wrong__repomem_correct"]
        ) / len(ids) if ids else 0.0,
        "net_gain": (
            buckets["baseline_wrong__repomem_correct"]
            - buckets["baseline_correct__repomem_wrong"]
        ),
        "empty_prediction_delta": {
            "baseline_empty": len(base_empty),
            "repomem_empty": len(mem_empty),
            "both_empty": len(base_empty & mem_empty),
            "fixed_empty": len(base_empty - mem_empty),
            "new_empty": len(mem_empty - base_empty),
        },
        "per_repo": {repo: dict(counts) for repo, counts in sorted(per_repo.items())},
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": rows}, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved to: {args.output}")


if __name__ == "__main__":
    main()
