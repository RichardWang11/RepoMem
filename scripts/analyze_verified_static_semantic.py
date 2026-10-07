#!/usr/bin/env python3
"""Paired localization analysis for the static Semantic-only reproduction."""

import argparse
import json
import math
import random
from pathlib import Path

from scripts.eval_verified_file_acc import files_from_patch, get_pred_files


def load_many(paths):
    rows = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[row["instance_id"]] = row
    return rows


def gold_files(row):
    metadata = row.get("meta_data", row)
    return list(dict.fromkeys(files_from_patch(metadata.get("patch", ""), py_only=True)))


def ranked_files(row):
    return list(dict.fromkeys(get_pred_files(row)))


def metrics(row):
    gold = set(gold_files(row))
    ranked = ranked_files(row)
    positions = [rank for rank, path in enumerate(ranked, 1) if path in gold]
    first = min(positions, default=None)
    result = {
        f"acc{k}": float(bool(gold) and gold.issubset(set(ranked[:k])))
        for k in (1, 3, 5)
    }
    result["mrr5"] = 1.0 / first if first is not None and first <= 5 else 0.0
    result["first_gold_rank5"] = float(first if first is not None and first <= 5 else 6)
    dcg = sum(1.0 / math.log2(rank + 1) for rank in positions if rank <= 5)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), 5) + 1))
    result["ndcg5"] = dcg / ideal if ideal else 0.0
    return result


def mean(values):
    return sum(values) / len(values) if values else None


def exact_mcnemar_p(rescued, regressed):
    discordant = rescued + regressed
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(rescued, regressed) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def paired_report(ids, baseline, semantic, seed=44):
    paired = [instance_id for instance_id in ids if instance_id in baseline and instance_id in semantic]
    bmetrics = {key: metrics(baseline[key]) for key in paired}
    smetrics = {key: metrics(semantic[key]) for key in paired}
    names = ("acc1", "acc3", "acc5", "mrr5", "first_gold_rank5", "ndcg5")
    aggregate = {}
    for name in names:
        baseline_values = [bmetrics[key][name] for key in paired]
        semantic_values = [smetrics[key][name] for key in paired]
        aggregate[name] = {
            "baseline": mean(baseline_values),
            "semantic": mean(semantic_values),
            "delta_semantic_minus_baseline": mean([
                svalue - bvalue for bvalue, svalue in zip(baseline_values, semantic_values)
            ]),
        }
    differences = [smetrics[key]["acc5"] - bmetrics[key]["acc5"] for key in paired]
    generator = random.Random(seed)
    bootstrap = sorted(
        mean([generator.choice(differences) for _ in differences])
        for _ in range(10000)
    ) if differences else []
    rescued = sum(bmetrics[key]["acc5"] == 0 and smetrics[key]["acc5"] == 1 for key in paired)
    regressed = sum(bmetrics[key]["acc5"] == 1 and smetrics[key]["acc5"] == 0 for key in paired)
    return {
        "expected_tasks": len(ids),
        "paired_tasks": len(paired),
        "completion_rate": len(paired) / len(ids) if ids else 0,
        "metrics": aggregate,
        "acc5_paired_bootstrap_95ci": [bootstrap[249], bootstrap[9749]] if bootstrap else None,
        "acc5_rescued": rescued,
        "acc5_regressed": regressed,
        "acc5_exact_mcnemar_p": exact_mcnemar_p(rescued, regressed),
        "missing_baseline": sorted(set(ids) - set(baseline)),
        "missing_semantic": sorted(set(ids) - set(semantic)),
    }


def markdown(result):
    overall = result["overall"]
    lines = [
        "# Verified Static Semantic-only Reproduction",
        "",
        f"- Paired completion: {overall['paired_tasks']}/{overall['expected_tasks']} "
        f"({overall['completion_rate']:.1%}).",
        f"- Primary Acc@5 delta: {overall['metrics']['acc5']['delta_semantic_minus_baseline']:+.4f}.",
        f"- Django gate: **{'GO' if result.get('django_gate', {}).get('passed') else 'NO-GO'}**.",
        "",
        "| Metric | Baseline | Semantic-only | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("acc1", "Acc@1"), ("acc3", "Acc@3"), ("acc5", "Acc@5"), ("mrr5", "MRR@5"), ("first_gold_rank5", "FirstGoldRank@5 (lower better)"), ("ndcg5", "NDCG@5")):
        row = overall["metrics"][key]
        lines.append(f"| {label} | {row['baseline']:.4f} | {row['semantic']:.4f} | {row['delta_semantic_minus_baseline']:+.4f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--semantic", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--django-gate", action="store_true")
    args = parser.parse_args()
    ids = json.loads(Path(args.ids).read_text(encoding="utf-8"))
    baseline = load_many(args.baseline)
    semantic = load_many(args.semantic)
    overall = paired_report(ids, baseline, semantic)
    repos = {}
    for instance_id in ids:
        row = baseline.get(instance_id) or semantic.get(instance_id)
        if row:
            repo = row.get("meta_data", row).get("repo")
            repos.setdefault(repo, []).append(instance_id)
    per_repo = {
        repo: paired_report(repo_ids, baseline, semantic)
        for repo, repo_ids in sorted(repos.items())
    }
    django = per_repo.get("django/django", overall if args.django_gate else None)
    gate = None
    if django is not None:
        gate = {
            "completion_at_least_99pct": django["completion_rate"] >= 0.99,
            "acc5_delta_at_least_minus_2pp": django["metrics"]["acc5"]["delta_semantic_minus_baseline"] >= -0.02,
            "passed": (
                django["completion_rate"] >= 0.99
                and django["metrics"]["acc5"]["delta_semantic_minus_baseline"] >= -0.02
            ),
            "note": "Bank integrity is enforced separately before runs; this gate covers completion and outcome.",
        }
    result = {
        "primary_endpoint": "complete GoldFile Acc@5",
        "paper_reference": {
            "locagent": {"acc1": 0.648, "acc3": 0.704, "acc5": 0.716},
            "semantic_only": {"acc1": 0.650, "acc3": 0.710, "acc5": 0.728},
        },
        "overall": overall,
        "per_repo": per_repo,
        "django_gate": gate,
        "known_deviations": [
            "static earliest-base repository bank interpretation",
            "gpt-4o-2024-11-20 instead of paper gpt-4o-2024-05-13",
            "BM25-code-v2 because the paper BM25(LLM) tokenizer is unavailable",
            "primary corpus excludes tests; include-tests is membership-only sensitivity",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
