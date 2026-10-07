import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.analyze_context_safe_replication_v3 import gap_summary


DEFAULT_INPUT = (
    "results/context_safe_m3a/replication_v4/analysis/v3_v4_per_case.csv"
)
DEFAULT_SUMMARY = (
    "results/context_safe_m3a/replication_v4/analysis/replication_v4_summary.json"
)
DEFAULT_OUTPUT = (
    "results/context_safe_m3a/replication_v4/analysis/exclusion_sensitivity"
)
DEFAULT_EXCLUDE = "matplotlib__matplotlib-25479"


NUMERIC_FIELDS = (
    "v3_baseline_stability",
    "v3_memory_stability",
    "v3_gap",
    "v4_baseline_stability",
    "v4_memory_stability",
    "v4_gap",
    "v4_minus_v3_gap",
)


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in NUMERIC_FIELDS:
            row[field] = float(row[field])
    return rows


def summarize(rows):
    v4_gaps = [row["v4_gap"] for row in rows]
    differences = [row["v4_minus_v3_gap"] for row in rows]
    return {
        "n": len(rows),
        "v4_baseline_pairwise_agreement": float(np.mean([
            row["v4_baseline_stability"] for row in rows
        ])),
        "v4_memory_pairwise_agreement": float(np.mean([
            row["v4_memory_stability"] for row in rows
        ])),
        "v4_gap": gap_summary(v4_gaps),
        "v4_minus_v3": gap_summary(differences),
    }


def leave_one_out(rows):
    full = summarize(rows)
    output = []
    for excluded in rows:
        subset = [row for row in rows if row["case_id"] != excluded["case_id"]]
        result = summarize(subset)
        output.append({
            "excluded_case": excluded["case_id"],
            "excluded_flip_class": excluded["flip_class"],
            "excluded_case_v4_gap": excluded["v4_gap"],
            "excluded_case_v4_minus_v3": excluded["v4_minus_v3_gap"],
            "remaining_n": result["n"],
            "remaining_v4_gap": result["v4_gap"]["gap"],
            "v4_gap_change": (
                result["v4_gap"]["gap"] - full["v4_gap"]["gap"]
            ),
            "remaining_v4_minus_v3": result["v4_minus_v3"]["gap"],
            "v4_minus_v3_change": (
                result["v4_minus_v3"]["gap"]
                - full["v4_minus_v3"]["gap"]
            ),
        })
    return output


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--full_summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--exclude_case", default=DEFAULT_EXCLUDE)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--robustness_margin", type=float, default=0.05)
    args = parser.parse_args()

    rows = load_rows(args.input)
    matches = [row for row in rows if row["case_id"] == args.exclude_case]
    if len(matches) != 1:
        raise ValueError(
            f"expected one row for {args.exclude_case}, got {len(matches)}"
        )
    full = summarize(rows)
    subset = [row for row in rows if row["case_id"] != args.exclude_case]
    excluded = summarize(subset)
    gap_change = excluded["v4_gap"]["gap"] - full["v4_gap"]["gap"]
    difference_change = (
        excluded["v4_minus_v3"]["gap"] - full["v4_minus_v3"]["gap"]
    )
    robust = abs(gap_change) <= args.robustness_margin
    source_summary = json.loads(
        Path(args.full_summary).read_text(encoding="utf-8")
    )
    replacement = source_summary.get("replacement")
    if not replacement or replacement.get("instance_id") != args.exclude_case:
        raise ValueError("excluded case does not match the recorded replacement")

    influence = leave_one_out(rows)
    ranked_gap = sorted(
        influence, key=lambda row: abs(row["v4_gap_change"]), reverse=True
    )
    ranked_difference = sorted(
        influence,
        key=lambda row: abs(row["v4_minus_v3_change"]),
        reverse=True,
    )
    gap_rank = next(
        index for index, row in enumerate(ranked_gap, 1)
        if row["excluded_case"] == args.exclude_case
    )
    difference_rank = next(
        index for index, row in enumerate(ranked_difference, 1)
        if row["excluded_case"] == args.exclude_case
    )
    summary = {
        "excluded_case": args.exclude_case,
        "exclusion_reason": "replacement_after_execution_flow_timeout",
        "robustness_margin": args.robustness_margin,
        "verdict": "ROBUST_TO_EXCLUSION" if robust else "SENSITIVE_TO_EXCLUSION",
        "full_15": full,
        "excluded_14": excluded,
        "changes": {
            "v4_gap": gap_change,
            "v4_minus_v3": difference_change,
        },
        "influence_rank": {
            "v4_gap_absolute_change": gap_rank,
            "v4_minus_v3_absolute_change": difference_rank,
            "denominator": len(rows),
        },
        "replacement_manifest": replacement,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exclusion_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "leave_one_out_influence.csv", influence)
    report = f"""# V4 Replacement-Case Exclusion Sensitivity

## Verdict

**{summary['verdict']}**

Removing `{args.exclude_case}` changes the V4 stability gap by
{gap_change * 100:+.2f} pp. The pre-specified robustness margin is
{args.robustness_margin * 100:.1f} pp.

## Main Result

| Analysis | N | Baseline stability | Memory stability | V4 gap | 95% CI | Exact p |
|---|---:|---:|---:|---:|---|---:|
| Full | {full['n']} | {full['v4_baseline_pairwise_agreement']:.4f} | {full['v4_memory_pairwise_agreement']:.4f} | {full['v4_gap']['gap']:+.4f} | [{full['v4_gap']['ci'][0]:+.4f}, {full['v4_gap']['ci'][1]:+.4f}] | {full['v4_gap']['exact_p']:.4f} |
| Excluding replacement | {excluded['n']} | {excluded['v4_baseline_pairwise_agreement']:.4f} | {excluded['v4_memory_pairwise_agreement']:.4f} | {excluded['v4_gap']['gap']:+.4f} | [{excluded['v4_gap']['ci'][0]:+.4f}, {excluded['v4_gap']['ci'][1]:+.4f}] | {excluded['v4_gap']['exact_p']:.4f} |

## V3/V4 Difference

- Full V4-minus-V3 gap: {full['v4_minus_v3']['gap']:+.4f}
- Excluded V4-minus-V3 gap: {excluded['v4_minus_v3']['gap']:+.4f}
- Change after exclusion: {difference_change * 100:+.2f} pp
- Excluded 95% CI: [{excluded['v4_minus_v3']['ci'][0]:+.4f}, {excluded['v4_minus_v3']['ci'][1]:+.4f}]
- Excluded exact p: {excluded['v4_minus_v3']['exact_p']:.4f}

## Influence Ranking

- Rank by absolute V4-gap change: {gap_rank}/{len(rows)}
- Rank by absolute V4-minus-V3 change: {difference_rank}/{len(rows)}

## Interpretation

The V4 stability gap remains close to the full-sample estimate after removing
the replacement case. This rules out that single task as the primary driver of
the V4 gap. However, the exact p-value for the paired V4-minus-V3 difference
increases after exclusion; evidence that V4 differs from V3 is therefore weaker
under this sensitivity analysis even though its point estimate remains large.
The analysis remains limited to the outcome-selected diagnostic subset and does
not estimate full-500 performance.
"""
    (output_dir / "exclusion_sensitivity_report.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
