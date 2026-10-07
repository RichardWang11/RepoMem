import argparse
import csv
import json
import math
import os

import pandas as pd


DEFAULT_INPUT = "results/full500_azure_gpt4o_repaired_compare/check5_retrieval_quality/retrieval_quality_per_case.jsonl"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/check6_7_coverage_decomposition"
NOMINAL_FULL500_N = 500


CHECK6_BUCKETS = [
    "coverage_eq_0",
    "coverage_0_0.25",
    "coverage_0.25_0.5",
    "coverage_0.5_0.75",
    "coverage_0.75_1.0",
]

CHECK7_BUCKETS = [
    "coverage_eq_0",
    "coverage_0_0.25",
    "coverage_0.25_0.5",
    "coverage_0.5_1.0",
    "coverage_gt_0",
]


def assign_coverage_bucket(value):
    value = float(value)
    if value == 0:
        return "coverage_eq_0"
    if 0 < value <= 0.25:
        return "coverage_0_0.25"
    if 0.25 < value <= 0.5:
        return "coverage_0.25_0.5"
    if 0.5 < value <= 0.75:
        return "coverage_0.5_0.75"
    if 0.75 < value <= 1.0:
        return "coverage_0.75_1.0"
    return "coverage_out_of_range"


def assign_check7_bucket(value):
    value = float(value)
    if value == 0:
        return "coverage_eq_0"
    if 0 < value <= 0.25:
        return "coverage_0_0.25"
    if 0.25 < value <= 0.5:
        return "coverage_0.25_0.5"
    if 0.5 < value <= 1.0:
        return "coverage_0.5_1.0"
    return "coverage_out_of_range"


def baseline_recall_bucket(value):
    value = float(value)
    if value <= 0.5:
        return "baseline_recall_0_0.5"
    if value <= 1.0:
        return "baseline_recall_0.5_1.0"
    return "baseline_recall_out_of_range"


def coverage_distribution_rows(df, nominal_n=NOMINAL_FULL500_N):
    work = df.copy()
    work["coverage_bucket"] = work["retrieval_gold_coverage"].apply(assign_coverage_bucket)
    total = len(work)
    rows = []
    for bucket in CHECK6_BUCKETS:
        n = int((work["coverage_bucket"] == bucket).sum())
        rows.append(
            {
                "bucket": bucket,
                "count": n,
                "share_analyzed": n / total if total else 0.0,
                "share_nominal_500": n / nominal_n if nominal_n else math.nan,
            }
        )
    return rows


def coverage_describe(df):
    desc = df["retrieval_gold_coverage"].describe()
    quantiles = df["retrieval_gold_coverage"].quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    row = {str(k): float(v) for k, v in desc.to_dict().items()}
    for q, value in quantiles.items():
        row[f"q{int(q * 100):02d}"] = float(value)
    return row


def coverage_baseline_crosstab_rows(df):
    work = df.copy()
    work["coverage_bucket"] = work["retrieval_gold_coverage"].apply(assign_coverage_bucket)
    work["baseline_recall_bucket"] = work["baseline_recall_at5"].apply(baseline_recall_bucket)
    rows = []
    for coverage_bucket in CHECK6_BUCKETS:
        sub = work[work["coverage_bucket"] == coverage_bucket]
        for recall_bucket in ["baseline_recall_0_0.5", "baseline_recall_0.5_1.0"]:
            rows.append(
                {
                    "coverage_bucket": coverage_bucket,
                    "baseline_recall_bucket": recall_bucket,
                    "count": int((sub["baseline_recall_bucket"] == recall_bucket).sum()),
                }
            )
    return rows


def bucket_frame(df, bucket):
    if bucket == "coverage_eq_0":
        return df[df["retrieval_gold_coverage"] == 0]
    if bucket == "coverage_0_0.25":
        return df[(df["retrieval_gold_coverage"] > 0) & (df["retrieval_gold_coverage"] <= 0.25)]
    if bucket == "coverage_0.25_0.5":
        return df[(df["retrieval_gold_coverage"] > 0.25) & (df["retrieval_gold_coverage"] <= 0.5)]
    if bucket == "coverage_0.5_1.0":
        return df[(df["retrieval_gold_coverage"] > 0.5) & (df["retrieval_gold_coverage"] <= 1.0)]
    if bucket == "coverage_gt_0":
        return df[df["retrieval_gold_coverage"] > 0]
    raise KeyError(bucket)


def compute_gain_by_bucket(df):
    total_n = len(df)
    total_acc1_gain_count = float((df["memory_precision_at1"] - df["baseline_precision_at1"]).sum())
    total_acc5_gain_count = float((df["memory_acc_at5_original"] - df["baseline_acc_at5_original"]).sum())
    rows = []
    for bucket in CHECK7_BUCKETS:
        sub = bucket_frame(df, bucket)
        n = len(sub)
        baseline_at1 = float(sub["baseline_precision_at1"].mean()) if n else math.nan
        memory_at1 = float(sub["memory_precision_at1"].mean()) if n else math.nan
        baseline_at5 = float(sub["baseline_acc_at5_original"].mean()) if n else math.nan
        memory_at5 = float(sub["memory_acc_at5_original"].mean()) if n else math.nan
        acc1_gain_count = float((sub["memory_precision_at1"] - sub["baseline_precision_at1"]).sum()) if n else 0.0
        acc5_gain_count = float((sub["memory_acc_at5_original"] - sub["baseline_acc_at5_original"]).sum()) if n else 0.0
        rows.append(
            {
                "bucket": bucket,
                "N": int(n),
                "baseline_acc1": baseline_at1,
                "memory_acc1": memory_at1,
                "delta_acc1": memory_at1 - baseline_at1 if n else math.nan,
                "baseline_acc5": baseline_at5,
                "memory_acc5": memory_at5,
                "delta_acc5": memory_at5 - baseline_at5 if n else math.nan,
                "acc1_gain_count": acc1_gain_count,
                "acc5_gain_count": acc5_gain_count,
                "acc1_gain_contribution_pp": acc1_gain_count / total_n if total_n else math.nan,
                "acc5_gain_contribution_pp": acc5_gain_count / total_n if total_n else math.nan,
                "acc1_gain_share": acc1_gain_count / total_acc1_gain_count if total_acc1_gain_count else math.nan,
                "acc5_gain_share": acc5_gain_count / total_acc5_gain_count if total_acc5_gain_count else math.nan,
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path, meta, describe, dist_rows, crosstab_rows, gain_rows):
    coverage_zero = next(r for r in dist_rows if r["bucket"] == "coverage_eq_0")
    gt0 = next(r for r in gain_rows if r["bucket"] == "coverage_gt_0")
    lines = [
        "# Check 6/7: Retrieval Coverage Distribution and Gain Decomposition",
        "",
        "This report uses the Check 5 per-case retrieval-quality file. It does not rerun LocAgent or RepoMem.",
        "",
        "## Key Statistics",
        "",
        f"- analyzed rows: {meta['n_analyzed']}",
        f"- nominal full500 missing rows: {meta['n_missing_nominal_500']}",
        f"- coverage=0: {coverage_zero['count']} / {meta['n_analyzed']} ({coverage_zero['share_analyzed']:.2%} analyzed)",
        f"- coverage=0 using nominal 500 denominator: {coverage_zero['share_nominal_500']:.2%}",
        f"- total Acc@1 gain: {meta['total_delta_acc1']:.4f}",
        f"- total Acc@5 gain: {meta['total_delta_acc5']:.4f}",
        f"- Acc@1 gain share from coverage>0: {gt0['acc1_gain_share']:.2%}",
        f"- Acc@5 gain share from coverage>0: {gt0['acc5_gain_share']:.2%}",
        "",
        "## Coverage Describe",
        "",
        markdown_table([describe], list(describe.keys())),
        "",
        "## Check 6 Coverage Distribution",
        "",
        markdown_table(dist_rows, ["bucket", "count", "share_analyzed", "share_nominal_500"]),
        "",
        "## Check 6 Coverage x Baseline Recall@5 Crosstab",
        "",
        markdown_table(crosstab_rows, ["coverage_bucket", "baseline_recall_bucket", "count"]),
        "",
        "## Check 7 Gain by Coverage Bucket",
        "",
        markdown_table(
            gain_rows,
            [
                "bucket",
                "N",
                "baseline_acc1",
                "memory_acc1",
                "delta_acc1",
                "baseline_acc5",
                "memory_acc5",
                "delta_acc5",
                "acc1_gain_contribution_pp",
                "acc1_gain_share",
                "acc5_gain_contribution_pp",
                "acc5_gain_share",
            ],
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nominal_n", type=int, default=NOMINAL_FULL500_N)
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)

    describe = coverage_describe(df)
    dist_rows = coverage_distribution_rows(df, nominal_n=args.nominal_n)
    crosstab_rows = coverage_baseline_crosstab_rows(df)
    gain_rows = compute_gain_by_bucket(df)

    meta = {
        "input": args.input,
        "n_analyzed": int(len(df)),
        "nominal_n": int(args.nominal_n),
        "n_missing_nominal_500": int(max(args.nominal_n - len(df), 0)),
        "total_delta_acc1": float(df["memory_precision_at1"].mean() - df["baseline_precision_at1"].mean()),
        "total_delta_acc5": float(df["memory_acc_at5_original"].mean() - df["baseline_acc_at5_original"].mean()),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "check6_coverage_distribution.csv"), dist_rows)
    write_csv(os.path.join(args.output_dir, "check6_crosstab.csv"), crosstab_rows)
    write_csv(os.path.join(args.output_dir, "check7_gain_by_bucket.csv"), gain_rows)

    summary = {
        "meta": meta,
        "coverage_describe": describe,
        "coverage_distribution": dist_rows,
        "coverage_baseline_recall_crosstab": crosstab_rows,
        "gain_by_bucket": gain_rows,
    }
    with open(os.path.join(args.output_dir, "check6_7_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_report(os.path.join(args.output_dir, "check6_7_report.md"), meta, describe, dist_rows, crosstab_rows, gain_rows)

    coverage_zero = next(r for r in dist_rows if r["bucket"] == "coverage_eq_0")
    gt0 = next(r for r in gain_rows if r["bucket"] == "coverage_gt_0")
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "n_analyzed": len(df),
                "coverage_zero_count": coverage_zero["count"],
                "coverage_zero_share_analyzed": coverage_zero["share_analyzed"],
                "coverage_zero_share_nominal_500": coverage_zero["share_nominal_500"],
                "acc1_gain_share_coverage_gt0": gt0["acc1_gain_share"],
                "acc5_gain_share_coverage_gt0": gt0["acc5_gain_share"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
