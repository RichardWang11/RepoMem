import argparse
import csv
import json
import math
import os
import re

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.optimize import minimize
from scipy.stats import fisher_exact, norm

from scripts.analyze_fragmentation_significance import assign_file_overlap_tiers


DEFAULT_INPUT = "results/full500_azure_gpt4o_repaired_compare/fragmentation/fragmentation_per_case.jsonl"
DEFAULT_SELECTED_FILE = "reproduce_lists/verified_full_500.json"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/fragmentation_difficulty"


def count_gold_files(patch_text):
    files = set(re.findall(r"^diff --git a/(\S+) b/", patch_text or "", re.MULTILINE))
    return len(files)


def load_selected_ids(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def load_gold_file_counts(dataset_name, split, selected_file):
    selected = load_selected_ids(selected_file)
    rows = []
    for inst in load_dataset(dataset_name, split=split):
        iid = inst["instance_id"]
        if selected is not None and iid not in selected:
            continue
        rows.append(
            {
                "instance_id": iid,
                "gold_files_count": count_gold_files(inst.get("patch", "")),
            }
        )
    return pd.DataFrame(rows)


def normalize_input_schema(df):
    df = df.copy()
    if "case_id" not in df.columns:
        df["case_id"] = df["instance_id"]
    if "file_overlap_score" not in df.columns:
        df["file_overlap_score"] = df["file_overlap_jaccard_mean"]
    if "file_overlap_jaccard_mean" not in df.columns:
        df["file_overlap_jaccard_mean"] = df["file_overlap_score"]
    if "baseline_correct_at1" not in df.columns:
        df["baseline_correct_at1"] = df["baseline_ok_k1"].astype(int)
    if "memory_correct_at1" not in df.columns:
        df["memory_correct_at1"] = df["repomem_ok_k1"].astype(int)
    if "baseline_correct_at5" not in df.columns:
        df["baseline_correct_at5"] = df["baseline_ok_k5"].astype(int)
    if "memory_correct_at5" not in df.columns:
        df["memory_correct_at5"] = df["repomem_ok_k5"].astype(int)
    return df


def tier_stats(group, k):
    n = len(group)
    baseline = int(group[f"baseline_correct_at{k}"].sum())
    memory = int(group[f"memory_correct_at{k}"].sum())
    rescue = int(((group[f"baseline_correct_at{k}"] == 0) & (group[f"memory_correct_at{k}"] == 1)).sum())
    regress = int(((group[f"baseline_correct_at{k}"] == 1) & (group[f"memory_correct_at{k}"] == 0)).sum())
    return {
        "N": n,
        "acc_baseline": baseline / n if n else 0.0,
        "acc_memory": memory / n if n else 0.0,
        "delta_acc": (memory - baseline) / n if n else 0.0,
        "rescue": rescue,
        "regress": regress,
        "rescue_rate": rescue / n if n else 0.0,
        "regress_rate": regress / n if n else 0.0,
    }


def compute_stratified_tier_stats(df):
    rows = []
    for difficulty in ["single_file", "multi_file"]:
        sub = df[df["difficulty"] == difficulty]
        for k in [1, 5]:
            for tier in ["low_frag", "mid_frag", "high_frag"]:
                group = sub[sub["frag_tier"] == tier]
                rows.append({"difficulty": difficulty, "k": k, "frag_tier": tier, **tier_stats(group, k)})
    return rows


def low_high_fisher(df, difficulty, k):
    sub = df[df["difficulty"] == difficulty]
    low = sub[sub["frag_tier"] == "low_frag"]
    high = sub[sub["frag_tier"] == "high_frag"]

    rescue_low = int(((low[f"baseline_correct_at{k}"] == 0) & (low[f"memory_correct_at{k}"] == 1)).sum())
    regress_low = int(((low[f"baseline_correct_at{k}"] == 1) & (low[f"memory_correct_at{k}"] == 0)).sum())
    rescue_high = int(((high[f"baseline_correct_at{k}"] == 0) & (high[f"memory_correct_at{k}"] == 1)).sum())
    regress_high = int(((high[f"baseline_correct_at{k}"] == 1) & (high[f"memory_correct_at{k}"] == 0)).sum())
    table = [[rescue_low, regress_low], [rescue_high, regress_high]]
    odds_ratio, p_value = fisher_exact(table)

    low_delta = tier_stats(low, k)["delta_acc"]
    high_delta = tier_stats(high, k)["delta_acc"]
    delta_gap = low_delta - high_delta
    return {
        "difficulty": difficulty,
        "k": k,
        "low_n": int(len(low)),
        "high_n": int(len(high)),
        "low_delta_acc": low_delta,
        "high_delta_acc": high_delta,
        "delta_gap_low_minus_high": delta_gap,
        "fisher_table": table,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "direction_consistent": bool(delta_gap > 0),
        "passes_threshold": bool(delta_gap >= 0.02 and p_value < 0.15),
    }


def safe_exp(x):
    if x > 709:
        return float("inf")
    if x < -745:
        return 0.0
    return float(math.exp(x))


def fit_logit_design(design, y, columns):
    y = np.asarray(y, dtype=float)
    x = np.asarray(design, dtype=float)
    if x.shape[0] != len(y):
        raise ValueError("design and y length mismatch")
    if len(np.unique(y)) < 2:
        raise ValueError("logistic regression needs both outcome classes")

    def negative_log_likelihood(beta):
        eta = x @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    result = minimize(negative_log_likelihood, np.zeros(x.shape[1]), method="BFGS")
    if not result.success:
        raise RuntimeError(f"logistic regression failed: {result.message}")

    beta = result.x
    p = 1.0 / (1.0 + np.exp(-(x @ beta)))
    weights = p * (1 - p)
    hessian = x.T @ (weights[:, None] * x)
    covariance = np.linalg.pinv(hessian)
    se = np.sqrt(np.diag(covariance))

    rows = []
    for name, coef, coef_se in zip(columns, beta, se):
        z_value = coef / coef_se
        p_value = float(2 * norm.sf(abs(z_value)))
        rows.append(
            {
                "term": name,
                "coef": float(coef),
                "se": float(coef_se),
                "z_value": float(z_value),
                "p_value": p_value,
                "odds_ratio": safe_exp(float(coef)),
                "odds_ratio_ci_low": safe_exp(float(coef - 1.96 * coef_se)),
                "odds_ratio_ci_high": safe_exp(float(coef + 1.96 * coef_se)),
            }
        )
    return rows


def interaction_model(df):
    work = df[df["file_overlap_score"].notna()].copy()
    score = work["file_overlap_score"].astype(float)
    score_z = (score - score.mean()) / score.std(ddof=0)
    multi = (work["difficulty"] == "multi_file").astype(float)
    work["rescued"] = ((work["baseline_correct_at1"] == 0) & (work["memory_correct_at1"] == 1)).astype(int)

    design = np.column_stack(
        [
            np.ones(len(work)),
            score_z,
            multi,
            score_z * multi,
        ]
    )
    columns = ["Intercept", "file_overlap_z", "C(difficulty)[multi_file]", "file_overlap_z:C(difficulty)[multi_file]"]
    rows = fit_logit_design(design, work["rescued"], columns)
    return rows, int(len(work)), int(work["rescued"].sum())


def difficulty_decision(fisher_rows):
    by_diff = {row["difficulty"]: row for row in fisher_rows if row["k"] == 1}
    single = by_diff["single_file"]["passes_threshold"]
    multi = by_diff["multi_file"]["passes_threshold"]
    if single and multi:
        return "strong_both_strata"
    if (not single) and multi:
        return "stronger_on_multi_file"
    if single and (not multi):
        return "confounded_by_single_file"
    return "weak_in_both_strata"


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path, args, meta, tier_rows, fisher_rows, interaction_rows):
    interaction = next(row for row in interaction_rows if row["term"].startswith("file_overlap_z:C"))
    lines = [
        "# Fragmentation by Task Difficulty",
        "",
        "This analysis stratifies the existing full500 fragmentation results by gold-patch file count. It does not modify RepoMem, LocAgent, or memory retrieval.",
        "",
        f"- input: `{args.input}`",
        f"- selected file: `{args.selected_file}`",
        f"- dataset: `{args.dataset}` / `{args.split}`",
        f"- analyzed rows: {meta['analyzed_rows']}",
        f"- dropped rows: {meta['dropped_rows']}",
        f"- missing gold joins: {meta['missing_gold']}",
        "- difficulty: `single_file` if gold patch touches exactly one file, otherwise `multi_file`",
        "- fragmentation direction: higher file overlap means lower fragmentation; therefore `low_frag` is the high-overlap tercile",
        f"- Acc@1 decision: `{meta['decision']}`",
        f"- interaction term p-value: {interaction['p_value']:.4f}",
        "",
        "## Gold File Count Distribution",
        "",
        markdown_table(meta["gold_files_hist"], ["gold_files_count", "count"]),
        "",
        "## Difficulty Distribution",
        "",
        markdown_table(meta["difficulty_hist"], ["difficulty", "count"]),
        "",
        "## Stratified Tercile Stats",
        "",
        markdown_table(
            tier_rows,
            [
                "difficulty",
                "k",
                "frag_tier",
                "N",
                "acc_baseline",
                "acc_memory",
                "delta_acc",
                "rescue",
                "regress",
                "rescue_rate",
                "regress_rate",
            ],
        ),
        "",
        "## Low-vs-High Fisher Tests",
        "",
        markdown_table(
            fisher_rows,
            [
                "difficulty",
                "k",
                "low_n",
                "high_n",
                "low_delta_acc",
                "high_delta_acc",
                "delta_gap_low_minus_high",
                "fisher_table",
                "odds_ratio",
                "p_value",
                "direction_consistent",
                "passes_threshold",
            ],
        ),
        "",
        "## Rescue Interaction Model",
        "",
        "Model: `rescued ~ file_overlap_z * C(difficulty)`, where `rescued` is 1 only for baseline-wrong / memory-correct Acc@1 cases.",
        "",
        markdown_table(
            interaction_rows,
            ["term", "coef", "se", "z_value", "p_value", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high"],
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--selected_file", default=DEFAULT_SELECTED_FILE)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df_all = pd.read_json(args.input, lines=True)
    df = normalize_input_schema(df_all)
    df = df[df["file_overlap_score"].notna()].copy()
    df = assign_file_overlap_tiers(df)

    gold_df = load_gold_file_counts(args.dataset, args.split, args.selected_file)
    df = df.merge(gold_df, left_on="case_id", right_on="instance_id", how="left", suffixes=("", "_gold"))
    missing_gold = int(df["gold_files_count"].isna().sum())
    if missing_gold:
        raise RuntimeError(f"join failed for {missing_gold} rows")

    df["difficulty"] = np.where(df["gold_files_count"] == 1, "single_file", "multi_file")

    tier_rows = compute_stratified_tier_stats(df)
    fisher_rows = []
    for difficulty in ["single_file", "multi_file"]:
        for k in [1, 5]:
            fisher_rows.append(low_high_fisher(df, difficulty, k))
    interaction_rows, interaction_n, interaction_positive = interaction_model(df)

    gold_hist = (
        df["gold_files_count"].value_counts().sort_index().rename_axis("gold_files_count").reset_index(name="count").to_dict("records")
    )
    difficulty_hist = df["difficulty"].value_counts().rename_axis("difficulty").reset_index(name="count").to_dict("records")

    meta = {
        "analyzed_rows": int(len(df)),
        "dropped_rows": int(len(df_all) - len(df)),
        "missing_gold": missing_gold,
        "gold_files_hist": [{**row, "gold_files_count": int(row["gold_files_count"]), "count": int(row["count"])} for row in gold_hist],
        "difficulty_hist": [{**row, "count": int(row["count"])} for row in difficulty_hist],
        "interaction_n": interaction_n,
        "interaction_positive": interaction_positive,
        "decision": difficulty_decision(fisher_rows),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "stratified_tercile_stats.csv"), tier_rows)
    write_csv(os.path.join(args.output_dir, "low_high_fisher.csv"), fisher_rows)
    write_csv(os.path.join(args.output_dir, "rescue_interaction_logit.csv"), interaction_rows)
    df.to_json(os.path.join(args.output_dir, "fragmentation_with_difficulty.jsonl"), orient="records", lines=True)

    summary = {
        "input": args.input,
        "selected_file": args.selected_file,
        "dataset": args.dataset,
        "split": args.split,
        **meta,
        "tier_stats": tier_rows,
        "fisher_tests": fisher_rows,
        "interaction_model": interaction_rows,
    }
    with open(os.path.join(args.output_dir, "difficulty_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_report(os.path.join(args.output_dir, "difficulty_report.md"), args, meta, tier_rows, fisher_rows, interaction_rows)

    print(json.dumps({"output_dir": args.output_dir, **meta}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
