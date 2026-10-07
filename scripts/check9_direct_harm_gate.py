import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from scripts.check9_gate_predictability import (
    DEFAULT_OUTPUT_DIR,
    LOCAL_FEATURE_COLUMNS,
    auc_score,
    brier_score,
    build_feature_dataframe,
    feature_columns,
    fit_full_feature_importance,
    gate_accuracy_from_mask,
    leave_one_repo_out_predictions,
    precision_at_top_fraction,
    recall_at_threshold,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_DIRECT_HARM_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "direct_harm")
DEFAULT_FEATURES_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "check9_features.jsonl")


def make_direct_labels(df, ks=(1, 3, 5)):
    out = df.copy()
    for k in ks:
        base = out[f"baseline_at{k}"].astype(int)
        mem = out[f"memory_at{k}"].astype(int)
        out[f"harm_at{k}"] = ((base == 1) & (mem == 0)).astype(int)
        out[f"rescue_at{k}"] = ((base == 0) & (mem == 1)).astype(int)
        out[f"agree_correct_at{k}"] = ((base == 1) & (mem == 1)).astype(int)
        out[f"agree_wrong_at{k}"] = ((base == 0) & (mem == 0)).astype(int)
    return out


def average_precision_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision = tp / ranks
    return float((precision * y_sorted).sum() / positives)


def summarize_predictions(pred, label_col):
    y = pred[label_col].astype(int).to_numpy()
    score = pred["risk_probability"].astype(float).to_numpy()
    undefined = pred[pred["fold_auc"].isna()] if "fold_auc" in pred else pd.DataFrame()
    return {
        "label": label_col,
        "n_cases": int(len(pred)),
        "positive_count": int(y.sum()),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "precision_top30_risky": precision_at_top_fraction(y, score, 0.30),
        "recall_threshold_0.5": recall_at_threshold(y, score, 0.5),
        "brier": brier_score(y, score),
        "undefined_fold_auc_case_count": int(len(undefined)),
        "undefined_fold_auc_repo_count": int(undefined["heldout_repo"].nunique()) if len(undefined) and "heldout_repo" in undefined else 0,
    }


def gate_percentile_curve(pred, label_col, fractions=(0.05, 0.10, 0.20, 0.30, 0.50), ks=(1, 3, 5)):
    rows = []
    ordered = pred.sort_values("risk_probability", ascending=False)
    for fraction in fractions:
        n = max(1, int(math.ceil(len(pred) * fraction)))
        risky = set(ordered.head(n).index)
        use_baseline = pred.index.to_series().isin(risky)
        row = {
            "label": label_col,
            "risk_fraction": float(fraction),
            "gated_case_count": int(n),
            "harm_capture_rate": float(pred.loc[use_baseline, label_col].sum() / max(1, pred[label_col].sum())),
            "harm_precision": float(pred.loc[use_baseline, label_col].mean()) if n else float("nan"),
        }
        for k in ks:
            base = pred[f"baseline_at{k}"].astype(float)
            mem = pred[f"memory_at{k}"].astype(float)
            gated = np.where(use_baseline, base, mem)
            row[f"gate_acc_at{k}"] = float(np.mean(gated))
            row[f"baseline_acc_at{k}"] = float(np.mean(base))
            row[f"memory_acc_at{k}"] = float(np.mean(mem))
            row[f"gate_delta_vs_memory_at{k}"] = row[f"gate_acc_at{k}"] - row[f"memory_acc_at{k}"]
            row[f"gate_delta_vs_baseline_at{k}"] = row[f"gate_acc_at{k}"] - row[f"baseline_acc_at{k}"]
        rows.append(row)
    return rows


def oracle_and_random_gate(pred, label_col, random_rate=0.30, n_random=100, seed=42, ks=(1, 3, 5)):
    rows = []
    policies = {
        "always_memory": np.zeros(len(pred), dtype=bool),
        "always_baseline": np.ones(len(pred), dtype=bool),
        "oracle_harm_gate": pred[label_col].astype(bool).to_numpy(),
    }
    for name, mask in policies.items():
        row = {"policy": name, "label": label_col, "risk_fraction": float(mask.mean())}
        for k in ks:
            values = np.where(mask, pred[f"baseline_at{k}"].astype(float), pred[f"memory_at{k}"].astype(float))
            row[f"acc_at{k}"] = float(np.mean(values))
            row[f"delta_vs_memory_at{k}"] = row[f"acc_at{k}"] - float(pred[f"memory_at{k}"].mean())
            row[f"delta_vs_baseline_at{k}"] = row[f"acc_at{k}"] - float(pred[f"baseline_at{k}"].mean())
        rows.append(row)

    rng = np.random.default_rng(seed)
    n_gate = max(1, int(math.ceil(len(pred) * random_rate)))
    random_acc = {k: [] for k in ks}
    for _ in range(n_random):
        mask = np.zeros(len(pred), dtype=bool)
        mask[rng.choice(len(pred), size=n_gate, replace=False)] = True
        for k in ks:
            values = np.where(mask, pred[f"baseline_at{k}"].astype(float), pred[f"memory_at{k}"].astype(float))
            random_acc[k].append(float(np.mean(values)))
    row = {"policy": "random_gate_mean", "label": label_col, "risk_fraction": float(n_gate / len(pred))}
    for k in ks:
        row[f"acc_at{k}"] = float(np.mean(random_acc[k]))
        row[f"acc_at{k}_std"] = float(np.std(random_acc[k]))
        row[f"delta_vs_memory_at{k}"] = row[f"acc_at{k}"] - float(pred[f"memory_at{k}"].mean())
        row[f"delta_vs_baseline_at{k}"] = row[f"acc_at{k}"] - float(pred[f"baseline_at{k}"].mean())
    rows.append(row)
    return rows


def coverage_harm_stats(df, label_col, coverage_col="coverage_zero_at5"):
    cov = df[coverage_col].astype(int)
    harm = df[label_col].astype(int)
    cells = {
        "coverage_zero_0_harm_0": int(((cov == 0) & (harm == 0)).sum()),
        "coverage_zero_0_harm_1": int(((cov == 0) & (harm == 1)).sum()),
        "coverage_zero_1_harm_0": int(((cov == 1) & (harm == 0)).sum()),
        "coverage_zero_1_harm_1": int(((cov == 1) & (harm == 1)).sum()),
    }
    table = np.array(
        [
            [cells["coverage_zero_0_harm_0"], cells["coverage_zero_0_harm_1"]],
            [cells["coverage_zero_1_harm_0"], cells["coverage_zero_1_harm_1"]],
        ],
        dtype=float,
    )
    n = table.sum()
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row @ col / n if n else np.zeros_like(table)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table - expected) ** 2 / expected)
    cramers_v = math.sqrt(float(chi2 / n)) if n else float("nan")
    joint = table / n if n else np.zeros_like(table)
    row_prob = joint.sum(axis=1, keepdims=True)
    col_prob = joint.sum(axis=0, keepdims=True)
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 0 and row_prob[i, 0] > 0 and col_prob[0, j] > 0:
                mi += float(joint[i, j] * math.log(joint[i, j] / (row_prob[i, 0] * col_prob[0, j])))
    precision_if_cov_zero = cells["coverage_zero_1_harm_1"] / max(1, cells["coverage_zero_1_harm_0"] + cells["coverage_zero_1_harm_1"])
    recall_if_cov_zero = cells["coverage_zero_1_harm_1"] / max(1, cells["coverage_zero_0_harm_1"] + cells["coverage_zero_1_harm_1"])
    return {
        "label": label_col,
        "coverage_col": coverage_col,
        "cells": cells,
        "cramers_v": cramers_v,
        "mutual_information": mi,
        "precision_if_coverage_zero": float(precision_if_cov_zero),
        "recall_if_coverage_zero": float(recall_if_cov_zero),
    }


def _prepare_for_model(df, feature_cols, label_col, repo_col):
    work = df.dropna(subset=feature_cols + [label_col, repo_col]).copy()
    work[label_col] = work[label_col].astype(int)
    return work.reset_index(drop=True)


def loro_prediction_summary(df, feature_cols, label_col, repo_col):
    work = _prepare_for_model(df, feature_cols, label_col, repo_col)
    pred = leave_one_repo_out_predictions(work, feature_cols, label_col, repo_col)
    return work, pred, summarize_predictions(pred, label_col)


def bootstrap_loro_auc(df, feature_cols, label_col, repo_col, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(df)
    for _ in range(n_boot):
        sample_idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[sample_idx].reset_index(drop=True)
        if sample[label_col].nunique() < 2 or sample[repo_col].nunique() < 2:
            continue
        try:
            pred = leave_one_repo_out_predictions(sample, feature_cols, label_col, repo_col)
            auc = auc_score(pred[label_col], pred["risk_probability"])
        except Exception:
            continue
        if not math.isnan(auc):
            aucs.append(float(auc))
    arr = np.array(aucs, dtype=float)
    return {
        "label": label_col,
        "n_bootstrap_requested": int(n_boot),
        "n_bootstrap_valid": int(len(arr)),
        "auc_mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "auc_ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
        "auc_ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
        "auc_values": [float(x) for x in arr],
    }


def permutation_auc_test(df, feature_cols, label_col, repo_col, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    observed_pred = leave_one_repo_out_predictions(df, feature_cols, label_col, repo_col)
    observed = auc_score(observed_pred[label_col], observed_pred["risk_probability"])
    aucs = []
    for _ in range(n_perm):
        shuffled = df.copy()
        shuffled[label_col] = rng.permutation(shuffled[label_col].to_numpy())
        if shuffled[label_col].nunique() < 2:
            continue
        try:
            pred = leave_one_repo_out_predictions(shuffled, feature_cols, label_col, repo_col)
            auc = auc_score(pred[label_col], pred["risk_probability"])
        except Exception:
            continue
        if not math.isnan(auc):
            aucs.append(float(auc))
    valid = np.array(aucs, dtype=float)
    p_value = float((np.sum(valid >= observed) + 1) / (len(valid) + 1)) if len(valid) else float("nan")
    return {
        "label": label_col,
        "observed_auc": float(observed),
        "n_permutations_requested": int(n_perm),
        "n_permutations_valid": int(len(valid)),
        "mean_permuted_auc": float(np.mean(valid)) if len(valid) else float("nan"),
        "std_permuted_auc": float(np.std(valid)) if len(valid) else float("nan"),
        "p_value": p_value,
        "permuted_auc": [float(x) for x in valid],
    }


def fit_label(df, feature_cols, label_col, repo_col):
    work, pred, summary = loro_prediction_summary(df, feature_cols, label_col, repo_col)
    feature_rows = fit_full_feature_importance(work, feature_cols, label_col) if work[label_col].nunique() == 2 else []
    return work, pred, summary, feature_rows


def load_or_build_features(args):
    if args.features_file and os.path.exists(args.features_file):
        return pd.read_json(args.features_file, lines=True)
    return build_feature_dataframe(args)


def decide_verdict(summary, oracle_rows):
    primary = next(row for row in summary["label_summaries"] if row["label"] == "harm_at5")
    oracle = next(row for row in oracle_rows if row["policy"] == "oracle_harm_gate")
    oracle_delta = oracle["delta_vs_memory_at5"]
    gate30 = next(row for row in summary["gate_curve"] if row["label"] == "harm_at5" and abs(row["risk_fraction"] - 0.30) < 1e-9)
    gate_delta = gate30["gate_delta_vs_memory_at5"]
    ci_low = summary["bootstrap"]["auc_ci_low"]
    ci_high = summary["bootstrap"]["auc_ci_high"]
    pr_auc = primary["pr_auc"]
    base = primary["base_rate"]

    if oracle_delta < 0.01:
        branch = "D_oracle_limited"
        reason = "oracle_delta_at5_below_1pp"
    elif oracle_delta <= 0.02 and gate_delta < 0.5 * oracle_delta:
        branch = "B_marginal_tight_ceiling"
        reason = "oracle_delta_1_to_2pp_but_gate_less_than_half_oracle"
    elif ci_low > 0.62 and pr_auc >= 3 * base and gate_delta > 0.01:
        branch = "A_strong"
        reason = "auc_ci_pr_auc_and_gate_delta_all_pass"
    elif ci_high < 0.60 and pr_auc < 2 * base and gate_delta < -0.005:
        branch = "C_no_signal"
        reason = "auc_ci_pr_auc_and_gate_delta_all_fail"
    else:
        branch = "B_marginal"
        reason = "default_non_strong_non_null_branch"
    return {
        "branch": branch,
        "reason": reason,
        "oracle_delta_vs_memory_at5": float(oracle_delta),
        "gate_top30_delta_vs_memory_at5": float(gate_delta),
        "auc_ci_low": float(ci_low),
        "auc_ci_high": float(ci_high),
        "pr_auc": float(pr_auc),
        "base_rate": float(base),
    }


def write_report(path, summary, harm_feature_rows, rescue_summary):
    verdict = summary["verdict"]
    lines = [
        "# Check 9a': Direct Harm Gate",
        "",
        "## Data",
        f"- analyzed cases: {summary['n_cases']}",
        f"- feature count: {summary['feature_count']}",
        "- primary label: `harm_at5 = baseline_at5 == 1 and memory_at5 == 0`",
        "- all gate simulations use leave-one-repo-out out-of-fold risk scores.",
        "",
        "## Label Summary",
        "| label | positives | base rate | ROC-AUC | PR-AUC | P@top30 | recall@0.5 | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["label_summaries"]:
        lines.append(
            f"| {row['label']} | {row['positive_count']} | {row['base_rate']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['precision_top30_risky']:.4f} | "
            f"{row['recall_threshold_0.5']:.4f} | {row['brier']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Primary Uncertainty",
            f"- full bootstrap ROC-AUC 95% CI: [{summary['bootstrap']['auc_ci_low']:.4f}, {summary['bootstrap']['auc_ci_high']:.4f}]",
            f"- permutation p-value: {summary['permutation']['p_value']:.4f}",
            "",
            "## Gate Curve At Acc@5",
            "| risky fraction | gated cases | harm precision | harm recall | gate Acc@5 | delta vs memory | delta vs baseline |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [r for r in summary["gate_curve"] if r["label"] == "harm_at5"]:
        lines.append(
            f"| {row['risk_fraction']:.2f} | {row['gated_case_count']} | {row['harm_precision']:.4f} | "
            f"{row['harm_capture_rate']:.4f} | {row['gate_acc_at5']:.4f} | "
            f"{row['gate_delta_vs_memory_at5']:.4f} | {row['gate_delta_vs_baseline_at5']:.4f} |"
        )
    lines.extend(["", "## Oracle And Random Baselines", "| policy | Acc@5 | delta vs memory | delta vs baseline |", "|---|---:|---:|---:|"])
    for row in summary["oracle_random"]:
        lines.append(
            f"| {row['policy']} | {row['acc_at5']:.4f} | {row['delta_vs_memory_at5']:.4f} | {row['delta_vs_baseline_at5']:.4f} |"
        )
    cov = summary["coverage_vs_harm"]
    lines.extend(
        [
            "",
            "## Coverage Zero vs Harm",
            f"- cells: {cov['cells']}",
            f"- Cramer's V: {cov['cramers_v']:.4f}",
            f"- mutual information: {cov['mutual_information']:.4f}",
            f"- precision if coverage_zero: {cov['precision_if_coverage_zero']:.4f}",
            f"- recall if coverage_zero: {cov['recall_if_coverage_zero']:.4f}",
            "",
            "## Harm Feature Coefficients",
        ]
    )
    for row in harm_feature_rows:
        lines.append(f"- `{row['feature']}`: {row['coef_on_standardized_feature']:.4f}")
    lines.extend(
        [
            "",
            "## Rescue Counterpart",
            f"- rescue_at5 ROC-AUC: {rescue_summary['roc_auc']:.4f}",
            f"- rescue_at5 PR-AUC: {rescue_summary['pr_auc']:.4f}",
            "",
            "## Verdict",
            f"- branch: {verdict['branch']}",
            f"- reason: {verdict['reason']}",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_direct_harm_gate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = make_direct_labels(load_or_build_features(args))
    cols = [c for c in feature_columns(df) if c in LOCAL_FEATURE_COLUMNS]
    if not cols:
        raise RuntimeError("No usable numeric feature columns found")
    df = _prepare_for_model(df, cols, "harm_at5", "repo")

    label_summaries = []
    predictions = []
    feature_rows_by_label = {}
    for label_col in ["harm_at5", "harm_at3", "harm_at1", "rescue_at5"]:
        _, pred, label_summary, feature_rows = fit_label(df, cols, label_col, "repo")
        label_summaries.append(label_summary)
        feature_rows_by_label[label_col] = feature_rows
        pred["label_name"] = label_col
        predictions.append(pred)

    primary_work, primary_pred, _, harm_feature_rows = fit_label(df, cols, "harm_at5", "repo")
    bootstrap = bootstrap_loro_auc(
        primary_work,
        cols,
        "harm_at5",
        "repo",
        n_boot=args.n_bootstrap,
        seed=args.seed,
    )
    permutation = permutation_auc_test(
        primary_work,
        cols,
        "harm_at5",
        "repo",
        n_perm=args.n_permutation,
        seed=args.seed + 1,
    )
    gate_curve = gate_percentile_curve(primary_pred, "harm_at5", args.gate_fractions)
    oracle_random = oracle_and_random_gate(
        primary_pred,
        "harm_at5",
        random_rate=args.random_gate_rate,
        n_random=args.n_random_gate,
        seed=args.seed + 2,
    )
    coverage_stats = coverage_harm_stats(primary_work, "harm_at5", "coverage_zero_at5")
    rescue_summary = next(row for row in label_summaries if row["label"] == "rescue_at5")
    stability_cases = (
        primary_work[primary_work["harm_at5"] == 1][["case_id", "repo", "baseline_at5", "memory_at5", "coverage_zero_at5"]]
        .sample(n=min(10, int(primary_work["harm_at5"].sum())), random_state=args.seed)
        .sort_values(["repo", "case_id"])
        .to_dict("records")
    )

    summary = {
        "n_cases": int(len(df)),
        "feature_count": int(len(cols)),
        "features": cols,
        "label_summaries": label_summaries,
        "bootstrap": bootstrap,
        "permutation": permutation,
        "gate_curve": gate_curve,
        "oracle_random": oracle_random,
        "coverage_vs_harm": coverage_stats,
        "label_stability_spot_check": {
            "status": "candidate_cases_only_not_rerun",
            "reason": "rerun requires external LLM/API calls",
            "case_count": len(stability_cases),
        },
    }
    summary["verdict"] = decide_verdict(summary, oracle_random)

    write_jsonl(os.path.join(args.output_dir, "direct_harm_features.jsonl"), df.to_dict("records"))
    write_jsonl(os.path.join(args.output_dir, "predictions_harm_by_k.jsonl"), pd.concat(predictions).to_dict("records"))
    write_csv(os.path.join(args.output_dir, "direct_harm_by_k.csv"), label_summaries)
    write_csv(os.path.join(args.output_dir, "harm_feature_importance.csv"), harm_feature_rows)
    write_csv(os.path.join(args.output_dir, "rescue_feature_importance.csv"), feature_rows_by_label["rescue_at5"])
    write_csv(os.path.join(args.output_dir, "gate_percentile_curve.csv"), gate_curve)
    write_csv(os.path.join(args.output_dir, "oracle_random_baselines.csv"), oracle_random)
    write_csv(
        os.path.join(args.output_dir, "coverage_vs_harm_crosstab.csv"),
        [{"cell": k, "count": v} for k, v in coverage_stats["cells"].items()],
    )
    write_json(os.path.join(args.output_dir, "coverage_vs_harm_stats.json"), coverage_stats)
    write_json(os.path.join(args.output_dir, "bootstrap_ci.json"), bootstrap)
    write_csv(os.path.join(args.output_dir, "bootstrap_ci.csv"), [bootstrap])
    write_json(os.path.join(args.output_dir, "permutation_test.json"), permutation)
    write_csv(os.path.join(args.output_dir, "rescue_classifier_summary.csv"), [rescue_summary])
    write_json(os.path.join(args.output_dir, "label_stability_harm10_cases.json"), stability_cases)
    write_json(os.path.join(args.output_dir, "direct_harm_summary.json"), summary)
    write_report(os.path.join(args.output_dir, "direct_harm_gate_report.md"), summary, harm_feature_rows, rescue_summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--output_dir", default=DEFAULT_DIRECT_HARM_DIR)
    parser.add_argument("--check5", default=None)
    parser.add_argument("--traj_files", nargs="+", default=None)
    parser.add_argument("--index_dirs", nargs="+", default=None)
    parser.add_argument("--coverage_k", type=int, default=5)
    parser.add_argument("--max_commits", type=int, default=20)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--n_permutation", type=int, default=1000)
    parser.add_argument("--n_random_gate", type=int, default=100)
    parser.add_argument("--random_gate_rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate_fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.50])
    args = parser.parse_args()
    if args.traj_files is None:
        from scripts.check9_gate_predictability import DEFAULT_TRAJ_FILES

        args.traj_files = DEFAULT_TRAJ_FILES
    if args.index_dirs is None:
        from scripts.analyze_memory_fragmentation import DEFAULT_INDEX_DIRS

        args.index_dirs = DEFAULT_INDEX_DIRS
    if args.check5 is None:
        from scripts.check9_gate_predictability import DEFAULT_CHECK5

        args.check5 = DEFAULT_CHECK5
    return args


if __name__ == "__main__":
    result = run_direct_harm_gate(parse_args())
    print(json.dumps(result["verdict"], indent=2, ensure_ascii=False))
