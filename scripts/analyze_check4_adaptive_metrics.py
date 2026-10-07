import argparse
import csv
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


DEFAULT_BASELINE = "results/full500_azure_gpt4o_repaired_compare/baseline_merged_loc_outputs_full500_repaired_mrr.jsonl"
DEFAULT_MEMORY = "results/full500_azure_gpt4o_repaired_compare/episodic_merged_loc_outputs_full500_repaired_mrr.jsonl"
DEFAULT_GOLD = "results/full500_azure_gpt4o_repaired_compare/fragmentation_difficulty/fragmentation_with_difficulty.jsonl"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/check4_adaptive"


def normalize_path(path):
    path = str(path).strip().strip("`'\" ,:;")
    for prefix in ("./", "a/", "b/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path.lstrip("/")


def normalize_pred_files(value):
    if value is None:
        return []
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: item[1], reverse=True)
        raw_files = [k for k, _ in items]
    elif isinstance(value, list):
        raw_files = value
    else:
        return []

    out = []
    seen = set()
    for item in raw_files:
        if isinstance(item, dict):
            path = item.get("file") or item.get("path") or item.get("name")
        elif isinstance(item, (list, tuple)) and item:
            path = item[0]
        else:
            path = item
        if not isinstance(path, str):
            continue
        path = normalize_path(path)
        if path and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def compute_metrics(pred_files, gold_files):
    pred_files = normalize_pred_files(pred_files)
    gold_set = {normalize_path(p) for p in gold_files or [] if isinstance(p, str)}
    if not gold_set:
        return {
            "recall_at5": np.nan,
            "precision_at1": np.nan,
            "adaptive_hit": np.nan,
            "acc_at5_original": np.nan,
        }

    top5 = set(pred_files[:5])
    k_adaptive = max(5, len(gold_set))
    topk = set(pred_files[:k_adaptive])
    return {
        "recall_at5": len(top5 & gold_set) / len(gold_set),
        "precision_at1": int(bool(pred_files) and pred_files[0] in gold_set),
        "adaptive_hit": int(gold_set.issubset(topk)),
        "acc_at5_original": int(gold_set.issubset(top5)),
    }


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_predictions(path, prefix):
    rows = []
    for row in load_jsonl(path):
        iid = row.get("instance_id") or row.get("case_id")
        rows.append(
            {
                "case_id": iid,
                f"{prefix}_predicted_files": normalize_pred_files(row.get("found_files") or row.get("predicted_files")),
            }
        )
    return pd.DataFrame(rows)


def merge_rare_repos(df, repo_col, min_n=5):
    df = df.copy()
    counts = df[repo_col].value_counts()
    rare = set(counts[counts < min_n].index)
    df["repo_fe"] = df[repo_col].where(~df[repo_col].isin(rare), "other")
    return df


def add_design_columns(df, predictors):
    columns = [np.ones(len(df))]
    names = ["Intercept"]
    for pred in predictors:
        if pred == "C(repo)":
            dummies = pd.get_dummies(df["repo_fe"], prefix="repo", drop_first=True, dtype=float)
            for col in dummies.columns:
                columns.append(dummies[col].to_numpy(dtype=float))
                names.append(col)
        elif pred == "file_overlap_z:is_multi":
            columns.append((df["file_overlap_z"].to_numpy(dtype=float) * df["is_multi"].to_numpy(dtype=float)))
            names.append(pred)
        elif pred == "np.log(gold_files_count)":
            columns.append(np.log(df["gold_files_count"].to_numpy(dtype=float)))
            names.append("log_gold_files_count")
        else:
            columns.append(df[pred].to_numpy(dtype=float))
            names.append(pred)
    return np.column_stack(columns), names


def fit_ols(df, outcome, predictors):
    needed = [outcome]
    for pred in predictors:
        if pred == "C(repo)":
            needed.append("repo_fe")
        elif pred == "file_overlap_z:is_multi":
            needed.extend(["file_overlap_z", "is_multi"])
        elif pred == "np.log(gold_files_count)":
            needed.append("gold_files_count")
        else:
            needed.append(pred)
    needed = list(dict.fromkeys(needed))
    work = df[needed].dropna().copy()
    if len(work) == 0:
        raise ValueError("No rows available for OLS")
    x, names = add_design_columns(work, predictors)
    y = work[outcome].to_numpy(dtype=float)

    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    n, p = x.shape
    dof = max(n - p, 1)
    sigma2 = float((residuals @ residuals) / dof)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(cov), 0))

    rows = []
    for name, coef, coef_se in zip(names, beta, se):
        t_value = float(coef / coef_se) if coef_se else float("inf")
        p_value = float(2 * student_t.sf(abs(t_value), dof))
        ci_low = float(coef - 1.96 * coef_se)
        ci_high = float(coef + 1.96 * coef_se)
        rows.append(
            {
                "term": name,
                "coef": float(coef),
                "se": float(coef_se),
                "t_value": t_value,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return {
        "n": int(n),
        "dof": int(dof),
        "outcome": outcome,
        "predictors": predictors,
        "terms": rows,
        "r2": float(1 - (residuals @ residuals) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 else np.nan,
    }


def term_row(model, term):
    for row in model["terms"]:
        if row["term"] == term:
            return row
    raise KeyError(term)


def model_summary_row(name, model, term="file_overlap_z"):
    row = term_row(model, term)
    return {
        "model": name,
        "N": model["n"],
        "coef": row["coef"],
        "se": row["se"],
        "p": row["p_value"],
        "ci_low": row["ci_low"],
        "ci_high": row["ci_high"],
        "r2": model["r2"],
    }


def build_outcome_dataframe(args):
    gold = pd.read_json(args.gold, lines=True)
    if "case_id" not in gold.columns:
        gold["case_id"] = gold["instance_id"]
    if "file_overlap_score" not in gold.columns:
        gold["file_overlap_score"] = gold["file_overlap_jaccard_mean"]
    if "repo_group" not in gold.columns:
        gold["repo_group"] = gold["repo"].astype(str).str.split("/").str[-1]

    base = load_predictions(args.baseline, "baseline")
    mem = load_predictions(args.memory, "memory")
    df = gold.merge(base, on="case_id", how="left").merge(mem, on="case_id", how="left")

    metric_rows = []
    for _, row in df.iterrows():
        gold_files = row.get("gt") or row.get("gold_files") or []
        base_metrics = compute_metrics(row.get("baseline_predicted_files"), gold_files)
        mem_metrics = compute_metrics(row.get("memory_predicted_files"), gold_files)
        out = {
            "case_id": row["case_id"],
            "repo": row.get("repo_group") or row.get("repo"),
            "gold_files": gold_files,
            "gold_files_count": int(row["gold_files_count"]) if not pd.isna(row["gold_files_count"]) else len(gold_files),
            "file_overlap_score": row["file_overlap_score"],
            "baseline_predicted_files": row.get("baseline_predicted_files") if isinstance(row.get("baseline_predicted_files"), list) else [],
            "memory_predicted_files": row.get("memory_predicted_files") if isinstance(row.get("memory_predicted_files"), list) else [],
        }
        for key, value in base_metrics.items():
            out[f"baseline_{key}"] = value
        for key, value in mem_metrics.items():
            out[f"memory_{key}"] = value
        metric_rows.append(out)

    out_df = pd.DataFrame(metric_rows)
    out_df = out_df[out_df["file_overlap_score"].notna()].copy()
    out_df = out_df[out_df["gold_files_count"] > 0].copy()
    mean = out_df["file_overlap_score"].mean()
    std = out_df["file_overlap_score"].std(ddof=0)
    out_df["file_overlap_z"] = (out_df["file_overlap_score"] - mean) / std
    out_df["log_gold_files_count"] = np.log(out_df["gold_files_count"])
    out_df["is_multi"] = (out_df["gold_files_count"] >= 2).astype(int)
    out_df = merge_rare_repos(out_df, "repo", min_n=args.min_repo_n)
    return out_df, {"file_overlap_mean": float(mean), "file_overlap_std": float(std)}


def add_improvement(df, metric):
    df = df.copy()
    df["improvement"] = df[f"memory_{metric}"].astype(float) - df[f"baseline_{metric}"].astype(float)
    return df


def run_nested_models(rescue_df):
    formulas = [
        ("M0 baseline", ["file_overlap_z"]),
        ("M1 +gold_count", ["file_overlap_z", "gold_files_count"]),
        ("M2 +log_gold", ["file_overlap_z", "np.log(gold_files_count)"]),
        ("M3 +repo_FE", ["file_overlap_z", "gold_files_count", "C(repo)"]),
        ("M4 +baseline_lvl", ["file_overlap_z", "gold_files_count", "C(repo)", "baseline_recall_at5"]),
    ]
    rows = []
    models = {}
    for name, predictors in formulas:
        model = fit_ols(rescue_df, "improvement", predictors)
        rows.append(model_summary_row(name, model))
        models[name] = model
    return rows, models


def run_interaction_model(rescue_df):
    predictors = ["file_overlap_z", "is_multi", "file_overlap_z:is_multi", "C(repo)"]
    model = fit_ols(rescue_df, "improvement", predictors)
    rows = []
    for term in ["file_overlap_z", "is_multi", "file_overlap_z:is_multi"]:
        rows.append({"model": "interaction", "N": model["n"], **term_row(model, term)})
    single = term_row(model, "file_overlap_z")
    interaction = term_row(model, "file_overlap_z:is_multi")
    rows.append(
        {
            "model": "multi_file_net_effect",
            "N": model["n"],
            "term": "file_overlap_z + file_overlap_z:is_multi",
            "coef": single["coef"] + interaction["coef"],
            "se": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    )
    return rows, model


def run_sensitivity_outcomes(df):
    rows = []
    for metric in ["precision_at1", "adaptive_hit", "acc_at5_original"]:
        sub = df[df[f"baseline_{metric}"] < 1.0].copy()
        sub = add_improvement(sub, metric)
        if len(sub) == 0:
            continue
        model = fit_ols(sub, "improvement", ["file_overlap_z", "gold_files_count", "C(repo)"])
        rows.append({"outcome": metric, **model_summary_row("M3 +repo_FE", model)})
    return rows


def run_trim_sensitivity(rescue_df):
    q05, q95 = rescue_df["file_overlap_score"].quantile([0.05, 0.95])
    sub = rescue_df[(rescue_df["file_overlap_score"] > q05) & (rescue_df["file_overlap_score"] < q95)].copy()
    model = fit_ols(sub, "improvement", ["file_overlap_z", "gold_files_count", "C(repo)"])
    row = model_summary_row("trim_5_95_M3", model)
    row["q05"] = float(q05)
    row["q95"] = float(q95)
    return [row], model


def run_bootstrap_ci(rescue_df, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(n_boot):
        sample = rescue_df.sample(frac=1, replace=True, random_state=int(rng.integers(1_000_000_000)))
        try:
            model = fit_ols(sample, "improvement", ["file_overlap_z", "gold_files_count", "C(repo)"])
            coefs.append(term_row(model, "file_overlap_z")["coef"])
        except Exception:
            continue
    arr = np.asarray(coefs, dtype=float)
    low, med, high = np.percentile(arr, [2.5, 50, 97.5])
    return [
        {
            "n_boot_requested": n_boot,
            "n_boot_success": int(len(arr)),
            "seed": seed,
            "ci_low": float(low),
            "median": float(med),
            "ci_high": float(high),
        }
    ]


def main_decision(nested_rows, sensitivity_rows, trim_rows, boot_rows):
    m0 = next(r for r in nested_rows if r["model"] == "M0 baseline")
    m3 = next(r for r in nested_rows if r["model"] == "M3 +repo_FE")
    shrink = 1 - (m3["coef"] / m0["coef"]) if m0["coef"] else float("inf")
    same_sign_sens = sum(1 for r in sensitivity_rows if math.copysign(1, r["coef"]) == math.copysign(1, m3["coef"]))
    sig_sens = sum(1 for r in sensitivity_rows if r["p"] < 0.15)
    boot = boot_rows[0]

    if m3["coef"] <= 0 or m3["p"] > 0.30:
        verdict = "FAIL"
    elif m3["coef"] > 0 and m3["p"] < 0.10 and shrink < 0.50 and same_sign_sens >= 2 and sig_sens >= 1 and boot["ci_low"] > 0:
        verdict = "PASS"
    elif m3["coef"] > 0 and (m3["p"] <= 0.30 or same_sign_sens >= 2 or boot["median"] > 0):
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "m0_coef": m0["coef"],
        "m3_coef": m3["coef"],
        "m3_p": m3["p"],
        "coef_shrinkage": float(shrink),
        "sensitivity_same_sign": int(same_sign_sens),
        "sensitivity_p_lt_0_15": int(sig_sens),
        "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_median": boot["median"],
        "bootstrap_ci_high": boot["ci_high"],
    }


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
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path, meta, nested_rows, interaction_rows, sensitivity_rows, trim_rows, boot_rows, decision):
    lines = [
        "# Check 4: Adaptive Metric Rescue Analysis",
        "",
        "This analysis tests whether the Check 1b rescue signal is an Acc@1 artifact. It uses existing repaired full500 outputs only.",
        "",
        "## Data",
        "",
        f"- N total: {meta['n_total']}",
        f"- N rescue subset (baseline_recall_at5 < 1.0): {meta['n_rescue']}",
        f"- N multi-file: {meta['n_multi']}",
        f"- file_overlap z-score mean/std source: {meta['file_overlap_mean']:.6f} / {meta['file_overlap_std']:.6f}",
        f"- multi-file baseline recall@5 mean: {meta['multi_baseline_recall_at5_mean']:.4f}",
        f"- multi-file baseline precision@1 mean: {meta['multi_baseline_precision_at1_mean']:.4f}",
        "",
        "## Main Result",
        "",
        markdown_table(nested_rows, ["model", "N", "coef", "se", "p", "ci_low", "ci_high", "r2"]),
        "",
        "## Interaction",
        "",
        markdown_table(interaction_rows, ["model", "N", "term", "coef", "se", "p_value", "ci_low", "ci_high"]),
        "",
        "## Sensitivity",
        "",
        markdown_table(sensitivity_rows, ["outcome", "N", "coef", "se", "p", "ci_low", "ci_high", "r2"]),
        "",
        "## Trimmed Sensitivity",
        "",
        markdown_table(trim_rows, ["model", "N", "coef", "se", "p", "ci_low", "ci_high", "q05", "q95"]),
        "",
        "## Bootstrap CI",
        "",
        markdown_table(boot_rows, ["n_boot_requested", "n_boot_success", "seed", "ci_low", "median", "ci_high"]),
        "",
        "## Verdict",
        "",
        f"- verdict: `{decision['verdict']}`",
        f"- M3 file_overlap_z coef: {decision['m3_coef']:.4f}",
        f"- M3 file_overlap_z p: {decision['m3_p']:.4f}",
        f"- coefficient shrinkage from M0 to M3: {decision['coef_shrinkage']:.4f}",
        f"- sensitivity same-sign outcomes: {decision['sensitivity_same_sign']}/3",
        f"- sensitivity outcomes with p < 0.15: {decision['sensitivity_p_lt_0_15']}/3",
        f"- bootstrap CI: [{decision['bootstrap_ci_low']:.4f}, {decision['bootstrap_ci_high']:.4f}], median={decision['bootstrap_median']:.4f}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_repo_n", type=int, default=5)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, overlap_meta = build_outcome_dataframe(args)
    df = add_improvement(df, "recall_at5")
    rescue_df = df[df["baseline_recall_at5"] < 1.0].copy()

    nested_rows, nested_models = run_nested_models(rescue_df)
    interaction_rows, interaction_model = run_interaction_model(rescue_df)
    sensitivity_rows = run_sensitivity_outcomes(df)
    trim_rows, trim_model = run_trim_sensitivity(rescue_df)
    boot_rows = run_bootstrap_ci(rescue_df, n_boot=args.n_boot, seed=args.seed)
    decision = main_decision(nested_rows, sensitivity_rows, trim_rows, boot_rows)

    multi = df[df["gold_files_count"] >= 2]
    meta = {
        "n_total": int(len(df)),
        "n_rescue": int(len(rescue_df)),
        "n_multi": int(len(multi)),
        "multi_baseline_recall_at5_mean": float(multi["baseline_recall_at5"].mean()) if len(multi) else np.nan,
        "multi_baseline_precision_at1_mean": float(multi["baseline_precision_at1"].mean()) if len(multi) else np.nan,
        **overlap_meta,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    outcome_path = os.path.join(args.output_dir, "outcome_metrics.jsonl")
    df.to_json(outcome_path, orient="records", lines=True)
    write_csv(os.path.join(args.output_dir, "nested_models.csv"), nested_rows)
    write_csv(os.path.join(args.output_dir, "interaction_model.csv"), interaction_rows)
    write_csv(os.path.join(args.output_dir, "sensitivity_outcomes.csv"), sensitivity_rows)
    write_csv(os.path.join(args.output_dir, "sensitivity_trim.csv"), trim_rows)
    write_csv(os.path.join(args.output_dir, "bootstrap_ci.csv"), boot_rows)

    summary = {
        "inputs": {
            "baseline": args.baseline,
            "memory": args.memory,
            "gold": args.gold,
        },
        "meta": meta,
        "decision": decision,
        "nested_models": nested_rows,
        "interaction_model": interaction_rows,
        "sensitivity_outcomes": sensitivity_rows,
        "sensitivity_trim": trim_rows,
        "bootstrap_ci": boot_rows,
    }
    with open(os.path.join(args.output_dir, "check4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_report(
        os.path.join(args.output_dir, "check4_report.md"),
        meta,
        nested_rows,
        interaction_rows,
        sensitivity_rows,
        trim_rows,
        boot_rows,
        decision,
    )

    print(json.dumps({"output_dir": args.output_dir, "meta": meta, "decision": decision}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
