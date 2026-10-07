import argparse
import csv
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


DEFAULT_INPUT = "results/full500_azure_gpt4o_repaired_compare/fragmentation/fragmentation_per_case.jsonl"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/fragmentation_logistic"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_exp(x):
    if x > 709:
        return float("inf")
    if x < -745:
        return 0.0
    return float(math.exp(x))


def fit_logit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(np.unique(y)) < 2:
        raise ValueError("logistic regression needs both outcome classes")

    x_mean = float(x.mean())
    x_std = float(x.std(ddof=0))
    if x_std == 0:
        raise ValueError("predictor has zero variance")

    x_z = (x - x_mean) / x_std
    design = np.column_stack([np.ones_like(x_z), x_z])

    def negative_log_likelihood(beta):
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    result = minimize(negative_log_likelihood, np.zeros(2), method="BFGS")
    if not result.success:
        raise RuntimeError(f"logistic regression failed: {result.message}")

    beta = result.x
    p = sigmoid(design @ beta)
    weights = p * (1 - p)
    hessian = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(hessian)
    se_z = np.sqrt(np.diag(covariance))

    coef_z = float(beta[1])
    se_coef_z = float(se_z[1])
    z_value = coef_z / se_coef_z
    p_value = float(2 * norm.sf(abs(z_value)))

    coef_raw = coef_z / x_std
    se_raw = se_coef_z / x_std
    intercept_raw = float(beta[0] - coef_z * x_mean / x_std)
    odds_ratio_1sd = safe_exp(coef_z)
    ci_low = safe_exp(coef_z - 1.96 * se_coef_z)
    ci_high = safe_exp(coef_z + 1.96 * se_coef_z)

    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "event_rate": float(y.mean()),
        "x_mean": x_mean,
        "x_std": x_std,
        "intercept_raw": intercept_raw,
        "coef_raw": float(coef_raw),
        "se_raw": float(se_raw),
        "coef_per_1sd": coef_z,
        "se_per_1sd": se_coef_z,
        "z_value": float(z_value),
        "p_value": p_value,
        "odds_ratio_per_1sd": odds_ratio_1sd,
        "odds_ratio_per_1sd_ci_low": ci_low,
        "odds_ratio_per_1sd_ci_high": ci_high,
    }


def get_columns(df):
    score_col = "file_overlap_score" if "file_overlap_score" in df.columns else "file_overlap_jaccard_mean"
    if score_col not in df.columns:
        raise KeyError("Need file_overlap_score or file_overlap_jaccard_mean")

    baseline_col = "baseline_correct_at1" if "baseline_correct_at1" in df.columns else "baseline_ok_k1"
    memory_col = "memory_correct_at1" if "memory_correct_at1" in df.columns else "repomem_ok_k1"
    for col in [baseline_col, memory_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    return score_col, baseline_col, memory_col


def build_models(df):
    score_col, baseline_col, memory_col = get_columns(df)
    work = df[[score_col, baseline_col, memory_col]].dropna().copy()
    work[baseline_col] = work[baseline_col].astype(bool)
    work[memory_col] = work[memory_col].astype(bool)

    rescue_sub = work[~work[baseline_col]].copy()
    rescue_sub["rescued"] = rescue_sub[memory_col].astype(int)

    regress_sub = work[work[baseline_col]].copy()
    regress_sub["regressed"] = (~regress_sub[memory_col]).astype(int)

    rescue = fit_logit(rescue_sub[score_col], rescue_sub["rescued"])
    rescue.update(
        {
            "model": "rescue_given_baseline_wrong",
            "outcome": "rescued",
            "predictor": score_col,
            "expected_direction": "positive",
            "interpretation": "higher file overlap means lower fragmentation; positive coefficient means memory rescue is more likely when retrieval is less fragmented",
        }
    )

    regress = fit_logit(regress_sub[score_col], regress_sub["regressed"])
    regress.update(
        {
            "model": "regress_given_baseline_correct",
            "outcome": "regressed",
            "predictor": score_col,
            "expected_direction": "negative",
            "interpretation": "higher file overlap means lower fragmentation; negative coefficient means memory regression is less likely when retrieval is less fragmented",
        }
    )

    return [rescue, regress], {
        "score_col": score_col,
        "baseline_col": baseline_col,
        "memory_col": memory_col,
        "analyzed_rows": int(len(work)),
        "dropped_rows": int(len(df) - len(work)),
    }


def decision(rows):
    rescue = next(r for r in rows if r["model"] == "rescue_given_baseline_wrong")
    regress = next(r for r in rows if r["model"] == "regress_given_baseline_correct")
    rescue_supported = rescue["coef_per_1sd"] > 0 and rescue["p_value"] < 0.10
    regress_supported = regress["coef_per_1sd"] < 0 and regress["p_value"] < 0.10
    if rescue_supported or regress_supported:
        return "continuous_signal_supported"
    return "continuous_signal_not_supported"


def write_csv(path, rows):
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


def write_report(path, input_path, meta, rows):
    verdict = decision(rows)
    lines = [
        "# Fragmentation Logistic Regression",
        "",
        "This analysis uses existing full500 fragmentation outputs only. It does not modify RepoMem, LocAgent, or memory retrieval.",
        "",
        f"- input: `{input_path}`",
        f"- analyzed rows: {meta['analyzed_rows']}",
        f"- dropped rows: {meta['dropped_rows']}",
        f"- predictor: `{meta['score_col']}`",
        "- metric direction: higher file overlap means lower fragmentation",
        f"- decision: `{verdict}`",
        "",
        "## Models",
        "",
        markdown_table(
            rows,
            [
                "model",
                "n",
                "positive",
                "event_rate",
                "coef_per_1sd",
                "se_per_1sd",
                "z_value",
                "p_value",
                "odds_ratio_per_1sd",
                "odds_ratio_per_1sd_ci_low",
                "odds_ratio_per_1sd_ci_high",
                "expected_direction",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- Rescue model: fitted only on baseline-wrong cases; positive coefficient supports the hypothesis.",
        "- Regression model: fitted only on baseline-correct cases; negative coefficient supports the hypothesis.",
        "- Acceptance rule: signal exists if either expected-direction coefficient has p < 0.10.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    rows, meta = build_models(df)
    summary = {
        "input": args.input,
        **meta,
        "models": rows,
        "decision": decision(rows),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "logistic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_csv(os.path.join(args.output_dir, "logistic_models.csv"), rows)
    write_report(os.path.join(args.output_dir, "logistic_report.md"), args.input, meta, rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
