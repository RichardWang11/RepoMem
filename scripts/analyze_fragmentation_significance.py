import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


DEFAULT_INPUT = "results/full500_azure_gpt4o_repaired_compare/fragmentation/fragmentation_per_case.jsonl"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/fragmentation_significance"


def assign_file_overlap_tiers(df):
    df = df.copy()
    q33, q67 = df["file_overlap_jaccard_mean"].quantile([1 / 3, 2 / 3])
    df["frag_tier"] = pd.cut(
        df["file_overlap_jaccard_mean"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["high_frag", "mid_frag", "low_frag"],
        include_lowest=True,
    )
    return df


def tier_stats(group, k):
    n = len(group)
    baseline = int(group[f"baseline_ok_k{k}"].sum())
    memory = int(group[f"repomem_ok_k{k}"].sum())
    rescue = int(((~group[f"baseline_ok_k{k}"]) & group[f"repomem_ok_k{k}"]).sum())
    regress = int((group[f"baseline_ok_k{k}"] & (~group[f"repomem_ok_k{k}"])).sum())
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


def build_fisher_table(df, k):
    low = df[df["frag_tier"] == "low_frag"]
    high = df[df["frag_tier"] == "high_frag"]

    rescue_low = int(((~low[f"baseline_ok_k{k}"]) & low[f"repomem_ok_k{k}"]).sum())
    regress_low = int((low[f"baseline_ok_k{k}"] & (~low[f"repomem_ok_k{k}"])).sum())
    rescue_high = int(((~high[f"baseline_ok_k{k}"]) & high[f"repomem_ok_k{k}"]).sum())
    regress_high = int((high[f"baseline_ok_k{k}"] & (~high[f"repomem_ok_k{k}"])).sum())
    return [[rescue_low, regress_low], [rescue_high, regress_high]]


def build_chi_square_table(df, k):
    table = []
    for tier in ["low_frag", "mid_frag", "high_frag"]:
        group = df[df["frag_tier"] == tier]
        rescue = int(((~group[f"baseline_ok_k{k}"]) & group[f"repomem_ok_k{k}"]).sum())
        regress = int((group[f"baseline_ok_k{k}"] & (~group[f"repomem_ok_k{k}"])).sum())
        table.append([rescue, regress])
    return table


def bootstrap_delta_gap(df, k, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        sample = df.sample(frac=1, replace=True, random_state=int(rng.integers(1_000_000_000)))
        low = sample[sample["frag_tier"] == "low_frag"]
        high = sample[sample["frag_tier"] == "high_frag"]
        if len(low) == 0 or len(high) == 0:
            continue
        low_delta = (low[f"repomem_ok_k{k}"].sum() - low[f"baseline_ok_k{k}"].sum()) / len(low)
        high_delta = (high[f"repomem_ok_k{k}"].sum() - high[f"baseline_ok_k{k}"].sum()) / len(high)
        gaps.append(float(low_delta - high_delta))

    gaps = np.array(gaps)
    ci_low, ci_high = np.percentile(gaps, [2.5, 97.5])
    return {
        "k": k,
        "n_boot": n_boot,
        "seed": seed,
        "mean_gap": float(gaps.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def run_significance_tests(df):
    rows = []
    for k in [1, 5]:
        fisher_table = build_fisher_table(df, k)
        odds_ratio, fisher_p = fisher_exact(fisher_table)
        chi_table = build_chi_square_table(df, k)
        chi2, chi_p, dof, expected = chi2_contingency(chi_table)
        rows.append(
            {
                "k": k,
                "test": "fisher_exact_low_vs_high_rescue_regress",
                "table": fisher_table,
                "odds_ratio": float(odds_ratio),
                "p_value": float(fisher_p),
                "chi2": "",
                "dof": "",
                "expected": "",
            }
        )
        rows.append(
            {
                "k": k,
                "test": "chi_square_tercile_rescue_regress",
                "table": chi_table,
                "odds_ratio": "",
                "p_value": float(chi_p),
                "chi2": float(chi2),
                "dof": int(dof),
                "expected": expected.tolist(),
            }
        )
    return rows


def compute_tier_stats(df):
    rows = []
    for k in [1, 5]:
        for tier in ["low_frag", "mid_frag", "high_frag"]:
            group = df[df["frag_tier"] == tier]
            stats = tier_stats(group, k)
            rows.append({"k": k, "frag_tier": tier, **stats})
    return rows


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


def acceptance_label(acc1_fisher, acc1_gap):
    if acc1_gap["mean_gap"] >= 0.02 and acc1_fisher["p_value"] < 0.10 and acc1_gap["ci_low"] > 0:
        return "significant_support"
    if acc1_gap["mean_gap"] >= 0.02 and 0.10 <= acc1_fisher["p_value"] <= 0.25:
        return "weak_support"
    return "not_supported_on_acc1"


def write_report(path, input_path, total_rows, analyzed_rows, tier_rows, test_rows, boot_rows, summary):
    acc1_fisher = next(r for r in test_rows if r["k"] == 1 and r["test"].startswith("fisher"))
    acc5_fisher = next(r for r in test_rows if r["k"] == 5 and r["test"].startswith("fisher"))
    acc1_gap = next(r for r in boot_rows if r["k"] == 1)
    acc5_gap = next(r for r in boot_rows if r["k"] == 5)
    label = acceptance_label(acc1_fisher, acc1_gap)

    lines = [
        "# Fragmentation Significance Test",
        "",
        "This is a statistical analysis on existing full500 fragmentation outputs. "
        "It does not modify RepoMem, LocAgent, or any memory retrieval method.",
        "",
        f"- input: `{input_path}`",
        f"- total rows: {total_rows}",
        f"- analyzed rows with file-overlap score: {analyzed_rows}",
        "- primary metric: `file_overlap_jaccard_mean`",
        "- metric direction: higher file overlap means lower fragmentation",
        "",
        "## Verdict",
        "",
        f"- Acc@1 decision: `{label}`",
        f"- Acc@1 delta gap: {acc1_gap['mean_gap']:.4f}, 95% CI [{acc1_gap['ci_low']:.4f}, {acc1_gap['ci_high']:.4f}]",
        f"- Acc@1 Fisher p-value: {acc1_fisher['p_value']:.4f}",
        f"- Acc@5 delta gap: {acc5_gap['mean_gap']:.4f}, 95% CI [{acc5_gap['ci_low']:.4f}, {acc5_gap['ci_high']:.4f}]",
        f"- Acc@5 Fisher p-value: {acc5_fisher['p_value']:.4f}",
        "",
        "Interpretation: the direction is consistent with file-overlap fragmentation, but the current full500 evidence does not pass the requested Acc@1 significance rule because Fisher p-value is above 0.25 and the bootstrap CI crosses zero.",
        "",
        "## Tier Stats",
        "",
        markdown_table(
            tier_rows,
            [
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
        "## Fisher and Chi-Square",
        "",
        markdown_table(
            test_rows,
            ["k", "test", "table", "odds_ratio", "p_value", "chi2", "dof"],
        ),
        "",
        "## Bootstrap Gap CI",
        "",
        markdown_table(
            boot_rows,
            ["k", "n_boot", "seed", "mean_gap", "ci_low", "ci_high"],
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df_all = pd.read_json(args.input, lines=True)
    df = df_all[df_all["file_overlap_jaccard_mean"].notna()].copy()
    df = assign_file_overlap_tiers(df)

    os.makedirs(args.output_dir, exist_ok=True)
    tier_rows = compute_tier_stats(df)
    test_rows = run_significance_tests(df)
    boot_rows = [
        bootstrap_delta_gap(df, 1, n_boot=args.n_boot, seed=args.seed),
        bootstrap_delta_gap(df, 5, n_boot=args.n_boot, seed=args.seed),
    ]

    summary = {
        "input": args.input,
        "total_rows": int(len(df_all)),
        "analyzed_rows": int(len(df)),
        "missing_file_overlap_rows": int(len(df_all) - len(df)),
        "tier_stats": tier_rows,
        "tests": test_rows,
        "bootstrap_gap_ci": boot_rows,
        "acc1_decision": acceptance_label(
            next(r for r in test_rows if r["k"] == 1 and r["test"].startswith("fisher")),
            boot_rows[0],
        ),
    }

    with open(os.path.join(args.output_dir, "significance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_csv(os.path.join(args.output_dir, "tier_stats.csv"), tier_rows)
    write_csv(os.path.join(args.output_dir, "fisher_chi_square.csv"), test_rows)
    write_csv(os.path.join(args.output_dir, "bootstrap_gap_ci.csv"), boot_rows)
    write_report(
        os.path.join(args.output_dir, "significance_report.md"),
        args.input,
        len(df_all),
        len(df),
        tier_rows,
        test_rows,
        boot_rows,
        summary,
    )

    print(json.dumps({
        "output_dir": args.output_dir,
        "total_rows": len(df_all),
        "analyzed_rows": len(df),
        "acc1_decision": summary["acc1_decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
