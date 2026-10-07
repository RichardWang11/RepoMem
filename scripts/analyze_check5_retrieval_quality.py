import argparse
import csv
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from scripts.analyze_check4_adaptive_metrics import (
    DEFAULT_BASELINE,
    DEFAULT_GOLD,
    DEFAULT_MEMORY,
    add_improvement,
    build_outcome_dataframe,
    merge_rare_repos,
)


DEFAULT_FRAGMENTATION = "results/full500_azure_gpt4o_repaired_compare/fragmentation/fragmentation_per_case.jsonl"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/check5_retrieval_quality"
DEFAULT_INDEX_DIRS = [
    "repomem_index/verified_200_episodic_with_issues",
    "repomem_index/verified_remaining_247_episodic_with_issues_v2",
]


def normalize_path(path):
    path = str(path).strip().strip("`'\" ,:;")
    for prefix in ("./", "a/", "b/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path.lstrip("/")


def normalize_repo(repo):
    return str(repo).replace("/", "__")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_commit_lookup_from_rows(rows):
    lookup = {}
    for row in rows:
        repo = row.get("repo")
        sha = row.get("sha")
        if not repo or not sha:
            continue
        files = [normalize_path(p) for p in row.get("changed_files") or []]
        for repo_key in {repo, normalize_repo(repo)}:
            lookup[(repo_key, sha)] = files
            for n in (7, 8, 9, 10, 12):
                lookup[(repo_key, sha[:n])] = files
    return lookup


def load_commit_lookup(index_dirs):
    rows = []
    for root in index_dirs:
        if not os.path.isdir(root):
            continue
        for repo_dir in sorted(os.listdir(root)):
            path = os.path.join(root, repo_dir, "commits.jsonl")
            if not os.path.exists(path):
                continue
            for row in load_jsonl(path):
                if not row.get("repo"):
                    row["repo"] = repo_dir.replace("__", "/")
                rows.append(row)
    return build_commit_lookup_from_rows(rows)


def lookup_commit_files(lookup, repo, sha):
    return (
        lookup.get((repo, sha))
        or lookup.get((normalize_repo(repo), sha))
        or []
    )


def repo_for_lookup(row):
    repo_full = row.get("repo_full")
    if isinstance(repo_full, str) and "/" in repo_full:
        return repo_full
    repo = row.get("repo")
    return repo if isinstance(repo, str) else ""


def compute_retrieval_gold_coverage(commits, gold_files):
    gold_set = {normalize_path(p) for p in gold_files or []}
    if not gold_set:
        return 0.0
    retrieved_files = set()
    for commit in commits:
        retrieved_files.update(normalize_path(p) for p in commit.get("changed_files") or [])
    return len(retrieved_files & gold_set) / max(1, len(gold_set))


def fit_ols(df, outcome, predictors):
    needed = [outcome]
    for pred in predictors:
        if pred == "C(repo)":
            needed.append("repo_fe")
        else:
            needed.append(pred)
    needed = list(dict.fromkeys(needed))
    work = df[needed].dropna().copy()
    if len(work) == 0:
        raise ValueError("No rows available for OLS")

    columns = [np.ones(len(work))]
    names = ["Intercept"]
    for pred in predictors:
        if pred == "C(repo)":
            dummies = pd.get_dummies(work["repo_fe"], prefix="repo", drop_first=True, dtype=float)
            for col in dummies.columns:
                columns.append(dummies[col].to_numpy(dtype=float))
                names.append(col)
        else:
            columns.append(work[pred].to_numpy(dtype=float))
            names.append(pred)

    x = np.column_stack(columns)
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
        rows.append(
            {
                "term": name,
                "coef": float(coef),
                "se": float(coef_se),
                "t_value": t_value,
                "p_value": p_value,
                "ci_low": float(coef - 1.96 * coef_se),
                "ci_high": float(coef + 1.96 * coef_se),
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


def model_summary_row(name, model, term="retrieval_gold_coverage"):
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


def build_retrieval_dataframe(args):
    class Obj:
        pass

    check4_args = Obj()
    check4_args.baseline = args.baseline
    check4_args.memory = args.memory
    check4_args.gold = args.gold
    check4_args.min_repo_n = args.min_repo_n
    df, overlap_meta = build_outcome_dataframe(check4_args)
    df = add_improvement(df, "recall_at5")

    frag = pd.read_json(args.fragmentation, lines=True)
    if "case_id" not in frag.columns:
        frag["case_id"] = frag["instance_id"]
    frag_cols = ["case_id", "top_commit_shas"]
    if "repo" in frag.columns:
        frag = frag.rename(columns={"repo": "repo_full"})
        frag_cols.append("repo_full")
    frag = frag[frag_cols].copy()
    df = df.merge(frag, on="case_id", how="left")

    lookup = load_commit_lookup(args.index_dirs)
    coverage_rows = []
    for _, row in df.iterrows():
        shas = row.get("top_commit_shas") or []
        lookup_repo = repo_for_lookup(row)
        commits = []
        missing = 0
        for sha in shas:
            files = lookup_commit_files(lookup, lookup_repo, sha)
            if files:
                commits.append({"sha": sha, "changed_files": files})
            else:
                missing += 1
                commits.append({"sha": sha, "changed_files": []})
        coverage_rows.append(
            {
                "retrieval_gold_coverage": compute_retrieval_gold_coverage(commits, row["gold_files"]),
                "retrieved_changed_files_count": len({p for c in commits for p in c.get("changed_files", [])}),
                "top_commit_count": len(shas),
                "missing_commit_file_count": missing,
                "retrieval_gold_hit": int(compute_retrieval_gold_coverage(commits, row["gold_files"]) > 0),
            }
        )
    coverage = pd.DataFrame(coverage_rows, index=df.index)
    df = pd.concat([df, coverage], axis=1)
    df = merge_rare_repos(df, "repo", min_n=args.min_repo_n)
    return df, {"commit_lookup_entries": int(len(lookup)), **overlap_meta}


def run_models(rescue_df):
    specs = [
        ("M0 coverage_only", ["retrieval_gold_coverage"]),
        ("M1 +gold_count", ["retrieval_gold_coverage", "gold_files_count"]),
        ("M2 +repo_FE", ["retrieval_gold_coverage", "gold_files_count", "C(repo)"]),
        ("M3 +baseline_lvl", ["retrieval_gold_coverage", "gold_files_count", "C(repo)", "baseline_recall_at5"]),
    ]
    rows = []
    models = {}
    for name, predictors in specs:
        model = fit_ols(rescue_df, "improvement", predictors)
        rows.append(model_summary_row(name, model))
        models[name] = model
    return rows, models


def coverage_bucket_stats(df):
    rows = []
    for label, sub in [
        ("coverage_eq_0", df[df["retrieval_gold_coverage"] == 0]),
        ("coverage_gt_0", df[df["retrieval_gold_coverage"] > 0]),
    ]:
        rows.append(
            {
                "bucket": label,
                "N": int(len(sub)),
                "mean_improvement": float(sub["improvement"].mean()) if len(sub) else np.nan,
                "rescue_rate_recall": float((sub["improvement"] > 0).mean()) if len(sub) else np.nan,
                "memory_better_count": int((sub["improvement"] > 0).sum()),
                "memory_worse_count": int((sub["improvement"] < 0).sum()),
                "memory_same_count": int((sub["improvement"] == 0).sum()),
            }
        )
    return rows


def main_decision(primary_row):
    if primary_row["coef"] > 0 and primary_row["p"] < 0.05:
        verdict = "PASS_RETRIEVAL_QUALITY_SIGNAL"
        conclusion = "retrieval_gold_coverage is a stronger mechanism candidate than fragmentation"
    elif primary_row["p"] > 0.20:
        verdict = "FAIL_NULL_RETRIEVAL_QUALITY"
        conclusion = "memory help/regression is not explained by direct gold-file coverage in top retrieved commits"
    else:
        verdict = "MARGINAL_RETRIEVAL_QUALITY"
        conclusion = "retrieval_gold_coverage is directionally plausible but statistically weak"
    return {"verdict": verdict, "conclusion": conclusion}


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


def write_report(path, meta, model_rows, bucket_rows, decision):
    primary = next(r for r in model_rows if r["model"] == "M2 +repo_FE")
    lines = [
        "# Check 5: Retrieval Gold Coverage",
        "",
        "This analysis directly tests whether SearchCommit top commits touch gold files. It uses existing trajectories/indexes only.",
        "",
        "## Data",
        "",
        f"- N total: {meta['n_total']}",
        f"- N rescue subset (baseline_recall_at5 < 1.0): {meta['n_rescue']}",
        f"- commit lookup entries: {meta['commit_lookup_entries']}",
        f"- mean retrieval_gold_coverage: {meta['mean_retrieval_gold_coverage']:.4f}",
        f"- coverage > 0 cases: {meta['coverage_gt_0_count']}",
        f"- missing commit-file links: {meta['missing_commit_file_count']}",
        "",
        "## Primary Model",
        "",
        "Outcome: `memory_recall_at5 - baseline_recall_at5` on baseline-imperfect cases.",
        "",
        markdown_table(model_rows, ["model", "N", "coef", "se", "p", "ci_low", "ci_high", "r2"]),
        "",
        "## Coverage Buckets",
        "",
        markdown_table(bucket_rows, ["bucket", "N", "mean_improvement", "rescue_rate_recall", "memory_better_count", "memory_worse_count", "memory_same_count"]),
        "",
        "## Verdict",
        "",
        f"- verdict: `{decision['verdict']}`",
        f"- conclusion: {decision['conclusion']}",
        f"- primary coef: {primary['coef']:.4f}",
        f"- primary p: {primary['p']:.4f}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--fragmentation", default=DEFAULT_FRAGMENTATION)
    parser.add_argument("--index_dirs", nargs="+", default=DEFAULT_INDEX_DIRS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_repo_n", type=int, default=5)
    args = parser.parse_args()

    df, extra_meta = build_retrieval_dataframe(args)
    rescue_df = df[df["baseline_recall_at5"] < 1.0].copy()
    model_rows, models = run_models(rescue_df)
    bucket_rows = coverage_bucket_stats(rescue_df)
    primary = next(r for r in model_rows if r["model"] == "M2 +repo_FE")
    decision = main_decision(primary)

    meta = {
        "n_total": int(len(df)),
        "n_rescue": int(len(rescue_df)),
        "mean_retrieval_gold_coverage": float(df["retrieval_gold_coverage"].mean()),
        "coverage_gt_0_count": int((df["retrieval_gold_coverage"] > 0).sum()),
        "missing_commit_file_count": int(df["missing_commit_file_count"].sum()),
        **extra_meta,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    df.to_json(os.path.join(args.output_dir, "retrieval_quality_per_case.jsonl"), orient="records", lines=True)
    write_csv(os.path.join(args.output_dir, "retrieval_quality_models.csv"), model_rows)
    write_csv(os.path.join(args.output_dir, "coverage_bucket_stats.csv"), bucket_rows)
    summary = {
        "inputs": {
            "baseline": args.baseline,
            "memory": args.memory,
            "gold": args.gold,
            "fragmentation": args.fragmentation,
            "index_dirs": args.index_dirs,
        },
        "meta": meta,
        "decision": decision,
        "models": model_rows,
        "coverage_buckets": bucket_rows,
    }
    with open(os.path.join(args.output_dir, "check5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_report(os.path.join(args.output_dir, "check5_report.md"), meta, model_rows, bucket_rows, decision)

    print(json.dumps({"output_dir": args.output_dir, "meta": meta, "decision": decision}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
