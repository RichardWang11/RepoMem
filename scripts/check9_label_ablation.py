import argparse
import json
import os

import pandas as pd

from scripts.check9_gate_predictability import (
    DEFAULT_CHECK5,
    DEFAULT_OUTPUT_DIR,
    LOCAL_FEATURE_COLUMNS,
    auc_score,
    brier_score,
    build_feature_dataframe,
    compute_topk_coverage,
    feature_columns,
    leave_one_repo_out_predictions,
    precision_at_top_fraction,
    recall_at_threshold,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_LABEL_ABLATION_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "label_ablation")


def add_coverage_zero_labels(df, ks):
    out = df.copy()
    if "top_commits" not in out.columns:
        raise KeyError("Need top_commits column to compute label ablation")
    if "gold_files" not in out.columns:
        raise KeyError("Need gold_files column to compute label ablation")
    for k in ks:
        out[f"coverage_at{k}"] = out.apply(
            lambda row: compute_topk_coverage(row["top_commits"], row["gold_files"], top_k=k),
            axis=1,
        )
        out[f"coverage_zero_at{k}"] = (out[f"coverage_at{k}"] == 0.0).astype(int)
    return out


def summarize_label_ablation(rows):
    out = []
    for row in rows:
        n = int(row["n_cases"])
        positive = int(row["positive_count"])
        new_row = dict(row)
        new_row["label"] = f"coverage_zero_at{row['k']}"
        new_row["positive_rate"] = positive / n if n else 0.0
        out.append(new_row)
    return out


def build_ablation_dataframe(args):
    df = build_feature_dataframe(args)
    check5 = pd.read_json(args.check5, lines=True)
    extra = check5[["case_id", "gold_files"]].copy()
    df = df.merge(extra, on="case_id", how="left")

    top_commits_by_case = {}
    # Reconstruct from feature source by asking build_feature_dataframe to keep only
    # derived columns would be insufficient, so use a narrow copy of the loader logic.
    from scripts.check9_gate_predictability import load_top_commits

    top_commit_data = load_top_commits(args.traj_files, args.index_dirs, max_commits=args.max_commits)
    for case_id, data in top_commit_data.items():
        top_commits_by_case[case_id] = data.get("commits") or []
    df["top_commits"] = df["case_id"].map(top_commits_by_case).apply(lambda x: x if isinstance(x, list) else [])
    return add_coverage_zero_labels(df, args.ks)


def run_label_ablation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = build_ablation_dataframe(args)
    cols = [c for c in feature_columns(df) if c in LOCAL_FEATURE_COLUMNS]
    rows = []
    prediction_frames = []
    for k in args.ks:
        label_col = f"coverage_zero_at{k}"
        work = df.dropna(subset=cols + [label_col, "repo"]).copy()
        pred = leave_one_repo_out_predictions(work, cols, label_col, "repo")
        pred["k"] = k
        prediction_frames.append(pred)
        y = pred[label_col]
        score = pred["risk_probability"]
        rows.append(
            {
                "k": int(k),
                "n_cases": int(len(work)),
                "positive_count": int(y.sum()),
                "auc": auc_score(y, score),
                "precision_top30_risky": precision_at_top_fraction(y, score, 0.30),
                "recall_threshold_0.5": recall_at_threshold(y, score, 0.5),
                "brier": brier_score(y, score),
            }
        )

    summary_rows = summarize_label_ablation(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    write_jsonl(os.path.join(args.output_dir, "features_with_label_ablation.jsonl"), df.to_dict("records"))
    write_jsonl(os.path.join(args.output_dir, "predictions_by_k.jsonl"), predictions.to_dict("records"))
    write_csv(os.path.join(args.output_dir, "label_ablation_summary.csv"), summary_rows)
    write_json(os.path.join(args.output_dir, "label_ablation_summary.json"), {"rows": summary_rows, "features": cols})
    write_report(os.path.join(args.output_dir, "label_ablation_report.md"), summary_rows, cols)
    return summary_rows


def write_report(path, rows, cols):
    lines = [
        "# Check 9 Label Ablation",
        "",
        "Same gold-free local features and LORO logistic classifier; only the coverage_zero label top-k changes.",
        "",
        f"- features: {', '.join(cols)}",
        "",
        "| label | N | positives | positive rate | LORO AUC | P@top30 risky | recall@0.5 | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        recall = row["recall_threshold_0.5"]
        lines.append(
            f"| {row['label']} | {row['n_cases']} | {row['positive_count']} | "
            f"{row['positive_rate']:.4f} | {row['auc']:.4f} | "
            f"{row['precision_top30_risky']:.4f} | {recall:.4f} | {row['brier']:.4f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check5", default=DEFAULT_CHECK5)
    parser.add_argument("--output_dir", default=DEFAULT_LABEL_ABLATION_DIR)
    parser.add_argument("--traj_files", nargs="+", default=None)
    parser.add_argument("--index_dirs", nargs="+", default=None)
    parser.add_argument("--coverage_k", type=int, default=5)
    parser.add_argument("--max_commits", type=int, default=20)
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()
    if args.traj_files is None:
        from scripts.check9_gate_predictability import DEFAULT_TRAJ_FILES

        args.traj_files = DEFAULT_TRAJ_FILES
    if args.index_dirs is None:
        from scripts.analyze_memory_fragmentation import DEFAULT_INDEX_DIRS

        args.index_dirs = DEFAULT_INDEX_DIRS
    return args


if __name__ == "__main__":
    print(json.dumps(run_label_ablation(parse_args()), indent=2))
