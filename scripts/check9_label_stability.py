import argparse
import json
import math
import os

import pandas as pd


DEFAULT_FEATURES_FILE = "results/full500_azure_gpt4o_repaired_compare/check9_gate/check9_features.jsonl"
DEFAULT_CHECK5_FILE = "results/full500_azure_gpt4o_repaired_compare/check5_retrieval_quality/retrieval_quality_per_case.jsonl"
DEFAULT_SELECTED_FILE = "reproduce_lists/verified_check9_harm30.json"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/check9_gate/label_stability"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_path(path):
    path = str(path).strip().strip("`'\" ,:;")
    for prefix in ("./", "a/", "b/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path.lstrip("/")


def flatten_found_files(value):
    if not value:
        return []
    if isinstance(value, list) and value and all(isinstance(x, list) for x in value):
        out = []
        for group in value:
            out.extend(group)
        return [normalize_path(x) for x in out if x]
    if isinstance(value, list):
        return [normalize_path(x) for x in value if x]
    return []


def complete_file_acc(predicted_files, gold_files, k):
    gold = {normalize_path(p) for p in gold_files or []}
    if not gold:
        return 0
    pred = {normalize_path(p) for p in list(predicted_files or [])[:k]}
    return int(gold.issubset(pred))


def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def build_harm_selection(df, k=5, limit=30):
    out = df.copy()
    label = f"harm_at{k}"
    out[label] = ((out[f"baseline_at{k}"].astype(int) == 1) & (out[f"memory_at{k}"].astype(int) == 0)).astype(int)
    selected = out[out[label] == 1].copy()
    selected = selected.sort_values(["repo", "case_id"], kind="mergesort").head(limit)
    return selected.reset_index(drop=True)


def write_selected_ids(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows["case_id"].tolist(), f, indent=2, ensure_ascii=False)


def load_gold_files(check5_file):
    df = pd.read_json(check5_file, lines=True)
    return {row["case_id"]: [normalize_path(p) for p in row.get("gold_files") or []] for _, row in df.iterrows()}


def load_loc_predictions(loc_file):
    preds = {}
    for row in load_jsonl(loc_file):
        iid = row.get("instance_id") or row.get("case_id")
        if iid:
            preds[iid] = flatten_found_files(row.get("found_files"))
    return preds


def compare_label_stability(original, baseline_rerun, memory_rerun, gold_files, k=5):
    rows = []
    stable = 0
    missing = 0
    for _, row in original.iterrows():
        case_id = row["case_id"]
        gold = gold_files.get(case_id, [])
        orig_harm = int((int(row[f"baseline_at{k}"]) == 1) and (int(row[f"memory_at{k}"]) == 0))
        base_pred = baseline_rerun.get(case_id)
        mem_pred = memory_rerun.get(case_id)
        if base_pred is None or mem_pred is None:
            missing += 1
            rerun_harm = None
            base_acc = None
            mem_acc = None
            is_stable = False
        else:
            base_acc = complete_file_acc(base_pred, gold, k)
            mem_acc = complete_file_acc(mem_pred, gold, k)
            rerun_harm = int(base_acc == 1 and mem_acc == 0)
            is_stable = orig_harm == rerun_harm
            stable += int(is_stable)
        rows.append(
            {
                "case_id": case_id,
                "repo": row.get("repo"),
                f"original_harm_at{k}": orig_harm,
                f"rerun_harm_at{k}": rerun_harm,
                f"rerun_baseline_at{k}": base_acc,
                f"rerun_memory_at{k}": mem_acc,
                "stable": bool(is_stable),
                "missing_rerun_prediction": base_pred is None or mem_pred is None,
            }
        )
    n = len(rows)
    lo, hi = wilson_interval(stable, n)
    rate = stable / n if n else float("nan")
    if rate >= 0.90:
        verdict = "stable_continue"
    elif rate >= 0.60:
        verdict = "mixed_discount_offline_results"
    else:
        verdict = "unstable_stop_or_majority_vote"
    summary = {
        "k": int(k),
        "n_cases": int(n),
        "stable_count": int(stable),
        "flip_count": int(n - stable - missing),
        "missing_prediction": int(missing),
        "stability_rate": float(rate),
        "wilson_ci_low": float(lo),
        "wilson_ci_high": float(hi),
        "verdict": verdict,
    }
    return summary, rows


def load_original_features(features_file):
    df = pd.read_json(features_file, lines=True)
    required = {"case_id", "repo", "baseline_at5", "memory_at5"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in {features_file}: {missing}")
    return df


def run_select(args):
    df = load_original_features(args.features_file)
    selected = build_harm_selection(df, k=args.k, limit=args.limit)
    write_selected_ids(selected, args.selected_file)
    write_jsonl(args.selection_details_file, selected.to_dict("records"))
    return {
        "selected_count": int(len(selected)),
        "selected_file": args.selected_file,
        "selection_details_file": args.selection_details_file,
    }


def run_compare(args):
    original = pd.read_json(args.selection_details_file, lines=True)
    if original.empty:
        raise RuntimeError(f"No selection rows found in {args.selection_details_file}")
    gold = load_gold_files(args.check5_file)
    baseline = load_loc_predictions(args.baseline_rerun_loc_file)
    memory = load_loc_predictions(args.memory_rerun_loc_file)
    summary, rows = compare_label_stability(original, baseline, memory, gold, k=args.k)
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "label_stability_summary.json"), summary)
    write_jsonl(os.path.join(args.output_dir, "label_stability_cases.jsonl"), rows)
    write_report(os.path.join(args.output_dir, "label_stability_report.md"), summary, rows)
    return summary


def write_report(path, summary, rows):
    flips = [r for r in rows if not r["stable"] and not r["missing_rerun_prediction"]]
    lines = [
        "# Check 9a' Label Stability",
        "",
        f"- k: {summary['k']}",
        f"- cases: {summary['n_cases']}",
        f"- stable: {summary['stable_count']}",
        f"- flips: {summary['flip_count']}",
        f"- missing rerun prediction: {summary['missing_prediction']}",
        f"- stability rate: {summary['stability_rate']:.4f}",
        f"- Wilson 95% CI: [{summary['wilson_ci_low']:.4f}, {summary['wilson_ci_high']:.4f}]",
        f"- verdict: {summary['verdict']}",
        "",
        "## Flipped Cases",
    ]
    for row in flips:
        lines.append(
            f"- {row['case_id']} ({row.get('repo')}): original={row[f'original_harm_at{summary['k']}']}, "
            f"rerun={row[f'rerun_harm_at{summary['k']}']}"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["select", "compare"], required=True)
    parser.add_argument("--features_file", default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--check5_file", default=DEFAULT_CHECK5_FILE)
    parser.add_argument("--selected_file", default=DEFAULT_SELECTED_FILE)
    parser.add_argument(
        "--selection_details_file",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "label_stability_harm30_selection.jsonl"),
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline_rerun_loc_file")
    parser.add_argument("--memory_rerun_loc_file")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action == "select":
        result = run_select(args)
    else:
        if not args.baseline_rerun_loc_file or not args.memory_rerun_loc_file:
            raise RuntimeError("--baseline_rerun_loc_file and --memory_rerun_loc_file are required for compare")
        result = run_compare(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
