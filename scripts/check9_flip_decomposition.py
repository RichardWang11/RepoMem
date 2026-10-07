import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SELECTION_FILE = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "label_stability/label_stability_harm30_selection.jsonl"
)
DEFAULT_BASELINE_RERUN = "results/check9_label_stability_gpt4o_baseline_harm30/merged_loc_outputs_mrr.repaired.jsonl"
DEFAULT_MEMORY_RERUN = "results/check9_label_stability_gpt4o_episodic_harm30/merged_loc_outputs_mrr.repaired.jsonl"
DEFAULT_CHECK5_FILE = "results/full500_azure_gpt4o_repaired_compare/check5_retrieval_quality/retrieval_quality_per_case.jsonl"
DEFAULT_INDEX_DIR = "repomem_index/verified_full500_episodic_with_issues_combined"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/check9_gate/flip_decomposition"
DEFAULT_ORIG_BASELINE_ARGS = "results/full500_azure_gpt4o_repaired_compare/baseline_args.json"
DEFAULT_ORIG_MEMORY_ARGS = "results/full500_azure_gpt4o_repaired_compare/episodic_args.json"


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


def normalize_selection_row(row):
    out = dict(row)
    if "case_id" not in out and "instance_id" in out:
        out["case_id"] = out["instance_id"]
    required = {"case_id", "repo", "baseline_at5", "memory_at5"}
    missing = sorted(required - set(out))
    if missing:
        case = out.get("case_id") or out.get("instance_id") or "<unknown>"
        raise ValueError(f"Selection row for {case} missing required fields: {missing}")
    return out


def classify_flip(orig_b, orig_m, rerun_b, rerun_m):
    if rerun_b is None or rerun_m is None:
        return "missing_rerun"
    if int(orig_b) != 1 or int(orig_m) != 0:
        raise ValueError(f"Expected original harm case (1, 0), got ({orig_b}, {orig_m})")
    pair = (int(rerun_b), int(rerun_m))
    if pair == (1, 0):
        return "stable_harm"
    if pair == (0, 0):
        return "baseline_flipped"
    if pair == (1, 1):
        return "memory_flipped"
    if pair == (0, 1):
        return "harm_to_rescue"
    raise ValueError(f"Invalid rerun correctness pair: {pair}")


def bucket_commit_count(count):
    if count < 10000:
        return "small"
    if count < 30000:
        return "mid"
    return "large"


def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_gold_files(check5_file):
    gold = {}
    for row in load_jsonl(check5_file):
        gold[row["case_id"]] = [normalize_path(p) for p in row.get("gold_files") or []]
    return gold


def load_loc_predictions(loc_file):
    preds = {}
    for row in load_jsonl(loc_file):
        case_id = row.get("instance_id") or row.get("case_id")
        if case_id:
            preds[case_id] = flatten_found_files(row.get("found_files"))
    return preds


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_commit_counts(index_dir):
    counts = {}
    for path in sorted(Path(index_dir).glob("*/commits.jsonl")):
        repo_dir = path.parent.name
        count = count_lines(path)
        keys = {repo_dir, repo_dir.replace("__", "/")}
        if "__" in repo_dir:
            keys.add(repo_dir.split("__", 1)[1])
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            first = json.loads(first_line)
            if first.get("repo"):
                keys.add(first["repo"])
                keys.add(str(first["repo"]).replace("/", "__"))
                keys.add(str(first["repo"]).split("/")[-1])
        except (IndexError, json.JSONDecodeError, OSError):
            pass
        for key in keys:
            counts[key] = count
    return counts


def repo_lookup_keys(row):
    keys = []
    for key in ("repo_full", "repo"):
        if row.get(key):
            value = str(row[key])
            keys.extend([value, value.replace("/", "__"), value.split("/")[-1]])
    case_id = row.get("case_id", "")
    if "__" in case_id:
        owner_repo = case_id.rsplit("-", 1)[0]
        if "__" in owner_repo:
            keys.extend([owner_repo, owner_repo.replace("__", "/"), owner_repo.split("__", 1)[1]])
    return keys


def lookup_commit_count(row, commit_counts):
    for key in repo_lookup_keys(row):
        if key in commit_counts:
            return commit_counts[key]
    return None


def read_args(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_args(label, args):
    if not args:
        return {
            "label": label,
            "available": False,
            "model": None,
            "num_samples": None,
            "ranking_method": None,
            "use_repomem": None,
        }
    return {
        "label": label,
        "available": True,
        "model": args.get("model"),
        "num_samples": args.get("num_samples"),
        "ranking_method": args.get("ranking_method"),
        "use_repomem": args.get("use_repomem"),
        "max_iteration_num": args.get("max_iteration_num"),
        "max_attempt_num": args.get("max_attempt_num"),
    }


def build_rows(selection_rows, baseline_preds, memory_preds, gold_files, commit_counts, k=5):
    out = []
    for raw in selection_rows:
        row = normalize_selection_row(raw)
        case_id = row["case_id"]
        gold = gold_files.get(case_id, [])
        base_pred = baseline_preds.get(case_id)
        mem_pred = memory_preds.get(case_id)
        rerun_b = complete_file_acc(base_pred, gold, k) if base_pred is not None else None
        rerun_m = complete_file_acc(mem_pred, gold, k) if mem_pred is not None else None
        commit_count = lookup_commit_count(row, commit_counts)
        flip_class = classify_flip(row[f"baseline_at{k}"], row[f"memory_at{k}"], rerun_b, rerun_m)
        out.append(
            {
                "case_id": case_id,
                "repo": row.get("repo"),
                "repo_full": row.get("repo_full"),
                "commit_count": commit_count,
                "repo_bucket": bucket_commit_count(commit_count) if commit_count is not None else "unknown",
                "flip_class": flip_class,
                f"orig_baseline_at{k}": int(row[f"baseline_at{k}"]),
                f"orig_memory_at{k}": int(row[f"memory_at{k}"]),
                f"rerun_baseline_at{k}": rerun_b,
                f"rerun_memory_at{k}": rerun_m,
                "gold_files_count": len(gold),
                "missing_rerun_prediction": base_pred is None or mem_pred is None,
            }
        )
    return out


def table_counts(rows, key):
    counts = Counter(row[key] for row in rows)
    return [{"name": name, "count": count, "pct": count / len(rows) if rows else 0.0} for name, count in sorted(counts.items())]


def key_rate(rows, key, value, denominator_rows=None):
    denom = len(denominator_rows if denominator_rows is not None else rows)
    count = sum(1 for row in rows if row.get(key) == value)
    lo, hi = wilson_interval(count, denom)
    return {"count": count, "denominator": denom, "pct": count / denom if denom else 0.0, "ci_low": lo, "ci_high": hi}


def build_summary(rows, sampling_args):
    flips = [r for r in rows if r["flip_class"] not in {"stable_harm", "missing_rerun"}]
    flip_count = len(flips)
    small_flips = [r for r in flips if r["repo_bucket"] == "small"]
    flags = {
        "baseline_flipped_ge_40pct_of_flips": key_rate(flips, "flip_class", "baseline_flipped"),
        "memory_flipped_ge_40pct_of_flips": key_rate(flips, "flip_class", "memory_flipped"),
        "harm_to_rescue_ge_10pct_of_all_cases": key_rate(rows, "flip_class", "harm_to_rescue"),
        "small_bucket_flips_gt_60pct_of_flips": {
            "count": len(small_flips),
            "denominator": flip_count,
            "pct": len(small_flips) / flip_count if flip_count else 0.0,
            "ci_low": wilson_interval(len(small_flips), flip_count)[0],
            "ci_high": wilson_interval(len(small_flips), flip_count)[1],
        },
    }
    flags["baseline_flipped_ge_40pct_of_flips"]["triggered"] = flags["baseline_flipped_ge_40pct_of_flips"]["pct"] >= 0.40
    flags["memory_flipped_ge_40pct_of_flips"]["triggered"] = flags["memory_flipped_ge_40pct_of_flips"]["pct"] >= 0.40
    flags["harm_to_rescue_ge_10pct_of_all_cases"]["triggered"] = flags["harm_to_rescue_ge_10pct_of_all_cases"]["pct"] >= 0.10
    flags["small_bucket_flips_gt_60pct_of_flips"]["triggered"] = flags["small_bucket_flips_gt_60pct_of_flips"]["pct"] > 0.60

    crosstab = defaultdict(Counter)
    per_repo = defaultdict(Counter)
    for row in rows:
        crosstab[row["repo_bucket"]][row["flip_class"]] += 1
        per_repo[row["repo"]][row["flip_class"]] += 1
    return {
        "n_cases": len(rows),
        "n_flips_excluding_stable_and_missing": flip_count,
        "class_counts": table_counts(rows, "flip_class"),
        "flip_only_class_counts": table_counts(flips, "flip_class") if flips else [],
        "repo_bucket_crosstab": {bucket: dict(counter) for bucket, counter in sorted(crosstab.items())},
        "per_repo_counts": {repo: dict(counter) for repo, counter in sorted(per_repo.items())},
        "flags": flags,
        "sampling_args": sampling_args,
        "caveat": "All percentages have wide CIs because the flip denominator is small.",
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{100 * value:.1f}%"


def write_markdown_report(path, summary):
    lines = [
        "# Check 9 Flip Decomposition",
        "",
        f"- cases: {summary['n_cases']}",
        f"- flips excluding stable/missing: {summary['n_flips_excluding_stable_and_missing']}",
        f"- caveat: {summary['caveat']}",
        "",
        "## Flip Classes",
        "",
        "| class | count | pct |",
        "|---|---:|---:|",
    ]
    for row in summary["class_counts"]:
        lines.append(f"| {row['name']} | {row['count']} | {fmt_pct(row['pct'])} |")

    lines.extend(["", "## Flip-Only Classes", "", "| class | count | pct |", "|---|---:|---:|"])
    for row in summary["flip_only_class_counts"]:
        lines.append(f"| {row['name']} | {row['count']} | {fmt_pct(row['pct'])} |")

    lines.extend(["", "## Flip Class x Repo Bucket", ""])
    classes = sorted({item["name"] for item in summary["class_counts"]})
    lines.append("| repo_bucket | " + " | ".join(classes) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(classes)) + "|")
    for bucket, counts in summary["repo_bucket_crosstab"].items():
        lines.append("| " + bucket + " | " + " | ".join(str(counts.get(cls, 0)) for cls in classes) + " |")

    lines.extend(["", "## Per-Repo Counts", ""])
    lines.append("| repo | " + " | ".join(classes) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(classes)) + "|")
    for repo, counts in summary["per_repo_counts"].items():
        lines.append("| " + str(repo) + " | " + " | ".join(str(counts.get(cls, 0)) for cls in classes) + " |")

    lines.extend(["", "## Interpretation Flags", ""])
    lines.append("| flag | triggered | count/denom | pct | Wilson 95% CI |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, value in summary["flags"].items():
        lines.append(
            f"| {name} | {value['triggered']} | {value['count']}/{value['denominator']} | "
            f"{fmt_pct(value['pct'])} | [{fmt_pct(value['ci_low'])}, {fmt_pct(value['ci_high'])}] |"
        )

    lines.extend(["", "## Sampling Parameter Parity", ""])
    lines.append("| run | available | model | num_samples | ranking_method | use_repomem | max_iteration_num | max_attempt_num |")
    lines.append("|---|---:|---|---:|---|---:|---:|---:|")
    for row in summary["sampling_args"]:
        lines.append(
            f"| {row['label']} | {row['available']} | {row.get('model')} | {row.get('num_samples')} | "
            f"{row.get('ranking_method')} | {row.get('use_repomem')} | {row.get('max_iteration_num')} | {row.get('max_attempt_num')} |"
        )
    lines.append("")
    lines.append(
        "If original full500 args are unavailable or differ from rerun args, interpret the stability rate as a diagnostic rather than a controlled repeat."
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Decompose direct-harm label flips for Check 9.")
    parser.add_argument("--selection_file", default=DEFAULT_SELECTION_FILE)
    parser.add_argument("--baseline_rerun_loc_file", default=DEFAULT_BASELINE_RERUN)
    parser.add_argument("--memory_rerun_loc_file", default=DEFAULT_MEMORY_RERUN)
    parser.add_argument("--check5_file", default=DEFAULT_CHECK5_FILE)
    parser.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--orig_baseline_args", default=DEFAULT_ORIG_BASELINE_ARGS)
    parser.add_argument("--orig_memory_args", default=DEFAULT_ORIG_MEMORY_ARGS)
    parser.add_argument("--baseline_rerun_args", default="results/check9_label_stability_gpt4o_baseline_harm30/args.json")
    parser.add_argument("--memory_rerun_args", default="results/check9_label_stability_gpt4o_episodic_harm30/args.json")
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    selection = load_jsonl(args.selection_file)
    baseline_preds = load_loc_predictions(args.baseline_rerun_loc_file)
    memory_preds = load_loc_predictions(args.memory_rerun_loc_file)
    gold = load_gold_files(args.check5_file)
    commit_counts = load_commit_counts(args.index_dir)

    rows = build_rows(selection, baseline_preds, memory_preds, gold, commit_counts, k=args.k)
    sampling_args = [
        summarize_args("orig_baseline", read_args(args.orig_baseline_args)),
        summarize_args("orig_memory", read_args(args.orig_memory_args)),
        summarize_args("rerun_baseline", read_args(args.baseline_rerun_args)),
        summarize_args("rerun_memory", read_args(args.memory_rerun_args)),
    ]
    summary = build_summary(rows, sampling_args)

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "flip_decomposition.csv"), rows)
    write_jsonl(os.path.join(args.output_dir, "flip_decomposition.jsonl"), rows)
    write_json(os.path.join(args.output_dir, "flip_summary.json"), summary)
    write_markdown_report(os.path.join(args.output_dir, "flip_summary.md"), summary)

    print(json.dumps({"output_dir": args.output_dir, "n_cases": len(rows), "summary": summary["class_counts"]}, indent=2))


if __name__ == "__main__":
    main()
