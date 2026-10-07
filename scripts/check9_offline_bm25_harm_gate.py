import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from scripts.check9_direct_harm_gate import (
    average_precision_score,
    bootstrap_loro_auc,
    coverage_harm_stats,
    gate_percentile_curve,
    oracle_and_random_gate,
    permutation_auc_test,
    summarize_predictions,
)
from scripts.check9_gate_predictability import (
    PATH_RE,
    auc_score,
    complete_file_acc,
    fit_full_feature_importance,
    issue_code_ratio,
    leave_one_repo_out_predictions,
    normalize_path,
    top_level_dir,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_CHECK5_FILE = "results/full500_azure_gpt4o_repaired_compare/check5_retrieval_quality/retrieval_quality_per_case.jsonl"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/check9_gate/offline_bm25_harm"
DEFAULT_INDEX_DIRS = [
    "repomem_index/verified_200_episodic_with_issues",
    "repomem_index/verified_remaining_247_episodic_with_issues_v2",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
OFFLINE_FEATURE_COLUMNS = [
    "offline_bm25_top1_score",
    "offline_bm25_top1_over_top5_mean",
    "offline_bm25_top20_std",
    "offline_top20_unique_files_log",
    "offline_top20_unique_dirs_log",
    "offline_subject_selfsim_mean",
    "issue_char_len",
    "issue_code_ratio",
    "mentioned_filename_count",
    "mentioned_symbol_count",
]


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_commits_by_repo(index_dirs):
    by_repo = {}
    for root in index_dirs:
        if not os.path.isdir(root):
            continue
        for repo_dir in sorted(os.listdir(root)):
            path = os.path.join(root, repo_dir, "commits.jsonl")
            if not os.path.exists(path):
                continue
            rows = load_jsonl(path)
            if not rows:
                continue
            repo = rows[0].get("repo") or repo_dir.replace("__", "/")
            # Keep the first configured index for duplicate repos so callers can
            # explicitly choose precedence by index_dirs order.
            if repo not in by_repo:
                by_repo[repo] = rows
                by_repo[repo_dir] = rows
    return by_repo


def bm25_scores(query, docs, k1=1.5, b=0.75):
    tokenized = [tokenize(doc) for doc in docs]
    query_terms = tokenize(query)
    n_docs = len(tokenized)
    if n_docs == 0:
        return []
    lengths = np.array([len(doc) for doc in tokenized], dtype=float)
    avgdl = float(lengths.mean()) if len(lengths) else 0.0
    doc_freq = Counter()
    term_freqs = []
    for doc in tokenized:
        tf = Counter(doc)
        term_freqs.append(tf)
        doc_freq.update(tf.keys())
    idf = {term: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0) for term, df in doc_freq.items()}
    scores = np.zeros(n_docs, dtype=float)
    for term in query_terms:
        term_idf = idf.get(term)
        if term_idf is None:
            continue
        for i, tf in enumerate(term_freqs):
            freq = tf.get(term, 0)
            if not freq:
                continue
            denom = freq + k1 * (1 - b + b * lengths[i] / max(avgdl, 1e-12))
            scores[i] += term_idf * (freq * (k1 + 1)) / denom
    return scores.tolist()


class BM25CommitIndex:
    def __init__(self, commits, k1=1.5, b=0.75):
        self.commits = list(commits or [])
        self.k1 = k1
        self.b = b
        docs = [" ".join([str(c.get("subject") or ""), str(c.get("message") or "")]) for c in self.commits]
        self.tokenized = [tokenize(doc) for doc in docs]
        self.n_docs = len(self.tokenized)
        self.lengths = np.array([len(doc) for doc in self.tokenized], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 0.0
        self.term_freqs = []
        postings = defaultdict(list)
        doc_freq = Counter()
        for idx, doc in enumerate(self.tokenized):
            tf = Counter(doc)
            self.term_freqs.append(tf)
            doc_freq.update(tf.keys())
            for term, freq in tf.items():
                postings[term].append((idx, freq))
        self.postings = dict(postings)
        self.idf = {
            term: math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

    def search(self, query, top_k=20):
        if self.n_docs == 0:
            return []
        scores = np.zeros(self.n_docs, dtype=float)
        for term in tokenize(query):
            term_idf = self.idf.get(term)
            if term_idf is None:
                continue
            for i, freq in self.postings.get(term, []):
                denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avgdl, 1e-12))
                scores[i] += term_idf * (freq * (self.k1 + 1)) / denom
        if top_k <= 0:
            return []
        order = np.argsort(-scores)[:top_k]
        ranked = []
        for idx in order:
            item = dict(self.commits[int(idx)])
            item["score"] = float(scores[int(idx)])
            ranked.append(item)
        return ranked


def bm25_search(query, commits, top_k=20):
    return BM25CommitIndex(commits).search(query, top_k=top_k)


def subject_self_similarity(commits):
    subjects = [set(tokenize(c.get("subject") or c.get("message") or "")) for c in commits[:20]]
    if len(subjects) < 2:
        return 0.0
    vals = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            union = subjects[i] | subjects[j]
            vals.append(len(subjects[i] & subjects[j]) / len(union) if union else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def offline_commit_features(issue_text, commits):
    scores = [float(c.get("score") or 0.0) for c in commits[:20]]
    top1 = scores[0] if scores else 0.0
    top5 = scores[:5]
    top5_mean = float(np.mean(top5)) if top5 else 0.0
    files = [normalize_path(f) for c in commits[:20] for f in c.get("changed_files") or []]
    dirs = [top_level_dir(f) for f in files]
    filenames = PATH_RE.findall(issue_text or "")
    symbols = re.findall(r"`([A-Za-z_][A-Za-z0-9_\.]*)`", issue_text or "")
    return {
        "offline_bm25_top1_score": float(top1),
        "offline_bm25_top1_over_top5_mean": float(top1 / top5_mean) if top5_mean > 1e-12 else 0.0,
        "offline_bm25_top20_std": float(np.std(scores)) if scores else 0.0,
        "offline_top20_unique_files_log": float(math.log1p(len(set(files)))),
        "offline_top20_unique_dirs_log": float(math.log1p(len(set(dirs)))),
        "offline_subject_selfsim_mean": subject_self_similarity(commits),
        "issue_char_len": int(len(issue_text or "")),
        "issue_code_ratio": float(issue_code_ratio(issue_text or "")),
        "mentioned_filename_count": int(len(set(filenames))),
        "mentioned_symbol_count": int(len(set(symbols))),
    }


def compute_topk_coverage(commits, gold_files, top_k):
    gold = {normalize_path(p) for p in gold_files or []}
    if not gold:
        return 0.0
    retrieved = set()
    for commit in commits[:top_k]:
        retrieved.update(normalize_path(p) for p in commit.get("changed_files") or [])
    return len(retrieved & gold) / len(gold)


def make_harm_rescue_labels(df, ks=(1, 3, 5)):
    out = df.copy()
    for k in ks:
        base = out[f"baseline_at{k}"].astype(int)
        mem = out[f"memory_at{k}"].astype(int)
        out[f"harm_at{k}"] = ((base == 1) & (mem == 0)).astype(int)
        out[f"rescue_at{k}"] = ((base == 0) & (mem == 1)).astype(int)
    return out


def build_offline_feature_dataframe(check5, commits_by_repo, max_commits=20, coverage_k=5):
    rows = []
    index_cache = {}
    for _, row in check5.iterrows():
        repo_full = row.get("repo_full")
        if not repo_full or not isinstance(repo_full, str):
            repo_full = row.get("repo")
        commits = commits_by_repo.get(repo_full) or commits_by_repo.get(str(repo_full).replace("/", "__")) or []
        issue_text = row.get("problem_statement") or row.get("issue_text") or ""
        if not issue_text:
            # The repaired loc outputs retain the original problem statement in
            # meta_data, but Check 5 may not. An empty query keeps the feature
            # extraction deterministic and exposes missing issue text clearly.
            issue_text = ""
        if repo_full not in index_cache:
            index_cache[repo_full] = BM25CommitIndex(commits)
        top_commits = index_cache[repo_full].search(issue_text, top_k=max_commits)
        gold_files = row.get("gold_files") or []
        out = {
            "case_id": row["case_id"],
            "repo": row.get("repo"),
            "repo_full": repo_full,
            "gold_files_count": int(row.get("gold_files_count") or len(gold_files) or 0),
            "offline_top_commit_count": int(len(top_commits)),
            "offline_retrieval_gold_coverage": compute_topk_coverage(top_commits, gold_files, top_k=max_commits),
            f"offline_coverage_zero_at{coverage_k}": int(compute_topk_coverage(top_commits, gold_files, top_k=coverage_k) == 0.0),
            "baseline_at1": complete_file_acc(row.get("baseline_predicted_files"), gold_files, 1),
            "memory_at1": complete_file_acc(row.get("memory_predicted_files"), gold_files, 1),
            "baseline_at3": complete_file_acc(row.get("baseline_predicted_files"), gold_files, 3),
            "memory_at3": complete_file_acc(row.get("memory_predicted_files"), gold_files, 3),
            "baseline_at5": complete_file_acc(row.get("baseline_predicted_files"), gold_files, 5),
            "memory_at5": complete_file_acc(row.get("memory_predicted_files"), gold_files, 5),
            "top_commit_shas": [c.get("short_sha") or c.get("sha") for c in top_commits],
            "top_commit_changed_files": [c.get("changed_files") or [] for c in top_commits],
        }
        out.update(offline_commit_features(issue_text, top_commits))
        rows.append(out)
    return make_harm_rescue_labels(pd.DataFrame(rows))


def feature_columns(df):
    return [c for c in OFFLINE_FEATURE_COLUMNS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def _prepare(df, feature_cols, label_col):
    work = df.dropna(subset=feature_cols + [label_col, "repo"]).copy().reset_index(drop=True)
    work[label_col] = work[label_col].astype(int)
    return work


def fit_label(df, feature_cols, label_col):
    work = _prepare(df, feature_cols, label_col)
    pred = leave_one_repo_out_predictions(work, feature_cols, label_col, "repo")
    summary = summarize_predictions(pred, label_col)
    features = fit_full_feature_importance(work, feature_cols, label_col) if work[label_col].nunique() == 2 else []
    return work, pred, summary, features


def run_offline_bm25(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.features_file and os.path.exists(args.features_file):
        df = pd.read_json(args.features_file, lines=True)
    else:
        check5 = pd.read_json(args.check5_file, lines=True)
        commits = load_commits_by_repo(args.index_dirs)
        df = build_offline_feature_dataframe(check5, commits, max_commits=args.max_commits, coverage_k=args.coverage_k)
        write_jsonl(args.features_output, df.to_dict("records"))
    cols = feature_columns(df)
    if "repo_commit_pool_size" in cols:
        raise RuntimeError("repo_commit_pool_size must not be used by the offline BM25 gate")
    if not cols:
        raise RuntimeError("No usable offline BM25 feature columns")

    label_summaries = []
    all_predictions = []
    feature_rows = {}
    for label in ["harm_at5", "harm_at3", "harm_at1", "rescue_at5"]:
        _, pred, summary, rows = fit_label(df, cols, label)
        pred["label_name"] = label
        all_predictions.append(pred)
        label_summaries.append(summary)
        feature_rows[label] = rows

    primary_work, primary_pred, _, harm_features = fit_label(df, cols, "harm_at5")
    bootstrap = bootstrap_loro_auc(primary_work, cols, "harm_at5", "repo", n_boot=args.n_bootstrap, seed=args.seed)
    permutation = permutation_auc_test(primary_work, cols, "harm_at5", "repo", n_perm=args.n_permutation, seed=args.seed + 1)
    gate_curve = gate_percentile_curve(primary_pred, "harm_at5", args.gate_fractions)
    oracle_random = oracle_and_random_gate(
        primary_pred,
        "harm_at5",
        random_rate=args.random_gate_rate,
        n_random=args.n_random_gate,
        seed=args.seed + 2,
    )
    coverage_stats = coverage_harm_stats(primary_work.rename(columns={f"offline_coverage_zero_at{args.coverage_k}": "coverage_zero_at5"}), "harm_at5", "coverage_zero_at5")
    rescue_summary = next(row for row in label_summaries if row["label"] == "rescue_at5")

    summary = {
        "n_cases": int(len(primary_work)),
        "feature_count": int(len(cols)),
        "features": cols,
        "label": "harm_at5",
        "label_summaries": label_summaries,
        "bootstrap": bootstrap,
        "permutation": permutation,
        "gate_curve": gate_curve,
        "oracle_random": oracle_random,
        "coverage_vs_harm": coverage_stats,
        "rescue_summary": rescue_summary,
        "notes": [
            "Offline BM25 uses problem_statement/issue_text directly rather than agent trajectory SearchCommit observations.",
            "repo_commit_pool_size is intentionally excluded.",
            "Gate simulation uses leave-one-repo-out out-of-fold risk predictions.",
        ],
    }

    write_json(os.path.join(args.output_dir, "offline_bm25_harm_summary.json"), summary)
    write_jsonl(os.path.join(args.output_dir, "offline_bm25_predictions_by_label.jsonl"), pd.concat(all_predictions).to_dict("records"))
    write_csv(os.path.join(args.output_dir, "offline_bm25_label_summaries.csv"), label_summaries)
    write_csv(os.path.join(args.output_dir, "offline_bm25_harm_feature_importance.csv"), harm_features)
    write_csv(os.path.join(args.output_dir, "offline_bm25_rescue_feature_importance.csv"), feature_rows["rescue_at5"])
    write_csv(os.path.join(args.output_dir, "offline_bm25_gate_percentile_curve.csv"), gate_curve)
    write_csv(os.path.join(args.output_dir, "offline_bm25_oracle_random_baselines.csv"), oracle_random)
    write_json(os.path.join(args.output_dir, "offline_bm25_bootstrap_ci.json"), bootstrap)
    write_json(os.path.join(args.output_dir, "offline_bm25_permutation_test.json"), permutation)
    write_report(os.path.join(args.output_dir, "offline_bm25_harm_report.md"), summary, harm_features)
    return summary


def write_report(path, summary, harm_features):
    primary = next(row for row in summary["label_summaries"] if row["label"] == "harm_at5")
    lines = [
        "# Check 9b: Offline BM25 Direct Harm Gate",
        "",
        f"- analyzed cases: {summary['n_cases']}",
        f"- feature count: {summary['feature_count']}",
        "- primary label: `harm_at5 = baseline_at5 == 1 and memory_at5 == 0`",
        "- retrieval source: offline BM25 over episodic `commits.jsonl` using issue text.",
        "- excluded feature: `repo_commit_pool_size`.",
        "",
        "## Primary Classifier",
        f"- ROC-AUC: {primary['roc_auc']:.4f}",
        f"- PR-AUC: {primary['pr_auc']:.4f}",
        f"- base rate: {primary['base_rate']:.4f}",
        f"- bootstrap ROC-AUC 95% CI: [{summary['bootstrap']['auc_ci_low']:.4f}, {summary['bootstrap']['auc_ci_high']:.4f}]",
        f"- permutation p-value: {summary['permutation']['p_value']:.4f}",
        "",
        "## Gate Curve At Acc@5",
        "| risky fraction | gate Acc@5 | delta vs memory | delta vs baseline | harm precision | harm recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in summary["gate_curve"] if r["label"] == "harm_at5"]:
        lines.append(
            f"| {row['risk_fraction']:.2f} | {row['gate_acc_at5']:.4f} | {row['gate_delta_vs_memory_at5']:.4f} | "
            f"{row['gate_delta_vs_baseline_at5']:.4f} | {row['harm_precision']:.4f} | {row['harm_capture_rate']:.4f} |"
        )
    lines.extend(["", "## Harm Feature Coefficients"])
    for row in harm_features:
        lines.append(f"- `{row['feature']}`: {row['coef_on_standardized_feature']:.4f}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check5_file", default=DEFAULT_CHECK5_FILE)
    parser.add_argument("--index_dirs", nargs="+", default=DEFAULT_INDEX_DIRS)
    parser.add_argument("--features_file")
    parser.add_argument(
        "--features_output",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "offline_bm25_features.jsonl"),
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_commits", type=int, default=20)
    parser.add_argument("--coverage_k", type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--n_permutation", type=int, default=1000)
    parser.add_argument("--n_random_gate", type=int, default=100)
    parser.add_argument("--random_gate_rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate_fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.50])
    return parser.parse_args()


if __name__ == "__main__":
    result = run_offline_bm25(parse_args())
    primary = next(row for row in result["label_summaries"] if row["label"] == "harm_at5")
    print(json.dumps(primary, indent=2, ensure_ascii=False))
