import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.analyze_memory_fragmentation import (
    DEFAULT_INDEX_DIRS,
    DEFAULT_TRAJ_FILES,
    enrich_commits_from_index,
    extract_search_commits_from_messages,
    load_commit_lookup,
    load_jsonl,
)


DEFAULT_CHECK5 = "results/full500_azure_gpt4o_repaired_compare/check5_retrieval_quality/retrieval_quality_per_case.jsonl"
DEFAULT_OUTPUT_DIR = "results/full500_azure_gpt4o_repaired_compare/check9_gate"
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PATH_RE = re.compile(r"[\w./-]+\.(?:py|go|js|ts|tsx|jsx|java|c|cc|cpp|h|hpp|rs|rb|php|scala|sh|yml|yaml|toml|ini|cfg|rst|md)")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
TRACE_RE = re.compile(r"Traceback|File\s+\"[^\"]+\"|^\s*File\s+", re.MULTILINE)
LOCAL_FEATURE_COLUMNS = [
    "bm25_top1_score",
    "bm25_top1_over_top5_mean",
    "bm25_top20_std",
    "top20_unique_files_count",
    "top20_unique_dirs_count",
    "issue_char_len",
    "issue_code_ratio",
    "mentioned_filename_count",
    "mentioned_symbol_count",
    "repo_commit_pool_size",
]


@dataclass
class LogisticModel:
    coef: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        x = (x - self.mean) / self.scale
        logits = np.column_stack([np.ones(len(x)), x]) @ self.coef
        return sigmoid(logits)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_path(path):
    path = str(path).strip().strip("`'\" ,:;")
    for prefix in ("./", "a/", "b/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path.lstrip("/")


def top_level_dir(path):
    path = normalize_path(path)
    if "/" not in path:
        return "."
    return path.split("/", 1)[0]


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def balanced_class_weights(y):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    weights = {}
    for label in (0, 1):
        weights[label] = len(y) / (2 * counts[label]) if counts[label] else 0.0
    return weights


def standardize_train(x):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def fit_logistic(x, y, c=1.0, feature_names=None, max_iter=1000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes to fit logistic regression")

    mean, scale = standardize_train(x)
    xs = (x - mean) / scale
    design = np.column_stack([np.ones(len(xs)), xs])
    weights_map = balanced_class_weights(y)
    sample_weights = np.array([weights_map[int(label)] for label in y], dtype=float)
    lambda_l2 = 1.0 / max(float(c), 1e-12)

    def objective(beta):
        logits = design @ beta
        loss = np.logaddexp(0, logits) - y * logits
        reg = 0.5 * lambda_l2 * np.sum(beta[1:] ** 2)
        return float(np.sum(sample_weights * loss) / len(y) + reg / len(y))

    def gradient(beta):
        prob = sigmoid(design @ beta)
        grad = design.T @ (sample_weights * (prob - y)) / len(y)
        grad[1:] += lambda_l2 * beta[1:] / len(y)
        return grad

    result = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"Logistic optimization failed: {result.message}")
    return LogisticModel(
        coef=result.x,
        mean=mean,
        scale=scale,
        feature_names=list(feature_names or [f"x{i}" for i in range(x.shape[1])]),
    )


def auc_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and y_score[order[j]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return float(np.mean((y_score - y_true) ** 2))


def compute_topk_coverage(commits, gold_files, top_k=5):
    gold = {normalize_path(p) for p in gold_files or []}
    if not gold:
        return 0.0
    retrieved = set()
    for commit in list(commits or [])[:top_k]:
        retrieved.update(normalize_path(p) for p in commit.get("changed_files") or [])
    return len(retrieved & gold) / len(gold)


def complete_file_acc(predicted_files, gold_files, k):
    gold = {normalize_path(p) for p in gold_files or []}
    if not gold:
        return 0
    pred = {normalize_path(p) for p in list(predicted_files or [])[:k]}
    return int(gold.issubset(pred))


def issue_code_ratio(issue_text):
    text = issue_text or ""
    if not text:
        return 0.0
    code_chars = sum(len(match.group(0)) for match in CODE_FENCE_RE.finditer(text))
    trace_chars = sum(len(match.group(0)) for match in TRACE_RE.finditer(text))
    path_chars = sum(len(match.group(0)) for match in PATH_RE.finditer(text))
    return min(1.0, (code_chars + trace_chars + path_chars) / max(1, len(text)))


def extract_case_features(row):
    commits = list(row.get("top_commits") or [])
    scores = [float(c.get("score") or 0.0) for c in commits]
    top5_scores = scores[:5]
    top1 = scores[0] if scores else 0.0
    top5_mean = float(np.mean(top5_scores)) if top5_scores else 0.0
    files = [normalize_path(f) for c in commits[:20] for f in c.get("changed_files") or []]
    dirs = [top_level_dir(f) for f in files]
    issue_text = row.get("issue_text") or ""
    filenames = PATH_RE.findall(issue_text)
    symbols = re.findall(r"`([A-Za-z_][A-Za-z0-9_\.]*)`", issue_text)

    return {
        "bm25_top1_score": top1,
        "bm25_top1_over_top5_mean": top1 / top5_mean if top5_mean > 1e-12 else 0.0,
        "bm25_top20_std": float(np.std(scores[:20])) if scores else 0.0,
        "top20_unique_files_count": len(set(files)),
        "top20_unique_dirs_count": len(set(dirs)),
        "issue_char_len": len(issue_text),
        "issue_code_ratio": issue_code_ratio(issue_text),
        "mentioned_filename_count": len(set(filenames)),
        "mentioned_symbol_count": len(set(symbols)),
        "repo_commit_pool_size": int(row.get("repo_commit_pool_size") or 0),
    }


def load_issue_texts_from_trajs(traj_files):
    texts = {}
    for path in traj_files:
        if not os.path.exists(path):
            continue
        for row in load_jsonl(path):
            iid = row.get("instance_id")
            text = row.get("meta_data", {}).get("problem_statement")
            if iid and text and iid not in texts:
                texts[iid] = text
    return texts


def load_top_commits(traj_files, index_dirs, max_commits=20):
    commit_lookup = load_commit_lookup(index_dirs)
    rows = {}
    for path in traj_files:
        if not os.path.exists(path):
            continue
        for row in load_jsonl(path):
            iid = row.get("instance_id")
            repo = row.get("meta_data", {}).get("repo", "")
            if not iid:
                continue
            best = rows.get(iid, {"commits": [], "search_calls": 0, "repo": repo})
            for traj in row.get("loc_trajs", {}).get("trajs", []):
                commits, search_calls = extract_search_commits_from_messages(
                    traj.get("messages", []),
                    max_commits=max_commits,
                )
                if len(commits) > len(best["commits"]):
                    best = {
                        "repo": repo,
                        "commits": enrich_commits_from_index(commits, repo, commit_lookup),
                        "search_calls": search_calls,
                    }
            rows[iid] = best
    return rows


def repo_commit_pool_sizes(index_dirs):
    sizes = {}
    for root in index_dirs:
        if not os.path.isdir(root):
            continue
        for repo_dir in sorted(os.listdir(root)):
            path = os.path.join(root, repo_dir, "commits.jsonl")
            if not os.path.exists(path):
                continue
            count = 0
            repo_name = repo_dir.replace("__", "/")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    count += 1
                    if count == 1:
                        try:
                            repo_name = json.loads(line).get("repo") or repo_name
                        except json.JSONDecodeError:
                            pass
            sizes[repo_name] = count
            sizes[repo_dir] = count
    return sizes


def build_feature_dataframe(args):
    check5 = pd.read_json(args.check5, lines=True)
    issue_texts = load_issue_texts_from_trajs(args.traj_files)
    top_commits = load_top_commits(args.traj_files, args.index_dirs, max_commits=args.max_commits)
    pool_sizes = repo_commit_pool_sizes(args.index_dirs)

    rows = []
    for _, row in check5.iterrows():
        case_id = row["case_id"]
        commit_data = top_commits.get(case_id, {})
        commits = commit_data.get("commits") or []
        repo_full = row.get("repo_full") or commit_data.get("repo")
        if not repo_full or not isinstance(repo_full, str):
            repo_full = row.get("repo")
        feature_input = {
            "issue_text": issue_texts.get(case_id, ""),
            "top_commits": commits,
            "repo_commit_pool_size": pool_sizes.get(repo_full, pool_sizes.get(str(repo_full).replace("/", "__"), 0)),
        }
        features = extract_case_features(feature_input)
        coverage_at5 = compute_topk_coverage(commits, row.get("gold_files") or [], top_k=args.coverage_k)
        out = {
            "case_id": case_id,
            "repo": row.get("repo"),
            "repo_full": repo_full,
            "gold_files_count": int(row.get("gold_files_count") or 0),
            "coverage_at5": coverage_at5,
            "coverage_zero_at5": int(coverage_at5 == 0.0),
            "top_commit_count": len(commits),
            "search_commit_calls": int(commit_data.get("search_calls") or 0),
            "baseline_at1": complete_file_acc(row.get("baseline_predicted_files"), row.get("gold_files"), 1),
            "memory_at1": complete_file_acc(row.get("memory_predicted_files"), row.get("gold_files"), 1),
            "baseline_at3": complete_file_acc(row.get("baseline_predicted_files"), row.get("gold_files"), 3),
            "memory_at3": complete_file_acc(row.get("memory_predicted_files"), row.get("gold_files"), 3),
            "baseline_at5": complete_file_acc(row.get("baseline_predicted_files"), row.get("gold_files"), 5),
            "memory_at5": complete_file_acc(row.get("memory_predicted_files"), row.get("gold_files"), 5),
        }
        out.update(features)
        rows.append(out)
    return pd.DataFrame(rows)


def feature_columns(df):
    return [c for c in LOCAL_FEATURE_COLUMNS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def make_loro_splits(df, repo_col):
    splits = []
    for repo in sorted(df[repo_col].dropna().unique()):
        test_idx = df.index[df[repo_col] == repo].to_numpy()
        train_idx = df.index[df[repo_col] != repo].to_numpy()
        splits.append((train_idx, test_idx, repo))
    return splits


def leave_one_repo_out_predictions(df, feature_cols, label_col, repo_col):
    base_cols = [
        "case_id",
        repo_col,
        label_col,
        "baseline_at1",
        "memory_at1",
        "baseline_at3",
        "memory_at3",
        "baseline_at5",
        "memory_at5",
    ]
    source = df.copy()
    if "case_id" not in source.columns:
        source["case_id"] = source.index.astype(str)
    for col in ["baseline_at1", "memory_at1", "baseline_at3", "memory_at3", "baseline_at5", "memory_at5"]:
        if col not in source.columns:
            source[col] = 0
    pred = source[base_cols].copy()
    pred["risk_probability"] = np.nan
    pred["heldout_repo"] = None
    pred["fold_auc"] = np.nan
    for train_idx, test_idx, repo in make_loro_splits(df, repo_col):
        train = df.loc[train_idx]
        test = df.loc[test_idx]
        if len(np.unique(train[label_col])) < 2:
            prob = np.full(len(test), train[label_col].mean())
        else:
            model = fit_logistic(
                train[feature_cols].to_numpy(dtype=float),
                train[label_col].to_numpy(dtype=int),
                feature_names=feature_cols,
            )
            prob = model.predict_proba(test[feature_cols].to_numpy(dtype=float))
        fold_auc = auc_score(test[label_col].to_numpy(dtype=int), prob)
        pred.loc[test_idx, "risk_probability"] = prob
        pred.loc[test_idx, "heldout_repo"] = repo
        pred.loc[test_idx, "fold_auc"] = fold_auc
    return pred


def random_cv_predictions(df, feature_cols, label_col, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.array(df.index)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    pred = np.full(len(df), np.nan)
    index_pos = {idx_value: i for i, idx_value in enumerate(df.index)}
    for fold in folds:
        train_idx = np.array([i for i in df.index if i not in set(fold)])
        train = df.loc[train_idx]
        test = df.loc[fold]
        model = fit_logistic(
            train[feature_cols].to_numpy(dtype=float),
            train[label_col].to_numpy(dtype=int),
            feature_names=feature_cols,
        )
        prob = model.predict_proba(test[feature_cols].to_numpy(dtype=float))
        for row_idx, value in zip(fold, prob):
            pred[index_pos[row_idx]] = value
    return pred


def precision_at_top_fraction(y, score, fraction=0.30):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n = max(1, int(math.ceil(len(y) * fraction)))
    idx = np.argsort(-score)[:n]
    return float(y[idx].mean())


def recall_at_threshold(y, score, threshold=0.5):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    positives = y == 1
    if positives.sum() == 0:
        return float("nan")
    return float((pred & positives).sum() / positives.sum())


def gate_accuracy(pred, threshold=0.5):
    use_baseline = pred["risk_probability"] >= threshold
    return gate_accuracy_from_mask(pred, use_baseline, prefix="gate")


def gate_accuracy_top_fraction(pred, fraction=0.30):
    n = max(1, int(math.ceil(len(pred) * fraction)))
    risky = set(pred.sort_values("risk_probability", ascending=False).head(n).index)
    use_baseline = pred.index.to_series().isin(risky)
    return gate_accuracy_from_mask(pred, use_baseline, prefix=f"gate_top{int(fraction * 100)}")


def gate_accuracy_from_mask(pred, use_baseline, prefix):
    rows = {}
    for k in (1, 3, 5):
        base = pred[f"baseline_at{k}"].astype(float)
        mem = pred[f"memory_at{k}"].astype(float)
        gated = np.where(use_baseline, base, mem)
        rows[f"{prefix}_acc_at{k}"] = float(np.mean(gated))
        rows[f"baseline_acc_at{k}"] = float(np.mean(base))
        rows[f"memory_acc_at{k}"] = float(np.mean(mem))
        rows[f"{prefix}_delta_vs_memory_at{k}"] = rows[f"{prefix}_acc_at{k}"] - rows[f"memory_acc_at{k}"]
        rows[f"{prefix}_delta_vs_baseline_at{k}"] = rows[f"{prefix}_acc_at{k}"] - rows[f"baseline_acc_at{k}"]
    return rows


def per_repo_auc_rows(pred):
    rows = []
    for repo, group in pred.groupby("repo"):
        rows.append(
            {
                "repo": repo,
                "N": int(len(group)),
                "positive_rate": float(group["coverage_zero_at5"].mean()),
                "auc": auc_score(group["coverage_zero_at5"], group["risk_probability"]),
                "mean_probability": float(group["risk_probability"].mean()),
            }
        )
    return rows


def fit_full_feature_importance(df, feature_cols, label_col):
    model = fit_logistic(
        df[feature_cols].to_numpy(dtype=float),
        df[label_col].to_numpy(dtype=int),
        feature_names=feature_cols,
    )
    rows = []
    for name, coef in zip(["Intercept"] + feature_cols, model.coef):
        rows.append(
            {
                "feature": name,
                "coef_on_standardized_feature": float(coef),
                "expected_sign": {
                    "bm25_top1_score": "negative",
                    "bm25_top1_over_top5_mean": "negative",
                    "top20_unique_files_count": "positive",
                    "top20_unique_dirs_count": "positive",
                }.get(name, ""),
            }
        )
    return rows


def ablation_rows(df, feature_cols, label_col, repo_col):
    groups = {
        "without_retrieval_scores": ["bm25_top1_score", "bm25_top1_over_top5_mean", "bm25_top20_std"],
        "without_retrieval_spread": ["top20_unique_files_count", "top20_unique_dirs_count"],
        "without_issue_features": [
            "issue_char_len",
            "issue_code_ratio",
            "mentioned_filename_count",
            "mentioned_symbol_count",
        ],
        "without_repo_pool": ["repo_commit_pool_size"],
    }
    full_pred = leave_one_repo_out_predictions(df, feature_cols, label_col, repo_col)
    full_auc = auc_score(full_pred[label_col], full_pred["risk_probability"])
    rows = [{"ablation": "full", "feature_count": len(feature_cols), "auc": full_auc, "auc_drop": 0.0}]
    for name, removed in groups.items():
        cols = [c for c in feature_cols if c not in removed]
        pred = leave_one_repo_out_predictions(df, cols, label_col, repo_col)
        auc = auc_score(pred[label_col], pred["risk_probability"])
        rows.append(
            {
                "ablation": name,
                "feature_count": len(cols),
                "auc": auc,
                "auc_drop": full_auc - auc,
            }
        )
    return rows


def permutation_test(df, feature_cols, label_col, repo_col, n_perm=100, seed=42):
    rng = np.random.default_rng(seed)
    observed = auc_score(
        df[label_col],
        leave_one_repo_out_predictions(df, feature_cols, label_col, repo_col)["risk_probability"],
    )
    aucs = []
    shuffled = df.copy()
    for _ in range(n_perm):
        shuffled[label_col] = rng.permutation(df[label_col].to_numpy())
        pred = leave_one_repo_out_predictions(shuffled, feature_cols, label_col, repo_col)
        aucs.append(auc_score(shuffled[label_col], pred["risk_probability"]))
    valid = np.array([x for x in aucs if not math.isnan(x)])
    p_value = float((np.sum(valid >= observed) + 1) / (len(valid) + 1)) if len(valid) else float("nan")
    return {
        "observed_auc": float(observed),
        "n_permutations": int(len(valid)),
        "mean_permuted_auc": float(np.mean(valid)) if len(valid) else float("nan"),
        "std_permuted_auc": float(np.std(valid)) if len(valid) else float("nan"),
        "p_value": p_value,
        "permuted_auc": [float(x) for x in valid],
    }


def report_markdown(summary, feature_rows, ablation, per_repo):
    sign_lines = []
    by_feature = {row["feature"]: row for row in feature_rows}
    for feature in ["bm25_top1_score", "bm25_top1_over_top5_mean", "top20_unique_files_count", "top20_unique_dirs_count"]:
        row = by_feature.get(feature, {})
        sign_lines.append(
            f"- `{feature}`: coef={row.get('coef_on_standardized_feature', float('nan')):.4f}, expected={row.get('expected_sign', '')}"
        )

    weak_repos = [
        row for row in per_repo
        if not math.isnan(row["auc"]) and row["auc"] < 0.55
    ]
    verdict = "strong" if summary["loro_auc"] > 0.75 else "weak" if summary["loro_auc"] >= 0.65 else "failed"
    return "\n".join(
        [
            "# Check 9: Gate Predictability",
            "",
            "## Data",
            f"- analyzed cases: {summary['n_cases']}",
            f"- coverage_zero positives: {summary['positive_count']} ({summary['positive_rate']:.4f})",
            f"- feature count: {summary['feature_count']}",
            "",
            "## Main Metrics",
            f"- LORO AUC: {summary['loro_auc']:.4f}",
            f"- random 5-fold AUC: {summary['random_cv_auc']:.4f}",
            f"- Precision@top30% risky: {summary['precision_top30_risky']:.4f}",
            f"- Recall@threshold 0.5: {summary['recall_threshold_0.5']:.4f}",
            f"- Brier score: {summary['brier']:.4f}",
            f"- permutation p-value: {summary['permutation_p_value']:.4f}",
            "",
            "## Gate Simulation",
            f"- Baseline Acc@1: {summary['baseline_acc_at1']:.4f}",
            f"- Memory Acc@1: {summary['memory_acc_at1']:.4f}",
            f"- Gate Acc@1 threshold 0.5: {summary['gate_acc_at1']:.4f}",
            f"- Gate Acc@1 top30 risky: {summary['gate_top30_acc_at1']:.4f}",
            f"- Baseline Acc@3: {summary['baseline_acc_at3']:.4f}",
            f"- Memory Acc@3: {summary['memory_acc_at3']:.4f}",
            f"- Gate Acc@3 threshold 0.5: {summary['gate_acc_at3']:.4f}",
            f"- Gate Acc@3 top30 risky: {summary['gate_top30_acc_at3']:.4f}",
            f"- Baseline Acc@5: {summary['baseline_acc_at5']:.4f}",
            f"- Memory Acc@5: {summary['memory_acc_at5']:.4f}",
            f"- Gate Acc@5 threshold 0.5: {summary['gate_acc_at5']:.4f}",
            f"- Gate Acc@5 top30 risky: {summary['gate_top30_acc_at5']:.4f}",
            "",
            "## Coefficient Sanity",
            *sign_lines,
            "",
            "## Ablation",
            *[
                f"- {row['ablation']}: AUC={row['auc']:.4f}, drop={row['auc_drop']:.4f}"
                for row in ablation
            ],
            "",
            "## Per-Repo Check",
            f"- repos with defined AUC < 0.55: {len(weak_repos)}",
            "",
            "## Verdict",
            f"- {verdict}",
        ]
    )


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = build_feature_dataframe(args)
    feature_cols = feature_columns(df)
    label_col = "coverage_zero_at5"
    repo_col = "repo"
    df = df.dropna(subset=feature_cols + [label_col, repo_col]).copy()

    pred = leave_one_repo_out_predictions(df, feature_cols, label_col, repo_col)
    random_prob = random_cv_predictions(df, feature_cols, label_col, n_folds=args.random_folds, seed=args.seed)
    loro_auc = auc_score(pred[label_col], pred["risk_probability"])
    random_auc = auc_score(df[label_col], random_prob)
    gate = gate_accuracy(pred, threshold=args.threshold)
    gate_top30 = gate_accuracy_top_fraction(pred, 0.30)
    per_repo = per_repo_auc_rows(pred)
    features = fit_full_feature_importance(df, feature_cols, label_col)
    ablation = ablation_rows(df, feature_cols, label_col, repo_col)
    perm = permutation_test(df, feature_cols, label_col, repo_col, n_perm=args.permutations, seed=args.seed)

    summary = {
        "n_cases": int(len(df)),
        "positive_count": int(df[label_col].sum()),
        "positive_rate": float(df[label_col].mean()),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "loro_auc": float(loro_auc),
        "random_cv_auc": float(random_auc),
        "precision_top30_risky": precision_at_top_fraction(pred[label_col], pred["risk_probability"], 0.30),
        "recall_threshold_0.5": recall_at_threshold(pred[label_col], pred["risk_probability"], args.threshold),
        "brier": brier_score(pred[label_col], pred["risk_probability"]),
        "permutation_p_value": perm["p_value"],
        **gate,
        **gate_top30,
    }

    write_jsonl(os.path.join(args.output_dir, "check9_features.jsonl"), df.to_dict("records"))
    write_jsonl(os.path.join(args.output_dir, "predictions_full500.jsonl"), pred.to_dict("records"))
    write_json(os.path.join(args.output_dir, "auc_summary.json"), summary)
    write_json(os.path.join(args.output_dir, "permutation_test.json"), perm)
    write_csv(os.path.join(args.output_dir, "per_repo_auc.csv"), per_repo)
    write_csv(os.path.join(args.output_dir, "feature_importance.csv"), features)
    write_csv(os.path.join(args.output_dir, "ablation.csv"), ablation)
    with open(os.path.join(args.output_dir, "check9_report.md"), "w", encoding="utf-8") as f:
        f.write(report_markdown(summary, features, ablation, per_repo) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check5", default=DEFAULT_CHECK5)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--traj_files", nargs="+", default=DEFAULT_TRAJ_FILES)
    parser.add_argument("--index_dirs", nargs="+", default=DEFAULT_INDEX_DIRS)
    parser.add_argument("--coverage_k", type=int, default=5)
    parser.add_argument("--max_commits", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random_folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
