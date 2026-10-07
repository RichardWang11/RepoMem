import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from scipy.stats import norm

from scripts.analyze_n2_memory_adoption import (
    extract_argument_features,
    files_from_patch,
    flatten_predicted_files,
    load_commit_lookup,
    lookup_commit,
    normalize_path,
    parse_examine_observation,
    parse_json_arguments,
    parse_search_observation,
)


DEFAULT_ARROW = (
    "/root/.cache/huggingface/datasets/princeton-nlp___swe-bench_verified/"
    "default/0.0.0/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/"
    "swe-bench_verified-test.arrow"
)
DEFAULT_OUTPUT = (
    "results/context_safe_m3a/replication_v4/analysis/early_memory_risk"
)
DEFAULT_INDEX_DIRS = [
    "repomem_index/verified_200_episodic_with_issues",
    "repomem_index/verified_remaining_247_episodic_with_issues_v2",
]
CODE_TOOLS = {
    "search_code_snippets",
    "get_entity_contents",
    "explore_tree_structure",
}
T0_FEATURES = [
    "bm25_top1_score",
    "bm25_relative_margin",
    "top5_file_consistency",
    "query_text_similarity",
    "inter_query_sha_jaccard",
    "module_conflict_rate",
    "retrieval_missing",
    "top2_missing",
    "multi_query_missing",
]
T1_RAW_FEATURES = T0_FEATURES + [
    "memory_code_file_overlap",
    "top1_file_overlap",
    "commits_examined_before_code",
    "rapid_unverified_top1_adoption",
    "code_evidence_missing",
]


def token_jaccard(left, right):
    left = {token.lower() for token in str(left or "").replace("_", " ").split()}
    right = {token.lower() for token in str(right or "").replace("_", " ").split()}
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right) if left and right else 0.0


def set_jaccard(left, right):
    left, right = set(left or []), set(right or [])
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right) if left and right else 0.0


def pairwise_mean(values, fn):
    pairs = [fn(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    return float(np.mean(pairs)) if pairs else np.nan


def complete_acc(row, gold_files):
    prediction = set(flatten_predicted_files(row.get("found_files") or [])[:5])
    return int(set(gold_files).issubset(prediction))


def trajectory_messages(row):
    trajectories = (row.get("loc_trajs") or {}).get("trajs") or []
    return trajectories[0].get("messages") or [] if trajectories else []


def call_and_observation_stream(messages):
    observations = {
        message.get("tool_call_id"): (index, message.get("content") or "")
        for index, message in enumerate(messages)
        if message.get("role") == "tool"
    }
    calls = []
    sequence = 0
    for message_index, message in enumerate(messages):
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            observation_index, observation = observations.get(tool_call.get("id"), (None, ""))
            calls.append(
                {
                    "name": function.get("name") or "",
                    "id": tool_call.get("id"),
                    "args_raw": function.get("arguments") or "{}",
                    "args": parse_json_arguments(function.get("arguments") or "{}"),
                    "message_index": message_index,
                    "observation_index": observation_index,
                    "observation": observation,
                    "sequence": sequence,
                }
            )
            sequence += 1
    return calls


def commit_files(commit_lookup, repo, sha, visible_files):
    commit = lookup_commit(commit_lookup, repo, sha)
    return set(commit["changed_files"] if commit else visible_files)


def extract_early_features(row, commit_lookup):
    repo = (row.get("meta_data") or {}).get("repo", "")
    calls = call_and_observation_stream(trajectory_messages(row))
    searches = [call for call in calls if call["name"] == "search_commit"]
    malformed = sum(call["args"] is None for call in calls)
    base = {
        "bm25_top1_score": np.nan,
        "bm25_relative_margin": np.nan,
        "top5_file_consistency": np.nan,
        "query_text_similarity": np.nan,
        "inter_query_sha_jaccard": np.nan,
        "module_conflict_rate": np.nan,
        "retrieval_missing": int(not searches),
        "top2_missing": 1,
        "multi_query_missing": 1,
        "memory_code_file_overlap": np.nan,
        "top1_file_overlap": np.nan,
        "commits_examined_before_code": 0,
        "rapid_unverified_top1_adoption": 0,
        "code_evidence_missing": 1,
        "search_commit_call_count": len(searches),
        "malformed_argument_count_at_checkpoint": malformed,
    }
    if not searches:
        return base

    search = searches[0]
    sections = parse_search_observation(search["observation"])
    if not sections:
        return base
    query_items = [section["items"][:5] for section in sections]
    first_items = query_items[0]
    if first_items:
        base["bm25_top1_score"] = first_items[0]["score"]
    if len(first_items) >= 2 and first_items[0]["score"] is not None and first_items[1]["score"] is not None:
        score1, score2 = first_items[0]["score"], first_items[1]["score"]
        base["bm25_relative_margin"] = (score1 - score2) / max(abs(score1), 1e-12)
        base["top2_missing"] = 0

    file_consistency = []
    query_sha_sets = []
    query_module_sets = []
    memory_files = set()
    for items in query_items:
        file_sets = []
        sha_set = set()
        modules = set()
        for item in items:
            files = commit_files(commit_lookup, repo, item["sha"], item["visible_files"])
            file_sets.append(files)
            memory_files.update(files)
            sha_set.add(item["sha"])
            modules.update(str(Path(path).parent).replace("\\", "/") for path in files if "/" in path)
        value = pairwise_mean(file_sets, set_jaccard)
        if not np.isnan(value):
            file_consistency.append(value)
        query_sha_sets.append(sha_set)
        query_module_sets.append(modules)
    base["top5_file_consistency"] = float(np.mean(file_consistency)) if file_consistency else np.nan

    queries = [section["query"] for section in sections]
    if len(queries) >= 2:
        base["multi_query_missing"] = 0
        base["query_text_similarity"] = pairwise_mean(queries, token_jaccard)
        base["inter_query_sha_jaccard"] = pairwise_mean(query_sha_sets, set_jaccard)
        conflicts = [
            int(bool(query_module_sets[i]) and bool(query_module_sets[j]) and not (query_module_sets[i] & query_module_sets[j]))
            for i in range(len(query_module_sets))
            for j in range(i + 1, len(query_module_sets))
        ]
        base["module_conflict_rate"] = float(np.mean(conflicts)) if conflicts else np.nan

    code_calls = [
        call
        for call in calls
        if call["name"] in CODE_TOOLS
        and call["message_index"] > (search["observation_index"] or search["message_index"])
    ]
    if not code_calls:
        return base
    code_call = code_calls[0]
    base["code_evidence_missing"] = 0
    arg_paths, _ = extract_argument_features(code_call["args"], code_call["args_raw"])
    observation_paths, _ = extract_argument_features({"observation": code_call["observation"]})
    code_files = set(arg_paths) | set(observation_paths)
    base["memory_code_file_overlap"] = set_jaccard(memory_files, code_files) if code_files else 0.0

    top1_sha = first_items[0]["sha"] if first_items else None
    top1_files = commit_files(commit_lookup, repo, top1_sha, first_items[0]["visible_files"]) if top1_sha else set()
    base["top1_file_overlap"] = int(bool(top1_files & code_files))

    examined = []
    for call in calls:
        if call["name"] != "examine_commit" or call["message_index"] >= code_call["message_index"]:
            continue
        examined.extend(block["sha"] for block in parse_examine_observation(call["observation"]))
    unique_examined = list(dict.fromkeys(examined))
    base["commits_examined_before_code"] = len(unique_examined)
    base["rapid_unverified_top1_adoption"] = int(
        bool(top1_sha)
        and unique_examined == [top1_sha]
        and bool(top1_files & code_files)
    )
    return base


def load_jsonl_map(run_dir, repaired=True):
    run_dir = Path(run_dir)
    repaired_path = run_dir / "loc_trajs.repaired.jsonl"
    path = repaired_path if repaired and repaired_path.exists() else run_dir / "loc_trajs.jsonl"
    return {row["instance_id"]: row for row in (json.loads(line) for line in path.open() if line.strip())}


def load_gold(dataset, split, arrow_path=None):
    rows = Dataset.from_file(arrow_path) if arrow_path else load_dataset(dataset, split=split)
    return {row["instance_id"]: files_from_patch(row["patch"]) for row in rows}


def add_pair_rows(records, source, repeat, baseline_rows, memory_rows, gold, commit_lookup):
    for case_id in sorted(memory_rows):
        memory = memory_rows[case_id]
        baseline = baseline_rows[case_id]
        memory_success = complete_acc(memory, gold[case_id])
        baseline_success = complete_acc(baseline, gold[case_id])
        records.append(
            {
                "case_id": case_id,
                "repo": (memory.get("meta_data") or {}).get("repo", ""),
                "source": source,
                "repeat": repeat,
                "baseline_success": baseline_success,
                "memory_success": memory_success,
                "paired_harm": int(baseline_success == 1 and memory_success == 0),
                "memory_failure": int(memory_success == 0),
                **extract_early_features(memory, commit_lookup),
            }
        )


def build_dataset(args, commit_lookup, gold):
    records = []
    for source, root in (
        ("v2", args.v2_root),
        ("v3", args.v3_root),
        ("v4", args.v4_root),
    ):
        for repeat in range(1, 5):
            add_pair_rows(
                records,
                source,
                repeat,
                load_jsonl_map(Path(root) / f"baseline_r{repeat}"),
                load_jsonl_map(Path(root) / f"memory_r{repeat}"),
                gold,
                commit_lookup,
            )

    prereg = json.loads(Path(args.pilot_preregistration).read_text())
    batches = {}
    for batch in prereg["design"]["execution_batches"]:
        name = f"batch_{batch['batch']:02d}_cohort_{batch['cohort']}_{batch['arm']}_r{batch['repeat']}"
        rows = load_jsonl_map(Path(args.pilot_root) / name)
        for case_id, row in rows.items():
            batches[(case_id, batch["repeat"], batch["arm"])] = row
    for case_id in prereg["sampling"]["selected_ids"]:
        for repeat in range(1, 5):
            add_pair_rows(
                records,
                "pilot",
                repeat,
                {case_id: batches[(case_id, repeat, "baseline")]},
                {case_id: batches[(case_id, repeat, "memory")]},
                gold,
                commit_lookup,
            )
    return pd.DataFrame(records)


def case_equal_weights(frame, label):
    counts = frame.groupby("case_id").size().to_dict()
    weights = np.asarray([1.0 / counts[case] for case in frame["case_id"]], dtype=float)
    y = frame[label].to_numpy(dtype=int)
    class_mass = {value: weights[y == value].sum() for value in (0, 1)}
    total = weights.sum()
    for value in (0, 1):
        if class_mass[value] > 0:
            weights[y == value] *= total / (2 * class_mass[value])
    return weights


def make_model(kind, seed=42):
    if kind == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=2,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_model(frame, features, label, kind="logistic", seed=42):
    model = make_model(kind, seed)
    weights = case_equal_weights(frame, label)
    fit_key = "model__sample_weight"
    model.fit(frame[features], frame[label], **{fit_key: weights})
    return model


def predict_model(model, frame, features):
    return model.predict_proba(frame[features])[:, 1]


def loco_predictions(frame, features, label, kind="logistic", seed=42):
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    for train, test, case_id in loco_splits(frame):
        if train[label].nunique() < 2:
            predictions.loc[test.index] = train[label].mean()
            continue
        model = fit_model(train, features, label, kind, seed)
        predictions.loc[test.index] = predict_model(model, test, features)
    return predictions.to_numpy()


def loco_splits(frame):
    for case_id in sorted(frame["case_id"].unique()):
        yield (
            frame[frame["case_id"] != case_id],
            frame[frame["case_id"] == case_id],
            case_id,
        )


def nested_t1_loco(frame, label, seed=42):
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for outer_case in sorted(frame["case_id"].unique()):
        train = frame[frame["case_id"] != outer_case].copy()
        test = frame[frame["case_id"] == outer_case].copy()
        inner_score = loco_predictions(train, T0_FEATURES, label, "logistic", seed)
        t0_model = fit_model(train, T0_FEATURES, label, "logistic", seed)
        train["t0_risk_logit"] = np.log(np.clip(inner_score, 1e-6, 1 - 1e-6) / np.clip(1 - inner_score, 1e-6, 1))
        test_score = predict_model(t0_model, test, T0_FEATURES)
        test["t0_risk_logit"] = np.log(np.clip(test_score, 1e-6, 1 - 1e-6) / np.clip(1 - test_score, 1e-6, 1))
        train["risk_x_rapid"] = train["t0_risk_logit"] * train["rapid_unverified_top1_adoption"]
        test["risk_x_rapid"] = test["t0_risk_logit"] * test["rapid_unverified_top1_adoption"]
        features = T1_RAW_FEATURES + ["t0_risk_logit", "risk_x_rapid"]
        model = fit_model(train, features, label, "logistic", seed)
        output.loc[test.index] = predict_model(model, test, features)
    return output.to_numpy()


def fit_t1_external(dev, external, label, seed=42):
    train = dev.copy()
    test = external.copy()
    oof = loco_predictions(train, T0_FEATURES, label, "logistic", seed)
    t0 = fit_model(train, T0_FEATURES, label, "logistic", seed)
    ext = predict_model(t0, test, T0_FEATURES)
    train["t0_risk_logit"] = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1))
    test["t0_risk_logit"] = np.log(np.clip(ext, 1e-6, 1 - 1e-6) / np.clip(1 - ext, 1e-6, 1))
    train["risk_x_rapid"] = train["t0_risk_logit"] * train["rapid_unverified_top1_adoption"]
    test["risk_x_rapid"] = test["t0_risk_logit"] * test["rapid_unverified_top1_adoption"]
    features = T1_RAW_FEATURES + ["t0_risk_logit", "risk_x_rapid"]
    model = fit_model(train, features, label, "logistic", seed)
    return predict_model(model, test, features), model, features


def top_fraction_metrics(y, score, fraction=0.20):
    n = max(1, math.ceil(len(y) * fraction))
    selected = np.argsort(-np.asarray(score))[:n]
    y = np.asarray(y, dtype=int)
    positives = y.sum()
    return {
        "top20_precision": float(y[selected].mean()),
        "top20_recall": float(y[selected].sum() / positives) if positives else None,
    }


def metrics(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    prevalence = float(y.mean())
    auroc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
    ap = float(average_precision_score(y, score)) if y.sum() else None
    brier = float(brier_score_loss(y, score))
    baseline_brier = prevalence * (1 - prevalence)
    return {
        "n": len(y),
        "positive_count": int(y.sum()),
        "prevalence": prevalence,
        "auroc": auroc,
        "average_precision": ap,
        "ap_lift": ap / prevalence if ap is not None and prevalence else None,
        "brier": brier,
        "brier_skill": 1 - brier / baseline_brier if baseline_brier else None,
        **top_fraction_metrics(y, score),
    }


def cluster_bootstrap(frame, label, score_col, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    cases = sorted(frame["case_id"].unique())
    values = defaultdict(list)
    for _ in range(n_boot):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        pieces = [frame[frame["case_id"] == case] for case in sampled]
        sample = pd.concat(pieces, ignore_index=True)
        if sample[label].nunique() < 2:
            continue
        result = metrics(sample[label], sample[score_col])
        for key in ("auroc", "ap_lift", "brier", "top20_recall"):
            if result[key] is not None:
                values[key].append(result[key])
    return {
        "requested_replicates": n_boot,
        "valid_replicates": len(values["auroc"]),
        **{
            f"{key}_95ci": np.percentile(vals, [2.5, 97.5]).tolist() if vals else [None, None]
            for key, vals in values.items()
        },
    }


def bca_auc_interval(labels, scores, bootstrap_aucs, alpha=0.05):
    """Return a case-level BCa interval for AUROC."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    bootstrap_aucs = np.asarray(bootstrap_aucs, dtype=float)
    observed = roc_auc_score(labels, scores)
    proportion_below = np.mean(bootstrap_aucs < observed)
    epsilon = 1 / (2 * len(bootstrap_aucs))
    z0 = norm.ppf(np.clip(proportion_below, epsilon, 1 - epsilon))

    jackknife = []
    for index in range(len(labels)):
        keep = np.arange(len(labels)) != index
        if len(np.unique(labels[keep])) == 2:
            jackknife.append(roc_auc_score(labels[keep], scores[keep]))
    jackknife = np.asarray(jackknife, dtype=float)
    centered = jackknife.mean() - jackknife
    denominator = 6 * np.sum(centered**2) ** 1.5
    acceleration = float(np.sum(centered**3) / denominator) if denominator else 0.0

    adjusted = []
    for probability in (alpha / 2, 1 - alpha / 2):
        z_alpha = norm.ppf(probability)
        divisor = 1 - acceleration * (z0 + z_alpha)
        adjusted.append(norm.cdf(z0 + (z0 + z_alpha) / divisor))
    interval = np.quantile(bootstrap_aucs, np.clip(adjusted, 0, 1))
    return {
        "auroc_bca_95ci": interval.tolist(),
        "bias_correction": float(z0),
        "acceleration": acceleration,
        "adjusted_quantiles": [float(value) for value in adjusted],
        "jackknife_replicates": len(jackknife),
    }


def bootstrap_distribution_diagnostics(values, seed=42):
    values = np.asarray(values, dtype=float)
    if len(np.unique(values)) < 2:
        return {
            "standard_deviation": float(np.std(values, ddof=1)),
            "gmm_bic_improvement_2_vs_1": None,
            "gmm_component_means": [float(values[0])],
            "gmm_component_weights": [1.0],
            "bimodal": False,
        }
    one = GaussianMixture(n_components=1, random_state=seed, n_init=10).fit(
        values.reshape(-1, 1)
    )
    two = GaussianMixture(n_components=2, random_state=seed, n_init=10).fit(
        values.reshape(-1, 1)
    )
    means = sorted(two.means_.ravel().tolist())
    weights = two.weights_[np.argsort(two.means_.ravel())].tolist()
    bic_improvement = float(one.bic(values.reshape(-1, 1)) - two.bic(values.reshape(-1, 1)))
    bimodal = bool(
        bic_improvement >= 10
        and means[1] - means[0] >= 0.15
        and min(weights) >= 0.10
    )
    return {
        "standard_deviation": float(np.std(values, ddof=1)),
        "gmm_bic_improvement_2_vs_1": bic_improvement,
        "gmm_component_means": means,
        "gmm_component_weights": weights,
        "bimodal": bimodal,
    }


def case_auc_bootstrap(frame, label, score_col, method, n_boot=10000, seed=42):
    """Bootstrap one-row-per-case predictions without refitting the model."""
    if frame["case_id"].nunique() != len(frame):
        raise ValueError("case AUROC bootstrap requires exactly one row per case")
    if frame[label].nunique() != 2:
        raise ValueError("case AUROC bootstrap requires two outcome classes")
    if method not in {"ordinary", "stratified"}:
        raise ValueError(f"unknown case bootstrap method: {method}")

    rng = np.random.default_rng(seed)
    labels = frame[label].to_numpy(dtype=int)
    scores = frame[score_col].to_numpy(dtype=float)
    all_indexes = np.arange(len(frame))
    positive = all_indexes[labels == 1]
    negative = all_indexes[labels == 0]
    rows = []
    valid_aucs = []
    for replicate in range(n_boot):
        if method == "ordinary":
            selected = rng.choice(all_indexes, size=len(all_indexes), replace=True)
        else:
            selected = np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
        sampled_labels = labels[selected]
        valid = len(np.unique(sampled_labels)) == 2
        auroc = float(roc_auc_score(sampled_labels, scores[selected])) if valid else np.nan
        if valid:
            valid_aucs.append(auroc)
        rows.append(
            {
                "method": method,
                "replicate": replicate,
                "valid": int(valid),
                "positive_count": int(sampled_labels.sum()),
                "negative_count": int(len(sampled_labels) - sampled_labels.sum()),
                "auroc": auroc,
            }
        )

    values = np.asarray(valid_aucs, dtype=float)
    quantile_levels = [2.5, 25, 50, 75, 97.5]
    quantiles = np.percentile(values, quantile_levels)
    bin_counts = {
        "lt_0_5": int(np.sum(values < 0.5)),
        "0_5_to_0_6": int(np.sum((values >= 0.5) & (values < 0.6))),
        "0_6_to_0_7": int(np.sum((values >= 0.6) & (values < 0.7))),
        "0_7_to_0_8": int(np.sum((values >= 0.7) & (values < 0.8))),
        "ge_0_8": int(np.sum(values >= 0.8)),
    }
    valid_count = len(values)
    diagnostics = bootstrap_distribution_diagnostics(values, seed)
    discard_rate = (n_boot - valid_count) / n_boot
    below_chance_rate = bin_counts["lt_0_5"] / valid_count
    bca_triggered = bool(
        method == "ordinary"
        and (diagnostics["standard_deviation"] > 0.15 or diagnostics["bimodal"])
    )
    summary = {
        "method": method,
        "requested_replicates": n_boot,
        "valid_replicates": valid_count,
        "discarded_replicates": n_boot - valid_count,
        "discard_rate": discard_rate,
        "class_imbalance_fragility": bool(discard_rate > 0.15),
        "percentiles": {
            str(level): float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
        "auroc_percentile_95ci": [float(quantiles[0]), float(quantiles[-1])],
        "ci_width": float(quantiles[-1] - quantiles[0]),
        "auroc_bin_counts": bin_counts,
        "auroc_bin_proportions": {
            key: value / valid_count for key, value in bin_counts.items()
        },
        "below_chance_rate": below_chance_rate,
        "signal_fragility": bool(below_chance_rate > 0.20),
        "distribution_diagnostics": diagnostics,
        "bca_triggered": bca_triggered,
        "bca": None,
    }
    if summary["bca_triggered"]:
        summary["bca"] = bca_auc_interval(labels, scores, values)
    return summary, rows


def case_flip_decision(dev_auc, external_auc, ordinary, stratified):
    directional_flip = bool(dev_auc < 0.55 and external_auc > 0.65)
    minimum_lower_ci = min(
        ordinary["auroc_percentile_95ci"][0],
        stratified["auroc_percentile_95ci"][0],
    )
    flags = []
    if directional_flip:
        flags.append("DIRECTIONAL_FLIP")
    if ordinary["class_imbalance_fragility"]:
        flags.append("CLASS_IMBALANCE_FRAGILITY")
    if ordinary["signal_fragility"]:
        flags.append("SIGNAL_FRAGILITY")
    if ordinary["distribution_diagnostics"]["bimodal"]:
        flags.append("BIMODAL_BOOTSTRAP")
    uncertainty_intervals = [ordinary["auroc_percentile_95ci"]]
    if ordinary.get("bca"):
        uncertainty_intervals.append(ordinary["bca"]["auroc_bca_95ci"])
    highly_uncertain = any(
        upper - lower >= 0.30 and lower <= 0.5 <= upper
        for lower, upper in uncertainty_intervals
    )
    if highly_uncertain:
        flags.append("HIGHLY_UNCERTAIN_PILOT_ESTIMATE")

    if directional_flip:
        status = "REPLICATION_REQUIRED_DIRECTIONAL_FLIP"
    elif external_auc >= 0.70 and minimum_lower_ci > 0.50:
        status = "CONTINUE_TO_PAPER2_CASE_LEVEL"
    else:
        status = "STOP_PREDICT_AND_GATE"
    return {
        "status": status,
        "directional_flip": directional_flip,
        "minimum_percentile_lower_ci": float(minimum_lower_ci),
        "flags": flags,
        "stage2_gating_authorized": False,
        "recommended_next_step": (
            "Independent second random pilot with at least 20 additional cases; "
            "evaluate separately before pooling."
            if directional_flip
            else "No Stage 2 gating experiment."
        ),
    }


def case_flip_influence_decision(
    dev_auc, original_auc, bca_lower, max_single_drop, max_pair_drop, min_signed_drop
):
    hard_stops = {
        "DIRECTIONAL_FLIP": bool(dev_auc <= 0.55 and original_auc >= 0.65),
        "BCA_CROSSES_CHANCE": bool(bca_lower <= 0.50),
        "SINGLE_CASE_DOMINATED": bool(max_single_drop > 0.20),
        "PAIR_DOMINANCE": bool(max_pair_drop > 0.20),
        "NOISE_SUPPRESSOR": bool(min_signed_drop < -0.15),
    }
    stop_reasons = [name for name, triggered in hard_stops.items() if triggered]
    category = (
        "SINGLE_CASE_DOMINATED" if max_single_drop > 0.20
        else "MODERATE_SINGLE_CASE_SENSITIVITY" if max_single_drop >= 0.10
        else "NO_SINGLE_CASE_DOMINANCE"
    )
    status = (
        "STOP_PREDICT_AND_GATE" if stop_reasons
        else "REPLICATION_ONLY_MODERATE_SENSITIVITY" if max_single_drop >= 0.10
        else "ELIGIBLE_FOR_INDEPENDENT_REPLICATION"
    )
    return hard_stops, stop_reasons, category, status


def case_flip_influence_audit(frame, label, score_col, dev_auc, bca_lower):
    original_auc = float(roc_auc_score(frame[label], frame[score_col]))
    single_rows = []
    for index, removed in frame.iterrows():
        kept = frame.drop(index=index)
        auc = float(roc_auc_score(kept[label], kept[score_col]))
        drop = original_auc - auc
        single_rows.append({
            "case_id": removed["case_id"], "case_flip": int(removed[label]),
            "predicted_risk": float(removed[score_col]),
            "remaining_positive": int(kept[label].sum()),
            "remaining_negative": int(len(kept) - kept[label].sum()),
            "auc_without_case": auc, "signed_drop": drop,
            "absolute_change": abs(drop),
        })
    single_rows.sort(key=lambda row: (-row["signed_drop"], row["case_id"]))
    for rank, row in enumerate(single_rows, 1):
        row["influence_rank"] = rank

    pair_rows = []
    for left, right in combinations(frame.index, 2):
        kept = frame.drop(index=[left, right])
        auc = float(roc_auc_score(kept[label], kept[score_col]))
        pair_rows.append({
            "case_id_left": frame.loc[left, "case_id"],
            "case_id_right": frame.loc[right, "case_id"],
            "case_flip_left": int(frame.loc[left, label]),
            "case_flip_right": int(frame.loc[right, label]),
            "remaining_positive": int(kept[label].sum()),
            "remaining_negative": int(len(kept) - kept[label].sum()),
            "auc_without_pair": auc, "signed_drop": original_auc - auc,
        })
    pair_rows.sort(key=lambda row: (-row["signed_drop"], row["case_id_left"], row["case_id_right"]))
    for rank, row in enumerate(pair_rows, 1):
        row["influence_rank"] = rank

    def class_summary(value):
        drops = [row["signed_drop"] for row in single_rows if row["case_flip"] == value]
        return {"n": len(drops), "max_drop": max(drops), "min_drop": min(drops), "mean_drop": float(np.mean(drops))}

    maximum, minimum, pair_maximum = single_rows[0], min(single_rows, key=lambda row: row["signed_drop"]), pair_rows[0]
    hard_stops, stop_reasons, category, status = case_flip_influence_decision(
        dev_auc, original_auc, bca_lower, maximum["signed_drop"],
        pair_maximum["signed_drop"], minimum["signed_drop"],
    )
    return {
        "method": "sklearn.metrics.roc_auc_score; default average-rank tie handling",
        "post_hoc": True, "original_auroc": original_auc,
        "positive_count": int(frame[label].sum()),
        "negative_count": int(len(frame) - frame[label].sum()),
        "concordant_pair_denominator": int(frame[label].sum() * (len(frame) - frame[label].sum())),
        "positive_deletions": class_summary(1),
        "negative_deletions": class_summary(0),
        "max_single_drop": maximum["signed_drop"], "max_single_case": maximum["case_id"],
        "min_signed_drop": minimum["signed_drop"], "min_signed_case": minimum["case_id"],
        "max_pair_drop": pair_maximum["signed_drop"],
        "max_pair_cases": [pair_maximum["case_id_left"], pair_maximum["case_id_right"]],
        "hard_stops": hard_stops, "stop_reasons": stop_reasons,
        "single_case_category": category,
        "status": status, "stage2_gating_authorized": False,
    }, single_rows, pair_rows


def reliability_rows(frame, label, score_col, method, bins=10):
    work = frame[["case_id", label, score_col]].sort_values(score_col).copy()
    work["bin"] = pd.qcut(np.arange(len(work)), q=min(bins, len(work)), labels=False)
    rows = []
    for bin_id, group in work.groupby("bin"):
        rows.append(
            {
                "method": method,
                "bin": int(bin_id),
                "n": len(group),
                "unique_cases": group["case_id"].nunique(),
                "mean_probability": group[score_col].mean(),
                "observed_rate": group[label].mean(),
            }
        )
    return rows


def ece(reliability):
    total = sum(row["n"] for row in reliability)
    return sum(row["n"] / total * abs(row["mean_probability"] - row["observed_rate"]) for row in reliability)


def prior_adjust(probability, source_prevalence, target_prevalence):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    odds = probability / (1 - probability)
    ratio = (target_prevalence / (1 - target_prevalence)) / (source_prevalence / (1 - source_prevalence))
    adjusted = odds * ratio
    return adjusted / (1 + adjusted)


def aggregate_case_features(frame):
    rows = []
    for (case_id, source), group in frame.groupby(["case_id", "source"]):
        item = {"case_id": case_id, "source": source}
        for feature in T0_FEATURES:
            item[f"{feature}_mean"] = group[feature].mean()
            item[f"{feature}_std"] = group[feature].std(ddof=0)
        memory = group.sort_values("repeat")["memory_success"].to_numpy()
        baseline = group.sort_values("repeat")["baseline_success"].to_numpy()
        memory_pairs = [int(memory[i] == memory[j]) for i in range(4) for j in range(i + 1, 4)]
        baseline_pairs = [int(baseline[i] == baseline[j]) for i in range(4) for j in range(i + 1, 4)]
        item["case_flip"] = int(len(set(memory)) > 1)
        item["memory_stability"] = np.mean(memory_pairs)
        item["baseline_stability"] = np.mean(baseline_pairs)
        item["stability_degradation"] = int(item["memory_stability"] < item["baseline_stability"])
        rows.append(item)
    return pd.DataFrame(rows)


def case_target_analysis(case_frame, target, seed=42):
    dev = case_frame[case_frame["source"] != "pilot"].copy()
    external = case_frame[case_frame["source"] == "pilot"].copy()
    features = [column for column in case_frame if column.endswith("_mean")]
    oof = loco_predictions(dev, features, target, "logistic", seed)
    model = fit_model(dev, features, target, "logistic", seed)
    ext = predict_model(model, external, features)
    predictions = external[["case_id", "source", target]].copy()
    predictions["predicted_risk"] = ext
    result = {
        "development_loco": metrics(dev[target], oof),
        "external": metrics(external[target], ext),
    }
    return result, predictions


def permutation_test(dev, external, label, observed_auc, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    cases = sorted(dev["case_id"].unique())
    label_vectors = {
        case: dev[dev["case_id"] == case].sort_values(["source", "repeat"])[label].to_numpy()
        for case in cases
    }
    aucs = []
    for _ in range(n_perm):
        permuted_cases = rng.permutation(cases)
        shuffled = dev.copy()
        for destination, source in zip(cases, permuted_cases):
            indexes = shuffled[shuffled["case_id"] == destination].sort_values(["source", "repeat"]).index
            shuffled.loc[indexes, label] = label_vectors[source]
        if shuffled[label].nunique() < 2:
            continue
        model = fit_model(shuffled, T0_FEATURES, label, "logistic", seed)
        score = predict_model(model, external, T0_FEATURES)
        aucs.append(roc_auc_score(external[label], score))
    return {
        "n_permutations": len(aucs),
        "observed_external_auc": observed_auc,
        "null_mean_auc": float(np.mean(aucs)),
        "p_value": float((sum(value >= observed_auc for value in aucs) + 1) / (len(aucs) + 1)),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def write_csv(path, frame_or_rows):
    if isinstance(frame_or_rows, pd.DataFrame):
        frame_or_rows.to_csv(path, index=False)
    else:
        pd.DataFrame(frame_or_rows).to_csv(path, index=False)


def write_jsonl(path, frame):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in frame.to_dict("records"):
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def plot_reliability(rows, path):
    plt.figure(figsize=(6, 5))
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        plt.plot([row["mean_probability"] for row in group], [row["observed_rate"] for row in group], marker="o", label=method)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed paired-harm rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_case_flip_bootstrap(rows, path):
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, method in zip(axes, ("ordinary", "stratified")):
        values = frame.loc[(frame["method"] == method) & (frame["valid"] == 1), "auroc"]
        axis.hist(values, bins=np.linspace(0, 1, 21), density=True, alpha=0.8)
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{method.title()} case bootstrap")
        axis.set_xlabel("AUROC")
    axes[0].set_ylabel("Density")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_case_flip_influence(single_rows, pair_rows, path):
    singles = pd.DataFrame(single_rows).sort_values("signed_drop")
    pairs = pd.DataFrame(pair_rows)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = singles["case_flip"].map({0: "tab:blue", 1: "tab:orange"})
    axes[0].hlines(range(len(singles)), 0, singles["signed_drop"], color=colors)
    axes[0].scatter(singles["signed_drop"], range(len(singles)), color=colors)
    axes[0].set_yticks(range(len(singles)), singles["case_id"], fontsize=7)
    for threshold in (-0.15, 0.10, 0.20):
        axes[0].axvline(threshold, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Original AUROC - AUROC without case")
    axes[0].set_title("Single-case influence")
    axes[1].hist(pairs["signed_drop"], bins=25, alpha=0.8)
    axes[1].axvline(0.20, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Original AUROC - AUROC without pair")
    axes[1].set_ylabel("Case-pair count")
    axes[1].set_title("Two-case deletion stress test")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def verdict(external_metrics, bootstrap, dev_auc, t1_metrics):
    auc_ci = bootstrap.get("auroc_95ci", [None, None])
    ap_ci = bootstrap.get("ap_lift_95ci", [None, None])
    promising = (
        external_metrics["auroc"] >= 0.70
        and auc_ci[0] is not None and auc_ci[0] >= 0.55
        and external_metrics["ap_lift"] >= 2.0
        and ap_ci[0] is not None and ap_ci[0] >= 1.25
        and external_metrics["top20_recall"] >= 0.50
        and dev_auc >= 0.60
    )
    no_signal = (
        auc_ci[1] is not None and auc_ci[1] <= 0.55
        and ap_ci[1] is not None and ap_ci[1] < 1.25
        and t1_metrics["auroc"] <= external_metrics["auroc"]
        and t1_metrics["ap_lift"] <= external_metrics["ap_lift"]
    )
    return "PROMISING_EARLY_SIGNAL" if promising else "NO_USEFUL_SIGNAL" if no_signal else "INCONCLUSIVE_PILOT"


def report(summary):
    primary = summary["paired_harm"]
    ext = primary["external_t0_raw"]
    boot = primary["external_bootstrap"]
    case_flip = summary["case_targets"]["case_flip"]
    case_external = case_flip["external"]
    case_boot = case_flip["external_bootstrap"]
    ordinary = case_boot["ordinary"]
    stratified = case_boot["stratified"]
    decision = case_flip["decision"]
    influence = case_flip["influence_audit"]
    bca = ordinary["bca"]
    bca_line = (
        f"- Ordinary BCa 95% CI: [{bca['auroc_bca_95ci'][0]:.3f}, "
        f"{bca['auroc_bca_95ci'][1]:.3f}]"
        if bca
        else "- Ordinary BCa 95% CI: not triggered by the preregistered diagnostics"
    )
    lines = [
            "# Stage 1: Early Memory Risk Detection",
            "",
            f"**Verdict: {summary['verdict']}**",
            "",
            "## Framing",
            "",
            "The primary paired-harm label uses matched Baseline and Memory outcomes. Baseline outcome is unavailable in single-arm deployment; paired harm is therefore diagnostic, not a deployment-ready gate target. Memory failure is reported separately as the baseline-independent, deployment-relevant target.",
            "",
            "Development is outcome-selected with 33.9% paired-harm prevalence, while the Random pilot prevalence is 7.5%. All pilot probability and threshold metrics are interpreted under this known base-rate shift.",
            "",
            "The six pilot harm trajectories come from only three unique cases. Case-cluster confidence intervals, rather than trajectory counts, determine the verdict.",
            "",
            "## Primary external result",
            "",
            f"- T0 AUROC: {ext['auroc']:.3f}, 95% case-cluster CI [{boot['auroc_95ci'][0]:.3f}, {boot['auroc_95ci'][1]:.3f}]",
            f"- T0 AP lift: {ext['ap_lift']:.3f}, 95% CI [{boot['ap_lift_95ci'][0]:.3f}, {boot['ap_lift_95ci'][1]:.3f}]",
            f"- Top-20% recall: {ext['top20_recall']:.3f}",
            f"- Development LOCO AUROC: {primary['development_loco_t0']['auroc']:.3f}",
            f"- T1 external AUROC: {primary['external_t1_raw']['auroc']:.3f}",
            f"- T1 AUROC delta: {primary['external_t1_raw']['auroc'] - ext['auroc']:+.3f}; AP-lift delta: {primary['external_t1_raw']['ap_lift'] - ext['ap_lift']:+.3f}",
            f"- Risk × rapid-adoption coefficient: {primary['t1_risk_x_rapid_coef']:+.3f}",
            f"- Case-level permutation p-value: {primary['permutation']['p_value']:.3f}",
            "",
            "## Calibration",
            "",
            f"- Raw Brier: {ext['brier']:.3f}; ECE: {primary['calibration']['raw_ece']:.3f}",
            f"- Development-isotonic Brier: {primary['external_t0_isotonic']['brier']:.3f}; ECE: {primary['calibration']['isotonic_ece']:.3f}",
            f"- Oracle prior-adjusted Brier: {primary['external_t0_prior_adjusted']['brier']:.3f}; ECE: {primary['calibration']['prior_adjusted_ece']:.3f}",
            f"- Raw/isotonic/prior-adjusted AUROC: {ext['auroc']:.3f} / {primary['external_t0_isotonic']['auroc']:.3f} / {primary['external_t0_prior_adjusted']['auroc']:.3f}",
            f"- Raw/isotonic/prior-adjusted AP lift: {ext['ap_lift']:.3f} / {primary['external_t0_isotonic']['ap_lift']:.3f} / {primary['external_t0_prior_adjusted']['ap_lift']:.3f}",
            "",
            "The prior-adjusted result uses the observed 6/80 pilot prevalence and is descriptive only. A prospective gate must obtain target prevalence from an independent calibration window.",
            "",
            "The result remains INCONCLUSIVE rather than NO_USEFUL_SIGNAL because the AP-lift confidence upper bound exceeds the predeclared 1.25 no-signal boundary, despite the poor AUROC interval.",
            "",
            "## Interpretation",
            "",
            "T0 evaluates pre-exposure retrieval risk. T1 evaluates whether early code evidence and rapid top-1 adoption add predictive information. Neither analysis establishes mediation, causality, or controller benefit. Accuracy–stability–cost trade-offs were not evaluated, and no Stage 2 gating experiment is authorized by these results.",
            "",
            "## Case-level directional consistency and uncertainty",
            "",
            f"**Case-level decision: {decision['status']}**",
            "",
            f"- Development LOCO case-flip AUROC: {case_flip['development_loco']['auroc']:.3f} on 45 case-version observations from 15 unique cases",
            f"- External Random-pilot case-flip AUROC: {case_external['auroc']:.3f} on 20 cases ({case_external['positive_count']} positive)",
            f"- Ordinary case-bootstrap percentile 95% CI: [{ordinary['auroc_percentile_95ci'][0]:.3f}, {ordinary['auroc_percentile_95ci'][1]:.3f}]",
            f"- Stratified case-bootstrap percentile 95% CI: [{stratified['auroc_percentile_95ci'][0]:.3f}, {stratified['auroc_percentile_95ci'][1]:.3f}]",
            bca_line,
            f"- Ordinary percentiles (2.5/25/50/75/97.5): {ordinary['percentiles']['2.5']:.3f} / {ordinary['percentiles']['25']:.3f} / {ordinary['percentiles']['50']:.3f} / {ordinary['percentiles']['75']:.3f} / {ordinary['percentiles']['97.5']:.3f}",
            f"- Stratified percentiles (2.5/25/50/75/97.5): {stratified['percentiles']['2.5']:.3f} / {stratified['percentiles']['25']:.3f} / {stratified['percentiles']['50']:.3f} / {stratified['percentiles']['75']:.3f} / {stratified['percentiles']['97.5']:.3f}",
            f"- Ordinary AUROC-bin counts (<.5/.5-.6/.6-.7/.7-.8/>=.8): {ordinary['auroc_bin_counts']['lt_0_5']} / {ordinary['auroc_bin_counts']['0_5_to_0_6']} / {ordinary['auroc_bin_counts']['0_6_to_0_7']} / {ordinary['auroc_bin_counts']['0_7_to_0_8']} / {ordinary['auroc_bin_counts']['ge_0_8']}",
            f"- Stratified AUROC-bin counts (<.5/.5-.6/.6-.7/.7-.8/>=.8): {stratified['auroc_bin_counts']['lt_0_5']} / {stratified['auroc_bin_counts']['0_5_to_0_6']} / {stratified['auroc_bin_counts']['0_6_to_0_7']} / {stratified['auroc_bin_counts']['0_7_to_0_8']} / {stratified['auroc_bin_counts']['ge_0_8']}",
            f"- Ordinary bootstrap SD: {ordinary['distribution_diagnostics']['standard_deviation']:.3f}; discard rate: {ordinary['discard_rate']:.3f}",
            f"- Below-chance ordinary replicates: {ordinary['below_chance_rate']:.3f}",
            f"- Flags: {', '.join(decision['flags']) if decision['flags'] else 'none'}",
            "",
            "The development and external AUROCs lie on opposite sides of chance. The preregistered directional-consistency gate therefore blocks continuation regardless of the bootstrap interval. These intervals describe point-estimate uncertainty conditional on this frozen model and these 20 pilot cases; they do not rule out lucky pilot sampling or development-specific noise.",
            "",
            "## Case-level influence audit",
            "",
            "This is a post-hoc, metric-only deletion audit using sklearn.metrics.roc_auc_score with default average-rank tie handling. The model is not refit, so this analysis can only decrease confidence in the observed pilot result.",
            "",
            f"- Final influence status: {influence['status']}",
            f"- Positive-case deletions (n={influence['positive_deletions']['n']}): max/min/mean drop = {influence['positive_deletions']['max_drop']:.3f} / {influence['positive_deletions']['min_drop']:.3f} / {influence['positive_deletions']['mean_drop']:.3f}",
            f"- Negative-case deletions (n={influence['negative_deletions']['n']}): max/min/mean drop = {influence['negative_deletions']['max_drop']:.3f} / {influence['negative_deletions']['min_drop']:.3f} / {influence['negative_deletions']['mean_drop']:.3f}",
            f"- Maximum single-case drop: {influence['max_single_drop']:.3f} ({influence['max_single_case']})",
            f"- Minimum signed drop: {influence['min_signed_drop']:.3f} ({influence['min_signed_case']})",
            f"- Maximum two-case drop: {influence['max_pair_drop']:.3f} ({' + '.join(influence['max_pair_cases'])})",
            f"- Single-case category: {influence['single_case_category']}",
            f"- Independent hard-stop checks: {', '.join(name + '=' + ('ON' if value else 'off') for name, value in influence['hard_stops'].items())}",
            f"- STOP reasons: {', '.join(influence['stop_reasons']) if influence['stop_reasons'] else 'none'}",
            "",
            "Paper 1 should report that trajectory-level early prediction was unsuccessful. No Stage 2 gating experiment is authorized. Before any Paper 2 commitment, the case-level result requires an independent second random pilot with at least 20 additional cases, evaluated separately before pooling.",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_root", default="results/context_safe_m3a/runs_v2")
    parser.add_argument("--v3_root", default="results/context_safe_m3a/replication_v3/runs")
    parser.add_argument("--v4_root", default="results/context_safe_m3a/replication_v4/runs")
    parser.add_argument("--pilot_root", default="results/paper2_pilot/runs")
    parser.add_argument("--pilot_preregistration", default="results/paper2_pilot/preregistration.json")
    parser.add_argument("--index_dirs", nargs="+", default=DEFAULT_INDEX_DIRS)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset_arrow", default=DEFAULT_ARROW)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gold = load_gold(args.dataset, args.split, args.dataset_arrow if Path(args.dataset_arrow).exists() else None)
    selected_repos = {case.rsplit("-", 1)[0].replace("__", "/") for case in gold}
    commit_lookup = load_commit_lookup(args.index_dirs, selected_repos)
    data = build_dataset(args, commit_lookup, gold)
    dev = data[data["source"] != "pilot"].copy()
    external = data[data["source"] == "pilot"].copy()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    expected = {
        "development_rows": len(dev),
        "external_rows": len(external),
        "development_harm": int(dev["paired_harm"].sum()),
        "external_harm": int(external["paired_harm"].sum()),
        "external_positive_cases": int(external.loc[external["paired_harm"] == 1, "case_id"].nunique()),
    }
    required = {"development_rows": 180, "external_rows": 80, "development_harm": 61, "external_harm": 6, "external_positive_cases": 3}
    if expected != required:
        raise ValueError(f"fixed-count audit failed: observed={expected}, expected={required}")

    summaries = {}
    prediction_tables = {}
    for label in ("paired_harm", "memory_failure"):
        dev_oof = loco_predictions(dev, T0_FEATURES, label, "logistic", args.seed)
        t0_model = fit_model(dev, T0_FEATURES, label, "logistic", args.seed)
        ext_t0 = predict_model(t0_model, external, T0_FEATURES)
        dev_t1 = nested_t1_loco(dev, label, args.seed)
        ext_t1, t1_model, t1_features = fit_t1_external(dev, external, label, args.seed)
        tree = fit_model(dev, T0_FEATURES, label, "tree", args.seed)
        ext_tree = predict_model(tree, external, T0_FEATURES)

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(dev_oof, dev[label], sample_weight=np.asarray([1 / 12] * len(dev)))
        ext_iso = isotonic.transform(ext_t0)
        ext_prior = prior_adjust(ext_iso, dev[label].mean(), external[label].mean())

        pred = external[["case_id", "source", "repeat", label]].copy()
        pred["t0_raw"] = ext_t0
        pred["t1_raw"] = ext_t1
        pred["tree_t0"] = ext_tree
        pred["t0_isotonic"] = ext_iso
        pred["t0_prior_adjusted"] = ext_prior
        prediction_tables[label] = pred

        dev_pred = dev[["case_id", "source", "repeat", label]].copy()
        dev_pred["t0_oof"] = dev_oof
        dev_pred["t1_oof"] = dev_t1
        write_csv(output / f"development_loco_predictions_{label}.csv", dev_pred)
        write_csv(output / f"external_predictions_{label}.csv", pred)

        bootstrap_frame = pred.rename(columns={label: "target"})
        boot = cluster_bootstrap(bootstrap_frame, "target", "t0_raw", args.bootstrap, args.seed)
        reliability = []
        for column, method in (("t0_raw", "raw"), ("t0_isotonic", "isotonic"), ("t0_prior_adjusted", "prior_adjusted")):
            reliability.extend(reliability_rows(pred, label, column, method))
        write_csv(output / f"reliability_{label}.csv", reliability)
        plot_reliability(reliability, output / f"reliability_{label}.png")
        methods = sorted({row["method"] for row in reliability})
        by_method = {
            method: [row for row in reliability if row["method"] == method]
            for method in methods
        }

        model_features = t1_features
        t1_coef = dict(zip(model_features, t1_model.named_steps["model"].coef_[0]))
        summaries[label] = {
            "development_loco_t0": metrics(dev[label], dev_oof),
            "development_loco_t1": metrics(dev[label], dev_t1),
            "external_t0_raw": metrics(external[label], ext_t0),
            "external_t1_raw": metrics(external[label], ext_t1),
            "external_tree_t0": metrics(external[label], ext_tree),
            "external_t0_isotonic": metrics(external[label], ext_iso),
            "external_t0_prior_adjusted": metrics(external[label], ext_prior),
            "external_bootstrap": boot,
            "calibration": {f"{method}_ece": ece(rows) for method, rows in by_method.items()},
            "t1_risk_x_rapid_coef": float(t1_coef["risk_x_rapid"]),
            "t0_logistic_coefficients": {
                feature: float(coef)
                for feature, coef in zip(T0_FEATURES, t0_model.named_steps["model"].coef_[0])
            },
            "tree_rules": export_text(tree.named_steps["model"], feature_names=T0_FEATURES),
        }

        ablations = {}
        for name, removed in {
            "full_t0": [],
            "without_bm25_scores": ["bm25_top1_score", "bm25_relative_margin"],
            "without_cross_query_consistency": [
                "top5_file_consistency",
                "query_text_similarity",
                "inter_query_sha_jaccard",
                "module_conflict_rate",
            ],
        }.items():
            columns = [feature for feature in T0_FEATURES if feature not in removed]
            ablation_model = fit_model(dev, columns, label, "logistic", args.seed)
            ablations[name] = metrics(
                external[label], predict_model(ablation_model, external, columns)
            )
        summaries[label]["external_ablations"] = ablations

    pooled = pd.concat([dev, external], ignore_index=True)
    pooled_score = loco_predictions(pooled, T0_FEATURES, "paired_harm", "logistic", args.seed)
    summaries["paired_harm"]["pooled_35_case_loco"] = metrics(pooled["paired_harm"], pooled_score)
    summaries["paired_harm"]["permutation"] = permutation_test(
        dev,
        external,
        "paired_harm",
        summaries["paired_harm"]["external_t0_raw"]["auroc"],
        args.permutations,
        args.seed,
    )

    case_frame = aggregate_case_features(data)
    case_results = {}
    for target in ("case_flip", "stability_degradation"):
        case_result, case_predictions = case_target_analysis(case_frame, target, args.seed)
        case_results[target] = case_result
        if target != "case_flip":
            continue
        case_audit = {
            "n": len(case_predictions),
            "unique_cases": case_predictions["case_id"].nunique(),
            "positive_count": int(case_predictions[target].sum()),
            "negative_count": int((1 - case_predictions[target]).sum()),
        }
        required_case_audit = {
            "n": 20,
            "unique_cases": 20,
            "positive_count": 6,
            "negative_count": 14,
        }
        if case_audit != required_case_audit:
            raise ValueError(
                "case-flip fixed-count audit failed: "
                f"observed={case_audit}, expected={required_case_audit}"
            )
        point_auc = case_result["external"]["auroc"]
        if not np.isclose(point_auc, 0.7619047619047619):
            raise ValueError(f"case-flip point AUROC audit failed: {point_auc}")
        ordinary, ordinary_rows = case_auc_bootstrap(
            case_predictions,
            target,
            "predicted_risk",
            "ordinary",
            args.bootstrap,
            args.seed,
        )
        stratified, stratified_rows = case_auc_bootstrap(
            case_predictions,
            target,
            "predicted_risk",
            "stratified",
            args.bootstrap,
            args.seed,
        )
        case_result["external_bootstrap"] = {
            "point_auroc": point_auc,
            "fixed_count_audit": case_audit,
            "ordinary": ordinary,
            "stratified": stratified,
        }
        case_result["decision"] = case_flip_decision(
            case_result["development_loco"]["auroc"],
            point_auc,
            ordinary,
            stratified,
        )
        if not ordinary.get("bca"):
            raise ValueError("case-flip influence audit requires the ordinary BCa interval")
        influence, influence_rows, pair_rows = case_flip_influence_audit(
            case_predictions,
            target,
            "predicted_risk",
            case_result["development_loco"]["auroc"],
            ordinary["bca"]["auroc_bca_95ci"][0],
        )
        case_result["influence_audit"] = influence
        case_result["decision"]["pre_influence_status"] = case_result["decision"]["status"]
        case_result["decision"]["status"] = influence["status"]
        case_result["decision"]["stop_reasons"] = influence["stop_reasons"]
        write_csv(output / "external_case_predictions_case_flip.csv", case_predictions)
        bootstrap_rows = ordinary_rows + stratified_rows
        write_csv(output / "case_flip_auroc_bootstrap.csv", bootstrap_rows)
        plot_case_flip_bootstrap(
            bootstrap_rows,
            output / "case_flip_auroc_bootstrap.png",
        )
        write_csv(output / "case_flip_influence_audit.csv", influence_rows)
        write_csv(output / "case_flip_pair_influence_audit.csv", pair_rows)
        plot_case_flip_influence(
            influence_rows,
            pair_rows,
            output / "case_flip_influence_audit.png",
        )
    write_csv(output / "early_risk_trajectory_features.csv", data)
    write_csv(output / "early_risk_case_features.csv", case_frame)
    write_jsonl(output / "early_risk_trajectory_features.jsonl", data)
    write_jsonl(output / "early_risk_case_features.jsonl", case_frame)
    availability = []
    for source, group in data.groupby("source"):
        availability.append(
            {
                "source": source,
                "n": len(group),
                "retrieval_missing": int(group["retrieval_missing"].sum()),
                "top2_missing": int(group["top2_missing"].sum()),
                "multi_query_missing": int(group["multi_query_missing"].sum()),
                "code_evidence_missing": int(group["code_evidence_missing"].sum()),
                "multi_retrieval_trajectory_count": int((group["search_commit_call_count"] > 1).sum()),
                "malformed_argument_count": int(group["malformed_argument_count_at_checkpoint"].sum()),
            }
        )
    write_csv(output / "feature_availability.csv", availability)

    primary = summaries["paired_harm"]
    final_verdict = verdict(
        primary["external_t0_raw"],
        primary["external_bootstrap"],
        primary["development_loco_t0"]["auroc"],
        primary["external_t1_raw"],
    )
    summary = {
        "verdict": final_verdict,
        "fixed_count_audit": expected,
        "development_prevalence": float(dev["paired_harm"].mean()),
        "external_prevalence": float(external["paired_harm"].mean()),
        "paired_harm": summaries["paired_harm"],
        "memory_failure": summaries["memory_failure"],
        "case_targets": case_results,
        "interpretation_boundary": {
            "paired_harm": "diagnostic; requires matched baseline outcome",
            "memory_failure": "baseline-independent and deployment-relevant, but not deployment-ready",
            "controller_benefit": "not evaluated in Stage 1",
        },
    }
    (output / "stage1_early_risk_summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n")
    (output / "stage1_early_risk_report.md").write_text(report(summary))
    print(json.dumps(json_safe({"verdict": final_verdict, "paired_harm": summaries["paired_harm"]}), indent=2))
    print(f"wrote outputs to {output}")


if __name__ == "__main__":
    main()
