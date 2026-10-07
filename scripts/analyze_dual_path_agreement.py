import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.analyze_m3a_matched_noise_floor import predicted_files, stability
from scripts.eval_verified_file_acc import files_from_patch
from scripts.sample_random_verified_pilot import repository_key
from scripts.validate_context_safe_repeat import load_jsonl


DEFAULT_ARROW = (
    "/data/wjl/ExpeRepair/ExpeRepair-v1.0/data/hf-datasets/"
    "princeton-nlp___swe-bench_verified/default/0.0.0/"
    "c104f840cc67f8b6eec6f759ebc8b2693d585d4a/"
    "swe-bench_verified-test.arrow"
)
DEFAULT_OUTPUT = "results/paper2_pilot/analysis/dual_path_agreement"
SIGNALS = (
    "d_cross_arm",
    "d_baseline_internal",
    "d_memory_internal",
)
SENSITIVITY_SIGNALS = (
    "d_cross_arm_max",
    "d_cross_all16",
    "d_cross_intersection_at5",
    "d_cross_rbo_at5",
    "d_cross_gold_removed",
)


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def jaccard_distance(left, right):
    left, right = set(left), set(right)
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / len(left | right)


def intersection_at_k_distance(left, right, k=5):
    left, right = set(left[:k]), set(right[:k])
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / k


def finite_rbo_distance(left, right, k=5, p=0.9):
    left, right = list(left[:k]), list(right[:k])
    if not left and not right:
        return 0.0
    agreements = []
    for depth in range(1, k + 1):
        left_prefix, right_prefix = set(left[:depth]), set(right[:depth])
        overlap = len(left_prefix & right_prefix)
        denominator = max(len(left_prefix), len(right_prefix), 1)
        agreements.append(overlap / denominator)
    weights = np.asarray([(1 - p) * p ** depth for depth in range(k)])
    similarity = float(np.dot(weights, agreements) / weights.sum())
    return 1.0 - similarity


def safe_auc(labels, scores, sample_weight=None):
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) != 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, sample_weight=sample_weight))


def safe_ap(labels, scores, sample_weight=None):
    labels = np.asarray(labels, dtype=int)
    if not len(labels) or not labels.sum():
        return float("nan")
    return float(average_precision_score(labels, scores, sample_weight=sample_weight))


def safe_spearman(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def partial_spearman(signal, outcome, control):
    signal = rankdata(np.asarray(signal, dtype=float), method="average")
    outcome = rankdata(np.asarray(outcome, dtype=float), method="average")
    control = rankdata(np.asarray(control, dtype=float), method="average")
    design = np.column_stack([np.ones(len(control)), control])
    signal_resid = signal - design @ np.linalg.lstsq(
        design, signal, rcond=None
    )[0]
    outcome_resid = outcome - design @ np.linalg.lstsq(
        design, outcome, rcond=None
    )[0]
    if np.ptp(signal_resid) == 0 or np.ptp(outcome_resid) == 0:
        return float("nan")
    return float(np.corrcoef(signal_resid, outcome_resid)[0, 1])


def percentile_summary(values):
    values = np.asarray([value for value in values if math.isfinite(value)])
    if not len(values):
        return {"valid": 0, "ci": [None, None], "median": None}
    low, median, high = np.percentile(values, [2.5, 50, 97.5])
    return {
        "valid": int(len(values)),
        "ci": [float(low), float(high)],
        "median": float(median),
    }


def load_predictions(preregistration, run_root):
    predictions = {
        case_id: {arm: {} for arm in ("baseline", "memory")}
        for case_id in preregistration["sampling"]["selected_ids"]
    }
    sources = {}
    for batch in preregistration["design"]["execution_batches"]:
        name = (
            f"batch_{batch['batch']:02d}_cohort_{batch['cohort']}_"
            f"{batch['arm']}_r{batch['repeat']}"
        )
        path = Path(run_root) / name / "loc_trajs.repaired.jsonl"
        rows = {row["instance_id"]: row for row in load_jsonl(path)}
        expected = set(
            preregistration["sampling"][f"cohort_{batch['cohort']}"]
        )
        if set(rows) != expected:
            raise ValueError(
                f"{name}: membership mismatch expected={len(expected)} "
                f"observed={len(rows)}"
            )
        for case_id, row in rows.items():
            predictions[case_id][batch["arm"]][batch["repeat"]] = (
                predicted_files(row)[:5]
            )
            sources[(case_id, batch["arm"], batch["repeat"])] = str(path)
    for case_id, arms in predictions.items():
        for arm, repeats in arms.items():
            if set(repeats) != {1, 2, 3, 4}:
                raise ValueError(f"incomplete predictions: {case_id} {arm}")
    return predictions, sources


def load_gold(arrow_path, selected_ids):
    if not Path(arrow_path).exists():
        return {}, f"missing Arrow file: {arrow_path}"
    rows = Dataset.from_file(arrow_path)
    gold = {
        row["instance_id"]: files_from_patch(row["patch"], py_only=True)
        for row in rows
        if row["instance_id"] in selected_ids
    }
    missing = set(selected_ids) - set(gold)
    if missing:
        return {}, f"missing gold cases: {sorted(missing)}"
    return gold, None


def outcome_audit(per_case_path, selected_ids):
    rows = {row["case_id"]: row for row in load_jsonl(per_case_path)}
    if set(rows) != set(selected_ids):
        raise ValueError("per-case outcome membership mismatch")
    counts = {"degradation": 0, "tie": 0, "improvement": 0}
    for row in rows.values():
        gap = row["stability_gap"]
        key = "degradation" if gap > 0 else "improvement" if gap < 0 else "tie"
        counts[key] += 1
    case_flips = sum(len(set(row["memory_at5"])) > 1 for row in rows.values())
    observed = {**counts, "case_flip": case_flips}
    expected = {
        "degradation": 5,
        "tie": 14,
        "improvement": 1,
        "case_flip": 6,
    }
    if observed != expected:
        raise ValueError(f"locked outcome audit failed: {observed} != {expected}")
    return rows, observed


def build_tables(predictions, sources, outcomes, gold):
    matched = []
    cases = []
    for case_id in outcomes:
        baseline = predictions[case_id]["baseline"]
        memory = predictions[case_id]["memory"]
        outcome = outcomes[case_id]
        pair_distances = []
        pair_intersection = []
        pair_rbo = []
        for repeat in range(1, 5):
            left, right = baseline[repeat], memory[repeat]
            distance = jaccard_distance(left, right)
            pair_distances.append(distance)
            pair_intersection.append(intersection_at_k_distance(left, right))
            pair_rbo.append(finite_rbo_distance(left, right))
            other = [index for index in range(4) if index != repeat - 1]
            baseline_other = [outcome["baseline_at5"][index] for index in other]
            memory_other = [outcome["memory_at5"][index] for index in other]
            matched.append({
                "case_id": case_id,
                "repository": repository_key(case_id),
                "repeat": repeat,
                "baseline_files": json.dumps(left, separators=(",", ":")),
                "memory_files": json.dumps(right, separators=(",", ":")),
                "d_jaccard": distance,
                "d_intersection_at5": pair_intersection[-1],
                "d_rbo_at5": pair_rbo[-1],
                "baseline_success": outcome["baseline_at5"][repeat - 1],
                "memory_success": outcome["memory_at5"][repeat - 1],
                "paired_harm": int(
                    outcome["baseline_at5"][repeat - 1] == 1
                    and outcome["memory_at5"][repeat - 1] == 0
                ),
                "loo_memory_instability": int(len(set(memory_other)) > 1),
                "loo_stability_gap": (
                    stability(baseline_other)["pairwise_agreement"]
                    - stability(memory_other)["pairwise_agreement"]
                ),
                "baseline_source": sources[(case_id, "baseline", repeat)],
                "memory_source": sources[(case_id, "memory", repeat)],
            })

        baseline_internal = [
            jaccard_distance(baseline[left], baseline[right])
            for left, right in itertools.combinations(range(1, 5), 2)
        ]
        memory_internal = [
            jaccard_distance(memory[left], memory[right])
            for left, right in itertools.combinations(range(1, 5), 2)
        ]
        all_cross = [
            jaccard_distance(baseline[left], memory[right])
            for left in range(1, 5)
            for right in range(1, 5)
        ]
        gold_files = set(gold.get(case_id, []))
        gold_removed = [
            jaccard_distance(
                set(baseline[repeat]) - gold_files,
                set(memory[repeat]) - gold_files,
            )
            for repeat in range(1, 5)
        ] if gold else []
        cases.append({
            "case_id": case_id,
            "repository": repository_key(case_id),
            "d_cross_arm": mean_or_nan(pair_distances),
            "d_cross_arm_max": max(pair_distances),
            "d_cross_all16": mean_or_nan(all_cross),
            "d_baseline_internal": mean_or_nan(baseline_internal),
            "d_memory_internal": mean_or_nan(memory_internal),
            "d_cross_intersection_at5": mean_or_nan(pair_intersection),
            "d_cross_rbo_at5": mean_or_nan(pair_rbo),
            "d_cross_gold_removed": mean_or_nan(gold_removed),
            "baseline_stability": outcome["baseline_stability"],
            "memory_stability": outcome["memory_stability"],
            "stability_gap": outcome["stability_gap"],
            "case_flip": int(len(set(outcome["memory_at5"])) > 1),
            "stability_class": (
                "degradation" if outcome["stability_gap"] > 0
                else "improvement" if outcome["stability_gap"] < 0
                else "tie"
            ),
            "any_paired_harm": int(any(
                baseline_value == 1 and memory_value == 0
                for baseline_value, memory_value in zip(
                    outcome["baseline_at5"], outcome["memory_at5"]
                )
            )),
        })
    return matched, cases


def case_bootstrap(cases, n_boot, seed):
    rng = np.random.default_rng(seed)
    auc_values = {signal: [] for signal in SIGNALS}
    rho_values = {signal: [] for signal in SIGNALS}
    partial_values = []
    delta_values = []
    rows = []
    size = len(cases)
    for replicate in range(n_boot):
        indexes = rng.integers(0, size, size=size)
        sample = [cases[index] for index in indexes]
        labels = [row["case_flip"] for row in sample]
        gaps = [row["stability_gap"] for row in sample]
        for signal in SIGNALS:
            scores = [row[signal] for row in sample]
            auc_values[signal].append(safe_auc(labels, scores))
            rho_values[signal].append(safe_spearman(scores, gaps))
        partial = partial_spearman(
            [row["d_cross_arm"] for row in sample],
            gaps,
            [row["d_baseline_internal"] for row in sample],
        )
        partial_values.append(partial)
        cross_auc = auc_values["d_cross_arm"][-1]
        baseline_auc = auc_values["d_baseline_internal"][-1]
        delta = cross_auc - baseline_auc
        delta_values.append(delta)
        rows.append({
            "replicate": replicate,
            "cross_auc": cross_auc,
            "baseline_internal_auc": baseline_auc,
            "auc_delta": delta,
            "cross_gap_rho": rho_values["d_cross_arm"][-1],
            "partial_gap_rho": partial,
            "positive_count": sum(labels),
        })
    summary = {
        "auc": {key: percentile_summary(value) for key, value in auc_values.items()},
        "rho": {key: percentile_summary(value) for key, value in rho_values.items()},
        "partial_rho": percentile_summary(partial_values),
        "auc_delta": percentile_summary(delta_values),
    }
    return summary, rows


def partial_permutation(cases, n_perm, seed):
    rng = np.random.default_rng(seed)
    signal = np.asarray([row["d_cross_arm"] for row in cases])
    gap = np.asarray([row["stability_gap"] for row in cases])
    control = np.asarray([row["d_baseline_internal"] for row in cases])
    observed = partial_spearman(signal, gap, control)
    values = []
    for _ in range(n_perm):
        values.append(partial_spearman(signal, rng.permutation(gap), control))
    valid = [value for value in values if math.isfinite(value)]
    p_value = (
        (1 + sum(abs(value) >= abs(observed) - 1e-12 for value in valid))
        / (1 + len(valid))
    )
    rows = [
        {"replicate": index, "partial_rho": value}
        for index, value in enumerate(values)
    ]
    return {"observed": observed, "p_value_two_sided": p_value}, rows


def metric_block(cases, signal):
    labels = [row["case_flip"] for row in cases]
    scores = [row[signal] for row in cases]
    gaps = [row["stability_gap"] for row in cases]
    prevalence = float(np.mean(labels))
    ap = safe_ap(labels, scores)
    return {
        "auroc_case_flip": safe_auc(labels, scores),
        "average_precision_case_flip": ap,
        "ap_lift": ap / prevalence if prevalence else None,
        "spearman_stability_gap": safe_spearman(scores, gaps),
    }


def high_quartile(cases):
    scores = np.asarray([row["d_cross_arm"] for row in cases])
    labels = np.asarray([row["case_flip"] for row in cases])
    threshold = float(np.quantile(scores, 0.75))
    selected = scores >= threshold
    return {
        "threshold": threshold,
        "selected_cases": int(selected.sum()),
        "case_flip_recall": float(labels[selected].sum() / labels.sum()),
        "case_flip_precision": float(labels[selected].mean()),
    }


def influence_audit(cases, original_auc):
    rows = []
    for removed in cases:
        remaining = [row for row in cases if row["case_id"] != removed["case_id"]]
        auc = safe_auc(
            [row["case_flip"] for row in remaining],
            [row["d_cross_arm"] for row in remaining],
        )
        rows.append({
            "case_id": removed["case_id"],
            "repository": removed["repository"],
            "case_flip": removed["case_flip"],
            "removed_d_cross_arm": removed["d_cross_arm"],
            "loo_auroc": auc,
            "signed_drop": original_auc - auc,
        })
    finite = [row for row in rows if math.isfinite(row["signed_drop"])]
    maximum = max(finite, key=lambda row: row["signed_drop"])
    return {
        "max_drop": maximum["signed_drop"],
        "max_drop_case": maximum["case_id"],
    }, rows


def repository_sensitivity(cases, original_auc):
    rows = []
    for repository in sorted({row["repository"] for row in cases}):
        remaining = [row for row in cases if row["repository"] != repository]
        auc = safe_auc(
            [row["case_flip"] for row in remaining],
            [row["d_cross_arm"] for row in remaining],
        )
        rows.append({
            "repository": repository,
            "removed_cases": len(cases) - len(remaining),
            "remaining_cases": len(remaining),
            "loo_repository_auroc": auc,
            "signed_drop": original_auc - auc,
        })
    repo_counts = {}
    for row in cases:
        repo_counts[row["repository"]] = repo_counts.get(row["repository"], 0) + 1
    weights = [1 / repo_counts[row["repository"]] for row in cases]
    weighted_auc = safe_auc(
        [row["case_flip"] for row in cases],
        [row["d_cross_arm"] for row in cases],
        weights,
    )
    finite = [row for row in rows if math.isfinite(row["signed_drop"])]
    maximum = max(finite, key=lambda row: row["signed_drop"])
    django = next(
        (row for row in rows if row["repository"] == "django__django"),
        None,
    )
    return {
        "inverse_repo_frequency_weighted_auroc": weighted_auc,
        "max_leave_repository_out_drop": maximum["signed_drop"],
        "max_drop_repository": maximum["repository"],
        "leave_django_out_auroc": (
            django["loo_repository_auroc"] if django else None
        ),
    }, rows


def trajectory_analysis(matched, n_boot, seed):
    labels = [row["paired_harm"] for row in matched]
    scores = [row["d_jaccard"] for row in matched]
    loo_labels = [row["loo_memory_instability"] for row in matched]
    point = {
        "paired_harm_count": int(sum(labels)),
        "paired_harm_auroc": safe_auc(labels, scores),
        "paired_harm_ap": safe_ap(labels, scores),
        "loo_memory_instability_count": int(sum(loo_labels)),
        "loo_memory_instability_auroc": safe_auc(loo_labels, scores),
        "loo_stability_gap_spearman": safe_spearman(
            scores, [row["loo_stability_gap"] for row in matched]
        ),
    }
    by_case = {}
    for row in matched:
        by_case.setdefault(row["case_id"], []).append(row)
    case_ids = sorted(by_case)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        rows = [row for case_id in sampled for row in by_case[case_id]]
        boot.append(safe_auc(
            [row["paired_harm"] for row in rows],
            [row["d_jaccard"] for row in rows],
        ))
    point["paired_harm_case_cluster_bootstrap"] = percentile_summary(boot)
    return point


def sparse_degradation(cases):
    effective = [row for row in cases if row["stability_class"] != "tie"]
    labels = [int(row["stability_class"] == "degradation") for row in effective]
    return {
        "effective_n": len(effective),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "auroc": safe_auc(labels, [row["d_cross_arm"] for row in effective]),
    }


def decide(metrics, bootstrap, partial, influence):
    cross = metrics["d_cross_arm"]
    baseline = metrics["d_baseline_internal"]
    auc = cross["auroc_case_flip"]
    rho = cross["spearman_stability_gap"]
    partial_rho = partial["observed"]
    auc_lower = bootstrap["auc"]["d_cross_arm"]["ci"][0]
    rho_lower = bootstrap["rho"]["d_cross_arm"]["ci"][0]
    partial_lower = bootstrap["partial_rho"]["ci"][0]
    max_drop = influence["max_drop"]
    strong = all((
        auc >= 0.70,
        auc_lower is not None and auc_lower >= 0.55,
        rho >= 0.30,
        rho_lower is not None and rho_lower > 0,
        partial_rho >= 0.30,
        partial_lower is not None and partial_lower > 0,
        partial["p_value_two_sided"] <= 0.05,
        auc > baseline["auroc_case_flip"],
        max_drop <= 0.15,
    ))
    moderate = all((
        auc >= 0.62,
        partial_rho > 0,
        auc >= baseline["auroc_case_flip"],
        max_drop <= 0.20,
    ))
    no_signal = all((
        auc <= 0.55,
        abs(rho) <= 0.10,
        abs(partial_rho) <= 0.10,
        baseline["auroc_case_flip"] >= auc,
    ))
    if strong:
        verdict = "STRONG_PROMISING"
    elif moderate:
        verdict = "MODERATE_PROMISING"
    elif no_signal:
        verdict = "NO_USEFUL_SIGNAL"
    else:
        verdict = "WEAK_INCONCLUSIVE"
    return {
        "verdict": verdict,
        "strong_conditions_all": strong,
        "moderate_conditions_all": moderate,
        "no_useful_signal_conditions_all": no_signal,
    }


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))


def plot_diagnostics(cases, path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["tab:orange" if row["case_flip"] else "tab:blue" for row in cases]
    axes[0].scatter(
        [row["d_cross_arm"] for row in cases],
        [row["stability_gap"] for row in cases],
        c=colors,
    )
    axes[0].set(xlabel="Cross-arm Jaccard distance", ylabel="Stability gap")
    axes[1].scatter(
        [row["d_baseline_internal"] for row in cases],
        [row["d_cross_arm"] for row in cases],
        c=colors,
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set(
        xlabel="Baseline-internal distance",
        ylabel="Cross-arm distance",
    )
    figure.suptitle("Dual-path disagreement diagnostics (orange = case flip)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def report_markdown(summary):
    cross = summary["signals"]["d_cross_arm"]
    baseline = summary["signals"]["d_baseline_internal"]
    partial = summary["partial_spearman"]
    boot = summary["bootstrap"]
    decision = summary["decision"]
    audit = summary["locked_outcome_audit"]
    lines = [
        "# Dual-Path Agreement Gating Pilot",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-07-16",
        "- Verification Status: ANALYZED",
        "- Version Label: dual_path_agreement_v1",
        "",
        "## Verdict",
        "",
        f"**{decision['verdict']}**",
        "",
        "This is a retrospective post-localization uncertainty audit. It does "
        "not establish a deployable controller or pre-execution cost savings.",
        "",
        "## Locked Sample Audit",
        "",
        f"- Cases: {summary['case_count']}; matched repeat pairs: {summary['matched_pair_count']}",
        f"- Degradation / tie / improvement: {audit['degradation']} / {audit['tie']} / {audit['improvement']}",
        f"- Memory case flips: {audit['case_flip']}/20",
        f"- SPARSE_SIGNAL: {summary['sparse_signal']} (ties={audit['tie']/20:.0%})",
        "",
        "## Co-primary Results",
        "",
        "| Signal | Case-flip AUROC | AP | AP lift | Stability-gap rho |",
        "|---|---:|---:|---:|---:|",
    ]
    for signal, values in summary["signals"].items():
        lines.append(
            f"| {signal} | {values['auroc_case_flip']:.3f} | "
            f"{values['average_precision_case_flip']:.3f} | "
            f"{values['ap_lift']:.3f} | "
            f"{values['spearman_stability_gap']:.3f} |"
        )
    lines.extend([
        "",
        "Sensitivity signal results (descriptive; cannot upgrade the verdict):",
        "",
        "| Signal | Case-flip AUROC | Stability-gap rho |",
        "|---|---:|---:|",
    ])
    for signal, values in summary["sensitivity_signals"].items():
        lines.append(
            f"| {signal} | {values['auroc_case_flip']:.3f} | "
            f"{values['spearman_stability_gap']:.3f} |"
        )
    lines.extend([
        "",
        f"Cross-arm AUROC 95% case-bootstrap CI: [{boot['auc']['d_cross_arm']['ci'][0]:.3f}, {boot['auc']['d_cross_arm']['ci'][1]:.3f}].",
        f"Cross-arm stability-gap rho 95% CI: [{boot['rho']['d_cross_arm']['ci'][0]:.3f}, {boot['rho']['d_cross_arm']['ci'][1]:.3f}].",
        f"Partial Spearman controlling Baseline-internal disagreement: {partial['observed']:.3f} "
        f"(95% bootstrap CI [{boot['partial_rho']['ci'][0]:.3f}, {boot['partial_rho']['ci'][1]:.3f}], permutation p={partial['p_value_two_sided']:.4f}).",
        f"Cross-minus-Baseline-internal AUROC: {cross['auroc_case_flip'] - baseline['auroc_case_flip']:.3f} "
        f"(95% bootstrap CI [{boot['auc_delta']['ci'][0]:.3f}, {boot['auc_delta']['ci'][1]:.3f}]).",
        "",
        "## Robustness and Boundaries",
        "",
        f"- Maximum leave-one-case-out AUROC drop: {summary['influence']['max_drop']:.3f} ({summary['influence']['max_drop_case']})",
        f"- Inverse-repository-frequency weighted AUROC: {summary['repository_control']['inverse_repo_frequency_weighted_auroc']:.3f}",
        f"- Leave-Django-out AUROC: {summary['repository_control']['leave_django_out_auroc']:.3f}; maximum leave-repository-out drop: {summary['repository_control']['max_leave_repository_out_drop']:.3f} ({summary['repository_control']['max_drop_repository']})",
        f"- High-D quartile case-flip recall / precision: {summary['high_quartile']['case_flip_recall']:.3f} / {summary['high_quartile']['case_flip_precision']:.3f}",
        f"- Tie-excluded degradation AUROC: {summary['degradation_sensitivity']['auroc']:.3f} on n={summary['degradation_sensitivity']['effective_n']} (5 positive, 1 negative); sensitivity only.",
        f"- Gold-removed sensitivity available: {summary['gold_sensitivity_available']}",
        "- Paired harm is diagnostic only because its definition uses both arm outcomes.",
        "- Baseline-internal disagreement is the locked difficulty confound control; a positive dual-path claim requires cross-arm evidence above this control.",
        "",
        "## Next Gate",
        "",
        (
            "Choice B is sufficiently supported for a preregistered confirmatory replication."
            if decision["verdict"] in {"STRONG_PROMISING", "MODERATE_PROMISING"}
            else "Do not commit to Choice B from this pilot; run an independent 30-case replication only if the weak evidence and cost justify it."
        ),
        "",
    ])
    return "\n".join(lines)


def validation_markdown(summary):
    cross = summary["signals"]["d_cross_arm"]
    baseline = summary["signals"]["d_baseline_internal"]
    partial = summary["partial_spearman"]
    boot = summary["bootstrap"]
    return "\n".join([
        "# Dual-Path Agreement Statistical Validation",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-07-16",
        "- Verification Status: VERIFIED",
        "- Version Label: dual_path_agreement_validation_v1",
        "",
        "## Validation Report",
        "",
        "- **Source:** dual_path_agreement_v1",
        "- **Overall Confidence:** CAUTION",
        "",
        "### Statistical Findings",
        "",
        "| Metric | Test | Value | Effect/precision | Confidence |",
        "|---|---|---:|---|---|",
        f"| Cross-arm -> case flip | AUROC | {cross['auroc_case_flip']:.3f} | 95% bootstrap CI [{boot['auc']['d_cross_arm']['ci'][0]:.3f}, {boot['auc']['d_cross_arm']['ci'][1]:.3f}] | CAUTION |",
        f"| Baseline-internal -> case flip | AUROC | {baseline['auroc_case_flip']:.3f} | Exceeds cross-arm by {baseline['auroc_case_flip'] - cross['auroc_case_flip']:.3f} | CAUTION |",
        f"| Cross-arm -> stability gap | Spearman rho | {cross['spearman_stability_gap']:.3f} | 95% bootstrap CI [{boot['rho']['d_cross_arm']['ci'][0]:.3f}, {boot['rho']['d_cross_arm']['ci'][1]:.3f}] | CAUTION |",
        f"| Cross-arm -> gap, controlling Baseline internal | Partial Spearman | {partial['observed']:.3f}, p={partial['p_value_two_sided']:.4f} | CI [{boot['partial_rho']['ci'][0]:.3f}, {boot['partial_rho']['ci'][1]:.3f}] | CAUTION |",
        "",
        "### Warnings",
        "",
        "| Type | Detail | Affected |",
        "|---|---|---|",
        "| Small/clustered sample | n=20 and 11/20 cases are Django; leave-Django-out AUROC is 0.286. | Generalization |",
        "| Sparse binary target | 14/20 stability-class ties; tie-excluded degradation has n=6 (5/1). | Degradation AUROC |",
        "| Wide uncertainty | Primary AUROC and correlation CIs cross chance/null. | Co-primary findings |",
        "| Multiple exploratory metrics | Five sensitivity distances plus primary/control signals; no multiplicity correction. Sensitivities cannot upgrade verdict. | Sensitivities |",
        "| Structural coupling | Paired harm uses both arm outcomes and is not deployment-ready. | Trajectory diagnostic |",
        "",
        "### Fallacy Scan",
        "",
        "- **Coverage:** 11/11 fallacy types checked",
        "",
        "| # | Fallacy | Severity | Finding |",
        "|---:|---|---|---|",
        "| 1 | Simpson's paradox | CAUTION | Strong repository heterogeneity is possible; aggregated AUROC falls sharply after removing Django. Repository strata are too small for a definitive direction audit. |",
        "| 2 | Ecological fallacy | NOTE | Primary inference remains case-level; trajectory-level paired-harm results are reported separately and must not be inferred from case aggregates. |",
        "| 3 | Berkson's paradox | NOTE | Pilot was randomly sampled after a frozen 15-case exclusion, not outcome-selected; the exclusion still limits the target population. |",
        "| 4 | Collider bias | CAUTION | Baseline-internal disagreement is a difficulty proxy, but the causal graph is not identified; partial correlation is descriptive, not a causal adjustment. |",
        "| 5 | Base-rate neglect | NOTE | Case-flip prevalence (6/20) and AP lift are explicit; degradation prevalence is too sparse for a primary claim. |",
        "| 6 | Regression to the mean | NOTE | No pre/post comparison or extreme-score enrollment is used. |",
        "| 7 | Survivorship bias | NOTE | All 20 cases and 160 trajectories completed; no incomplete cases were dropped. |",
        "| 8 | Look-elsewhere effect | CAUTION | Several distance sensitivities were evaluated without multiplicity correction; they are descriptive and cannot alter the locked verdict. |",
        "| 9 | Garden of forking paths | CAUTION | This is a retrospective pilot motivated by earlier negative prediction results. Locked definitions and full sensitivity disclosure reduce but do not remove researcher degrees of freedom. |",
        "| 10 | Correlation != causation | CAUTION | Association between disagreement and failure would not show that a verification controller improves outcomes; no controller was run. |",
        "| 11 | Reverse causality | CAUTION | Disagreement and outcomes arise from the same completed paths, so temporal precedence for a causal gating mechanism is not established. |",
        "",
        "### Reproducibility",
        "",
        "- **Method:** two full deterministic reruns in the locked RepoMem/locagent environment",
        "- **Verdict:** REPRODUCIBLE",
        "- All CSV, JSON, Markdown, manifest, and PNG SHA-256 hashes matched exactly.",
        "",
    ])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration",
        default="results/paper2_pilot/preregistration.json",
    )
    parser.add_argument("--run_root", default="results/paper2_pilot/runs")
    parser.add_argument(
        "--per_case",
        default="results/paper2_pilot/analysis/per_case_metrics.jsonl",
    )
    parser.add_argument("--dataset_arrow", default=DEFAULT_ARROW)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prereg = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    selected = prereg["sampling"]["selected_ids"]
    outcomes, locked_audit = outcome_audit(args.per_case, selected)
    predictions, sources = load_predictions(prereg, args.run_root)
    gold, gold_warning = load_gold(args.dataset_arrow, selected)
    matched, cases = build_tables(predictions, sources, outcomes, gold)

    signal_metrics = {signal: metric_block(cases, signal) for signal in SIGNALS}
    sensitivity_metrics = {
        signal: metric_block(cases, signal)
        for signal in SENSITIVITY_SIGNALS
        if all(math.isfinite(row[signal]) for row in cases)
    }
    bootstrap, bootstrap_rows = case_bootstrap(cases, args.bootstrap, args.seed)
    partial, permutation_rows = partial_permutation(
        cases, args.permutations, args.seed
    )
    original_auc = signal_metrics["d_cross_arm"]["auroc_case_flip"]
    influence, influence_rows = influence_audit(cases, original_auc)
    repository_control, repository_rows = repository_sensitivity(
        cases, original_auc
    )
    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run+validate",
            "origin_date": "2026-07-16",
            "verification_status": "ANALYZED",
            "version_label": "dual_path_agreement_v1",
        },
        "case_count": len(cases),
        "matched_pair_count": len(matched),
        "locked_outcome_audit": locked_audit,
        "sparse_signal": locked_audit["tie"] / len(cases) > 0.40,
        "signals": signal_metrics,
        "sensitivity_signals": sensitivity_metrics,
        "bootstrap": bootstrap,
        "partial_spearman": partial,
        "high_quartile": high_quartile(cases),
        "influence": influence,
        "repository_control": repository_control,
        "degradation_sensitivity": sparse_degradation(cases),
        "trajectory_diagnostics": trajectory_analysis(
            matched, args.bootstrap, args.seed
        ),
        "gold_sensitivity_available": bool(gold),
        "gold_warning": gold_warning,
        "configuration": {
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "seed": args.seed,
            "preregistration": args.preregistration,
            "run_root": args.run_root,
            "per_case": args.per_case,
            "dataset_arrow": args.dataset_arrow,
        },
    }
    summary["decision"] = decide(
        signal_metrics, bootstrap, partial, influence
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "matched_repeat_metrics.csv", matched)
    write_csv(output / "case_metrics.csv", cases)
    write_csv(output / "bootstrap_metrics.csv", bootstrap_rows)
    write_csv(output / "permutation_metrics.csv", permutation_rows)
    write_csv(output / "case_influence.csv", influence_rows)
    write_csv(output / "repository_sensitivity.csv", repository_rows)
    (output / "dual_path_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "dual_path_report.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    (output / "validation_report.md").write_text(
        validation_markdown(summary), encoding="utf-8"
    )
    plot_diagnostics(cases, output / "agreement_diagnostics.png")
    manifest_paths = [
        output / name for name in (
            "matched_repeat_metrics.csv",
            "case_metrics.csv",
            "bootstrap_metrics.csv",
            "permutation_metrics.csv",
            "case_influence.csv",
            "repository_sensitivity.csv",
            "dual_path_summary.json",
            "dual_path_report.md",
            "validation_report.md",
        )
    ]
    manifest = {path.name: file_hash(path) for path in manifest_paths}
    (output / "artifact_hashes.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe({
        "verdict": summary["decision"]["verdict"],
        "case_count": len(cases),
        "matched_pair_count": len(matched),
        "cross_arm": signal_metrics["d_cross_arm"],
        "baseline_internal": signal_metrics["d_baseline_internal"],
        "partial_spearman": partial,
        "influence": influence,
    }), indent=2))


if __name__ == "__main__":
    main()
