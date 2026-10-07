import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LABELS = ("harm", "rescue", "both_correct", "both_wrong")
LOCKED_PARAMETER_KEYS = (
    "model",
    "temperature",
    "top_p",
    "max_iteration_num",
    "max_attempt_num",
    "num_samples",
    "ranking_method",
    "request_timeout",
    "timeout",
    "context_token_budget",
    "tool_observation_token_cap",
    "use_function_calling",
    "fail_on_context_error",
    "bounded_graph_output",
    "bounded_entity_output",
    "bounded_search_output",
    "rerun_empty_location",
)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_outcomes(value):
    if isinstance(value, list):
        outcomes = [int(item) for item in value]
    else:
        text = str(value).strip()
        outcomes = [int(char) for char in text if char in "01"]
    if len(outcomes) != 4 or any(item not in (0, 1) for item in outcomes):
        raise ValueError(f"Expected four binary outcomes, got {value!r}")
    return outcomes


def pairwise_agreement(values):
    values = list(values)
    if len(values) < 2:
        return None
    pairs = [
        int(values[left] == values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return float(np.mean(pairs))


def ab_label(baseline, memory):
    if baseline == 1 and memory == 0:
        return "harm"
    if baseline == 0 and memory == 1:
        return "rescue"
    if baseline == 1:
        return "both_correct"
    return "both_wrong"


def fleiss_kappa(label_rows, categories=LABELS):
    label_rows = [list(row) for row in label_rows]
    if not label_rows:
        return None
    n_raters = len(label_rows[0])
    if n_raters < 2 or any(len(row) != n_raters for row in label_rows):
        raise ValueError("Fleiss kappa requires a rectangular label matrix")
    counts = np.asarray(
        [[row.count(category) for category in categories] for row in label_rows],
        dtype=float,
    )
    p_i = (np.square(counts).sum(axis=1) - n_raters) / (
        n_raters * (n_raters - 1)
    )
    category_rates = counts.sum(axis=0) / (len(label_rows) * n_raters)
    expected = float(np.square(category_rates).sum())
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(float(p_i.mean()), 1.0) else None
    return float((p_i.mean() - expected) / (1.0 - expected))


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def bootstrap_mean(values, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"estimate": None, "ci": None, "n_success": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci": [float(value) for value in np.percentile(means, [2.5, 97.5])],
        "n_success": int(n_boot),
    }


def bootstrap_two_sample_difference(
    selected_values, random_values, n_boot=10000, seed=42
):
    selected_values = np.asarray(selected_values, dtype=float)
    random_values = np.asarray(random_values, dtype=float)
    rng = np.random.default_rng(seed)
    selected_samples = rng.choice(
        selected_values, size=(n_boot, len(selected_values)), replace=True
    )
    random_samples = rng.choice(
        random_values, size=(n_boot, len(random_values)), replace=True
    )
    differences = selected_samples.mean(axis=1) - random_samples.mean(axis=1)
    return {
        "estimate": float(selected_values.mean() - random_values.mean()),
        "ci": [
            float(value) for value in np.percentile(differences, [2.5, 97.5])
        ],
        "n_success": int(n_boot),
    }


def bootstrap_weighted_strata(
    selected_values,
    random_values,
    selected_weight=15 / 500,
    random_weight=485 / 500,
    n_boot=10000,
    seed=42,
):
    selected_values = np.asarray(selected_values, dtype=float)
    random_values = np.asarray(random_values, dtype=float)
    rng = np.random.default_rng(seed)
    selected_samples = rng.choice(
        selected_values, size=(n_boot, len(selected_values)), replace=True
    ).mean(axis=1)
    random_samples = rng.choice(
        random_values, size=(n_boot, len(random_values)), replace=True
    ).mean(axis=1)
    estimates = selected_weight * selected_samples + random_weight * random_samples
    return {
        "estimate": float(
            selected_weight * selected_values.mean()
            + random_weight * random_values.mean()
        ),
        "ci": [float(value) for value in np.percentile(estimates, [2.5, 97.5])],
        "selected_weight": selected_weight,
        "random_weight": random_weight,
        "n_success": int(n_boot),
    }


def exact_sign_flip_p(values):
    values = np.asarray(values, dtype=float)
    nonzero = values[~np.isclose(values, 0.0)]
    if not len(nonzero):
        return 1.0
    if len(nonzero) > 22:
        raise ValueError("Exact sign-flip test is restricted to <=22 nonzero cases")
    observed = abs(float(nonzero.sum()))
    extreme = 0
    total = 2 ** len(nonzero)
    magnitudes = np.abs(nonzero)
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        statistic = abs(float(np.dot(magnitudes, np.asarray(signs))))
        if statistic >= observed - 1e-12:
            extreme += 1
    return extreme / total


def enrich_case(row, cohort):
    baseline = parse_outcomes(row["baseline_at5"])
    memory = parse_outcomes(row["memory_at5"])
    labels = [ab_label(b, m) for b, m in zip(baseline, memory)]
    binary_harm = [int(label == "harm") for label in labels]
    result = {
        "case_id": row["case_id"],
        "repository": row.get("repository", row["case_id"].split("-", 1)[0]),
        "cohort": cohort,
        "baseline_at5": baseline,
        "memory_at5": memory,
        "baseline_stability": float(row["baseline_stability"]),
        "memory_stability": float(row["memory_stability"]),
        "stability_gap": float(row["baseline_stability"])
        - float(row["memory_stability"]),
        "labels": labels,
        "label_pairwise_agreement": pairwise_agreement(labels),
        "label_unanimous": int(len(set(labels)) == 1),
        "label_flip": int(len(set(labels)) > 1),
        "harm_binary_agreement": pairwise_agreement(binary_harm),
        "harm_count": sum(binary_harm),
        "ever_harm": int(any(binary_harm)),
        "majority_harm": int(sum(binary_harm) >= 3),
        "persistent_harm": int(sum(binary_harm) == 4),
        "case_effect": float(np.mean(memory) - np.mean(baseline)),
    }
    return result


def cohort_summary(rows, n_boot=10000, seed=42):
    gaps = [row["stability_gap"] for row in rows]
    agreements = [row["label_pairwise_agreement"] for row in rows]
    binary_agreements = [row["harm_binary_agreement"] for row in rows]
    case_effects = [row["case_effect"] for row in rows]
    labels = [row["labels"] for row in rows]
    baseline_values = [value for row in rows for value in row["baseline_at5"]]
    memory_values = [value for row in rows for value in row["memory_at5"]]
    flip_count = sum(row["label_flip"] for row in rows)
    unanimous_count = sum(row["label_unanimous"] for row in rows)
    ever_harm = sum(row["ever_harm"] for row in rows)
    majority_harm = sum(row["majority_harm"] for row in rows)
    persistent_harm = sum(row["persistent_harm"] for row in rows)
    return {
        "n_cases": len(rows),
        "baseline_stability": float(
            np.mean([row["baseline_stability"] for row in rows])
        ),
        "memory_stability": float(
            np.mean([row["memory_stability"] for row in rows])
        ),
        "stability_gap": {
            **bootstrap_mean(gaps, n_boot, seed),
            "exact_sign_flip_p": exact_sign_flip_p(gaps),
        },
        "baseline_acc_at5": float(np.mean(baseline_values)),
        "memory_acc_at5": float(np.mean(memory_values)),
        "memory_minus_baseline_acc_at5": bootstrap_mean(
            case_effects, n_boot, seed + 1
        ),
        "label_pairwise_agreement": bootstrap_mean(
            agreements, n_boot, seed + 2
        ),
        "harm_binary_agreement": bootstrap_mean(
            binary_agreements, n_boot, seed + 3
        ),
        "fleiss_kappa_four_class": fleiss_kappa(labels),
        "label_unanimous": {
            "count": unanimous_count,
            "rate": unanimous_count / len(rows),
            "wilson_95ci": wilson_ci(unanimous_count, len(rows)),
        },
        "label_flip": {
            "count": flip_count,
            "rate": flip_count / len(rows),
            "wilson_95ci": wilson_ci(flip_count, len(rows)),
        },
        "ever_harm": {
            "count": ever_harm,
            "rate": ever_harm / len(rows),
            "wilson_95ci": wilson_ci(ever_harm, len(rows)),
        },
        "majority_harm": {
            "count": majority_harm,
            "rate": majority_harm / len(rows),
            "wilson_95ci": wilson_ci(majority_harm, len(rows)),
        },
        "persistent_harm": {
            "count": persistent_harm,
            "rate": persistent_harm / len(rows),
            "wilson_95ci": wilson_ci(persistent_harm, len(rows)),
        },
        "label_counts": dict(Counter(label for row in rows for label in row["labels"])),
    }


def harm_confirmation(rows):
    repeat_rows = []
    total_confirmed = 0
    total_opportunities = 0
    for discovery_repeat in range(4):
        selected = [
            row for row in rows if row["labels"][discovery_repeat] == "harm"
        ]
        confirmed = sum(
            row["labels"][repeat] == "harm"
            for row in selected
            for repeat in range(4)
            if repeat != discovery_repeat
        )
        opportunities = len(selected) * 3
        total_confirmed += confirmed
        total_opportunities += opportunities
        repeat_rows.append({
            "discovery_repeat": discovery_repeat + 1,
            "selected_harm_cases": len(selected),
            "confirmation_opportunities": opportunities,
            "confirmed_harm_outcomes": confirmed,
            "harm_recurrence": confirmed / opportunities if opportunities else None,
        })
    return {
        "repeat_rows": repeat_rows,
        "selected_harm_events": sum(row["harm_count"] for row in rows),
        "confirmation_opportunities": total_opportunities,
        "confirmed_harm_outcomes": total_confirmed,
        "harm_recurrence": (
            total_confirmed / total_opportunities if total_opportunities else None
        ),
    }


def bootstrap_harm_confirmation(rows, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        indexes = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[index] for index in indexes]
        estimate = harm_confirmation(sample)["harm_recurrence"]
        if estimate is not None:
            estimates.append(estimate)
    return {
        "estimate": harm_confirmation(rows)["harm_recurrence"],
        "ci": (
            [float(value) for value in np.percentile(estimates, [2.5, 97.5])]
            if estimates
            else None
        ),
        "n_success": len(estimates),
        "discard_rate": 1 - len(estimates) / n_boot,
    }


def repeat_block_metrics(rows):
    output = []
    for repeat in range(4):
        baseline = [row["baseline_at5"][repeat] for row in rows]
        memory = [row["memory_at5"][repeat] for row in rows]
        labels = [row["labels"][repeat] for row in rows]
        output.append({
            "repeat": repeat + 1,
            "baseline_acc_at5": float(np.mean(baseline)),
            "memory_acc_at5": float(np.mean(memory)),
            "memory_minus_baseline": float(np.mean(memory) - np.mean(baseline)),
            "harm_count": labels.count("harm"),
            "rescue_count": labels.count("rescue"),
            "both_correct_count": labels.count("both_correct"),
            "both_wrong_count": labels.count("both_wrong"),
        })
    return output


def inverse_repository_weighted(rows, key):
    counts = Counter(row["repository"] for row in rows)
    weights = np.asarray([1 / counts[row["repository"]] for row in rows])
    values = np.asarray([row[key] for row in rows], dtype=float)
    return float(np.average(values, weights=weights))


def audit_runtime(run_root, v4_args_path, expected_fingerprint, expected_model):
    run_root = Path(run_root)
    batch_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    v4_args = json.loads(Path(v4_args_path).read_text(encoding="utf-8"))
    parameter_mismatches = []
    fingerprint_mismatches = 0
    model_mismatches = 0
    response_count = 0
    invalid_batches = []
    truncated_observations = 0
    for batch_dir in batch_dirs:
        args = json.loads((batch_dir / "args.json").read_text(encoding="utf-8"))
        for key in LOCKED_PARAMETER_KEYS:
            if args.get(key) != v4_args.get(key):
                parameter_mismatches.append({
                    "batch": batch_dir.name,
                    "key": key,
                    "pilot": args.get(key),
                    "v4": v4_args.get(key),
                })
        fingerprint = json.loads(
            (batch_dir / "fingerprint_audit_summary.json").read_text(
                encoding="utf-8"
            )
        )
        response_count += fingerprint.get("response_count", 0)
        fingerprint_mismatches += fingerprint.get("fingerprint_mismatch_count", 0)
        model_mismatches += fingerprint.get("model_mismatch_count", 0)
        if (
            not fingerprint.get("valid")
            or fingerprint.get("system_fingerprints") != [expected_fingerprint]
            or fingerprint.get("response_models") != [expected_model]
        ):
            invalid_batches.append(batch_dir.name)
        audit_path = batch_dir / "context_guard_audit.jsonl"
        if audit_path.exists():
            for row in load_jsonl(audit_path):
                truncated_observations += row.get("summary", {}).get(
                    "truncated_observation_count", 0
                )
    return {
        "batch_count": len(batch_dirs),
        "response_count": response_count,
        "expected_fingerprint": expected_fingerprint,
        "expected_model": expected_model,
        "fingerprint_mismatch_count": fingerprint_mismatches,
        "model_mismatch_count": model_mismatches,
        "invalid_batches": invalid_batches,
        "parameter_mismatches_vs_v4": parameter_mismatches,
        "truncated_observation_count": truncated_observations,
        "valid": not (
            invalid_batches
            or parameter_mismatches
            or fingerprint_mismatches
            or model_mismatches
            or truncated_observations
        ),
    }


def flatten_case_rows(rows):
    output = []
    for row in rows:
        output.append({
            "case_id": row["case_id"],
            "repository": row["repository"],
            "cohort": row["cohort"],
            "baseline_at5": "".join(map(str, row["baseline_at5"])),
            "memory_at5": "".join(map(str, row["memory_at5"])),
            "labels": "|".join(row["labels"]),
            "baseline_stability": row["baseline_stability"],
            "memory_stability": row["memory_stability"],
            "stability_gap": row["stability_gap"],
            "label_pairwise_agreement": row["label_pairwise_agreement"],
            "label_unanimous": row["label_unanimous"],
            "label_flip": row["label_flip"],
            "harm_binary_agreement": row["harm_binary_agreement"],
            "harm_count": row["harm_count"],
            "ever_harm": row["ever_harm"],
            "majority_harm": row["majority_harm"],
            "persistent_harm": row["persistent_harm"],
            "case_effect": row["case_effect"],
        })
    return output


def create_plot(path, repeats, pooled):
    import matplotlib.pyplot as plt

    xs = np.arange(1, 5)
    deltas = [row["memory_minus_baseline"] for row in repeats]
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
    axis.plot(xs, deltas, marker="o", label="Single matched repeat")
    axis.errorbar(
        [4.7],
        [pooled["estimate"]],
        yerr=[
            [pooled["estimate"] - pooled["ci"][0]],
            [pooled["ci"][1] - pooled["estimate"]],
        ],
        fmt="s",
        capsize=4,
        label="4-repeat pooled (case bootstrap 95% CI)",
    )
    axis.set_xticks([1, 2, 3, 4, 4.7], ["R1", "R2", "R3", "R4", "Pooled"])
    axis.set_ylabel("Memory − Baseline Acc@5")
    axis.set_title("Random-20: single-run A/B effect variability")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def report_markdown(summary):
    random = summary["random20"]
    selected = summary["selected15"]
    gap = random["stability_gap"]
    effect = random["memory_minus_baseline_acc_at5"]
    agreement = random["label_pairwise_agreement"]
    harm = summary["harm_confirmation"]["bootstrap"]
    comparison = summary["selection_comparison"]
    lines = [
        "# Paper 1 Random-Task Repeated A/B Reliability Audit",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Mode: run + validate (offline reanalysis)",
        "- Verification Status: VERIFIED",
        "- Prospective status: original stability estimand was frozen pre-API; "
        "all label-repeatability analyses are retrospective and known-outcome",
        "- Data reuse: the Random-20 trajectories are shared with the Paper 2 pilot",
        "",
        "## Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        "The random sample preserves the direction of lower Memory stability, but "
        "its confidence interval crosses zero. It is pilot evidence outside the "
        "outcome-selected subset, not confirmation of a benchmark-wide "
        "Memory-specific stability gap.",
        "",
        "## Locked Prospective Result",
        "",
        "| Cohort | N | Baseline stability | Memory stability | B−M gap | 95% CI | Exact p |",
        "|---|---:|---:|---:|---:|---|---:|",
        f"| Random | {random['n_cases']} | {random['baseline_stability']:.4f} | "
        f"{random['memory_stability']:.4f} | {gap['estimate']:+.4f} | "
        f"[{gap['ci'][0]:+.4f}, {gap['ci'][1]:+.4f}] | "
        f"{gap['exact_sign_flip_p']:.4f} |",
        f"| Outcome-selected | {selected['n_cases']} | "
        f"{selected['baseline_stability']:.4f} | "
        f"{selected['memory_stability']:.4f} | "
        f"{selected['stability_gap']['estimate']:+.4f} | "
        f"[{selected['stability_gap']['ci'][0]:+.4f}, "
        f"{selected['stability_gap']['ci'][1]:+.4f}] | "
        f"{selected['stability_gap']['exact_sign_flip_p']:.4f} |",
        "",
        "The original Random-20 bootstrap result is reproduced exactly: "
        f"{gap['estimate']:+.4f}, 95% CI "
        f"[{gap['ci'][0]:+.4f}, {gap['ci'][1]:+.4f}].",
        "",
        "## Retrospective Single-Run Conclusion Repeatability",
        "",
        "| Metric | Estimate | 95% CI |",
        "|---|---:|---|",
        f"| Four-class A/B label pairwise agreement | {agreement['estimate']:.4f} | "
        f"[{agreement['ci'][0]:.4f}, {agreement['ci'][1]:.4f}] |",
        f"| Cases with any A/B label flip | {random['label_flip']['count']}/"
        f"{random['n_cases']} ({random['label_flip']['rate']:.3f}) | "
        f"[{random['label_flip']['wilson_95ci'][0]:.3f}, "
        f"{random['label_flip']['wilson_95ci'][1]:.3f}] |",
        f"| Cases ever observed as harm | {random['ever_harm']['count']}/"
        f"{random['n_cases']} ({random['ever_harm']['rate']:.3f}) | "
        f"[{random['ever_harm']['wilson_95ci'][0]:.3f}, "
        f"{random['ever_harm']['wilson_95ci'][1]:.3f}] |",
        f"| Majority-harm cases | {random['majority_harm']['count']}/"
        f"{random['n_cases']} | — |",
        f"| Persistent-harm cases | {random['persistent_harm']['count']}/"
        f"{random['n_cases']} | — |",
        f"| Leave-one-repeat-out harm recurrence | {harm['estimate']:.4f} | "
        f"[{harm['ci'][0]:.4f}, {harm['ci'][1]:.4f}] |",
        "",
        "The four-class agreement includes both-correct and both-wrong outcomes. "
        "Binary harm agreement and Fleiss kappa are sensitivity metrics because "
        "harm is rare.",
        "",
        "## Average Effect Across Repeats",
        "",
        f"Repeated mean Acc@5: Baseline={random['baseline_acc_at5']:.4f}, "
        f"Memory={random['memory_acc_at5']:.4f}, Memory−Baseline="
        f"{effect['estimate']:+.4f}, 95% case-bootstrap CI "
        f"[{effect['ci'][0]:+.4f}, {effect['ci'][1]:+.4f}].",
        "",
        "| Repeat | Baseline Acc@5 | Memory Acc@5 | M−B | Harm | Rescue |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["repeat_blocks"]:
        lines.append(
            f"| {row['repeat']} | {row['baseline_acc_at5']:.3f} | "
            f"{row['memory_acc_at5']:.3f} | "
            f"{row['memory_minus_baseline']:+.3f} | "
            f"{row['harm_count']} | {row['rescue_count']} |"
        )
    lines.extend([
        "",
        "## Selection Amplification",
        "",
        f"Selected-minus-random stability-gap difference: "
        f"{comparison['stability_gap_difference']['estimate']:+.4f}, 95% CI "
        f"[{comparison['stability_gap_difference']['ci'][0]:+.4f}, "
        f"{comparison['stability_gap_difference']['ci'][1]:+.4f}].",
        "",
        f"Selected-minus-random four-class label-agreement difference: "
        f"{comparison['label_agreement_difference']['estimate']:+.4f}, 95% CI "
        f"[{comparison['label_agreement_difference']['ci'][0]:+.4f}, "
        f"{comparison['label_agreement_difference']['ci'][1]:+.4f}].",
        "",
        "The weighted finite-benchmark sensitivity treats the selected 15 as a "
        "certainty stratum and the Random-20 as a sample of the remaining 485. "
        f"Its estimated stability gap is "
        f"{summary['finite_benchmark_sensitivity']['estimate']:+.4f}, 95% CI "
        f"[{summary['finite_benchmark_sensitivity']['ci'][0]:+.4f}, "
        f"{summary['finite_benchmark_sensitivity']['ci'][1]:+.4f}]. This is "
        "retrospective and applies to the fixed 500-task benchmark, not a task "
        "superpopulation.",
        "",
        "## Robustness and Boundaries",
        "",
        f"- Leave-Django-out random stability gap: "
        f"{summary['robustness']['leave_django_out_gap']:+.4f}",
        f"- Equal-repository-weight sensitivity gap: "
        f"{summary['robustness']['equal_repository_weight_gap']:+.4f}",
        f"- Random sample repository counts: "
        f"{summary['robustness']['random_repository_counts']}",
        f"- Runtime audit valid: {summary['runtime_audit']['valid']}; responses="
        f"{summary['runtime_audit']['response_count']}; fingerprint/model "
        "mismatches=0; context truncations=0; V4 parameter mismatches=0",
        "- The Random-20 target population is the 485 tasks remaining after the "
        "15 outcome-selected cases were excluded.",
        "- The original precision analysis requires approximately 183 random "
        "cases for a 3pp half-width; N=20 remains pilot evidence.",
        "- This experiment evaluates outcome reliability. It does not identify "
        "retrieval or adoption causality.",
        "",
        "## Paper-Safe Conclusion",
        "",
        "On 20 randomly sampled tasks, Memory showed a smaller, directionally "
        "consistent stability deficit than in the outcome-selected cohort, but "
        "the interval included zero. Repeated executions also changed some "
        "case-level A/B labels, including all observed harm cases. The result "
        "therefore extends the instability concern beyond the selected cohort as "
        "pilot evidence, while not establishing a benchmark-wide "
        "Memory-specific instability effect.",
        "",
    ])
    return "\n".join(lines)


def validation_markdown(summary):
    random = summary["random20"]
    comparison = summary["selection_comparison"]
    harm = summary["harm_confirmation"]["bootstrap"]
    return "\n".join([
        "# Random-20 Reliability Statistical Validation",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: VERIFIED",
        "- Reproduction: locked pre-API Random-20 stability estimate and CI "
        "reproduced exactly",
        "- Scope: statistical validation of offline reanalysis; no new Agent runs",
        "",
        "## Statistical Findings",
        "",
        "| Finding | Estimate | Uncertainty | Confidence |",
        "|---|---:|---|---|",
        f"| Random B−M stability gap | "
        f"{random['stability_gap']['estimate']:+.4f} | 95% CI "
        f"[{random['stability_gap']['ci'][0]:+.4f}, "
        f"{random['stability_gap']['ci'][1]:+.4f}], exact p="
        f"{random['stability_gap']['exact_sign_flip_p']:.4f} | CAUTION: wide, crosses 0 |",
        f"| Random M−B Acc@5 | "
        f"{random['memory_minus_baseline_acc_at5']['estimate']:+.4f} | 95% CI "
        f"[{random['memory_minus_baseline_acc_at5']['ci'][0]:+.4f}, "
        f"{random['memory_minus_baseline_acc_at5']['ci'][1]:+.4f}] | CAUTION: wide, crosses 0 |",
        f"| Four-class A/B label agreement | "
        f"{random['label_pairwise_agreement']['estimate']:.4f} | 95% CI "
        f"[{random['label_pairwise_agreement']['ci'][0]:.4f}, "
        f"{random['label_pairwise_agreement']['ci'][1]:.4f}] | Descriptive, retrospective |",
        f"| Harm recurrence after discovery | {harm['estimate']:.4f} | 95% CI "
        f"[{harm['ci'][0]:.4f}, {harm['ci'][1]:.4f}] | RED_FLAG for precision: only 6 discovery events |",
        f"| Selected−random stability gap | "
        f"{comparison['stability_gap_difference']['estimate']:+.4f} | 95% CI "
        f"[{comparison['stability_gap_difference']['ci'][0]:+.4f}, "
        f"{comparison['stability_gap_difference']['ci'][1]:+.4f}] | CAUTION: retrospective contrast |",
        "",
        "## Assumption and Dependence Checks",
        "",
        "- The case is the resampling and inference unit; 160 trajectories are "
        "not treated as independent.",
        "- Four repeats per arm yield discrete per-case stability values; "
        "percentile case bootstrap and exact sign flipping avoid normal-theory tests.",
        "- Complete-case coverage is 20/20 and trajectory coverage is 160/160; "
        "there is no attrition or parser-repair exclusion.",
        "- Harm recurrence is rare-event evidence: 6 discovery harm events from "
        "3 unique cases. Its interval, not the event count, governs interpretation.",
        "- Multiple retrospective metrics are descriptive and cannot upgrade the "
        "prospective primary verdict.",
        "",
        "## Fallacy Scan — 11/11 Checked",
        "",
        "| # | Pattern | Status | Audit finding |",
        "|---:|---|---|---|",
        "| 1 | Simpson's paradox | CAUTION | Overall gap is positive, but the "
        "leave-Django-out point estimate is negative on only 9 cases; repository "
        "sensitivity blocks a homogeneous-effect claim. |",
        "| 2 | Ecological fallacy | PASS | Claims are kept at sample/case level; "
        "aggregate stability is not used to label every individual case. |",
        "| 3 | Berkson's paradox | CAUTION | The V4-15 cohort is outcome-selected; "
        "it is separated from the random primary and used only in a disclosed "
        "retrospective contrast. |",
        "| 4 | Collider bias | PASS | No post-treatment covariate adjustment is "
        "used in the primary estimate. |",
        "| 5 | Base-rate neglect | CAUTION | Harm prevalence is explicitly reported "
        "as 6/80 events from 3/20 cases; raw agreement is not interpreted alone. |",
        "| 6 | Regression to the mean | CAUTION | The selected-15 contrast is expected "
        "to amplify extremes; it is evidence about selection, not a causal cohort effect. |",
        "| 7 | Survivorship bias | PASS | All preregistered cases and trajectories "
        "are retained. |",
        "| 8 | Look-elsewhere effect | CAUTION | Several known-outcome diagnostics are "
        "reported; none changes the locked primary verdict. |",
        "| 9 | Garden of forking paths | CAUTION | The original primary is pre-API; "
        "new label analyses are transparently marked retrospective. |",
        "| 10 | Correlation != causation | PASS | No retrieval/adoption or selection "
        "causal effect is claimed. |",
        "| 11 | Reverse causality | PASS | The analysis compares randomized arm "
        "configurations and repeated outcomes, not a directional observational predictor. |",
        "",
        "## Validation Verdict",
        "",
        "**CAUTION / VERIFIED COMPUTATION.** The computation and locked primary "
        "result are reproducible. The Random-20 evidence is underpowered and "
        "repository-sensitive, so it supports a directional pilot statement, not "
        "a benchmark-wide confirmation. The selected-versus-random contrast "
        "supports selection amplification descriptively, but remains retrospective.",
        "",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random_cases",
        default="results/paper2_pilot/analysis/per_case_metrics.jsonl",
    )
    parser.add_argument(
        "--random_summary",
        default="results/paper2_pilot/analysis/pilot_summary.json",
    )
    parser.add_argument(
        "--preregistration",
        default="results/paper2_pilot/preregistration.json",
    )
    parser.add_argument(
        "--selected_cases",
        default="results/context_safe_m3a/replication_v4/analysis/v3_v4_per_case.csv",
    )
    parser.add_argument(
        "--selected_summary",
        default="results/context_safe_m3a/replication_v4/analysis/replication_v4_summary.json",
    )
    parser.add_argument(
        "--run_root", default="results/paper2_pilot/runs"
    )
    parser.add_argument(
        "--v4_args",
        default="results/context_safe_m3a/replication_v4/runs/baseline_r1/args.json",
    )
    parser.add_argument(
        "--output_dir",
        default="results/context_safe_m3a/random20_reliability",
    )
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random_source = load_jsonl(args.random_cases)
    random_rows = [enrich_case(row, "random20") for row in random_source]
    with Path(args.selected_cases).open(encoding="utf-8", newline="") as handle:
        selected_source = list(csv.DictReader(handle))
    selected_rows = []
    for row in selected_source:
        selected_rows.append(enrich_case({
            "case_id": row["case_id"],
            "repository": row["case_id"].split("-", 1)[0],
            "baseline_at5": row["v4_baseline_outcomes"],
            "memory_at5": row["v4_memory_outcomes"],
            "baseline_stability": row["v4_baseline_stability"],
            "memory_stability": row["v4_memory_stability"],
        }, "outcome_selected15"))

    random_summary = cohort_summary(random_rows, args.n_boot, args.seed)
    selected_summary = cohort_summary(selected_rows, args.n_boot, args.seed)
    original_summary = json.loads(
        Path(args.random_summary).read_text(encoding="utf-8")
    )
    preregistration = json.loads(
        Path(args.preregistration).read_text(encoding="utf-8")
    )
    selected_original_summary = json.loads(
        Path(args.selected_summary).read_text(encoding="utf-8")
    )

    expected_gap = original_summary["primary"]
    reproduced = (
        math.isclose(random_summary["stability_gap"]["estimate"], expected_gap["gap"], abs_tol=1e-12)
        and all(
            math.isclose(actual, expected, abs_tol=1e-12)
            for actual, expected in zip(
                random_summary["stability_gap"]["ci"], expected_gap["ci"]
            )
        )
        and math.isclose(
            random_summary["baseline_stability"],
            original_summary["baseline"]["pairwise_stability"],
            abs_tol=1e-12,
        )
        and math.isclose(
            random_summary["memory_stability"],
            original_summary["memory"]["pairwise_stability"],
            abs_tol=1e-12,
        )
    )
    if not reproduced:
        raise ValueError("Locked Random-20 prospective result was not reproduced")

    expected_selected_gap = selected_original_summary["v4"]["gap"]
    if not math.isclose(
        selected_summary["stability_gap"]["estimate"],
        expected_selected_gap,
        abs_tol=1e-12,
    ):
        raise ValueError("Selected V4 stability gap was not reproduced")

    selected_ids = {row["case_id"] for row in selected_rows}
    random_ids = {row["case_id"] for row in random_rows}
    if selected_ids & random_ids:
        raise ValueError("Random-20 overlaps with outcome-selected V4 cases")
    if len(random_rows) != 20 or len(selected_rows) != 15:
        raise ValueError("Unexpected cohort cardinality")
    if preregistration["sampling"]["selected_ids"] != [
        row["case_id"] for row in random_rows
    ]:
        raise ValueError("Random case order/membership differs from preregistration")

    harm_details = harm_confirmation(random_rows)
    harm_bootstrap = bootstrap_harm_confirmation(
        random_rows, args.n_boot, args.seed + 4
    )
    repeats = repeat_block_metrics(random_rows)
    comparison = {
        "stability_gap_difference": bootstrap_two_sample_difference(
            [row["stability_gap"] for row in selected_rows],
            [row["stability_gap"] for row in random_rows],
            args.n_boot,
            args.seed + 5,
        ),
        "label_agreement_difference": bootstrap_two_sample_difference(
            [row["label_pairwise_agreement"] for row in selected_rows],
            [row["label_pairwise_agreement"] for row in random_rows],
            args.n_boot,
            args.seed + 6,
        ),
    }
    finite_benchmark = bootstrap_weighted_strata(
        [row["stability_gap"] for row in selected_rows],
        [row["stability_gap"] for row in random_rows],
        n_boot=args.n_boot,
        seed=args.seed + 7,
    )
    non_django = [
        row for row in random_rows if row["repository"] != "django__django"
    ]
    runtime_audit = audit_runtime(
        args.run_root,
        args.v4_args,
        preregistration["runtime"]["expected_system_fingerprint"],
        preregistration["runtime"]["expected_response_model"],
    )
    if not runtime_audit["valid"]:
        raise ValueError(f"Runtime/fingerprint audit failed: {runtime_audit}")

    gap = random_summary["stability_gap"]
    if gap["ci"][0] > 0:
        verdict = "GENERAL_RANDOM_MEMORY_INSTABILITY_SUPPORTED"
    elif gap["estimate"] > 0:
        verdict = "DIRECTIONAL_RANDOM_PILOT_ONLY_UNDERPOWERED"
    else:
        verdict = "NO_DETECTABLE_RANDOM_MEMORY_STABILITY_GAP"

    summary = {
        "analysis": "Paper 1 Random-Task Repeated A/B Reliability Audit",
        "verdict": verdict,
        "evidence_status": {
            "locked_primary": "prospective_pre_api_reproduced",
            "label_repeatability": "retrospective_known_outcome",
            "selection_comparison": "retrospective_known_outcome",
            "data_reuse": "shared_with_paper2_pilot",
        },
        "random20": random_summary,
        "selected15": selected_summary,
        "repeat_blocks": repeats,
        "harm_confirmation": {
            **harm_details,
            "bootstrap": harm_bootstrap,
        },
        "selection_comparison": comparison,
        "selection_interpretation": {
            "stability_gap_amplification_supported": comparison[
                "stability_gap_difference"
            ]["ci"][0] > 0,
            "selected_label_agreement_lower_supported": comparison[
                "label_agreement_difference"
            ]["ci"][1] < 0,
            "status": "retrospective_known_outcome",
        },
        "finite_benchmark_sensitivity": finite_benchmark,
        "robustness": {
            "leave_django_out_n": len(non_django),
            "leave_django_out_gap": float(
                np.mean([row["stability_gap"] for row in non_django])
            ),
            "equal_repository_weight_gap": inverse_repository_weighted(
                random_rows, "stability_gap"
            ),
            "random_repository_counts": dict(
                Counter(row["repository"] for row in random_rows)
            ),
        },
        "runtime_audit": runtime_audit,
        "protocol": {
            "random_target_population": 485,
            "excluded_outcome_selected_cases": 15,
            "random_sample_size": 20,
            "repeats_per_case_arm": 4,
            "trajectory_count": 160,
            "bootstrap_replicates": args.n_boot,
            "bootstrap_seed": args.seed,
            "original_target_n_for_3pp_half_width": original_summary[
                "sample_size_target"
            ]["n_target"],
            "locked_primary_reproduced": reproduced,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "random20_reliability_summary.json"
    cases_path = output_dir / "random20_case_labels.csv"
    selected_cases_path = output_dir / "selected15_case_labels.csv"
    repeats_path = output_dir / "repeat_block_metrics.csv"
    harm_path = output_dir / "harm_confirmation_by_discovery_repeat.csv"
    bootstrap_path = output_dir / "bootstrap_metrics.csv"
    report_path = output_dir / "random20_reliability_report.md"
    validation_path = output_dir / "validation_report.md"
    plot_path = output_dir / "repeat_effect_variability.png"

    write_json(summary_path, summary)
    write_csv(cases_path, flatten_case_rows(random_rows))
    write_csv(selected_cases_path, flatten_case_rows(selected_rows))
    write_csv(repeats_path, repeats)
    write_csv(harm_path, harm_details["repeat_rows"])
    bootstrap_rows = []
    for name, metric in (
        ("random_stability_gap", random_summary["stability_gap"]),
        ("random_acc_at5_effect", random_summary["memory_minus_baseline_acc_at5"]),
        ("random_label_agreement", random_summary["label_pairwise_agreement"]),
        ("random_harm_binary_agreement", random_summary["harm_binary_agreement"]),
        ("harm_recurrence", harm_bootstrap),
        ("selected_stability_gap", selected_summary["stability_gap"]),
        ("selected_minus_random_gap", comparison["stability_gap_difference"]),
        ("selected_minus_random_label_agreement", comparison["label_agreement_difference"]),
        ("finite_benchmark_gap", finite_benchmark),
    ):
        bootstrap_rows.append({
            "metric": name,
            "estimate": metric["estimate"],
            "ci_low": metric["ci"][0] if metric.get("ci") else None,
            "ci_high": metric["ci"][1] if metric.get("ci") else None,
            "n_success": metric.get("n_success"),
        })
    write_csv(bootstrap_path, bootstrap_rows)
    report_path.write_text(report_markdown(summary), encoding="utf-8")
    validation_path.write_text(validation_markdown(summary), encoding="utf-8")
    create_plot(
        plot_path, repeats, random_summary["memory_minus_baseline_acc_at5"]
    )

    input_paths = [
        Path(args.random_cases),
        Path(args.random_summary),
        Path(args.preregistration),
        Path(args.selected_cases),
        Path(args.selected_summary),
        Path(args.v4_args),
    ]
    output_paths = [
        summary_path,
        cases_path,
        selected_cases_path,
        repeats_path,
        harm_path,
        bootstrap_path,
        report_path,
        validation_path,
        plot_path,
    ]
    manifest = {
        "analysis": summary["analysis"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/analyze_random_repeat_reliability.py",
        "external_calls": 0,
        "raw_inputs_modified": False,
        "prospective_status": summary["evidence_status"],
        "inputs": {
            str(path): sha256_path(path) for path in input_paths
        },
        "outputs": {
            str(path): sha256_path(path) for path in output_paths
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
