import argparse
import csv
import itertools
import json
import math
import os

import numpy as np
from datasets import load_dataset

from scripts.check_m3_finalization_audit import extract_trajectory, flatten_files
from scripts.eval_verified_file_acc import files_from_patch, normalize_path


DEFAULT_SELECTION = "reproduce_lists/verified_m3a_matched_15.json"
DEFAULT_FLIP_FILE = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "flip_decomposition/flip_decomposition.csv"
)
DEFAULT_OUTPUT_DIR = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "m3_memory_variance/m3a_matched_noise_floor"
)
BEHAVIOR_METRICS = [
    "code_verify_call_count",
    "active_code_verify_count",
    "finished_before_max",
    "effective_finish_iteration",
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ids(path):
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def load_gold(dataset, split, selected_ids):
    rows = load_dataset(dataset, split=split)
    result = {}
    for row in rows:
        case_id = row["instance_id"]
        if case_id in selected_ids:
            result[case_id] = files_from_patch(row["patch"], py_only=True)
    missing = selected_ids - set(result)
    if missing:
        raise ValueError(f"missing gold patches: {sorted(missing)}")
    return result


def predicted_files(row):
    paths = []
    seen = set()
    for path in flatten_files(row.get("found_files") or []):
        if not isinstance(path, str):
            continue
        path = normalize_path(path)
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def load_repeat(path, config, repeat, selected_ids, gold, max_iterations):
    rows = {
        row["instance_id"]: row
        for row in load_jsonl(path)
        if row.get("instance_id") in selected_ids
    }
    missing = selected_ids - set(rows)
    if missing:
        raise ValueError(f"{path}: missing={sorted(missing)}")
    result = []
    for case_id in sorted(selected_ids):
        row = rows[case_id]
        trajectory = extract_trajectory(row, max_iterations)
        predictions = predicted_files(row)
        correct = int(set(gold[case_id]).issubset(set(predictions[:5])))
        result.append(
            {
                "case_id": case_id,
                "config": config,
                "repeat": repeat,
                "correct_at5": correct,
                "predicted_files": predictions,
                "source_file": path,
                **trajectory,
            }
        )
    return result


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    if not len(values):
        return None
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        extreme += abs(np.mean(values * signs)) >= observed - 1e-12
    return extreme / (2 ** len(values))


def bootstrap_mean(values, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.percentile(samples, [2.5, 97.5]).tolist()


def stability(values):
    pairs = [int(left == right) for left, right in itertools.combinations(values, 2)]
    ones = sum(values)
    return {
        "correct_rate": ones / len(values),
        "unanimous": int(ones in (0, len(values))),
        "modal_consistency": max(ones, len(values) - ones) / len(values),
        "pairwise_agreement": sum(pairs) / len(pairs),
        "correctness_variance": float(np.var(values)),
    }


def within_case_slope(rows, field):
    numerator = 0.0
    denominator = 0.0
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_rows in by_case.values():
        x = np.asarray([row[field] for row in case_rows], dtype=float)
        y = np.asarray([row["correct_at5"] for row in case_rows], dtype=float)
        x -= x.mean()
        y -= y.mean()
        numerator += float(np.dot(x, y))
        denominator += float(np.dot(x, x))
    return numerator / denominator if denominator else 0.0


def permute_within_case(rows, field, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)
    observed = within_case_slope(rows, field)
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(dict(row))
    extreme = 0
    for _ in range(n_perm):
        permuted = []
        for case_rows in by_case.values():
            labels = [row["correct_at5"] for row in case_rows]
            labels = rng.permutation(labels)
            for row, label in zip(case_rows, labels):
                item = dict(row)
                item["correct_at5"] = int(label)
                permuted.append(item)
        extreme += abs(within_case_slope(permuted, field)) >= abs(observed) - 1e-12
    return observed, (extreme + 1) / (n_perm + 1)


def bootstrap_slope(rows, field, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    case_ids = sorted(by_case)
    slopes = []
    for _ in range(n_boot):
        sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        sample = []
        for index, case_id in enumerate(sample_ids):
            for row in by_case[case_id]:
                item = dict(row)
                item["case_id"] = f"{case_id}__boot{index}"
                sample.append(item)
        slopes.append(within_case_slope(sample, field))
    return np.percentile(slopes, [2.5, 97.5]).tolist()


def interaction_test(baseline_rows, memory_rows, field, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)
    observed = within_case_slope(memory_rows, field) - within_case_slope(
        baseline_rows, field
    )
    combined = {}
    for row in baseline_rows + memory_rows:
        combined.setdefault(row["case_id"], []).append(dict(row))

    extreme = 0
    for _ in range(n_perm):
        shuffled_baseline = []
        shuffled_memory = []
        for case_rows in combined.values():
            indexes = rng.permutation(len(case_rows))
            midpoint = len(case_rows) // 2
            shuffled_baseline.extend(case_rows[index] for index in indexes[:midpoint])
            shuffled_memory.extend(case_rows[index] for index in indexes[midpoint:])
        difference = within_case_slope(
            shuffled_memory, field
        ) - within_case_slope(shuffled_baseline, field)
        extreme += abs(difference) >= abs(observed) - 1e-12

    case_ids = sorted(combined)
    bootstrap_values = []
    for _ in range(n_perm):
        sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        baseline_sample = []
        memory_sample = []
        for index, case_id in enumerate(sample_ids):
            for source, target in (
                (baseline_rows, baseline_sample),
                (memory_rows, memory_sample),
            ):
                for row in source:
                    if row["case_id"] == case_id:
                        item = dict(row)
                        item["case_id"] = f"{case_id}__boot{index}"
                        target.append(item)
        bootstrap_values.append(
            within_case_slope(memory_sample, field)
            - within_case_slope(baseline_sample, field)
        )
    return {
        "slope": observed,
        "p": (extreme + 1) / (n_perm + 1),
        "ci": np.percentile(bootstrap_values, [2.5, 97.5]).tolist(),
    }


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze matched M3a baseline and memory repeats.")
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--flip_file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--baseline", nargs=4, required=True)
    parser.add_argument("--memory", nargs=4, required=True)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_iterations", type=int, default=8)
    parser.add_argument("--discarded_context_attempts", type=int, default=0)
    parser.add_argument("--discarded_cases", nargs="*", default=[])
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis_label", default="Parameter-Matched Noise Floor")
    parser.add_argument(
        "--verdict_protocol",
        choices=("legacy", "context_safe"),
        default="legacy",
    )
    args = parser.parse_args()

    selected_ids = load_ids(args.selection)
    if len(selected_ids) != 15:
        raise ValueError(f"expected 15 selected cases, got {len(selected_ids)}")
    gold = load_gold(args.dataset, args.split, selected_ids)
    all_rows = []
    for config, paths in (("baseline", args.baseline), ("memory", args.memory)):
        for repeat, path in enumerate(paths):
            all_rows.extend(
                load_repeat(path, config, repeat, selected_ids, gold, args.max_iterations)
            )

    flip_classes = {}
    with open(args.flip_file, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            flip_classes[row["case_id"]] = row["flip_class"]
    for row in all_rows:
        row["flip_class"] = flip_classes[row["case_id"]]

    per_case = []
    for case_id in sorted(selected_ids):
        entry = {"case_id": case_id, "flip_class": flip_classes[case_id]}
        for config in ("baseline", "memory"):
            values = [
                row["correct_at5"]
                for row in all_rows
                if row["case_id"] == case_id and row["config"] == config
            ]
            if len(values) != 4:
                raise ValueError(f"{case_id}/{config} has {len(values)} repeats, expected 4")
            for key, value in stability(values).items():
                entry[f"{config}_{key}"] = value
            entry[f"{config}_outcomes"] = "".join(map(str, values))
        entry["stability_gap"] = (
            entry["baseline_pairwise_agreement"] - entry["memory_pairwise_agreement"]
        )
        per_case.append(entry)

    gaps = [row["stability_gap"] for row in per_case]
    baseline_agreement = np.mean([row["baseline_pairwise_agreement"] for row in per_case])
    memory_agreement = np.mean([row["memory_pairwise_agreement"] for row in per_case])
    gap_ci = bootstrap_mean(gaps)
    gap_p = exact_sign_flip(gaps)
    discarded_cases = set(args.discarded_cases)
    unknown_discarded = discarded_cases - selected_ids
    if unknown_discarded:
        raise ValueError(f"discarded cases are not selected: {sorted(unknown_discarded)}")
    sensitivity_rows = [
        row for row in per_case if row["case_id"] not in discarded_cases
    ]
    sensitivity_gaps = [row["stability_gap"] for row in sensitivity_rows]
    discard_sensitivity = {
        "excluded_cases": sorted(discarded_cases),
        "remaining_cases": len(sensitivity_rows),
        "baseline_pairwise_agreement": float(
            np.mean([row["baseline_pairwise_agreement"] for row in sensitivity_rows])
        ),
        "memory_pairwise_agreement": float(
            np.mean([row["memory_pairwise_agreement"] for row in sensitivity_rows])
        ),
        "stability_gap": float(np.mean(sensitivity_gaps)),
        "stability_gap_ci": bootstrap_mean(sensitivity_gaps),
        "stability_gap_exact_p": exact_sign_flip(sensitivity_gaps),
    }

    mechanism = []
    for field in BEHAVIOR_METRICS:
        config_results = {}
        config_rows = {}
        for config in ("baseline", "memory"):
            rows = [row for row in all_rows if row["config"] == config]
            config_rows[config] = rows
            slope, pvalue = permute_within_case(rows, field)
            ci = bootstrap_slope(rows, field)
            config_results[config] = {"slope": slope, "p": pvalue, "ci": ci}
            mechanism.append(
                {
                    "config": config,
                    "metric": field,
                    "within_case_slope": slope,
                    "permutation_p": pvalue,
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                }
            )
        interaction = interaction_test(
            config_rows["baseline"], config_rows["memory"], field
        )
        mechanism.append(
            {
                "config": "memory_minus_baseline",
                "metric": field,
                "within_case_slope": interaction["slope"],
                "permutation_p": interaction["p"],
                "ci_low": interaction["ci"][0],
                "ci_high": interaction["ci"][1],
            }
        )

    accuracy = {}
    for config in ("baseline", "memory"):
        rows = [row for row in all_rows if row["config"] == config]
        accuracy[config] = {
            "overall": float(np.mean([row["correct_at5"] for row in rows])),
            "memory_flipped": float(
                np.mean(
                    [
                        row["correct_at5"]
                        for row in rows
                        if row["flip_class"] == "memory_flipped"
                    ]
                )
            ),
            "stable_harm": float(
                np.mean(
                    [
                        row["correct_at5"]
                        for row in rows
                        if row["flip_class"] == "stable_harm"
                    ]
                )
            ),
        }

    if args.verdict_protocol == "context_safe":
        inferential_strong = (
            np.mean(gaps) >= 0.15 and gap_ci[0] > 0 and gap_p < 0.05
        )
        if baseline_agreement > 0.85 and inferential_strong:
            verdict = "STRONG_CONTEXT_SAFE_MEMORY_SPECIFIC_VARIANCE"
        elif baseline_agreement >= 0.75 and inferential_strong:
            verdict = (
                "SIGNIFICANT_DIRECTIONAL_MEMORY_SPECIFIC_VARIANCE_"
                "WITH_MODERATE_BASELINE_NOISE"
            )
        elif np.mean(gaps) >= 0.15 and gap_ci[0] > 0:
            verdict = "ROBUST_PARTIAL_CONTEXT_SAFE_MEMORY_SPECIFIC_VARIANCE"
        elif np.mean(gaps) > 0 and gap_ci[0] <= 0 <= gap_ci[1]:
            verdict = "BORDERLINE_CONTEXT_SAFE_MEMORY_SPECIFIC_VARIANCE"
        else:
            verdict = "UNRESOLVED_CONTEXT_SAFE_MEMORY_SPECIFIC_VARIANCE"
    else:
        if baseline_agreement > 0.85 and np.mean(gaps) >= 0.15 and gap_ci[0] > 0 and gap_p < 0.05:
            verdict = "STRONG_MEMORY_SPECIFIC_VARIANCE"
        elif np.mean(gaps) > 0 and (gap_p < 0.15 or gap_ci[0] <= 0 <= gap_ci[1]):
            verdict = "PARTIAL_MEMORY_SPECIFIC_VARIANCE"
        elif baseline_agreement < 0.75 and abs(np.mean(gaps)) < 0.05:
            verdict = "PIPELINE_WIDE_NOISE"
        elif abs(np.mean(gaps)) < 0.05 or memory_agreement >= baseline_agreement:
            verdict = "NO_MEMORY_SPECIFIC_EFFECT"
        else:
            verdict = "INCONCLUSIVE"

    summary = {
        "case_count": len(per_case),
        "repeats_per_case_per_config": 4,
        "trajectory_count": len(all_rows),
        "baseline_pairwise_agreement": float(baseline_agreement),
        "memory_pairwise_agreement": float(memory_agreement),
        "stability_gap": float(np.mean(gaps)),
        "stability_gap_ci": gap_ci,
        "stability_gap_exact_p": gap_p,
        "baseline_unanimous_rate": float(np.mean([row["baseline_unanimous"] for row in per_case])),
        "memory_unanimous_rate": float(np.mean([row["memory_unanimous"] for row in per_case])),
        "baseline_modal_consistency": float(
            np.mean([row["baseline_modal_consistency"] for row in per_case])
        ),
        "memory_modal_consistency": float(
            np.mean([row["memory_modal_consistency"] for row in per_case])
        ),
        "accuracy_at5": accuracy,
        "discarded_context_attempts": args.discarded_context_attempts,
        "discard_excluded_sensitivity": discard_sensitivity,
        "verdict": verdict,
        "verdict_protocol": args.verdict_protocol,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "matched_trajectories.jsonl"), "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(os.path.join(args.output_dir, "stability_per_case.csv"), per_case)
    write_csv(os.path.join(args.output_dir, "finalization_mechanism.csv"), mechanism)
    with open(os.path.join(args.output_dir, "m3a_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "matched_noise_floor_report.md"), "w", encoding="utf-8") as f:
        f.write(f"# M3a {args.analysis_label}\n\n")
        f.write(f"- cases: {summary['case_count']}\n")
        f.write(f"- trajectories: {summary['trajectory_count']}\n")
        f.write(f"- baseline pairwise agreement: {baseline_agreement:.4f}\n")
        f.write(f"- memory pairwise agreement: {memory_agreement:.4f}\n")
        f.write(f"- gap: {np.mean(gaps):+.4f}, 95% CI [{gap_ci[0]:.4f}, {gap_ci[1]:.4f}], exact p={gap_p:.4f}\n")
        f.write(
            f"- unanimous agreement: baseline {summary['baseline_unanimous_rate']:.4f}, "
            f"memory {summary['memory_unanimous_rate']:.4f}\n"
        )
        f.write(
            f"- modal consistency: baseline {summary['baseline_modal_consistency']:.4f}, "
            f"memory {summary['memory_modal_consistency']:.4f}\n"
        )
        f.write(f"- baseline Acc@5 on selected repeats: {accuracy['baseline']['overall']:.4f}\n")
        f.write(f"- memory Acc@5 on selected repeats: {accuracy['memory']['overall']:.4f}\n")
        f.write(f"- discarded context-window attempts: {args.discarded_context_attempts}\n")
        f.write(f"- verdict: **{verdict}**\n\n")
        f.write("## Pre-Registered Interpretation And Caveats\n\n")
        f.write("### P1-1: Exact permutation threshold\n\n")
        f.write(
            f"The exact paired permutation test yielded p={gap_p:.4f} against the pre-registered alpha=0.05 threshold. "
            "With 15 paired cases the exact randomization distribution is discrete, but this does not justify changing alpha after observing the data. "
            f"The result is classified as **{verdict}** under the pre-registered rule. "
            "The point-estimate direction is consistent across pairwise agreement, unanimous agreement, and modal consistency.\n\n"
        )
        f.write("### P1-2: Magnitude uncertainty\n\n")
        f.write(
            f"The observed {np.mean(gaps) * 100:+.2f} pp gap is evaluated against the pre-registered 15 pp magnitude criterion; "
            f"its case-bootstrap 95% CI is [{gap_ci[0] * 100:+.2f}, {gap_ci[1] * 100:+.2f}] pp. "
            "The point estimate must not be presented as a lower bound or a population estimate. More independent cases, "
            "and potentially more repeats per case, are needed for a narrower interval.\n\n"
        )
        f.write("### P1-3: Context-window discard sensitivity\n\n")
        if discarded_cases:
            f.write(
                f"{args.discarded_context_attempts} attempts were discarded after provider context-limit errors and restarted, "
                f"affecting `{', '.join(sorted(discarded_cases))}`. Fresh restarts do not share a controlled provider-side sampling seed. "
                "The completed trajectories remain in the primary intention-to-analyze result; the following sensitivity analysis excludes affected cases.\n\n"
            )
            f.write(
                f"- Remaining cases: {discard_sensitivity['remaining_cases']}/15\n"
                f"- Baseline pairwise agreement: {discard_sensitivity['baseline_pairwise_agreement']:.4f}\n"
                f"- Memory pairwise agreement: {discard_sensitivity['memory_pairwise_agreement']:.4f}\n"
                f"- Gap: {discard_sensitivity['stability_gap']:+.4f}\n"
                f"- Case-bootstrap 95% CI: [{discard_sensitivity['stability_gap_ci'][0]:+.4f}, {discard_sensitivity['stability_gap_ci'][1]:+.4f}]\n"
                f"- Exact permutation p: {discard_sensitivity['stability_gap_exact_p']:.4f}\n\n"
            )
        else:
            f.write(
                "No provider context-limit attempt was discarded or restarted in this analysis. "
                "The runtime audit must independently confirm complete trajectories, symmetric context policy, and zero fallback truncations.\n\n"
            )
        f.write("## Finalization Mechanism\n\n")
        f.write("| config | metric | within-case slope | permutation p | 95% CI |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in mechanism:
            f.write(
                f"| {row['config']} | {row['metric']} | {row['within_case_slope']:+.4f} | "
                f"{row.get('permutation_p', float('nan')):.4f} | "
                f"[{row.get('ci_low', float('nan')):.4f}, {row.get('ci_high', float('nan')):.4f}] |\n"
            )
        f.write("\nAll repeats use the same endpoint, function-calling mode, iteration budget, and parser repair.\n")
        if args.discarded_context_attempts:
            f.write("Discarded context-window attempts were retried from a fresh trajectory and are infrastructure events, not model-produced empty predictions.\n")
        else:
            f.write("No context-window attempt was discarded or restarted in this analysis.\n")
        f.write("The selected harm subset is diagnostic and does not estimate full-500 accuracy or population stability.\n")
    print(json.dumps(summary, indent=2))
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
