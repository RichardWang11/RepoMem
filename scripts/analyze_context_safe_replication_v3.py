import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
from scipy.stats import chi2

from scripts.analyze_m3a_matched_noise_floor import (
    bootstrap_mean,
    exact_sign_flip,
    load_gold,
    load_ids,
    load_repeat,
    stability,
)


DEFAULT_SELECTION = "reproduce_lists/verified_m3a_matched_15.json"
DEFAULT_FLIP_FILE = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "flip_decomposition/flip_decomposition.csv"
)
DEFAULT_OUTPUT = "results/context_safe_m3a/replication_v3/analysis"
RUN_ORDER = [
    ("baseline", 1),
    ("memory", 1),
    ("memory", 2),
    ("baseline", 2),
    ("memory", 3),
    ("baseline", 3),
    ("baseline", 4),
    ("memory", 4),
]


def load_outcomes(paths, config, selected_ids, gold):
    outcomes = {}
    rows = []
    for repeat, path in enumerate(paths, 1):
        repeat_rows = load_repeat(path, config, repeat, selected_ids, gold, 8)
        rows.extend(repeat_rows)
        for row in repeat_rows:
            outcomes.setdefault(row["case_id"], {})[repeat] = row["correct_at5"]
    return outcomes, rows


def case_stabilities(outcomes):
    result = {}
    for case_id, repeats in outcomes.items():
        values = [repeats[index] for index in range(1, 5)]
        result[case_id] = {**stability(values), "outcomes": "".join(map(str, values))}
    return result


def cross_run_agreement(left, right):
    pairs = [int(a == b) for a in left for b in right]
    if len(pairs) != 16:
        raise ValueError(f"expected 16 cross-run pairs, got {len(pairs)}")
    return sum(pairs) / len(pairs)


def gap_summary(gaps):
    return {
        "gap": float(np.mean(gaps)),
        "ci": bootstrap_mean(gaps),
        "exact_p": exact_sign_flip(gaps),
    }


def aggregate_arm_metrics(case_metrics):
    return {
        "pairwise_agreement": float(np.mean([
            row["pairwise_agreement"] for row in case_metrics.values()
        ])),
        "unanimous_rate": float(np.mean([
            row["unanimous"] for row in case_metrics.values()
        ])),
        "modal_consistency": float(np.mean([
            row["modal_consistency"] for row in case_metrics.values()
        ])),
        "accuracy_at5": float(np.mean([
            row["correct_rate"] for row in case_metrics.values()
        ])),
    }


def naive_fisher(pvalues):
    statistic = -2 * sum(math.log(max(value, 1e-300)) for value in pvalues)
    return {"statistic": statistic, "df": 2 * len(pvalues), "p": float(chi2.sf(statistic, 2 * len(pvalues)))}


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def endpoint_hash():
    endpoint = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    return hashlib.sha256((endpoint or "").encode()).hexdigest() if endpoint else None


def response_metadata(paths):
    models = set()
    fingerprints = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                trajectories = row.get("loc_trajs", {}).get("trajs", [])
                if not trajectories:
                    continue
                provider = trajectories[0].get("provider_metadata", {})
                for response in provider.get("responses", []):
                    if response.get("model"):
                        models.add(response["model"])
                    if response.get("system_fingerprint"):
                        fingerprints.add(response["system_fingerprint"])
    return {"response_models": sorted(models), "system_fingerprints": sorted(fingerprints)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--flip_file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--v3_baseline", nargs=4, required=True)
    parser.add_argument("--v3_memory", nargs=4, required=True)
    parser.add_argument("--v2_baseline", nargs=4, required=True)
    parser.add_argument("--v2_memory", nargs=4, required=True)
    parser.add_argument("--legacy_baseline", nargs=4, required=True)
    parser.add_argument("--legacy_memory", nargs=4, required=True)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selected_ids = load_ids(args.selection)
    if len(selected_ids) != 15:
        raise ValueError(f"expected 15 cases, got {len(selected_ids)}")
    gold = load_gold(args.dataset, args.split, selected_ids)
    flip_classes = {}
    with open(args.flip_file, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            flip_classes[row["case_id"]] = row["flip_class"]

    path_groups = {
        "v3": (args.v3_baseline, args.v3_memory),
        "v2": (args.v2_baseline, args.v2_memory),
        "legacy": (args.legacy_baseline, args.legacy_memory),
    }
    outcomes = {}
    trajectory_rows = {}
    stabilities = {}
    run_summaries = {}
    for run, (baseline_paths, memory_paths) in path_groups.items():
        outcomes[run] = {}
        trajectory_rows[run] = {}
        stabilities[run] = {}
        for arm, paths in (("baseline", baseline_paths), ("memory", memory_paths)):
            config_outcomes, rows = load_outcomes(
                paths, f"{run}_{arm}", selected_ids, gold
            )
            outcomes[run][arm] = config_outcomes
            trajectory_rows[run][arm] = rows
            stabilities[run][arm] = case_stabilities(config_outcomes)
        gaps = [
            stabilities[run]["baseline"][case_id]["pairwise_agreement"]
            - stabilities[run]["memory"][case_id]["pairwise_agreement"]
            for case_id in sorted(selected_ids)
        ]
        baseline_metrics = aggregate_arm_metrics(stabilities[run]["baseline"])
        memory_metrics = aggregate_arm_metrics(stabilities[run]["memory"])
        run_summaries[run] = {
            "baseline_pairwise_agreement": baseline_metrics["pairwise_agreement"],
            "memory_pairwise_agreement": memory_metrics["pairwise_agreement"],
            "baseline_unanimous_rate": baseline_metrics["unanimous_rate"],
            "memory_unanimous_rate": memory_metrics["unanimous_rate"],
            "baseline_modal_consistency": baseline_metrics["modal_consistency"],
            "memory_modal_consistency": memory_metrics["modal_consistency"],
            "baseline_accuracy_at5": baseline_metrics["accuracy_at5"],
            "memory_accuracy_at5": memory_metrics["accuracy_at5"],
            **gap_summary(gaps),
        }

    v3 = run_summaries["v3"]
    if v3["gap"] >= 0.15 and v3["ci"][0] > 0 and v3["exact_p"] < 0.05:
        verdict = "REPLICATED"
    elif v3["gap"] > 0:
        verdict = "DIRECTIONAL_PARTIAL"
    else:
        verdict = "NOT_REPLICATED"

    per_case = []
    mean_gaps = []
    pooled_gaps = []
    cross_gaps = []
    three_run_gaps = []
    cross_rows = []
    for case_id in sorted(selected_ids):
        row = {"case_id": case_id, "flip_class": flip_classes[case_id]}
        run_gaps = {}
        for run in ("legacy", "v2", "v3"):
            baseline = stabilities[run]["baseline"][case_id]
            memory = stabilities[run]["memory"][case_id]
            row[f"{run}_baseline_outcomes"] = baseline["outcomes"]
            row[f"{run}_memory_outcomes"] = memory["outcomes"]
            run_gaps[run] = baseline["pairwise_agreement"] - memory["pairwise_agreement"]
            row[f"{run}_gap"] = run_gaps[run]

        mean_gap = (run_gaps["v2"] + run_gaps["v3"]) / 2
        mean_gaps.append(mean_gap)
        row["context_safe_mean_gap"] = mean_gap

        pooled = {}
        cross = {}
        for arm in ("baseline", "memory"):
            v2_values = [outcomes["v2"][arm][case_id][index] for index in range(1, 5)]
            v3_values = [outcomes["v3"][arm][case_id][index] for index in range(1, 5)]
            pooled[arm] = stability(v2_values + v3_values)["pairwise_agreement"]
            cross[arm] = cross_run_agreement(v2_values, v3_values)
            cross_rows.append({
                "case_id": case_id,
                "flip_class": flip_classes[case_id],
                "arm": arm,
                "v2_within": stabilities["v2"][arm][case_id]["pairwise_agreement"],
                "v3_within": stabilities["v3"][arm][case_id]["pairwise_agreement"],
                "cross_run_16pair": cross[arm],
                "pooled_28pair": pooled[arm],
            })
        pooled_gap = pooled["baseline"] - pooled["memory"]
        cross_gap = cross["baseline"] - cross["memory"]
        pooled_gaps.append(pooled_gap)
        cross_gaps.append(cross_gap)
        row["pooled8_gap"] = pooled_gap
        row["cross_run_gap"] = cross_gap
        three_gap = float(np.mean(list(run_gaps.values())))
        three_run_gaps.append(three_gap)
        row["three_run_mean_gap"] = three_gap
        per_case.append(row)

    context_safe_mean = gap_summary(mean_gaps)
    pooled_baseline = []
    pooled_memory = []
    cross_baseline = []
    cross_memory = []
    for row in cross_rows:
        if row["arm"] == "baseline":
            pooled_baseline.append(row["pooled_28pair"])
            cross_baseline.append(row["cross_run_16pair"])
        else:
            pooled_memory.append(row["pooled_28pair"])
            cross_memory.append(row["cross_run_16pair"])
    pooled8 = {
        **gap_summary(pooled_gaps),
        "baseline_pairwise_agreement": float(np.mean(pooled_baseline)),
        "memory_pairwise_agreement": float(np.mean(pooled_memory)),
        "pair_decomposition": {"v2_within": 6, "v3_within": 6, "cross_run": 16, "total": 28},
        "cross_run_gap": {
            **gap_summary(cross_gaps),
            "baseline_pairwise_agreement": float(np.mean(cross_baseline)),
            "memory_pairwise_agreement": float(np.mean(cross_memory)),
        },
    }
    three_run = gap_summary(three_run_gaps)
    fisher = naive_fisher([
        run_summaries["legacy"]["exact_p"],
        run_summaries["v2"]["exact_p"],
        run_summaries["v3"]["exact_p"],
    ])

    position_map = {item: index for index, item in enumerate(RUN_ORDER, 1)}
    position_rows = []
    for arm in ("baseline", "memory"):
        for repeat in range(1, 5):
            values = [
                outcomes["v3"][arm][case_id][repeat] for case_id in selected_ids
            ]
            position = position_map[(arm, repeat)]
            position_rows.append({
                "arm": arm,
                "repeat": repeat,
                "execution_position": position,
                "position_squared": position ** 2,
                "accuracy_at5": float(np.mean(values)),
            })

    position_summary = {}
    for arm in ("baseline", "memory"):
        arm_rows = [row for row in position_rows if row["arm"] == arm]
        x = np.asarray([row["execution_position"] for row in arm_rows], dtype=float)
        y = np.asarray([row["accuracy_at5"] for row in arm_rows], dtype=float)
        design = np.column_stack([np.ones(len(x)), x, x ** 2])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        position_summary[arm] = {
            "linear_correlation": float(np.corrcoef(x, y)[0, 1]),
            "quadratic_correlation": float(np.corrcoef(x ** 2, y)[0, 1]),
            "quadratic_fit_intercept": float(coefficients[0]),
            "quadratic_fit_position": float(coefficients[1]),
            "quadratic_fit_position_squared": float(coefficients[2]),
            "note": "Descriptive only: four batch-level positions per arm.",
        }

    subgroup_summary = {}
    for flip_class in ("memory_flipped", "stable_harm"):
        subset = [row for row in per_case if row["flip_class"] == flip_class]
        subgroup_summary[flip_class] = {
            "n": len(subset),
            "v3_gap": float(np.mean([row["v3_gap"] for row in subset])),
            "v2_gap": float(np.mean([row["v2_gap"] for row in subset])),
            "context_safe_mean_gap": float(np.mean([
                row["context_safe_mean_gap"] for row in subset
            ])),
        }

    summary = {
        "v3": {**v3, "verdict": verdict},
        "context_safe_mean_of_runs": context_safe_mean,
        "context_safe_pooled8": pooled8,
        "three_run_dependence_aware": three_run,
        "naive_fisher_nonconfirmatory": {
            **fisher,
            "warning": "Dependent p-values and non-identical legacy configuration; appendix only.",
        },
        "run_summaries": run_summaries,
        "v3_subgroups": subgroup_summary,
        "position_drift": position_summary,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "stability_per_case.csv"), per_case)
    write_csv(os.path.join(args.output_dir, "cross_run_agreement.csv"), cross_rows)
    write_csv(os.path.join(args.output_dir, "position_drift_analysis.csv"), position_rows)
    write_csv(os.path.join(args.output_dir, "three_run_robustness.csv"), [
        {"run": run, **values} for run, values in run_summaries.items()
    ])
    for filename, payload in (
        ("replication_v3_summary.json", summary),
        ("context_safe_mean_of_runs.json", context_safe_mean),
        ("context_safe_pooled8.json", pooled8),
    ):
        with open(os.path.join(args.output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    metadata = response_metadata(args.v3_baseline + args.v3_memory)
    manifest = {
        "experiment": "Third M3a-style run; second request-matched context-safe replication",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": args.selection,
        "cases": 15,
        "fresh_trajectories": 120,
        "execution_order": [f"{arm[0].upper()}{repeat}" for arm, repeat in RUN_ORDER],
        "position_balance": {
            "baseline": {"positions": [1, 4, 6, 7], "mean": 4.5, "variance": 5.25},
            "memory": {"positions": [2, 3, 5, 8], "mean": 4.5, "variance": 5.25},
        },
        "request": {
            "model": "gpt-4o",
            "temperature": 1.0,
            "top_p": None,
            "endpoint_sha256": endpoint_hash(),
        },
        "provider_metadata": metadata,
        "v2_parity": {
            "effective_temperature": 1.0,
            "top_p": "omitted/provider-default",
            "requested_model": "gpt-4o",
            "backend_snapshot_parity": "not retrospectively verifiable",
        },
        "credentials_recorded": False,
    }
    with open(
        os.path.join(os.path.dirname(args.output_dir), "experiment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(args.output_dir, "replication_v3_report.md"), "w", encoding="utf-8") as f:
        f.write("# Context-Safe Replication V3\n\n")
        f.write(f"- V3 baseline stability: {v3['baseline_pairwise_agreement']:.4f}\n")
        f.write(f"- V3 memory stability: {v3['memory_pairwise_agreement']:.4f}\n")
        f.write(f"- V3 gap: {v3['gap']:+.4f}, 95% CI [{v3['ci'][0]:+.4f}, {v3['ci'][1]:+.4f}], exact p={v3['exact_p']:.4f}\n")
        f.write(f"- Verdict: **{verdict}**\n\n")
        f.write(f"- Unanimous agreement: baseline {v3['baseline_unanimous_rate']:.4f}, memory {v3['memory_unanimous_rate']:.4f}\n")
        f.write(f"- Modal consistency: baseline {v3['baseline_modal_consistency']:.4f}, memory {v3['memory_modal_consistency']:.4f}\n")
        f.write(f"- Selected-subset Acc@5: baseline {v3['baseline_accuracy_at5']:.4f}, memory {v3['memory_accuracy_at5']:.4f}\n\n")
        f.write("## Original-Stratum Breakdown\n\n")
        f.write("| Stratum | N | V2 gap | V3 gap | Mean gap |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for label, item in subgroup_summary.items():
            f.write(
                f"| {label} | {item['n']} | {item['v2_gap']:+.4f} | "
                f"{item['v3_gap']:+.4f} | {item['context_safe_mean_gap']:+.4f} |\n"
            )
        f.write("\n")
        f.write("## Dependence-Aware Combined Results\n\n")
        f.write(f"- Mean-of-runs gap: {context_safe_mean['gap']:+.4f}, 95% CI [{context_safe_mean['ci'][0]:+.4f}, {context_safe_mean['ci'][1]:+.4f}], p={context_safe_mean['exact_p']:.4f}\n")
        f.write(f"- Pooled-8 gap: {pooled8['gap']:+.4f}, 95% CI [{pooled8['ci'][0]:+.4f}, {pooled8['ci'][1]:+.4f}], p={pooled8['exact_p']:.4f}\n")
        f.write(f"- Cross-run agreement gap: {pooled8['cross_run_gap']['gap']:+.4f}\n")
        f.write(f"- Three-run mean gap: {three_run['gap']:+.4f}, p={three_run['exact_p']:.4f}\n\n")
        f.write("## Non-Confirmatory Fisher Sensitivity\n\n")
        f.write(f"Naive Fisher p={fisher['p']:.6f}. This is appendix-only because the p-values share the same selected cases and legacy uses a different context policy.\n\n")
        f.write("## Interpretation Limit\n\n")
        f.write("These 15 cases were selected from an original baseline-correct/memory-wrong pool. The results measure stability on a diagnostic harm subset, not full-500 performance.\n")
        f.write("V2 did not retain a provider backend snapshot identifier, so this is request-matched rather than snapshot-identical replication.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
