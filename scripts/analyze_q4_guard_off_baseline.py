import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

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
DEFAULT_OUTPUT_DIR = "results/context_safe_m3a/q4_guard_off_baseline/analysis"


def summarize_config(rows, selected_ids):
    by_case = []
    for case_id in sorted(selected_ids):
        values = [
            row["correct_at5"] for row in rows if row["case_id"] == case_id
        ]
        if len(values) != 4:
            raise ValueError(f"{case_id}: expected four repeats, got {len(values)}")
        by_case.append({"case_id": case_id, "outcomes": "".join(map(str, values)), **stability(values)})
    return by_case


def compare(reference, candidate, label):
    ref = {row["case_id"]: row for row in reference}
    candidate_map = {row["case_id"]: row for row in candidate}
    deltas = [
        candidate_map[case_id]["pairwise_agreement"]
        - ref[case_id]["pairwise_agreement"]
        for case_id in sorted(ref)
    ]
    return {
        "reference": label,
        "reference_agreement": float(
            np.mean([row["pairwise_agreement"] for row in reference])
        ),
        "q4_agreement": float(
            np.mean([row["pairwise_agreement"] for row in candidate])
        ),
        "q4_minus_reference": float(np.mean(deltas)),
        "ci": bootstrap_mean(deltas),
        "exact_p": exact_sign_flip(deltas),
    }


def classify(q4, guard_on, original):
    delta_on = q4 - guard_on["reference_agreement"]
    if (
        q4 >= 0.85
        and delta_on >= 0.08
        and guard_on["ci"][0] > 0
        and abs(q4 - original["reference_agreement"]) <= 0.05
    ):
        return "CONSISTENT_WITH_GUARD_POLICY_EFFECT"
    if abs(delta_on) <= 0.05 and q4 <= original["reference_agreement"] - 0.08:
        return "CONSISTENT_WITH_SAMPLING_OR_ENDPOINT_DRIFT"
    return "INCONCLUSIVE_SEQUENTIAL_REPLICATION"


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_args(paths):
    result = []
    for path in paths:
        args_path = os.path.join(os.path.dirname(path), "args.json")
        with open(args_path, encoding="utf-8") as f:
            run_args = json.load(f)
        result.append(
            {
                "run_dir": os.path.dirname(path),
                "model": run_args["model"],
                "num_samples": run_args["num_samples"],
                "num_processes": run_args["num_processes"],
                "max_iteration_num": run_args["max_iteration_num"],
                "max_attempt_num": run_args["max_attempt_num"],
                "context_token_budget": run_args["context_token_budget"],
                "tool_observation_token_cap": run_args["tool_observation_token_cap"],
                "bounded_graph_output": run_args["bounded_graph_output"],
                "bounded_entity_output": run_args["bounded_entity_output"],
                "bounded_search_output": run_args["bounded_search_output"],
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--flip_file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--q4", nargs=4, required=True)
    parser.add_argument("--guard_on", nargs=4, required=True)
    parser.add_argument("--original", nargs=4, required=True)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    selected_ids = load_ids(args.selection)
    if len(selected_ids) != 15:
        raise ValueError(f"expected 15 cases, got {len(selected_ids)}")
    gold = load_gold(args.dataset, args.split, selected_ids)

    configs = {}
    for config, paths in (
        ("q4_guard_off", args.q4),
        ("context_safe_guard_on", args.guard_on),
        ("original_guard_off", args.original),
    ):
        rows = []
        for repeat, path in enumerate(paths):
            rows.extend(
                load_repeat(path, config, repeat, selected_ids, gold, 8)
            )
        configs[config] = summarize_config(rows, selected_ids)

    flip_classes = {}
    with open(args.flip_file, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            flip_classes[row["case_id"]] = row["flip_class"]

    per_case = []
    for case_id in sorted(selected_ids):
        row = {"case_id": case_id, "flip_class": flip_classes[case_id]}
        for config, config_rows in configs.items():
            item = next(x for x in config_rows if x["case_id"] == case_id)
            row[f"{config}_outcomes"] = item["outcomes"]
            row[f"{config}_pairwise_agreement"] = item["pairwise_agreement"]
            row[f"{config}_unanimous"] = item["unanimous"]
            row[f"{config}_modal_consistency"] = item["modal_consistency"]
            row[f"{config}_acc_at5"] = item["correct_rate"]
        row["q4_minus_guard_on"] = (
            row["q4_guard_off_pairwise_agreement"]
            - row["context_safe_guard_on_pairwise_agreement"]
        )
        row["q4_minus_original"] = (
            row["q4_guard_off_pairwise_agreement"]
            - row["original_guard_off_pairwise_agreement"]
        )
        per_case.append(row)

    guard_on = compare(
        configs["context_safe_guard_on"], configs["q4_guard_off"], "guard_on"
    )
    original = compare(
        configs["original_guard_off"], configs["q4_guard_off"], "original_guard_off"
    )
    q4_agreement = guard_on["q4_agreement"]
    verdict = classify(q4_agreement, guard_on, original)

    subgroup = {}
    for flip_class in ("memory_flipped", "stable_harm"):
        subset = [row for row in per_case if row["flip_class"] == flip_class]
        subgroup[flip_class] = {
            "n": len(subset),
            "q4_pairwise_agreement": float(
                np.mean([row["q4_guard_off_pairwise_agreement"] for row in subset])
            ),
            "guard_on_pairwise_agreement": float(
                np.mean([row["context_safe_guard_on_pairwise_agreement"] for row in subset])
            ),
            "original_pairwise_agreement": float(
                np.mean([row["original_guard_off_pairwise_agreement"] for row in subset])
            ),
        }

    q4_rows = configs["q4_guard_off"]
    summary = {
        "case_count": 15,
        "q4_trajectory_count": 60,
        "q4_pairwise_agreement": q4_agreement,
        "q4_unanimous_rate": float(np.mean([row["unanimous"] for row in q4_rows])),
        "q4_modal_consistency": float(
            np.mean([row["modal_consistency"] for row in q4_rows])
        ),
        "q4_accuracy_at5": float(np.mean([row["correct_rate"] for row in q4_rows])),
        "comparisons": {"guard_on": guard_on, "original_guard_off": original},
        "subgroups": subgroup,
        "verdict": verdict,
        "causal_limit": (
            "The sequential design cannot distinguish sampling variance from "
            "Azure endpoint drift and does not identify a causal guard effect."
        ),
    }

    manifest = {
        "experiment": "Q4 guard-off baseline sequential replication",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "selection": args.selection,
        "selection_sha256": sha256_file(args.selection),
        "model": "gpt-4o",
        "credentials_recorded": False,
        "design": {
            "cases": 15,
            "arm": "LocAgent baseline",
            "fresh_repeats_per_case": 4,
            "fresh_trajectories": 60,
            "sequential_comparison": True,
        },
        "q4_runs": load_run_args(args.q4),
        "reference_inputs": {
            "context_safe_guard_on": args.guard_on,
            "original_guard_off": args.original,
        },
        "result": {
            "pairwise_agreement": q4_agreement,
            "verdict": verdict,
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "q4_per_case.csv"), per_case)
    comparison_rows = []
    for item in (guard_on, original):
        comparison_rows.append(
            {
                **{key: value for key, value in item.items() if key != "ci"},
                "ci_low": item["ci"][0],
                "ci_high": item["ci"][1],
            }
        )
    write_csv(
        os.path.join(args.output_dir, "q4_reference_comparison.csv"),
        comparison_rows,
    )
    with open(os.path.join(args.output_dir, "q4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(
        os.path.join(os.path.dirname(args.output_dir), "experiment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.output_dir, "q4_guard_off_report.md"), "w", encoding="utf-8") as f:
        f.write("# Q4 Guard-Off Baseline Replication\n\n")
        f.write(f"- Cases: 15\n- Fresh trajectories: 60\n")
        f.write(f"- Q4 pairwise agreement: {q4_agreement:.4f}\n")
        f.write(f"- Q4 unanimous agreement: {summary['q4_unanimous_rate']:.4f}\n")
        f.write(f"- Q4 modal consistency: {summary['q4_modal_consistency']:.4f}\n")
        f.write(f"- Q4 selected-subset Acc@5: {summary['q4_accuracy_at5']:.4f}\n")
        f.write(f"- Verdict: **{verdict}**\n\n")
        f.write("## Reference Comparisons\n\n")
        f.write("| Reference | Reference | Q4 | Delta | 95% CI | Exact p |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for item in (guard_on, original):
            f.write(
                f"| {item['reference']} | {item['reference_agreement']:.4f} | "
                f"{item['q4_agreement']:.4f} | {item['q4_minus_reference']:+.4f} | "
                f"[{item['ci'][0]:+.4f}, {item['ci'][1]:+.4f}] | "
                f"{item['exact_p']:.4f} |\n"
            )
        f.write("\n## Subgroups\n\n")
        f.write("| Original class | N | Q4 off | Guard-on | Original off |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for label, item in subgroup.items():
            f.write(
                f"| {label} | {item['n']} | {item['q4_pairwise_agreement']:.4f} | "
                f"{item['guard_on_pairwise_agreement']:.4f} | "
                f"{item['original_pairwise_agreement']:.4f} |\n"
            )
        f.write("\n## Interpretation Limit\n\n")
        f.write(summary["causal_limit"] + " Failure to reject is not evidence of equivalence.\n")
        f.write("The selected harm subset is diagnostic and does not estimate full-500 population stability.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
