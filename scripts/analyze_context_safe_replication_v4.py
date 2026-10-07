import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.analyze_context_safe_replication_v3 import (
    aggregate_arm_metrics,
    case_stabilities,
    cross_run_agreement,
    gap_summary,
    load_outcomes,
    write_csv,
)
from scripts.analyze_m3a_matched_noise_floor import load_gold, load_ids, stability
from scripts.validate_replication_v4 import (
    EXPECTED_FINGERPRINT,
    EXPECTED_MODEL,
    validate_run,
)


DEFAULT_SELECTION = "reproduce_lists/verified_m3a_matched_15.json"
DEFAULT_FLIP_FILE = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "flip_decomposition/flip_decomposition.csv"
)
DEFAULT_V3 = "results/context_safe_m3a/replication_v3/runs"
DEFAULT_V4 = "results/context_safe_m3a/replication_v4/runs"
DEFAULT_OUTPUT = "results/context_safe_m3a/replication_v4/analysis"
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


def endpoint_hash():
    endpoint = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
        "OPENAI_API_BASE"
    )
    return hashlib.sha256((endpoint or "").encode()).hexdigest() if endpoint else None


def paths_for(root, arm, filename="loc_trajs.repaired.jsonl"):
    return [str(Path(root) / f"{arm}_r{repeat}" / filename) for repeat in range(1, 5)]


def run_metrics(root, selected_ids, gold, label):
    outcomes = {}
    stabilities = {}
    rows = {}
    for arm in ("baseline", "memory"):
        arm_outcomes, arm_rows = load_outcomes(
            paths_for(root, arm), arm, selected_ids, gold
        )
        outcomes[arm] = arm_outcomes
        rows[arm] = arm_rows
        stabilities[arm] = case_stabilities(arm_outcomes)
    gaps = [
        stabilities["baseline"][case_id]["pairwise_agreement"]
        - stabilities["memory"][case_id]["pairwise_agreement"]
        for case_id in sorted(selected_ids)
    ]
    baseline = aggregate_arm_metrics(stabilities["baseline"])
    memory = aggregate_arm_metrics(stabilities["memory"])
    return {
        "label": label,
        "outcomes": outcomes,
        "stabilities": stabilities,
        "rows": rows,
        "summary": {
            "baseline_pairwise_agreement": baseline["pairwise_agreement"],
            "memory_pairwise_agreement": memory["pairwise_agreement"],
            "baseline_unanimous_rate": baseline["unanimous_rate"],
            "memory_unanimous_rate": memory["unanimous_rate"],
            "baseline_modal_consistency": baseline["modal_consistency"],
            "memory_modal_consistency": memory["modal_consistency"],
            "baseline_accuracy_at5": baseline["accuracy_at5"],
            "memory_accuracy_at5": memory["accuracy_at5"],
            **gap_summary(gaps),
        },
    }


def point_verdict(v4_gap, delta):
    if abs(delta) > 0.30:
        return "EXTREME_WITHIN_SNAPSHOT_VARIANCE"
    if v4_gap < -0.05:
        return "DIRECTION_REVERSAL"
    if abs(delta) <= 0.08:
        return "SMALL_GAP_REPLICATED"
    if 0.15 <= v4_gap <= 0.25:
        return "LARGE_WITHIN_SNAPSHOT_VARIANCE"
    return "INCONCLUSIVE_INTERMEDIATE"


def audit_fingerprints(root):
    summaries = []
    response_rows = []
    for arm, repeat in RUN_ORDER:
        summary, rows = validate_run(
            Path(root) / f"{arm}_r{repeat}", arm, repeat, 15
        )
        summaries.append(summary)
        response_rows.extend(rows)
    eligible = all(row["same_fingerprint_eligible"] for row in summaries)
    return {
        "same_fingerprint_eligible": eligible,
        "trajectory_count": sum(row["trajectory_count"] for row in summaries),
        "response_count": sum(row["response_count"] for row in summaries),
        "fingerprint_mismatch_count": sum(
            row["fingerprint_mismatch_count"] for row in summaries
        ),
        "fingerprint_missing_count": sum(
            row["fingerprint_missing_count"] for row in summaries
        ),
        "model_mismatch_count": sum(
            row["model_mismatch_count"] for row in summaries
        ),
        "batch_summaries": summaries,
    }, response_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--flip_file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--v3_root", default=DEFAULT_V3)
    parser.add_argument("--v4_root", default=DEFAULT_V4)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    selected_ids = load_ids(args.selection)
    if len(selected_ids) != 15:
        raise ValueError(f"expected 15 selected cases, got {len(selected_ids)}")
    gold = load_gold(args.dataset, args.split, selected_ids)
    flip_classes = {}
    with open(args.flip_file, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            flip_classes[row["case_id"]] = row["flip_class"]

    v3 = run_metrics(args.v3_root, selected_ids, gold, "v3")
    v4 = run_metrics(args.v4_root, selected_ids, gold, "v4")
    per_case = []
    delta_gaps = []
    pooled_gaps = []
    cross_gaps = []
    for case_id in sorted(selected_ids):
        item = {"case_id": case_id, "flip_class": flip_classes[case_id]}
        gaps = {}
        pooled = {}
        cross = {}
        for version, run in (("v3", v3), ("v4", v4)):
            for arm in ("baseline", "memory"):
                metric = run["stabilities"][arm][case_id]
                item[f"{version}_{arm}_outcomes"] = metric["outcomes"]
                item[f"{version}_{arm}_stability"] = metric["pairwise_agreement"]
            gaps[version] = (
                run["stabilities"]["baseline"][case_id]["pairwise_agreement"]
                - run["stabilities"]["memory"][case_id]["pairwise_agreement"]
            )
            item[f"{version}_gap"] = gaps[version]
        delta = gaps["v4"] - gaps["v3"]
        delta_gaps.append(delta)
        item["v4_minus_v3_gap"] = delta
        for arm in ("baseline", "memory"):
            left = [v3["outcomes"][arm][case_id][index] for index in range(1, 5)]
            right = [v4["outcomes"][arm][case_id][index] for index in range(1, 5)]
            pooled[arm] = stability(left + right)["pairwise_agreement"]
            cross[arm] = cross_run_agreement(left, right)
            item[f"{arm}_pooled8_stability"] = pooled[arm]
            item[f"{arm}_cross_run_agreement"] = cross[arm]
        item["pooled8_gap"] = pooled["baseline"] - pooled["memory"]
        item["cross_run_gap"] = cross["baseline"] - cross["memory"]
        pooled_gaps.append(item["pooled8_gap"])
        cross_gaps.append(item["cross_run_gap"])
        per_case.append(item)

    difference = gap_summary(delta_gaps)
    difference["equivalence_margin"] = 0.10
    difference["equivalence_supported"] = (
        difference["ci"][0] >= -0.10 and difference["ci"][1] <= 0.10
    )
    fingerprint, response_rows = audit_fingerprints(args.v4_root)
    replacement_path = Path(args.v4_root) / "memory_r3" / "replacement_manifest.json"
    replacement = (
        json.loads(replacement_path.read_text(encoding="utf-8"))
        if replacement_path.exists()
        else None
    )
    descriptive_verdict = point_verdict(v4["summary"]["gap"], difference["gap"])
    primary_verdict = (
        descriptive_verdict
        if fingerprint["same_fingerprint_eligible"]
        else "SNAPSHOT_CONTAMINATED"
    )

    subgroups = {}
    for label in ("memory_flipped", "stable_harm"):
        subset = [row for row in per_case if row["flip_class"] == label]
        subgroups[label] = {
            "n": len(subset),
            "v3_gap": float(np.mean([row["v3_gap"] for row in subset])),
            "v4_gap": float(np.mean([row["v4_gap"] for row in subset])),
            "v4_minus_v3_gap": float(np.mean([
                row["v4_minus_v3_gap"] for row in subset
            ])),
        }

    summary = {
        "primary_verdict": primary_verdict,
        "descriptive_point_verdict": descriptive_verdict,
        "v3": v3["summary"],
        "v4": v4["summary"],
        "v4_minus_v3": difference,
        "pooled8": gap_summary(pooled_gaps),
        "cross_run": gap_summary(cross_gaps),
        "fingerprint_audit": fingerprint,
        "replacement": replacement,
        "uninterrupted_run": replacement is None,
        "subgroups": subgroups,
        "interpretation_limit": (
            "Outcome-selected diagnostic harm subset; not a full-500 "
            "performance estimate."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "v3_v4_per_case.csv", per_case)
    with (output_dir / "fingerprint_audit_per_response.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in response_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "replication_v4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "experiment": "V4 same-fingerprint replication of V3",
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
        "expected_provider_metadata": {
            "response_model": EXPECTED_MODEL,
            "system_fingerprint": EXPECTED_FINGERPRINT,
        },
        "observed_fingerprint_audit": fingerprint,
        "replacement": replacement,
        "uninterrupted_run": replacement is None,
        "credentials_recorded": False,
    }
    (output_dir.parent / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = f"""# Context-Safe Same-Fingerprint Replication V4

## Verdict

**{primary_verdict}**

- Fingerprint-clean: {fingerprint['same_fingerprint_eligible']}
- V3 gap: {v3['summary']['gap']:+.4f}
- V4 gap: {v4['summary']['gap']:+.4f}, 95% CI [{v4['summary']['ci'][0]:+.4f}, {v4['summary']['ci'][1]:+.4f}], exact p={v4['summary']['exact_p']:.4f}
- V4 minus V3: {difference['gap']:+.4f}, 95% CI [{difference['ci'][0]:+.4f}, {difference['ci'][1]:+.4f}], exact p={difference['exact_p']:.4f}
- Equivalence within +/-10 pp supported: {difference['equivalence_supported']}

## V4 Arm Metrics

- Baseline pairwise stability: {v4['summary']['baseline_pairwise_agreement']:.4f}
- Memory pairwise stability: {v4['summary']['memory_pairwise_agreement']:.4f}
- Baseline diagnostic-subset Acc@5: {v4['summary']['baseline_accuracy_at5']:.4f}
- Memory diagnostic-subset Acc@5: {v4['summary']['memory_accuracy_at5']:.4f}

## Fingerprint Audit

- Responses: {fingerprint['response_count']}
- Missing fingerprints: {fingerprint['fingerprint_missing_count']}
- Mismatched fingerprints: {fingerprint['fingerprint_mismatch_count']}
- Mismatched response models: {fingerprint['model_mismatch_count']}

## Execution Exception

- Uninterrupted run: {replacement is None}
- Replacement trajectory: {replacement['instance_id'] if replacement else 'none'}
- Replacement reason: {replacement['reason'] if replacement else 'none'}

The replacement was run under the same model, sampling, context, and memory
configuration after the original trajectory hit execution-flow timeout. Its
source and the pre-replacement 14-row files are retained in the replacement
manifest. The replacement ran later and without concurrent peer cases, so this
remains a documented execution caveat.

## Interpretation Limit

These 15 cases were selected from an original baseline-correct/memory-wrong
pool. Results measure run-to-run stability on a diagnostic harm subset and do
not estimate full-500 localization performance.
"""
    (output_dir / "replication_v4_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
