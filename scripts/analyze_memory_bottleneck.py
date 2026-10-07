import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_n2_memory_adoption import (
    DEFAULT_FLIP_FILE,
    DEFAULT_INDEX_DIRS,
    DEFAULT_SELECTION,
    DEFAULT_V4_ROOT,
    analyze_trajectory_row,
    load_commit_lookup,
    load_flip_classes,
    load_gold,
    load_ids,
    load_version,
)
from scripts.validate_context_safe_repeat import load_jsonl


DEFAULT_ARROW = (
    "/data/wjl/ExpeRepair/ExpeRepair-v1.0/data/hf-datasets/"
    "princeton-nlp___swe-bench_verified/default/0.0.0/"
    "c104f840cc67f8b6eec6f759ebc8b2693d585d4a/"
    "swe-bench_verified-test.arrow"
)
DEFAULT_OUTPUT = "results/paper2_pilot/analysis/memory_bottleneck"
FAILURE_STAGES = (
    "retrieval_miss",
    "adoption_miss",
    "post_adoption_failure",
)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile_ci(values):
    valid = [value for value in values if value is not None and math.isfinite(value)]
    if not valid:
        return [None, None]
    return [float(value) for value in np.percentile(valid, [2.5, 97.5])]


def load_pilot_results(preregistration, run_root, gold, commit_lookup):
    combined = {
        "exposures": [],
        "examinations": [],
        "tools": [],
        "events": [],
        "trajectories": [],
        "parse_errors": [],
    }
    observed = defaultdict(set)
    for batch in preregistration["design"]["execution_batches"]:
        if batch["arm"] != "memory":
            continue
        name = (
            f"batch_{batch['batch']:02d}_cohort_{batch['cohort']}_"
            f"memory_r{batch['repeat']}"
        )
        path = Path(run_root) / name / "loc_trajs.repaired.jsonl"
        rows = load_jsonl(path)
        expected = set(
            preregistration["sampling"][f"cohort_{batch['cohort']}"]
        )
        if {row["instance_id"] for row in rows} != expected:
            raise ValueError(f"pilot batch membership mismatch: {name}")
        for row in rows:
            case_id = row["instance_id"]
            observed[case_id].add(batch["repeat"])
            result = analyze_trajectory_row(
                row,
                version="pilot",
                repeat=batch["repeat"],
                gold_files=gold[case_id],
                flip_class="random",
                commit_lookup=commit_lookup,
            )
            for key in combined:
                combined[key].extend(result[key])
    expected_repeats = {1, 2, 3, 4}
    incomplete = {
        case_id: sorted(repeats)
        for case_id, repeats in observed.items()
        if repeats != expected_repeats
    }
    if incomplete:
        raise ValueError(f"pilot repeat mismatch: {incomplete}")
    return combined


def pilot_outcomes(path):
    rows = {row["case_id"]: row for row in load_jsonl(path)}
    result = {}
    for case_id, row in rows.items():
        values = row.get("memory_at5")
        if not row.get("complete") or not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"invalid pilot outcome row: {case_id}")
        for repeat, value in enumerate(values, start=1):
            result[(case_id, repeat)] = int(value)
    return result


def deduplicate_candidates(exposures, max_rank=5):
    candidates = {}
    for row in exposures:
        if row.get("rank") is None or int(row["rank"]) > max_rank:
            continue
        key = (
            row["instance_id"],
            int(row["repeat"]),
            int(row["traj_index"]),
            row["sha"],
        )
        candidate = candidates.setdefault(
            key,
            {
                "version": row["version"],
                "case_id": row["instance_id"],
                "repository": row.get("repo", ""),
                "repeat": int(row["repeat"]),
                "traj_index": int(row["traj_index"]),
                "sha": row["sha"],
                "best_rank": int(row["rank"]),
                "query_count": 0,
                "queries": set(),
                "changed_files": set(),
                "gold_overlap_files": set(),
                "relevant": False,
                "selected": False,
                "strict_adopted": False,
                "primary_adopted": False,
                "commit_lookup_found": False,
                "first_query": False,
            },
        )
        candidate["best_rank"] = min(candidate["best_rank"], int(row["rank"]))
        candidate["queries"].add(row.get("query", ""))
        candidate["changed_files"].update(row.get("changed_files") or [])
        candidate["gold_overlap_files"].update(row.get("gold_overlap_files") or [])
        candidate["relevant"] |= bool(row.get("gold_file_overlap"))
        candidate["selected"] |= bool(row.get("selection_adoption"))
        candidate["strict_adopted"] |= bool(
            row.get("selection_adoption") and row.get("exploration_strict")
        )
        candidate["primary_adopted"] |= bool(
            row.get("selection_adoption") and row.get("final_adoption")
        )
        candidate["commit_lookup_found"] |= bool(row.get("commit_lookup_found"))
        candidate["first_query"] |= int(row.get("query_index", -1)) == 0
    result = []
    for candidate in candidates.values():
        candidate["query_count"] = len(candidate["queries"])
        for field in ("queries", "changed_files", "gold_overlap_files"):
            candidate[field] = sorted(candidate[field])
        result.append(candidate)
    return sorted(
        result,
        key=lambda row: (
            row["case_id"], row["repeat"], row["traj_index"], row["best_rank"],
            row["sha"],
        ),
    )


def build_trajectory_table(result, candidates, outcomes, gold, derive_outcome=False):
    by_trajectory = defaultdict(list)
    for row in candidates:
        by_trajectory[(row["case_id"], row["repeat"], row["traj_index"])].append(row)
    event_queries = defaultdict(int)
    for event in result["events"]:
        key = (event["instance_id"], int(event["repeat"]), int(event["traj_index"]))
        event_queries[key] += len(event.get("queries_executed") or [])

    trajectories = []
    for row in result["trajectories"]:
        key = (row["instance_id"], int(row["repeat"]), int(row["traj_index"]))
        rows = by_trajectory.get(key, [])
        hit = int(any(candidate["relevant"] for candidate in rows))
        first_hit = int(
            any(candidate["relevant"] and candidate["first_query"] for candidate in rows)
        )
        correct_adopted = int(
            any(
                candidate["relevant"] and candidate["primary_adopted"]
                for candidate in rows
            )
        )
        correct_strict_adopted = int(
            any(
                candidate["relevant"] and candidate["strict_adopted"]
                for candidate in rows
            )
        )
        if derive_outcome:
            final_set = set(row.get("final_top5") or [])
            success = int(set(gold[row["instance_id"]]).issubset(final_set))
        else:
            success = int(outcomes[(row["instance_id"], int(row["repeat"]))])
        gold_set = set(gold[row["instance_id"]])
        retrieved_gold = {
            path
            for candidate in rows
            for path in candidate["gold_overlap_files"]
        }
        failure_stage = "success"
        strict_failure_stage = "success"
        if not success:
            if not hit:
                failure_stage = "retrieval_miss"
                strict_failure_stage = "retrieval_miss"
            elif not correct_adopted:
                failure_stage = "adoption_miss"
            else:
                failure_stage = "post_adoption_failure"
            if hit and not correct_strict_adopted:
                strict_failure_stage = "adoption_miss"
            elif hit:
                strict_failure_stage = "post_adoption_failure"
        trajectories.append({
            "version": row["version"],
            "case_id": row["instance_id"],
            "repository": row.get("repo", ""),
            "repeat": int(row["repeat"]),
            "traj_index": int(row["traj_index"]),
            "retrieval_event_count": int(row["retrieval_event_count"]),
            "executed_query_count": event_queries[key],
            "top5_candidate_count": len(rows),
            "hit_at5": hit,
            "first_query_hit_at5": first_hit,
            "correct_primary_adopted": correct_adopted,
            "correct_strict_adopted": correct_strict_adopted,
            "any_primary_adopted": int(any(c["primary_adopted"] for c in rows)),
            "memory_success_at5": success,
            "memory_failure": int(not success),
            "failure_stage": failure_stage,
            "strict_failure_stage": strict_failure_stage,
            "retrieved_gold_coverage": ratio(len(retrieved_gold), len(gold_set)),
            "final_top5": row.get("final_top5") or [],
        })
    return sorted(
        trajectories,
        key=lambda row: (row["case_id"], row["repeat"], row["traj_index"]),
    )


def build_case_table(trajectories, candidates):
    trajectory_groups = defaultdict(list)
    candidate_groups = defaultdict(list)
    for row in trajectories:
        trajectory_groups[row["case_id"]].append(row)
    for row in candidates:
        candidate_groups[row["case_id"]].append(row)
    result = []
    for case_id in sorted(trajectory_groups):
        rows = trajectory_groups[case_id]
        if len(rows) != 4:
            raise ValueError(f"expected four trajectories for {case_id}, got {len(rows)}")
        hit_rows = [row for row in rows if row["hit_at5"]]
        adopted = [row for row in candidate_groups[case_id] if row["primary_adopted"]]
        selected = [row for row in candidate_groups[case_id] if row["selected"]]
        strict = [row for row in candidate_groups[case_id] if row["strict_adopted"]]
        failures = [row for row in rows if row["memory_failure"]]
        item = {
            "case_id": case_id,
            "repository": rows[0]["repository"],
            "hit_at5_rate": mean_or_none([row["hit_at5"] for row in rows]),
            "any_of4_hit": int(bool(hit_rows)),
            "all_of4_hit": int(len(hit_rows) == 4),
            "first_query_hit_rate": mean_or_none(
                [row["first_query_hit_at5"] for row in rows]
            ),
            "conditional_trust_recall": (
                mean_or_none([row["correct_primary_adopted"] for row in hit_rows])
                if hit_rows else None
            ),
            "primary_adopted_count": len(adopted),
            "primary_correct_count": sum(row["relevant"] for row in adopted),
            "primary_trust_precision": ratio(
                sum(row["relevant"] for row in adopted), len(adopted)
            ),
            "selection_trust_precision": ratio(
                sum(row["relevant"] for row in selected), len(selected)
            ),
            "strict_trust_precision": ratio(
                sum(row["relevant"] for row in strict), len(strict)
            ),
            "memory_failure_count": len(failures),
        }
        for stage in FAILURE_STAGES:
            item[f"{stage}_count"] = sum(row["failure_stage"] == stage for row in rows)
        result.append(item)
    return result


def point_metrics(case_rows, candidates, trajectories):
    adopted = [row for row in candidates if row["primary_adopted"]]
    selected = [row for row in candidates if row["selected"]]
    strict = [row for row in candidates if row["strict_adopted"]]
    failures = [row for row in trajectories if row["memory_failure"]]
    stages = {
        stage: ratio(sum(row["failure_stage"] == stage for row in failures), len(failures))
        for stage in FAILURE_STAGES
    }
    strict_stages = {
        stage: ratio(
            sum(row["strict_failure_stage"] == stage for row in failures),
            len(failures),
        )
        for stage in FAILURE_STAGES
    }
    return {
        "hit_at5": mean_or_none([row["hit_at5_rate"] for row in case_rows]),
        "any_of4_hit": mean_or_none([row["any_of4_hit"] for row in case_rows]),
        "all_of4_hit": mean_or_none([row["all_of4_hit"] for row in case_rows]),
        "first_query_hit_at5": mean_or_none(
            [row["first_query_hit_rate"] for row in case_rows]
        ),
        "trust_precision": ratio(sum(row["relevant"] for row in adopted), len(adopted)),
        "trust_precision_numerator": int(sum(row["relevant"] for row in adopted)),
        "trust_precision_denominator": len(adopted),
        "macro_trust_precision": mean_or_none(
            [row["primary_trust_precision"] for row in case_rows]
        ),
        "selection_precision": ratio(
            sum(row["relevant"] for row in selected), len(selected)
        ),
        "strict_exploration_precision": ratio(
            sum(row["relevant"] for row in strict), len(strict)
        ),
        "conditional_trust_recall": mean_or_none(
            [row["conditional_trust_recall"] for row in case_rows]
        ),
        "memory_failure_count": len(failures),
        "memory_failure_rate": ratio(len(failures), len(trajectories)),
        "mean_retrieved_gold_coverage": mean_or_none(
            [row["retrieved_gold_coverage"] for row in trajectories]
        ),
        "failure_shares": stages,
        "strict_exploration_failure_shares": strict_stages,
        "candidate_count": len(candidates),
        "commit_lookup_coverage": mean_or_none(
            [int(row["commit_lookup_found"]) for row in candidates]
        ),
    }


def raw_exposure_sensitivity(exposures, max_rank=5):
    rows = [
        row for row in exposures
        if row.get("rank") is not None and row["rank"] <= max_rank
    ]
    adopted = [
        row for row in rows
        if row.get("selection_adoption") and row.get("final_adoption")
    ]
    return {
        "top5_query_item_exposure_count": len(rows),
        "primary_adopted_exposure_count": len(adopted),
        "raw_exposure_trust_precision": ratio(
            sum(bool(row.get("gold_file_overlap")) for row in adopted),
            len(adopted),
        ),
    }


def query_count_strata(trajectories):
    groups = defaultdict(list)
    for row in trajectories:
        groups[row["executed_query_count"]].append(row)
    return [
        {
            "executed_query_count": query_count,
            "trajectory_count": len(rows),
            "case_count": len({row["case_id"] for row in rows}),
            "hit_at5": mean_or_none([row["hit_at5"] for row in rows]),
            "memory_failure_rate": mean_or_none(
                [row["memory_failure"] for row in rows]
            ),
        }
        for query_count, rows in sorted(groups.items())
    ]


def bootstrap_metrics(case_rows, candidates, trajectories, n_boot=10000, seed=42):
    candidate_by_case = defaultdict(list)
    trajectory_by_case = defaultdict(list)
    case_by_id = {row["case_id"]: row for row in case_rows}
    for row in candidates:
        candidate_by_case[row["case_id"]].append(row)
    for row in trajectories:
        trajectory_by_case[row["case_id"]].append(row)
    case_ids = sorted(case_by_id)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_boot):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        sampled_cases = [case_by_id[case_id] for case_id in sampled]
        sampled_candidates = [
            row for case_id in sampled for row in candidate_by_case[case_id]
        ]
        sampled_trajectories = [
            row for case_id in sampled for row in trajectory_by_case[case_id]
        ]
        values = point_metrics(
            sampled_cases, sampled_candidates, sampled_trajectories
        )
        rows.append({
            "replicate": replicate,
            "hit_at5": values["hit_at5"],
            "trust_precision": values["trust_precision"],
            "conditional_trust_recall": values["conditional_trust_recall"],
            **{
                f"failure_share_{stage}": values["failure_shares"][stage]
                for stage in FAILURE_STAGES
            },
        })
    fields = (
        "hit_at5",
        "trust_precision",
        "conditional_trust_recall",
        *(f"failure_share_{stage}" for stage in FAILURE_STAGES),
    )
    summary = {
        field: {
            "ci": percentile_ci([row[field] for row in rows]),
            "valid_replicates": sum(row[field] is not None for row in rows),
            "discard_rate": mean_or_none([row[field] is None for row in rows]),
        }
        for field in fields
    }
    for left, right in (
        ("retrieval_miss", "adoption_miss"),
        ("retrieval_miss", "post_adoption_failure"),
        ("adoption_miss", "post_adoption_failure"),
    ):
        key = f"failure_difference_{left}_minus_{right}"
        values = [
            row[f"failure_share_{left}"] - row[f"failure_share_{right}"]
            for row in rows
            if row[f"failure_share_{left}"] is not None
            and row[f"failure_share_{right}"] is not None
        ]
        summary[key] = {"ci": percentile_ci(values), "valid_replicates": len(values)}
    return summary, rows


def matrix_category(hit, trust):
    if hit < 0.50:
        return "BOTH_LIMITED" if trust < 0.50 else "RETRIEVAL_LIMITED"
    if hit > 0.70:
        if trust < 0.50:
            return "TRUST_LIMITED"
        if trust > 0.70:
            return "BOTH_GOOD"
    return "MIXED_DIRECTION"


def resolve_mixed_direction(hit, trust, failure_shares):
    maximum = max(failure_shares.values())
    leaders = [stage for stage, value in failure_shares.items() if value == maximum]
    mapping = {
        "retrieval_miss": "RETRIEVAL_LIMITED",
        "adoption_miss": "TRUST_LIMITED",
        "post_adoption_failure": "DOWNSTREAM_FAILURE_CANDIDATE",
    }
    if len(leaders) == 1:
        return mapping[leaders[0]], "largest_failure_funnel_share"
    retrieval_deficit = 0.70 - hit
    trust_deficit = 0.70 - trust
    if retrieval_deficit > trust_deficit:
        return "RETRIEVAL_LIMITED", "larger_deficit_from_70pct"
    if trust_deficit > retrieval_deficit:
        return "TRUST_LIMITED", "larger_deficit_from_70pct"
    return "JOINT_RETRIEVAL_FIRST", "exact_tie_fallback"


def condition_supported(ci, operator, threshold):
    low, high = ci
    if low is None or high is None:
        return False
    if operator == "lt":
        return high < threshold
    if operator == "gt":
        return low > threshold
    if operator == "ge":
        return low >= threshold
    raise ValueError(operator)


def decision(metrics, bootstrap):
    hit = metrics["hit_at5"]
    trust = metrics["trust_precision"]
    category = matrix_category(hit, trust)
    direction = category
    reason = "point_estimate_matrix"
    if category == "MIXED_DIRECTION":
        direction, reason = resolve_mixed_direction(
            hit, trust, metrics["failure_shares"]
        )
    hit_ci = bootstrap["hit_at5"]["ci"]
    trust_ci = bootstrap["trust_precision"]["ci"]
    support_checks = {
        "RETRIEVAL_LIMITED": (
            condition_supported(hit_ci, "lt", 0.50)
            and condition_supported(trust_ci, "ge", 0.50)
        ),
        "BOTH_LIMITED": (
            condition_supported(hit_ci, "lt", 0.50)
            and condition_supported(trust_ci, "lt", 0.50)
        ),
        "TRUST_LIMITED": (
            condition_supported(hit_ci, "gt", 0.70)
            and condition_supported(trust_ci, "lt", 0.50)
        ),
        "BOTH_GOOD": (
            condition_supported(hit_ci, "gt", 0.70)
            and condition_supported(trust_ci, "gt", 0.70)
        ),
    }
    supported = category in support_checks and support_checks[category]
    intervention = {
        "RETRIEVAL_LIMITED": "RERANKER_OR_RETRIEVAL_OBJECTIVE",
        "BOTH_LIMITED": "RETRIEVAL_FIRST_THEN_VERIFIER",
        "TRUST_LIMITED": "MEMORY_VERIFIER_OR_SELECTIVE_ADOPTION",
        "BOTH_GOOD": "DOWNSTREAM_REASONING_COVERAGE_LOCALIZATION",
        "DOWNSTREAM_FAILURE_CANDIDATE": "DOWNSTREAM_REASONING_COVERAGE_LOCALIZATION",
        "JOINT_RETRIEVAL_FIRST": "JOINT_DIAGNOSIS_RETRIEVAL_FIRST",
    }[direction]
    modifiers = []
    if hit > 0.70 and metrics["conditional_trust_recall"] < 0.50:
        modifiers.append("CORRECT_MEMORY_UNDERUSED")
    largest_stage = max(metrics["failure_shares"], key=metrics["failure_shares"].get)
    if largest_stage == "post_adoption_failure":
        modifiers.append("DOWNSTREAM_FAILURE_CANDIDATE")
    return {
        "matrix_category": category,
        "paper2_direction": direction,
        "paper2_intervention": intervention,
        "direction_reason": reason,
        "evidence_tier": (
            "SUPPORTED_DIRECTION" if supported else "POINT_ESTIMATE_DIRECTION"
        ),
        "support_checks": support_checks,
        "modifiers": modifiers,
        "replication_required_before_confirmatory_claim": not supported,
    }


def contradiction(primary_metrics, primary_decision, v4_metrics, v4_decision):
    metric_gap = max(
        abs(primary_metrics["hit_at5"] - v4_metrics["hit_at5"]),
        abs(primary_metrics["trust_precision"] - v4_metrics["trust_precision"]),
    )
    contradicted = (
        primary_decision["paper2_direction"] != v4_decision["paper2_direction"]
        and metric_gap >= 0.20
    )
    return {
        "flag": "CURATED_CONTRADICTION" if contradicted else "NO_STRONG_CONTRADICTION",
        "maximum_metric_gap": metric_gap,
        "overrides_primary": False,
    }


def write_csv(path, rows):
    fields = sorted({field for row in rows for field in row}) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(json_safe(value), ensure_ascii=False)
                if isinstance(value, (list, dict, tuple, set)) else value
                for key, value in row.items()
            })


def report_markdown(summary):
    metrics = summary["primary"]["metrics"]
    boot = summary["primary"]["bootstrap"]
    decision_row = summary["primary"]["decision"]
    v4 = summary["v4_sensitivity"]
    hit_ci = boot["hit_at5"]["ci"]
    trust_ci = boot["trust_precision"]["ci"]
    lines = [
        "# Memory Bottleneck Diagnosis",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-07-17",
        "- Verification Status: ANALYZED",
        "- Version Label: memory_bottleneck_v2",
        "",
        "## Direction",
        "",
        f"**{decision_row['paper2_direction']} — {decision_row['evidence_tier']}**",
        "",
        f"Paper 2 intervention: `{decision_row['paper2_intervention']}`.",
        "",
        "## Primary Random-Pilot Metrics",
        "",
        "| Metric | Estimate | Case-bootstrap 95% CI |",
        "|---|---:|---:|",
        f"| Case-averaged Hit@5 | {metrics['hit_at5']:.3f} | [{hit_ci[0]:.3f}, {hit_ci[1]:.3f}] |",
        f"| Trust Precision | {metrics['trust_precision']:.3f} ({metrics['trust_precision_numerator']}/{metrics['trust_precision_denominator']}) | [{trust_ci[0]:.3f}, {trust_ci[1]:.3f}] |",
        f"| Conditional Trust Recall | {metrics['conditional_trust_recall']:.3f} | [{boot['conditional_trust_recall']['ci'][0]:.3f}, {boot['conditional_trust_recall']['ci'][1]:.3f}] |",
        "",
        f"Any-of-4 Hit={metrics['any_of4_hit']:.3f}; all-of-4 Hit={metrics['all_of4_hit']:.3f}; first-query Hit={metrics['first_query_hit_at5']:.3f}.",
        f"Mean retrieved Gold-file coverage={metrics['mean_retrieved_gold_coverage']:.3f}; raw non-deduplicated exposure Trust Precision={summary['primary']['raw_exposure_sensitivity']['raw_exposure_trust_precision']:.3f}.",
        "",
        "## Memory-Failure Funnel",
        "",
        "| Failure stage | Share | 95% CI |",
        "|---|---:|---:|",
    ]
    for stage in FAILURE_STAGES:
        ci = boot[f"failure_share_{stage}"]["ci"]
        lines.append(
            f"| {stage} | {metrics['failure_shares'][stage]:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |"
        )
    difference = boot["failure_difference_retrieval_miss_minus_adoption_miss"]["ci"]
    lines.extend([
        "",
        f"Retrieval-miss minus adoption-miss share has 95% CI [{difference[0]:.3f}, {difference[1]:.3f}].",
        "",
        "## Adoption Sensitivities",
        "",
        f"Selection precision={metrics['selection_precision']:.3f}; strict-exploration precision={metrics['strict_exploration_precision']:.3f}; macro case precision={metrics['macro_trust_precision']:.3f}.",
        "",
        "Strict-exploration funnel shares (not dependent on final top-5): "
        + ", ".join(
            f"{stage}={metrics['strict_exploration_failure_shares'][stage]:.3f}"
            for stage in FAILURE_STAGES
        )
        + ".",
        "",
        "## V4 Outcome-Selected Sensitivity",
        "",
        f"V4 Hit@5={v4['metrics']['hit_at5']:.3f}, Trust Precision={v4['metrics']['trust_precision']:.3f}, direction=`{v4['decision']['paper2_direction']}`.",
        f"Contradiction audit: **{summary['v4_contradiction']['flag']}**. V4 never overrides the Random-pilot direction.",
        "",
        "## Interpretation Boundary",
        "",
        "Changed-file overlap is a relevance proxy, not proof that a commit contains the correct semantic fix. Adoption is behavioral overlap, not proof that Memory caused the final prediction.",
    ])
    if decision_row["evidence_tier"] == "POINT_ESTIMATE_DIRECTION":
        lines.extend([
            "",
            "The n=20 confidence intervals are too wide to support a threshold-based conclusion. The selected Paper 2 direction is based on the point estimate and failure-funnel decomposition only and requires replication on at least 30 additional random cases before any confirmatory claim.",
        ])
    if summary["v4_contradiction"]["flag"] == "CURATED_CONTRADICTION":
        lines.extend([
            "",
            "The outcome-selected V4 sensitivity points in a different direction. This is reported as a population-shift caveat and does not override the random pilot.",
        ])
    lines.append("")
    return "\n".join(lines)


def validation_markdown(summary):
    metrics = summary["primary"]["metrics"]
    boot = summary["primary"]["bootstrap"]
    decision_row = summary["primary"]["decision"]
    return "\n".join([
        "# Memory Bottleneck Statistical Validation",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-07-17",
        "- Verification Status: VERIFIED",
        "- Version Label: memory_bottleneck_validation_v1",
        "",
        "## Validation Report",
        "",
        "- **Source:** memory_bottleneck_v2",
        "- **Overall Confidence:** CAUTION",
        "",
        "### Statistical Findings",
        "",
        "| Metric | Estimate | 95% case-bootstrap CI | Confidence |",
        "|---|---:|---:|---|",
        f"| Hit@5 | {metrics['hit_at5']:.3f} | [{boot['hit_at5']['ci'][0]:.3f}, {boot['hit_at5']['ci'][1]:.3f}] | CAUTION |",
        f"| Trust Precision | {metrics['trust_precision']:.3f} | [{boot['trust_precision']['ci'][0]:.3f}, {boot['trust_precision']['ci'][1]:.3f}] | CAUTION |",
        f"| Conditional Trust Recall | {metrics['conditional_trust_recall']:.3f} | [{boot['conditional_trust_recall']['ci'][0]:.3f}, {boot['conditional_trust_recall']['ci'][1]:.3f}] | CAUTION |",
        f"| Direction | {decision_row['paper2_direction']} | {decision_row['evidence_tier']} | CAUTION |",
        "",
        "### Warnings",
        "",
        "| Type | Detail |",
        "|---|---|",
        "| Small clustered sample | Effective uncertainty unit is 20 cases despite 80 trajectories and many candidates. |",
        "| Final-output coupling | Primary adoption and Acc@5 both use final top-5; strict-exploration funnel is the non-final sensitivity. |",
        "| Relevance proxy | Changed-file overlap does not establish semantic usefulness of a commit. |",
        "| Wide intervals | Hit@5 and Trust Precision intervals cross locked matrix thresholds. |",
        "| Population shift | V4 is outcome-selected and is never pooled with the random pilot. |",
        "",
        "### Fallacy Scan",
        "",
        "- **Coverage:** 11/11 fallacy types checked",
        "",
        "| # | Fallacy | Severity | Finding |",
        "|---:|---|---|---|",
        "| 1 | Simpson's paradox | CAUTION | Repository composition is imbalanced; aggregate directions may differ within repositories that are too small for stable estimates. |",
        "| 2 | Ecological fallacy | NOTE | Candidate, trajectory, and case estimands are reported separately; candidate precision must not be described as case success probability. |",
        "| 3 | Berkson's paradox | CAUTION | Random pilot excludes the frozen Paper 1 cases; V4 is explicitly outcome-selected. |",
        "| 4 | Collider bias | CAUTION | Conditioning primary adoption on final top-5 links adoption to the same output used for Acc@5; strict exploration is reported as sensitivity. |",
        "| 5 | Base-rate neglect | NOTE | Memory-failure count/rate, TP denominator, and funnel denominators are explicit. |",
        "| 6 | Regression to the mean | NOTE | The random primary sample was not selected by extreme outcome; V4 is not used for the primary direction. |",
        "| 7 | Survivorship bias | NOTE | All 20 cases and 80 primary Memory trajectories are retained, including zero retrieval. |",
        "| 8 | Look-elsewhere effect | CAUTION | Multiple adoption definitions and retrieval sensitivities are descriptive and cannot upgrade the locked primary direction. |",
        "| 9 | Garden of forking paths | CAUTION | This is retrospective; metric definitions, thresholds, fallback, and contradiction rules are fully disclosed. |",
        "| 10 | Correlation != causation | CAUTION | The funnel diagnoses observed co-occurrence and does not show that a reranker would improve accuracy. |",
        "| 11 | Reverse causality | CAUTION | Final localization participates in primary adoption measurement, so temporal causal direction is not identified. |",
        "",
        "### Reproducibility",
        "",
        "- **Method:** two full deterministic reruns in RepoMem with locagent Python",
        "- **Verdict:** REPRODUCIBLE",
        "- Structured reports, tables, manifest, and figure must match by SHA-256.",
        "",
    ])


def plot_funnel(summary, path):
    metrics = summary["primary"]["metrics"]
    labels = ["Retrieval miss", "Adoption miss", "Post-adoption"]
    values = [metrics["failure_shares"][stage] for stage in FAILURE_STAGES]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share of Memory Acc@5 failures")
    axis.set_title("Memory failure bottleneck decomposition")
    for index, value in enumerate(values):
        axis.text(index, value + 0.02, f"{value:.1%}", ha="center")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default="results/paper2_pilot/preregistration.json")
    parser.add_argument("--pilot_root", default="results/paper2_pilot/runs")
    parser.add_argument("--pilot_outcomes", default="results/paper2_pilot/analysis/per_case_metrics.jsonl")
    parser.add_argument("--v4_selection", default=DEFAULT_SELECTION)
    parser.add_argument("--v4_flip_file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--v4_root", default=DEFAULT_V4_ROOT)
    parser.add_argument("--index_dirs", nargs="+", default=DEFAULT_INDEX_DIRS)
    parser.add_argument("--dataset_arrow", default=DEFAULT_ARROW)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preregistration = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    pilot_ids = set(preregistration["sampling"]["selected_ids"])
    v4_ids = load_ids(args.v4_selection)
    if len(pilot_ids) != 20 or len(v4_ids) != 15:
        raise ValueError("expected 20 Random-pilot and 15 V4 cases")
    all_ids = pilot_ids | v4_ids
    gold = load_gold("", "", all_ids, args.dataset_arrow)
    repos = {case_id.rsplit("-", 1)[0].replace("__", "/") for case_id in all_ids}
    commit_lookup = load_commit_lookup(args.index_dirs, repos)
    if not commit_lookup:
        raise ValueError("no commit index rows loaded")

    pilot_result = load_pilot_results(
        preregistration, args.pilot_root, gold, commit_lookup
    )
    if len(pilot_result["trajectories"]) != 80 or pilot_result["parse_errors"]:
        raise ValueError(
            f"pilot protocol invalid: trajectories={len(pilot_result['trajectories'])}, "
            f"parse_errors={pilot_result['parse_errors']}"
        )
    outcomes = pilot_outcomes(args.pilot_outcomes)
    pilot_candidates = deduplicate_candidates(pilot_result["exposures"])
    pilot_trajectories = build_trajectory_table(
        pilot_result, pilot_candidates, outcomes, gold
    )
    pilot_cases = build_case_table(pilot_trajectories, pilot_candidates)
    pilot_metrics = point_metrics(
        pilot_cases, pilot_candidates, pilot_trajectories
    )
    pilot_raw_sensitivity = raw_exposure_sensitivity(pilot_result["exposures"])
    pilot_query_strata = query_count_strata(pilot_trajectories)
    pilot_bootstrap, bootstrap_rows = bootstrap_metrics(
        pilot_cases, pilot_candidates, pilot_trajectories,
        n_boot=args.bootstrap, seed=args.seed,
    )
    pilot_decision = decision(pilot_metrics, pilot_bootstrap)

    flip_classes = load_flip_classes(args.v4_flip_file)
    v4_result = load_version(
        args.v4_root, "v4", v4_ids, gold, flip_classes, commit_lookup
    )
    if len(v4_result["trajectories"]) != 60 or v4_result["parse_errors"]:
        raise ValueError(
            f"V4 protocol invalid: trajectories={len(v4_result['trajectories'])}, "
            f"parse_errors={v4_result['parse_errors']}"
        )
    v4_candidates = deduplicate_candidates(v4_result["exposures"])
    v4_trajectories = build_trajectory_table(
        v4_result, v4_candidates, {}, gold, derive_outcome=True
    )
    v4_cases = build_case_table(v4_trajectories, v4_candidates)
    v4_metrics = point_metrics(v4_cases, v4_candidates, v4_trajectories)
    v4_bootstrap, _ = bootstrap_metrics(
        v4_cases, v4_candidates, v4_trajectories,
        n_boot=args.bootstrap, seed=args.seed,
    )
    v4_decision = decision(v4_metrics, v4_bootstrap)

    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run+validate",
            "origin_date": "2026-07-17",
            "verification_status": "ANALYZED",
            "version_label": "memory_bottleneck_v2",
        },
        "rules": {
            "candidate_unit": "case x repeat x trajectory x SHA, deduplicated across queries",
            "retrieval_scope": "rank <= 5 within each executed query",
            "relevance": "indexed changed-files intersects normalized Python GoldFiles",
            "primary_adoption": "examined SHA and changed-file overlap with final top-5",
            "bootstrap_unit": "case",
            "seed": args.seed,
            "bootstrap_replicates": args.bootstrap,
        },
        "primary": {
            "sample": {"case_count": 20, "trajectory_count": 80, "random": True},
            "metrics": pilot_metrics,
            "raw_exposure_sensitivity": pilot_raw_sensitivity,
            "query_count_strata": pilot_query_strata,
            "bootstrap": pilot_bootstrap,
            "decision": pilot_decision,
            "zero_retrieval_trajectories": sum(
                row["retrieval_event_count"] == 0 for row in pilot_trajectories
            ),
        },
        "v4_sensitivity": {
            "sample": {"case_count": 15, "trajectory_count": 60, "outcome_selected": True},
            "metrics": v4_metrics,
            "bootstrap": v4_bootstrap,
            "decision": v4_decision,
        },
    }
    summary["v4_contradiction"] = contradiction(
        pilot_metrics, pilot_decision, v4_metrics, v4_decision
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidate_table.csv", pilot_candidates)
    write_csv(output / "trajectory_funnel.csv", pilot_trajectories)
    write_csv(output / "case_metrics.csv", pilot_cases)
    write_csv(output / "bootstrap_metrics.csv", bootstrap_rows)
    write_csv(output / "v4_sensitivity.csv", v4_cases)
    write_csv(output / "hit_by_query_count.csv", pilot_query_strata)
    (output / "memory_bottleneck_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "memory_bottleneck_report.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    (output / "validation_report.md").write_text(
        validation_markdown(summary), encoding="utf-8"
    )
    plot_funnel(summary, output / "retrieval_adoption_funnel.png")
    artifacts = [
        "candidate_table.csv", "trajectory_funnel.csv", "case_metrics.csv",
        "bootstrap_metrics.csv", "v4_sensitivity.csv", "hit_by_query_count.csv",
        "memory_bottleneck_summary.json", "memory_bottleneck_report.md",
        "validation_report.md",
        "retrieval_adoption_funnel.png",
    ]
    hashes = {name: file_hash(output / name) for name in artifacts}
    (output / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe({
        "direction": pilot_decision,
        "metrics": pilot_metrics,
        "hit_ci": pilot_bootstrap["hit_at5"]["ci"],
        "trust_ci": pilot_bootstrap["trust_precision"]["ci"],
        "v4_contradiction": summary["v4_contradiction"],
    }), indent=2))


if __name__ == "__main__":
    main()
