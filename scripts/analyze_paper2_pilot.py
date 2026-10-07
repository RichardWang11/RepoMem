import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from scripts.analyze_m3a_matched_noise_floor import (
    load_gold,
    predicted_files,
    stability,
)
from scripts.sample_random_verified_pilot import repository_key
from scripts.validate_context_safe_repeat import load_jsonl


COST_RE = re.compile(
    r"cost_analysis\.py INFO input_tokens=(\d+), output_tokens=(\d+), cost=([0-9.]+)"
)


def bootstrap_ci(values, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def sample_size_target(sd_gap, pilot_half_width, pilot_n=20, population_n=485):
    if sd_gap < 0 or pilot_half_width < 0:
        raise ValueError("standard deviation and half-width must be nonnegative")
    n_0 = (1.96 * sd_gap / 0.03) ** 2
    n_fpc = n_0 / (1 + (n_0 - 1) / population_n)
    n_target = math.ceil(max(n_fpc, pilot_n))
    projected = pilot_half_width * math.sqrt(pilot_n / n_target)
    normal_fpc_half_width = (
        1.96
        * sd_gap
        / math.sqrt(n_target)
        * math.sqrt((population_n - n_target) / (population_n - 1))
    )
    return {
        "sd_gap": sd_gap,
        "n_0_uncorrected": n_0,
        "n_fpc": n_fpc,
        "n_target": n_target,
        "pilot_ci_half_width": pilot_half_width,
        "projected_target_half_width": projected,
        "normal_fpc_target_half_width": normal_fpc_half_width,
        "projection_mismatch": (
            normal_fpc_half_width <= 0.03 < projected
        ),
    }


def parse_cost(log_path):
    totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    if not Path(log_path).exists():
        return totals
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = COST_RE.search(line)
        if match:
            totals["input_tokens"] += int(match.group(1))
            totals["output_tokens"] += int(match.group(2))
            totals["cost_usd"] += float(match.group(3))
    totals["cost_usd"] = round(totals["cost_usd"], 2)
    return totals


def load_batch(run_dir, expected_ids):
    run_dir = Path(run_dir)
    repaired_path = run_dir / "loc_trajs.repaired.jsonl"
    rows = load_jsonl(repaired_path) if repaired_path.exists() else []
    raw_path = run_dir / "loc_trajs.jsonl"
    raw_rows = load_jsonl(raw_path) if raw_path.exists() else []
    raw_by_id = {row.get("instance_id"): row for row in raw_rows}
    by_id = {row.get("instance_id"): row for row in rows}
    fingerprint_path = run_dir / "fingerprint_audit_summary.json"
    fingerprint = (
        json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if fingerprint_path.exists()
        else None
    )
    audit_path = run_dir / "context_guard_audit.jsonl"
    audits = load_jsonl(audit_path) if audit_path.exists() else []
    repair_count = sum(
        raw_by_id.get(case_id, {}).get("found_files") != row.get("found_files")
        for case_id, row in by_id.items()
    )
    return {
        "rows": by_id,
        "missing_ids": sorted(set(expected_ids) - set(by_id)),
        "extra_ids": sorted(set(by_id) - set(expected_ids)),
        "fingerprint": fingerprint,
        "context_audits": audits,
        "repair_count": repair_count,
        "cost": parse_cost(run_dir / "localize.log"),
    }


def aggregate_arm(case_rows, arm):
    rows = [row for row in case_rows if row[f"{arm}_stability"] is not None]
    if not rows:
        return None
    return {
        "n_cases": len(rows),
        "pairwise_stability": float(np.mean([
            row[f"{arm}_stability"] for row in rows
        ])),
        "unanimous_rate": float(np.mean([
            row[f"{arm}_unanimous"] for row in rows
        ])),
        "acc_at1": float(np.mean([
            value for row in rows for value in row[f"{arm}_at1"]
        ])),
        "acc_at3": float(np.mean([
            value for row in rows for value in row[f"{arm}_at3"]
        ])),
        "acc_at5": float(np.mean([
            value for row in rows for value in row[f"{arm}_at5"]
        ])),
        "empty_predictions": int(sum(
            value for row in rows for value in row[f"{arm}_empty"]
        )),
    }


def decision_status(protocol_failures, target):
    if protocol_failures:
        return "PROTOCOL_INVALID"
    n_target = target["n_target"]
    if n_target <= 20:
        return "PILOT_SUFFICIENT"
    if n_target <= 200:
        return "EXPAND_TO_TARGET"
    return "REDESIGN_FOR_SCALE"


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(summary):
    primary = summary.get("primary") or {}
    target = summary.get("sample_size_target") or {}
    lines = [
        "# Paper 2 Random-Task Pilot",
        "",
        "## Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        f"Valid complete cases: {summary['valid_complete_cases']}/20.",
        f"Completed trajectories: {summary['completed_trajectories']}/160.",
        f"Observed cost: ${summary['cost']['cost_usd']:.2f}.",
        "",
        "## Stability And Accuracy",
        "",
        "| Arm | Stability | Acc@1 | Acc@3 | Acc@5 | Empty |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("baseline", "memory"):
        row = summary.get(arm)
        if row:
            lines.append(
                f"| {arm} | {row['pairwise_stability']:.4f} | "
                f"{row['acc_at1']:.4f} | {row['acc_at3']:.4f} | "
                f"{row['acc_at5']:.4f} | {row['empty_predictions']} |"
            )
    if primary:
        lines.extend([
            "",
            f"Baseline-minus-memory stability gap: {primary['gap']:.4f} "
            f"(95% bootstrap CI [{primary['ci'][0]:.4f}, "
            f"{primary['ci'][1]:.4f}]).",
        ])
    lines.extend(["", "## Sample-Size Calibration", ""])
    if target:
        lines.extend([
            f"- Pilot SD of case gaps: {target['sd_gap']:.6f}",
            f"- Uncorrected n0: {target['n_0_uncorrected']:.3f}",
            f"- FPC-adjusted n: {target['n_fpc']:.3f}",
            f"- Locked integer target: {target['n_target']}",
            f"- Projected target half-width: "
            f"{target['projected_target_half_width']:.4f}",
            f"- Normal/FPC half-width at target: "
            f"{target['normal_fpc_target_half_width']:.4f}",
            f"- Projection mismatch: {target['projection_mismatch']}",
            f"- Sixty sufficient: {summary['sixty_sufficient']}",
            f"- Projected total cost at n=60: "
            f"${summary['projected_cost_at_60']:.2f}",
            f"- Projected total cost at n_target: "
            f"${summary['projected_cost_at_target']:.2f}",
        ])
    else:
        lines.append("Unavailable because the complete-case protocol failed.")
    lines.extend(["", "## Protocol Audit", ""])
    failures = summary["protocol_failures"]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("All preregistered protocol checks passed.")
    lines.extend([
        "",
        "Per-case stability is discrete and descriptive. The paired aggregate "
        "gap is the primary estimand; this pilot is not a significance test.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration", default="results/paper2_pilot/preregistration.json"
    )
    parser.add_argument("--run_root", default="results/paper2_pilot/runs")
    parser.add_argument("--output_dir", default="results/paper2_pilot/analysis")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    prereg = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    selected = prereg["sampling"]["selected_ids"]
    selected_set = set(selected)
    gold = load_gold(args.dataset, args.split, selected_set)
    predictions = {
        case_id: {arm: {} for arm in ("baseline", "memory")}
        for case_id in selected
    }
    batch_rows = []
    protocol_failures = []
    total_cost = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    total_repairs = 0
    fingerprint_responses = 0
    max_prompt_tokens = 0
    completed_trajectories = 0

    for batch in prereg["design"]["execution_batches"]:
        cohort = batch["cohort"]
        arm = batch["arm"]
        repeat = batch["repeat"]
        expected_ids = prereg["sampling"][f"cohort_{cohort}"]
        name = (
            f"batch_{batch['batch']:02d}_cohort_{cohort}_{arm}_r{repeat}"
        )
        run_dir = Path(args.run_root) / name
        if not run_dir.exists():
            protocol_failures.append(f"missing batch directory: {name}")
            continue
        loaded = load_batch(run_dir, expected_ids)
        completed_trajectories += len(loaded["rows"])
        total_repairs += loaded["repair_count"]
        for key in total_cost:
            total_cost[key] += loaded["cost"][key]
        fingerprint = loaded["fingerprint"]
        if fingerprint:
            fingerprint_responses += fingerprint.get("response_count", 0)
        if not fingerprint or not fingerprint.get("valid"):
            protocol_failures.append(f"invalid fingerprint/runtime audit: {name}")
        if loaded["missing_ids"] or loaded["extra_ids"]:
            protocol_failures.append(
                f"trajectory membership mismatch: {name} "
                f"missing={loaded['missing_ids']} extra={loaded['extra_ids']}"
            )
        truncated = [
            row.get("instance_id")
            for row in loaded["context_audits"]
            if row.get("summary", {}).get("truncated_observation_count")
        ]
        if truncated:
            protocol_failures.append(f"context truncation in {name}: {truncated}")
        if loaded["context_audits"]:
            max_prompt_tokens = max(
                max_prompt_tokens,
                max(
                    row.get("summary", {}).get("max_prompt_tokens", 0)
                    for row in loaded["context_audits"]
                ),
            )
        for case_id, row in loaded["rows"].items():
            predictions[case_id][arm][repeat] = predicted_files(row)
        batch_rows.append({
            **batch,
            "name": name,
            "trajectory_count": len(loaded["rows"]),
            **loaded["cost"],
            "parser_repairs": loaded["repair_count"],
            "valid": bool(fingerprint and fingerprint.get("valid"))
            and not loaded["missing_ids"]
            and not loaded["extra_ids"]
            and not truncated,
        })

    total_cost["cost_usd"] = round(total_cost["cost_usd"], 2)
    case_rows = []
    for case_id in selected:
        row = {
            "case_id": case_id,
            "repository": repository_key(case_id),
        }
        complete = True
        for arm in ("baseline", "memory"):
            files_by_repeat = predictions[case_id][arm]
            if set(files_by_repeat) != {1, 2, 3, 4}:
                complete = False
                row[f"{arm}_at1"] = None
                row[f"{arm}_at3"] = None
                row[f"{arm}_at5"] = None
                row[f"{arm}_empty"] = None
                row[f"{arm}_stability"] = None
                row[f"{arm}_unanimous"] = None
                continue
            outcomes = {k: [] for k in (1, 3, 5)}
            empties = []
            gold_set = set(gold[case_id])
            for repeat in range(1, 5):
                files = files_by_repeat[repeat]
                empties.append(int(not files))
                for k in outcomes:
                    outcomes[k].append(int(gold_set.issubset(set(files[:k]))))
            metric = stability(outcomes[5])
            row[f"{arm}_at1"] = outcomes[1]
            row[f"{arm}_at3"] = outcomes[3]
            row[f"{arm}_at5"] = outcomes[5]
            row[f"{arm}_empty"] = empties
            row[f"{arm}_stability"] = metric["pairwise_agreement"]
            row[f"{arm}_unanimous"] = metric["unanimous"]
        row["complete"] = complete
        row["stability_gap"] = (
            row["baseline_stability"] - row["memory_stability"]
            if complete
            else None
        )
        case_rows.append(row)

    valid_rows = [row for row in case_rows if row["complete"]]
    if len(valid_rows) != 20:
        protocol_failures.append(
            f"complete cases={len(valid_rows)}, preregistered=20"
        )
    gaps = [row["stability_gap"] for row in valid_rows]
    primary = None
    target = None
    if len(valid_rows) == 20:
        ci = bootstrap_ci(gaps)
        primary = {"gap": float(np.mean(gaps)), "ci": ci}
        half_width = (ci[1] - ci[0]) / 2
        target = sample_size_target(
            float(np.std(gaps, ddof=1)), half_width
        )

    summary = {
        "verdict": decision_status(protocol_failures, target) if target else "PROTOCOL_INVALID",
        "valid_complete_cases": len(valid_rows),
        "completed_trajectories": completed_trajectories,
        "baseline": aggregate_arm(case_rows, "baseline"),
        "memory": aggregate_arm(case_rows, "memory"),
        "primary": primary,
        "sample_size_target": target,
        "sixty_sufficient": bool(target and target["n_target"] <= 60),
        "protocol_failures": protocol_failures,
        "cost": total_cost,
        "parser_repairs": total_repairs,
        "fingerprint_response_count": fingerprint_responses,
        "max_prompt_tokens": max_prompt_tokens,
        "projected_cost_at_target": (
            total_cost["cost_usd"] / completed_trajectories
            * target["n_target"] * 8
            if target and completed_trajectories
            else None
        ),
        "projected_cost_at_60": (
            total_cost["cost_usd"] / completed_trajectories * 60 * 8
            if completed_trajectories
            else None
        ),
        "batch_count": len(batch_rows),
        "preregistration": args.preregistration,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pilot_report.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    with (output_dir / "per_case_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "batch_audit.csv", batch_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
