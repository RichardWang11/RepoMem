"""Compare blind GPT taxonomy labels with frozen human adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.memory_transfer_taxonomy import (
    DEFAULT_OUTPUT,
    nominal_metrics,
    safe_kappa,
    taxonomy_primary,
)


SEED = 20260720


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def binary_label(label: str) -> str:
    return "App" if label in {"Direct-help", "Indirect-help"} else "NotApp"


def gpt_as_row(label: dict, case_id: str) -> dict[str, str]:
    return {
        "sample_id": label["sample_id"],
        "case_id": case_id,
        "applicability_label": label["applicability_label"],
        "transfer_bridge_tags": "|".join(label["transfer_bridge_tags"]),
        "primary_transfer_bridge": label["primary_transfer_bridge"] or "",
        "contribution_role_tags": "|".join(label["contribution_role_tags"]),
        "nontransfer_reason_tags": "|".join(label["nontransfer_reason_tags"]),
    }


def cluster_bootstrap_kappa(
    pairs: list[tuple[dict, dict]],
    value_fn,
    replicates: int = 10000,
) -> dict:
    by_case = defaultdict(list)
    for human, gpt in pairs:
        by_case[human["case_id"]].append((human, gpt))
    cases = sorted(by_case)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        left = []
        right = []
        for case_id in sampled:
            for human, gpt in by_case[case_id]:
                human_value, gpt_value = value_fn(human, gpt)
                left.append(human_value)
                right.append(gpt_value)
        kappa = safe_kappa(left, right)
        if kappa is not None:
            values.append(kappa)
    return {
        "replicates": replicates,
        "defined": len(values),
        "undefined_rate": 1 - len(values) / replicates,
        "ci": [float(value) for value in np.percentile(values, [2.5, 97.5])]
        if values else [None, None],
    }


def run(output: Path, replicates: int = 10000) -> dict:
    human = read_csv(output / "holdout_adjudicated.csv")
    human_by_id = {row["sample_id"]: row for row in human}
    labels = read_jsonl(output / "gpt_labels.jsonl")
    accepted = [row for row in labels if row.get("accepted")]
    if len(accepted) != len(human):
        raise ValueError(
            f"expected {len(human)} accepted GPT labels, got {len(accepted)}"
        )
    gpt_by_id = {
        row["sample_id"]: gpt_as_row(
            row, human_by_id[row["sample_id"]]["case_id"]
        )
        for row in accepted
    }
    pairs = [(row, gpt_by_id[row["sample_id"]]) for row in human]
    strata = {
        row["sample_id"]: row["sampling_stratum"]
        for row in read_csv(output / "sealed_machine_features_after_freeze.csv")
    }

    human_app = [row["applicability_label"] for row, _ in pairs]
    gpt_app = [row["applicability_label"] for _, row in pairs]
    human_binary = [binary_label(value) for value in human_app]
    gpt_binary = [binary_label(value) for value in gpt_app]
    human_primary = [
        taxonomy_primary(row, strata[row["sample_id"]]) for row, _ in pairs
    ]
    gpt_primary = [
        taxonomy_primary(row, strata[row["sample_id"]]) for _, row in pairs
    ]

    app_values = ["Direct-help", "Indirect-help", "Irrelevant", "Unclear"]
    confusion = {
        human_label: {
            gpt_label: sum(
                left == human_label and right == gpt_label
                for left, right in zip(human_app, gpt_app)
            )
            for gpt_label in app_values
        }
        for human_label in app_values
    }
    audits = read_jsonl(output / "gpt_judge_audit.jsonl")
    prior_failure_path = output / "gpt_judge_attempt_failures.jsonl"
    prior_failures = (
        read_jsonl(prior_failure_path) if prior_failure_path.exists() else []
    )
    prompt_counts = Counter(row["prompt_sha256"] for row in audits)
    model_counts = Counter(row["model"] for row in audits)
    fingerprint_counts = Counter(row["system_fingerprint"] for row in audits)
    result = {
        "protocol": "memory_transfer_taxonomy_gpt_sensitivity_v1",
        "role": "sensitivity_only_human_verdict_unchanged",
        "n": len(pairs),
        "unique_cases": len({row["case_id"] for row, _ in pairs}),
        "human_distribution": dict(Counter(human_app)),
        "gpt_distribution": dict(Counter(gpt_app)),
        "applicability_exact": nominal_metrics(human_app, gpt_app),
        "applicability_binary": nominal_metrics(human_binary, gpt_binary),
        "primary_taxonomy": nominal_metrics(human_primary, gpt_primary),
        "applicability_confusion_human_rows_gpt_columns": confusion,
        "binary_case_cluster_bootstrap": cluster_bootstrap_kappa(
            pairs,
            lambda human_row, gpt_row: (
                binary_label(human_row["applicability_label"]),
                binary_label(gpt_row["applicability_label"]),
            ),
            replicates,
        ),
        "primary_case_cluster_bootstrap": cluster_bootstrap_kappa(
            pairs,
            lambda human_row, gpt_row: (
                taxonomy_primary(
                    human_row, strata[human_row["sample_id"]]
                ),
                taxonomy_primary(gpt_row, strata[gpt_row["sample_id"]]),
            ),
            replicates,
        ),
        "audit": {
            "persisted_responses": len(audits),
            "prior_responses_without_raw_audit": len(prior_failures),
            "external_attempts_total": len(audits) + len(prior_failures),
            "accepted": sum(bool(row.get("accepted")) for row in audits),
            "rejected_format_total": len(prior_failures) + sum(
                not bool(row.get("accepted")) for row in audits
            ),
            "models": dict(model_counts),
            "fingerprints": dict(fingerprint_counts),
            "prompt_hashes": dict(prompt_counts),
            "prior_prompt_hashes": dict(Counter(
                row["prompt_sha256"] for row in prior_failures
            )),
            "format_only_amendments": 2,
            "input_tokens": sum(row.get("input_tokens") or 0 for row in audits),
            "output_tokens": sum(row.get("output_tokens") or 0 for row in audits),
            "captured_estimated_cost_usd_lower_bound": sum(
                row.get("estimated_cost_usd") or 0 for row in audits
            ),
        },
    }
    json_path = output / "gpt_judge_sensitivity_analysis.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    binary = result["applicability_binary"]
    report = f"""# Blind GPT Judge Sensitivity Analysis

- Role: sensitivity only; the frozen human verdict is unchanged
- Accepted labels: {len(accepted)}/{len(human)}
- Human applicability distribution: {result['human_distribution']}
- GPT applicability distribution: {result['gpt_distribution']}
- Binary App/NotApp agreement: {binary['raw_agreement']:.1%}
- Binary Cohen's kappa: {binary['kappa']}
- Binary case-cluster 95% CI: {result['binary_case_cluster_bootstrap']['ci']}
- Primary taxonomy agreement: {result['primary_taxonomy']['raw_agreement']:.1%}
- Primary taxonomy kappa: {result['primary_taxonomy']['kappa']}
- Primary case-cluster 95% CI: {result['primary_case_cluster_bootstrap']['ci']}
- Model identity: {dict(model_counts)}
- Fingerprint identity: {dict(fingerprint_counts)}
- External attempts / accepted: {result['audit']['external_attempts_total']} / {result['audit']['accepted']}
- Format-only rejected responses: {result['audit']['rejected_format_total']}
- Captured API cost lower bound: ${result['audit']['captured_estimated_cost_usd_lower_bound']:.6f}

The GPT judge labeled substantially more memories applicable than the human
adjudication. It is therefore reported as a sensitivity disagreement, not as a
replacement label source or evidence that the human bottleneck verdict is
replicated by an independent judge.
"""
    report_path = output / "gpt_judge_sensitivity_report.md"
    report_path.write_text(report, encoding="utf-8")

    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gpt_judge_sensitivity"] = result
    for path in (
        output / "gpt_labels.jsonl",
        output / "gpt_judge_audit.jsonl",
        json_path,
        report_path,
    ):
        manifest["artifact_sha256"][str(path)] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.bootstrap_replicates), sort_keys=True))


if __name__ == "__main__":
    main()
