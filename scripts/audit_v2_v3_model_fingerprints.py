import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_V2 = "results/context_safe_m3a/runs_v2"
DEFAULT_V3 = "results/context_safe_m3a/replication_v3/runs"
DEFAULT_OUTPUT = "results/context_safe_m3a/fingerprint_audit"
FINGERPRINT_KEY = "system_fingerprint"
LOG_FINGERPRINT_RE = re.compile(
    r"system_fingerprint\s*[:=]\s*[\"']?(fp_[A-Za-z0-9]+)",
    re.IGNORECASE,
)
STRICT_FINGERPRINT_RE = re.compile(r"\bfp_[0-9a-f]{8,}\b", re.IGNORECASE)


def iter_runs(root):
    root = Path(root)
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"(baseline|memory)_r(\d+)", path.name)
        if match:
            yield match.group(1), int(match.group(2)), path


def recursive_values(value, target_key):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == target_key and child:
                found.append(str(child))
            found.extend(recursive_values(child, target_key))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_values(child, target_key))
    return found


def log_fingerprints(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    values = LOG_FINGERPRINT_RE.findall(text)
    values.extend(STRICT_FINGERPRINT_RE.findall(text))
    return sorted(set(values))


def provider_details(trajectory):
    provider = trajectory.get("provider_metadata") or {}
    responses = provider.get("responses") or []
    models = []
    fingerprints = []
    missing_fingerprint = 0
    for response in responses:
        model = response.get("model")
        fingerprint = response.get(FINGERPRINT_KEY)
        if model:
            models.append(str(model))
        if fingerprint:
            fingerprints.append(str(fingerprint))
        else:
            missing_fingerprint += 1
    return {
        "requested_model": provider.get("requested_model"),
        "response_count": len(responses),
        "response_models": models,
        "fingerprints": fingerprints,
        "missing_response_fingerprint_count": missing_fingerprint,
    }


def audit_version(version, root):
    rows = []
    run_log_values = {}
    for arm, repeat, run_dir in iter_runs(root):
        log_values = log_fingerprints(run_dir / "localize.log")
        run_log_values[run_dir.name] = log_values
        configured_model = None
        args_path = run_dir / "args.json"
        if args_path.exists():
            configured_model = json.loads(
                args_path.read_text(encoding="utf-8")
            ).get("model")
        trajectory_path = run_dir / "loc_trajs.jsonl"
        if not trajectory_path.exists():
            continue
        with trajectory_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                trajectories = record.get("loc_trajs", {}).get("trajs", [])
                for sample_index, trajectory in enumerate(trajectories, 1):
                    details = provider_details(trajectory)
                    recursive = recursive_values(trajectory, FINGERPRINT_KEY)
                    fingerprints = sorted(set(details["fingerprints"] + recursive))
                    sources = []
                    if details["fingerprints"]:
                        sources.append("provider_metadata")
                    if recursive and not details["fingerprints"]:
                        sources.append("recursive_json")
                    rows.append({
                        "version": version,
                        "arm": arm,
                        "repeat": repeat,
                        "run": run_dir.name,
                        "instance_id": record.get("instance_id"),
                        "sample_index": sample_index,
                        "line_number": line_number,
                        "configured_model": configured_model,
                        "requested_model": details["requested_model"],
                        "response_count": details["response_count"],
                        "response_models": sorted(set(details["response_models"])),
                        "system_fingerprints": fingerprints,
                        "missing_response_fingerprint_count": details[
                            "missing_response_fingerprint_count"
                        ],
                        "fingerprint_sources": sources,
                        "_response_model_values": details["response_models"],
                        "_response_fingerprint_values": details["fingerprints"],
                    })
    return rows, run_log_values


def summarize_version(rows, log_values):
    fingerprints = Counter()
    response_models = Counter()
    requested_models = Counter()
    configured_models = Counter()
    response_count = 0
    missing_response_fingerprints = 0
    trajectories_with_fingerprint = 0
    for row in rows:
        response_count += row["response_count"]
        missing_response_fingerprints += row["missing_response_fingerprint_count"]
        if row["system_fingerprints"]:
            trajectories_with_fingerprint += 1
        fingerprints.update(row.get("_response_fingerprint_values", []))
        response_models.update(row.get("_response_model_values", []))
        if row.get("configured_model"):
            configured_models[row["configured_model"]] += 1
        if row["requested_model"]:
            requested_models[row["requested_model"]] += 1
    log_fingerprints_flat = sorted({
        value for values in log_values.values() for value in values
    })
    all_fingerprints = sorted(set(fingerprints) | set(log_fingerprints_flat))
    trajectory_count = len(rows)
    response_coverage = (
        (response_count - missing_response_fingerprints) / response_count
        if response_count else None
    )
    return {
        "trajectory_count": trajectory_count,
        "trajectories_with_fingerprint": trajectories_with_fingerprint,
        "trajectory_fingerprint_coverage": (
            trajectories_with_fingerprint / trajectory_count
            if trajectory_count else 0.0
        ),
        "response_count": response_count,
        "responses_with_fingerprint": response_count - missing_response_fingerprints,
        "response_fingerprint_coverage": response_coverage,
        "configured_models": dict(configured_models),
        "requested_models": dict(requested_models),
        "response_models": dict(response_models),
        "system_fingerprints": dict(fingerprints),
        "log_system_fingerprints": log_fingerprints_flat,
        "all_system_fingerprints": all_fingerprints,
        "mixed_within_version": len(all_fingerprints) > 1,
    }


def classify(v2, v3):
    if v2["mixed_within_version"] or v3["mixed_within_version"]:
        return "MIXED_WITHIN_RUN"
    v2_values = set(v2["all_system_fingerprints"])
    v3_values = set(v3["all_system_fingerprints"])
    if not v2_values and v3_values:
        return "V2_FINGERPRINT_NOT_RECORDED"
    if not v2_values or not v3_values:
        return "INSUFFICIENT_COVERAGE"
    complete = all(
        summary["trajectory_fingerprint_coverage"] == 1.0
        and summary["response_fingerprint_coverage"] == 1.0
        for summary in (v2, v3)
    )
    if not complete:
        return "INSUFFICIENT_COVERAGE"
    if v2_values == v3_values:
        return "SAME_SNAPSHOT"
    return "DIFFERENT_SNAPSHOT"


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_rows(path, rows):
    fields = [
        "version", "arm", "repeat", "run", "instance_id", "sample_index",
        "line_number", "configured_model", "requested_model", "response_count",
        "response_models", "system_fingerprints",
        "missing_response_fingerprint_count", "fingerprint_sources",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json_cell(row[key]) if isinstance(row[key], (list, dict))
                else row[key]
                for key in fields
            })


def report_text(summary):
    v2 = summary["versions"]["v2"]
    v3 = summary["versions"]["v3"]
    verdict = summary["verdict"]
    v3_models = format_values(v3["response_models"])
    v3_fingerprints = format_values(v3["system_fingerprints"])
    interpretation = {
        "MIXED_WITHIN_RUN": (
            "At least one experiment version contains multiple fingerprints. "
            "A rolling deployment or mixed backend cannot be excluded."
        ),
        "SAME_SNAPSHOT": (
            "V2 and V3 have complete response-level coverage and the same "
            "fingerprint set. The observed difference is not explained by a "
            "recorded model-snapshot change."
        ),
        "DIFFERENT_SNAPSHOT": (
            "V2 and V3 have complete response-level coverage but different "
            "fingerprints. Model snapshot is a confirmed confound."
        ),
        "V2_FINGERPRINT_NOT_RECORDED": (
            "V3 records provider fingerprints, but V2 does not. A model-snapshot "
            "change between V2 and V3 cannot be confirmed or excluded."
        ),
        "INSUFFICIENT_COVERAGE": (
            "Fingerprint evidence is absent or incomplete, so snapshot parity "
            "cannot be established."
        ),
    }[verdict]
    return f"""# V2 vs V3 Model Fingerprint Audit

## Verdict

**{verdict}**

{interpretation}

## Coverage

| Version | Configured model | Trajectories | Trajectory coverage | Responses | Response coverage | Response models | Fingerprints |
|---|---|---:|---:|---:|---:|---|---|
| V2 | {format_values(v2['configured_models'])} | {v2['trajectory_count']} | {v2['trajectory_fingerprint_coverage']:.1%} | {v2['response_count']} | {format_coverage(v2['response_fingerprint_coverage'])} | {format_values(v2['response_models'])} | {format_values(v2['system_fingerprints'])} |
| V3 | {format_values(v3['configured_models'])} | {v3['trajectory_count']} | {v3['trajectory_fingerprint_coverage']:.1%} | {v3['response_count']} | {format_coverage(v3['response_fingerprint_coverage'])} | {v3_models} | {v3_fingerprints} |

## Interpretation Boundary

The requested model name (`gpt-4o`), LiteLLM provider, deployment name, and
endpoint identity are not model-snapshot evidence. Only response-level
`system_fingerprint` values are used for snapshot comparison. Source-code tokens
such as `fp_groups.py` are excluded by the fingerprint parser.

## Paper-Ready Caveat

V3 responses resolve to `{v3_models}` with fingerprint `{v3_fingerprints}`.
V2 did not serialize provider fingerprints, so a model-snapshot
change between V2 and V3 cannot be excluded. The V3 replication difference
therefore remains confounded by possible endpoint snapshot drift.
"""


def format_coverage(value):
    return "not recorded" if value is None else f"{value:.1%}"


def format_values(value):
    return ", ".join(value) if value else "not recorded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_root", default=DEFAULT_V2)
    parser.add_argument("--v3_root", default=DEFAULT_V3)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    versions = {}
    log_evidence = {}
    for version, root in (("v2", args.v2_root), ("v3", args.v3_root)):
        rows, logs = audit_version(version, root)
        if not rows:
            raise ValueError(f"no trajectories found for {version}: {root}")
        all_rows.extend(rows)
        versions[version] = summarize_version(rows, logs)
        log_evidence[version] = logs

    summary = {
        "verdict": classify(versions["v2"], versions["v3"]),
        "versions": versions,
        "log_evidence": log_evidence,
        "comparison_uses_response_fingerprints_only": True,
    }
    write_rows(output_dir / "fingerprint_per_trajectory.csv", all_rows)
    with (output_dir / "fingerprint_audit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output_dir / "fingerprint_audit_report.md").write_text(
        report_text(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
