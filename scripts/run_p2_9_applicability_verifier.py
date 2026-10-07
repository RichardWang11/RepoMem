"""Run the locked P2-9 applicability-aware verifier offline pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import keyword
import pickle
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

from scripts.analyze_n2_memory_adoption import DEFAULT_INDEX_DIRS
from scripts.validate_applicability_labels import (
    DEFAULT_GRAPH_ROOT,
    graph_indexes,
    load_commit_catalog,
)


DEFAULT_INPUT = Path(
    "results/paper2_pilot/analysis/memory_transfer_taxonomy"
)
DEFAULT_OUTPUT = DEFAULT_INPUT / "p2_9_applicability_verifier"
DEFAULT_PREREGISTRATION = Path(
    "p2_9_applicability_verifier_preregistration.md"
)
BUDGETS = (5, 10, 20, 40)
SEED = 20260720
RELATION_TYPES = {"imports", "invokes", "inherits"}
APPLICABLE = {"Direct-help", "Indirect-help"}
GENERIC_NAMES = {
    "args",
    "assert",
    "class",
    "cls",
    "def",
    "dict",
    "else",
    "except",
    "False",
    "for",
    "from",
    "get",
    "if",
    "import",
    "in",
    "is",
    "kwargs",
    "len",
    "list",
    "None",
    "not",
    "object",
    "raise",
    "return",
    "self",
    "set",
    "str",
    "super",
    "test",
    "True",
    "try",
    "tuple",
    "value",
    "values",
    "with",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_anchor(name: str) -> bool:
    return (
        len(name) > 1
        and name not in GENERIC_NAMES
        and not keyword.iskeyword(name)
        and not (name.startswith("__") and name.endswith("__"))
    )


def hunk_local_anchors(patch: str) -> list[str]:
    """Return deterministic symbols from hunk headers and changed lines."""
    tier_counts = {tier: Counter() for tier in range(1, 5)}
    header_pattern = re.compile(r"^@@.*?@@\s*(.*)$")
    definition_pattern = re.compile(r"\b(?:async\s+def|def|class)\s+([A-Za-z_]\w*)")
    call_pattern = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
    attribute_pattern = re.compile(r"\.([A-Za-z_]\w*)")
    identifier_pattern = re.compile(r"\b[A-Za-z_]\w*\b")
    for raw_line in str(patch or "").splitlines():
        header = header_pattern.match(raw_line)
        if header:
            match = definition_pattern.search(header.group(1))
            if match and valid_anchor(match.group(1)):
                tier_counts[1][match.group(1)] += 1
            continue
        if not raw_line.startswith(("+", "-")):
            continue
        if raw_line.startswith(("+++", "---")):
            continue
        line = raw_line[1:]
        definitions = definition_pattern.findall(line)
        for name in definitions:
            if valid_anchor(name):
                tier_counts[2][name] += 1
        for name in call_pattern.findall(line) + attribute_pattern.findall(line):
            if valid_anchor(name):
                tier_counts[3][name] += 1
        for name in identifier_pattern.findall(line):
            if valid_anchor(name):
                tier_counts[4][name] += 1
    ranked = []
    seen = set()
    for tier in range(1, 5):
        for name, _ in sorted(
            tier_counts[tier].items(), key=lambda item: (-item[1], item[0])
        ):
            if name not in seen:
                ranked.append(name)
                seen.add(name)
    return ranked


def normalize_rows(input_dir: Path) -> list[dict]:
    cohorts = (
        ("discovery", "discovery_adjudicated.csv"),
        ("pilot_evaluation", "holdout_adjudicated.csv"),
    )
    result = []
    for cohort, filename in cohorts:
        for row in read_csv(input_dir / filename):
            current_field = (
                "current_code_context_json"
                if cohort == "discovery"
                else "current_base_code_json"
            )
            current = json.loads(row[current_field])
            result.append({
                "cohort": cohort,
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "repository": row["repository"],
                "memory_sha": row["memory_sha"].lower(),
                "memory_files": json.loads(row["memory_changed_files"]),
                "memory_patch": row["memory_patch"],
                "base_commit": current["base_commit"].lower(),
                "applicability_label": row["applicability_label"],
            })
    return result


def repo_candidates(repo: str, root: Path) -> list[Path]:
    name = repo.replace("/", "_")
    return sorted({path.parent for path in root.glob(f"**/{name}/.git")})


def git_has_commit(repo_dir: Path, sha: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "cat-file", "-e", f"{sha}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    return completed.returncode == 0


def git_temporal_status(
    repo: str,
    memory_sha: str,
    base_sha: str,
    root: Path,
    cache: dict,
) -> tuple[str, str | None]:
    key = (repo, memory_sha, base_sha)
    if key in cache:
        return cache[key]
    for repo_dir in repo_candidates(repo, root):
        if not (
            git_has_commit(repo_dir, memory_sha)
            and git_has_commit(repo_dir, base_sha)
        ):
            continue
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "merge-base",
                "--is-ancestor",
                memory_sha,
                base_sha,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        status = "BEFORE_BASE" if completed.returncode == 0 else "AFTER_OR_DIVERGENT"
        cache[key] = (status, str(repo_dir))
        return cache[key]
    cache[key] = ("UNKNOWN", None)
    return cache[key]


def temporal_status(
    row: dict,
    catalog: dict,
    meta: dict,
    repo_root: Path,
    git_cache: dict,
) -> tuple[str, str, str | None]:
    repo = row["repository"]
    memory = catalog.get((repo, row["memory_sha"]))
    base = catalog.get((repo, row["base_commit"]))
    sha_map = meta.get(repo, {}).get("sha_to_history_index", {})
    memory_index = sha_map.get(row["memory_sha"])
    base_index = sha_map.get(row["base_commit"])
    if memory_index is None and memory:
        memory_index = memory.get("history_index")
    if base_index is None and base:
        base_index = base.get("history_index")
    if memory_index is not None and base_index is not None:
        status = (
            "BEFORE_BASE"
            if int(memory_index) < int(base_index)
            else "AFTER_OR_DIVERGENT"
        )
        return status, "commit_catalog", None
    status, path = git_temporal_status(
        repo,
        row["memory_sha"],
        row["base_commit"],
        repo_root,
        git_cache,
    )
    return status, "local_git" if path else "unresolved", path


def resolve_structure(graph, indexes, memory_files: list[str], anchors: list[str]) -> dict:
    files, by_file, by_name = indexes
    surviving = sorted(path for path in memory_files if path in files)
    missing = sorted(set(memory_files) - set(surviving))
    exact_nodes = set()
    moved_nodes = set()
    resolved = []
    for anchor in anchors:
        exact = []
        for path in memory_files:
            exact.extend(node for node, _ in by_file.get((path, anchor), []))
        global_nodes = by_name.get(anchor, [])
        moved = [] if exact or len(global_nodes) != 1 else [global_nodes[0][0]]
        exact_nodes.update(exact)
        moved_nodes.update(moved)
        resolved.append({
            "anchor": anchor,
            "same_file_nodes": sorted(exact),
            "unique_moved_nodes": sorted(moved),
        })
    relation_types = set()
    for node in exact_nodes | moved_nodes:
        for _, _, data in graph.in_edges(node, data=True):
            if data.get("type") in RELATION_TYPES:
                relation_types.add(data["type"])
        for _, _, data in graph.out_edges(node, data=True):
            if data.get("type") in RELATION_TYPES:
                relation_types.add(data["type"])
    return {
        "surviving_files": surviving,
        "missing_files": missing,
        "file_survival_rate": (
            len(surviving) / len(memory_files) if memory_files else None
        ),
        "same_file_anchor_count": len(exact_nodes),
        "moved_anchor_count": len(moved_nodes),
        "anchor_resolution": resolved,
        "current_relation_types": sorted(relation_types),
    }


def verdict(features: dict, method: str) -> tuple[bool, list[str]]:
    reasons = []
    if features["temporal_status"] == "AFTER_OR_DIVERGENT":
        reasons.append("temporal_contradiction")
    all_files_missing = (
        bool(features["memory_file_count"])
        and not features["surviving_files"]
    )
    any_moved = features["moved_anchor_count"] > 0
    if all_files_missing and not any_moved:
        reasons.append("all_files_missing_no_unique_move")
    if method == "V-Struct":
        anchors_exist = features["anchor_count"] > 0
        any_anchor = (
            features["same_file_anchor_count"] > 0
            or features["moved_anchor_count"] > 0
        )
        if anchors_exist and not any_anchor:
            reasons.append("anchors_unresolved")
    return not reasons, reasons


def exact_binomial_ci(successes: int, total: int, alpha: float = 0.05) -> list:
    if total == 0:
        return [None, None]
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2, successes, total - successes + 1)
    )
    upper = 1.0 if successes == total else float(
        beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    )
    return [lower, upper]


def metrics(rows: list[dict], method: str, budget: int) -> dict:
    usable = [row for row in rows if row["applicability_label"] != "Unclear"]
    applicable = [row for row in usable if row["applicability_label"] in APPLICABLE]
    irrelevant = [row for row in usable if row["applicability_label"] == "Irrelevant"]
    if method == "BM25":
        retained = [True for _ in usable]
    else:
        retained = [row[f"{method}_{budget}_retained"] for row in usable]
    irrelevant_retained = sum(
        row[f"{method}_{budget}_retained"] if method != "BM25" else True
        for row in irrelevant
    )
    applicable_retained = sum(
        row[f"{method}_{budget}_retained"] if method != "BM25" else True
        for row in applicable
    )
    rejection_reasons = Counter()
    if method != "BM25":
        for row in usable:
            if not row[f"{method}_{budget}_retained"]:
                rejection_reasons.update(row[f"{method}_{budget}_reasons"])
    baseline_exposure = len(irrelevant) / len(usable) if usable else None
    exposure = irrelevant_retained / len(usable) if usable else None
    return {
        "n": len(usable),
        "applicable_n": len(applicable),
        "irrelevant_n": len(irrelevant),
        "retained_n": sum(retained),
        "irrelevant_retained_n": irrelevant_retained,
        "applicable_retained_n": applicable_retained,
        "irrelevant_exposure": exposure,
        "exposure_reduction_pp": (
            100 * (baseline_exposure - exposure)
            if exposure is not None else None
        ),
        "useful_retention": (
            applicable_retained / len(applicable) if applicable else None
        ),
        "useful_retention_exact_95_ci": exact_binomial_ci(
            applicable_retained, len(applicable)
        ),
        "rejection_reason_counts": dict(rejection_reasons),
    }


def bootstrap(rows: list[dict], method: str, budget: int, replicates: int) -> dict:
    by_case = defaultdict(list)
    for row in rows:
        if row["applicability_label"] != "Unclear":
            by_case[row["case_id"]].append(row)
    cases = sorted(by_case)
    rng = np.random.default_rng(SEED)
    reductions = []
    retentions = []
    for _ in range(replicates):
        sample = []
        for case_id in rng.choice(cases, size=len(cases), replace=True):
            sample.extend(by_case[case_id])
        value = metrics(sample, method, budget)
        reductions.append(value["exposure_reduction_pp"])
        if value["useful_retention"] is not None:
            retentions.append(value["useful_retention"])
    return {
        "replicates": replicates,
        "exposure_reduction_pp_ci": [
            float(value) for value in np.percentile(reductions, [2.5, 97.5])
        ],
        "useful_retention_ci": [
            float(value) for value in np.percentile(retentions, [2.5, 97.5])
        ] if retentions else [None, None],
    }


def run(args) -> dict:
    input_dir = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prereg = Path(args.preregistration)
    text = prereg.read_text(encoding="utf-8")
    if "LOCKED BEFORE P2-9 FEATURE EXTRACTION" not in text:
        raise ValueError("P2-9 preregistration is not locked")
    rows = normalize_rows(input_dir)
    repos = {row["repository"] for row in rows}
    catalog, meta = load_commit_catalog(args.index_dirs, repos)
    graph_cache = {}
    git_cache = {}
    feature_rows = []
    internal = []
    for row in rows:
        case_id = row["case_id"]
        if case_id not in graph_cache:
            path = Path(args.graph_root) / f"{case_id}.pkl"
            with path.open("rb") as handle:
                graph = pickle.load(handle)
            graph_cache[case_id] = (graph, graph_indexes(graph))
        graph, indexes = graph_cache[case_id]
        anchors = hunk_local_anchors(row["memory_patch"])
        temporal, temporal_source, git_path = temporal_status(
            row, catalog, meta, Path(args.repo_root), git_cache
        )
        item = {
            **row,
            "all_anchors": anchors,
            "temporal_status": temporal,
            "temporal_source": temporal_source,
            "git_checkout_used": git_path,
        }
        public = {
            "cohort": row["cohort"],
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "repository": row["repository"],
            "memory_sha": row["memory_sha"],
            "base_commit": row["base_commit"],
            "memory_file_count": len(row["memory_files"]),
            "hunk_anchor_count_unbounded": len(anchors),
            "temporal_status": temporal,
            "temporal_source": temporal_source,
            "current_relation_is_historical_proxy_only": True,
        }
        for budget in BUDGETS:
            structure = resolve_structure(
                graph, indexes, row["memory_files"], anchors[:budget]
            )
            features = {
                **structure,
                "memory_file_count": len(row["memory_files"]),
                "anchor_count": len(anchors[:budget]),
                "temporal_status": temporal,
            }
            public[f"budget_{budget}_anchors"] = json.dumps(
                anchors[:budget], ensure_ascii=False
            )
            for key in (
                "file_survival_rate",
                "same_file_anchor_count",
                "moved_anchor_count",
            ):
                public[f"budget_{budget}_{key}"] = features[key]
            public[f"budget_{budget}_current_relation_types"] = json.dumps(
                features["current_relation_types"]
            )
            for method in ("V-Relaxed", "V-Struct"):
                retained, reasons = verdict(features, method)
                item[f"{method}_{budget}_retained"] = retained
                item[f"{method}_{budget}_reasons"] = reasons
        feature_rows.append(public)
        internal.append(item)

    discovery = [row for row in internal if row["cohort"] == "discovery"]
    development = {}
    eligible_budgets = []
    for budget in BUDGETS:
        value = metrics(discovery, "V-Struct", budget)
        development[str(budget)] = value
        if value["useful_retention"] is not None and value["useful_retention"] >= 0.85:
            eligible_budgets.append(budget)
    if eligible_budgets:
        chosen_budget = sorted(
            eligible_budgets,
            key=lambda budget: (
                -development[str(budget)]["exposure_reduction_pp"],
                budget,
            ),
        )[0]
        development_verdict = "DEVELOPMENT_SOLUTION_FOUND"
    else:
        chosen_budget = max(BUDGETS)
        development_verdict = "NO_DEVELOPMENT_SOLUTION"

    cohorts = {
        "discovery": discovery,
        "pilot_evaluation": [
            row for row in internal if row["cohort"] == "pilot_evaluation"
        ],
        "pooled_descriptive": internal,
    }
    results = {}
    for cohort, cohort_rows in cohorts.items():
        results[cohort] = {}
        for method in ("BM25", "V-Relaxed", "V-Struct"):
            value = metrics(cohort_rows, method, chosen_budget)
            value["case_cluster_bootstrap"] = bootstrap(
                cohort_rows, method, chosen_budget, args.bootstrap_replicates
            ) if method != "BM25" else None
            results[cohort][method] = value

    pilot_struct = results["pilot_evaluation"]["V-Struct"]
    if development_verdict == "NO_DEVELOPMENT_SOLUTION":
        pilot_verdict = "NO_DEVELOPMENT_SOLUTION"
    elif (
        pilot_struct["exposure_reduction_pp"] >= 15
        and pilot_struct["applicable_retained_n"]
        == pilot_struct["applicable_n"]
    ):
        pilot_verdict = "PILOT_GO"
    else:
        pilot_verdict = "PILOT_NO_GO"

    evaluation_rows = []
    for row in internal:
        evaluation_rows.append({
            "cohort": row["cohort"],
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "applicability_label": row["applicability_label"],
            "chosen_budget": chosen_budget,
            "V-Relaxed_retained": row[f"V-Relaxed_{chosen_budget}_retained"],
            "V-Relaxed_reasons": "|".join(
                row[f"V-Relaxed_{chosen_budget}_reasons"]
            ),
            "V-Struct_retained": row[f"V-Struct_{chosen_budget}_retained"],
            "V-Struct_reasons": "|".join(
                row[f"V-Struct_{chosen_budget}_reasons"]
            ),
        })
    write_csv(output / "p2_9_nonoracle_features.csv", feature_rows)
    write_csv(output / "p2_9_evaluation_rows.csv", evaluation_rows)
    result = {
        "protocol": "p2_9_applicability_verifier_pilot_v1",
        "preregistration_sha256": sha256(prereg),
        "development_verdict": development_verdict,
        "chosen_symbol_budget": chosen_budget,
        "budget_candidates": list(BUDGETS),
        "development_budget_metrics": development,
        "pilot_verdict": pilot_verdict,
        "results": results,
        "temporal_status_counts": dict(Counter(
            row["temporal_status"] for row in internal
        )),
        "temporal_source_counts": dict(Counter(
            row["temporal_source"] for row in internal
        )),
        "current_relation_claim_boundary": (
            "current-state proxy only; no memory-era consistency claim"
        ),
        "confirmatory_claim": False,
    }
    json_path = output / "p2_9_results.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    pilot_relaxed = results["pilot_evaluation"]["V-Relaxed"]
    report = f"""## Material Passport

- Protocol: `p2_9_applicability_verifier_pilot_v1`
- Verification Status: ANALYZED
- Confirmatory claim: no

# P2-9 Applicability Verifier Offline Pilot

- Development verdict: **{development_verdict}**
- Chosen hunk-anchor budget: **{chosen_budget}**
- Pilot verdict: **{pilot_verdict}**
- Fresh applicable denominator: {pilot_struct['applicable_n']} (pilot only)
- V-Struct Fresh exposure reduction: {pilot_struct['exposure_reduction_pp']:.1f}pp
- V-Struct Fresh useful retention: {pilot_struct['applicable_retained_n']}/{pilot_struct['applicable_n']}
- V-Struct Fresh exact retention 95% CI: {pilot_struct['useful_retention_exact_95_ci']}
- V-Struct Fresh rejection reasons: {pilot_struct['rejection_reason_counts']}
- V-Relaxed Fresh exposure reduction: {pilot_relaxed['exposure_reduction_pp']:.1f}pp
- V-Relaxed Fresh useful retention: {pilot_relaxed['applicable_retained_n']}/{pilot_relaxed['applicable_n']}
- Temporal statuses: {result['temporal_status_counts']}
- Temporal sources: {result['temporal_source_counts']}

`Current relation evidence` is a current-state proxy only. It does not verify
that dependency edges were consistent at the Memory commit. This pilot uses
previously inspected cohorts and cannot support a confirmatory method claim.
"""
    report_path = output / "p2_9_report.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "protocol": result["protocol"],
        "preregistration": str(prereg),
        "preregistration_sha256": result["preregistration_sha256"],
        "inputs": {
            str(input_dir / "discovery_adjudicated.csv"): sha256(
                input_dir / "discovery_adjudicated.csv"
            ),
            str(input_dir / "holdout_adjudicated.csv"): sha256(
                input_dir / "holdout_adjudicated.csv"
            ),
        },
        "artifacts": {
            str(path): sha256(path)
            for path in (
                output / "p2_9_nonoracle_features.csv",
                output / "p2_9_evaluation_rows.csv",
                json_path,
                report_path,
            )
        },
        "status": "ANALYZED_PILOT",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--preregistration", default=str(DEFAULT_PREREGISTRATION))
    parser.add_argument("--graph-root", default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--repo-root", default="playground")
    parser.add_argument("--index-dirs", nargs="+", default=DEFAULT_INDEX_DIRS)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
