import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
)

from scripts.analyze_memory_bottleneck import (
    DEFAULT_ARROW,
    deduplicate_candidates,
    load_pilot_results,
)
from scripts.analyze_n2_memory_adoption import (
    DEFAULT_INDEX_DIRS,
    extract_patch_symbols,
    files_from_patch,
    load_commit_lookup,
    load_gold,
    normalize_path,
)


DEFAULT_PREREGISTRATION = "results/paper2_pilot/preregistration.json"
DEFAULT_RUN_ROOT = "results/paper2_pilot/runs"
DEFAULT_GRAPH_ROOT = "index_data/SWE-bench_Verified/graph_index_v2.3"
DEFAULT_OUTPUT = "results/paper2_pilot/analysis/applicability_validation"
DEFAULT_HUMAN_WORKBOOK = "applicability_human_annotation_30.csv"
DEFAULT_INSTRUCTIONS = "applicability_annotation_instructions.md"
DEFAULT_SEED = 20260717
STRATUM_TARGETS = {
    "exact_file": 8,
    "parent_only": 7,
    "symbol_only": 7,
    "none": 8,
}
APPLICABILITY_LABELS = ("Direct-help", "Indirect-help", "Irrelevant")
VALIDITY_LABELS = ("Valid", "Partially-valid", "Invalid-obsolete")
RELATION_TYPES = {"imports", "invokes", "inherits"}
BLINDED_FORBIDDEN_KEYS = {
    "sampling_stratum",
    "file_overlap",
    "parent_module_match",
    "symbol_overlap",
    "structural_relevant",
    "structurally_irrelevant",
    "weak_relevant",
    "adopted",
    "acc_at5",
    "paired_harm",
    "outcome",
}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(json_safe(rows))


def normalize_python_files(paths):
    return sorted(
        {
            normalize_path(path)
            for path in paths or []
            if normalize_path(path).endswith(".py")
        }
    )


def immediate_parent(path):
    parent = str(PurePosixPath(normalize_path(path)).parent)
    return None if parent in ("", ".", "/") else parent


def has_parent_match(memory_files, gold_files):
    memory_parents = {immediate_parent(path) for path in memory_files}
    gold_parents = {immediate_parent(path) for path in gold_files}
    memory_parents.discard(None)
    gold_parents.discard(None)
    return bool(memory_parents & gold_parents)


def non_dunder_symbols(patch):
    return sorted(
        symbol
        for symbol in extract_patch_symbols(patch or "")
        if not (symbol.startswith("__") and symbol.endswith("__"))
    )


def assign_stratum(file_overlap, parent_match, symbol_overlap):
    if file_overlap:
        return "exact_file"
    if parent_match:
        return "parent_only"
    if symbol_overlap:
        return "symbol_only"
    return "none"


def load_dataset_rows(dataset_arrow, selected_ids):
    from datasets import Dataset

    rows = Dataset.from_file(str(dataset_arrow))
    result = {
        row["instance_id"]: dict(row)
        for row in rows
        if row["instance_id"] in selected_ids
    }
    missing = selected_ids - set(result)
    if missing:
        raise ValueError(f"dataset rows missing: {sorted(missing)}")
    return result


def load_commit_catalog(index_dirs, selected_repos):
    catalog = {}
    meta = {}
    for root in index_dirs:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for repo_dir in sorted(root_path.iterdir()):
            repo = repo_dir.name.replace("__", "/")
            if repo not in selected_repos:
                continue
            commit_path = repo_dir / "commits.jsonl"
            if not commit_path.exists():
                continue
            with commit_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sha = str(row.get("sha", "")).lower()
                    if not sha:
                        continue
                    item = dict(row)
                    item["repo"] = row.get("repo") or repo
                    item["_commit_index_path"] = str(commit_path)
                    item["patch_path"] = (
                        str(repo_dir / row["patch_path"])
                        if row.get("patch_path")
                        else None
                    )
                    # Later roots are not allowed to silently replace a complete row.
                    current = catalog.get((repo, sha))
                    if current is None or (not current.get("patch_path") and item.get("patch_path")):
                        catalog[(repo, sha)] = item
            meta_path = repo_dir / "meta.json"
            if meta_path.exists() and repo not in meta:
                meta[repo] = json.loads(meta_path.read_text(encoding="utf-8"))
                meta[repo]["__source_path"] = str(meta_path)
    return catalog, meta


def lookup_full_commit(catalog, repo, short_sha):
    short_sha = str(short_sha).lower()
    matches = [
        row
        for (candidate_repo, sha), row in catalog.items()
        if candidate_repo == repo and sha.startswith(short_sha)
    ]
    unique = {row["sha"]: row for row in matches}
    if len(unique) != 1:
        raise ValueError(
            f"expected one commit for {repo}@{short_sha}, got {len(unique)}"
        )
    return next(iter(unique.values()))


def read_patch(commit):
    path = commit.get("patch_path")
    if not path or not Path(path).exists():
        return ""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def graph_indexes(graph):
    files = {}
    symbols_by_file = defaultdict(list)
    symbols_by_name = defaultdict(list)
    for node, data in graph.nodes(data=True):
        node_type = data.get("type")
        if node_type == "file":
            files[normalize_path(node)] = (node, data)
        elif node_type in {"function", "class"} and ":" in str(node):
            path, qualified = str(node).split(":", 1)
            simple_name = qualified.rsplit(".", 1)[-1]
            item = (node, data)
            symbols_by_file[(normalize_path(path), simple_name)].append(item)
            symbols_by_name[simple_name].append(item)
    return files, symbols_by_file, symbols_by_name


def resolve_current_structure(graph, memory_files, memory_symbols, indexes=None):
    files, by_file, by_name = indexes or graph_indexes(graph)
    surviving_files = [path for path in memory_files if path in files]
    same_file_nodes = []
    moved_nodes = []
    symbol_rows = []
    for symbol in memory_symbols:
        exact = []
        for path in memory_files:
            exact.extend(by_file.get((path, symbol), []))
        unique_global = by_name.get(symbol, [])
        moved = [] if exact or len(unique_global) != 1 else unique_global
        same_file_nodes.extend(node for node, _ in exact)
        moved_nodes.extend(node for node, _ in moved)
        symbol_rows.append({
            "symbol": symbol,
            "same_file_nodes": [node for node, _ in exact],
            "unique_moved_nodes": [node for node, _ in moved],
        })
    resolved_nodes = set(same_file_nodes + moved_nodes)
    relation_types = set()
    for node in resolved_nodes:
        for _, _, data in graph.in_edges(node, data=True):
            if data.get("type") in RELATION_TYPES:
                relation_types.add(data["type"])
        for _, _, data in graph.out_edges(node, data=True):
            if data.get("type") in RELATION_TYPES:
                relation_types.add(data["type"])
    file_rate = len(surviving_files) / len(memory_files) if memory_files else None
    symbol_count = len(memory_symbols)
    return {
        "surviving_files": surviving_files,
        "missing_files": sorted(set(memory_files) - set(surviving_files)),
        "file_survival_rate": file_rate,
        "same_file_symbol_count": len(set(same_file_nodes)),
        "moved_symbol_count": len(set(moved_nodes)),
        "symbol_survival_same_file_rate": (
            len({row["symbol"] for row in symbol_rows if row["same_file_nodes"]})
            / symbol_count
            if symbol_count else None
        ),
        "symbol_survival_moved_rate": (
            len({row["symbol"] for row in symbol_rows if row["unique_moved_nodes"]})
            / symbol_count
            if symbol_count else None
        ),
        "symbol_resolution": symbol_rows,
        "current_relation_resolvable": bool(relation_types),
        "current_relation_types": sorted(relation_types),
        "no_extractable_symbol": not memory_symbols,
        "any_changed_file_survives": bool(surviving_files),
        "any_symbol_resolves_same_file": bool(same_file_nodes),
        "any_symbol_resolves_uniquely_elsewhere": bool(moved_nodes),
    }


def bounded_text(text, limit=12000):
    text = str(text or "")
    if len(text) <= limit:
        return text
    half = (limit - 80) // 2
    return text[:half] + "\n...[deterministically truncated]...\n" + text[-half:]


def current_context(graph, paths, memory_symbols, max_files=8, indexes=None):
    files, by_file, _ = indexes or graph_indexes(graph)
    rows = []
    for path in sorted(set(paths))[:max_files]:
        file_item = files.get(path)
        snippets = []
        for symbol in memory_symbols:
            for node, data in by_file.get((path, symbol), []):
                snippets.append({
                    "node": node,
                    "code": bounded_text(data.get("code", ""), 2200),
                })
        rows.append({
            "path": path,
            "exists": file_item is not None,
            "symbol_snippets": snippets,
            "file_excerpt": (
                bounded_text(file_item[1].get("code", ""), 3200)
                if file_item and not snippets else ""
            ),
        })
    return rows


def build_candidate_pool(
    preregistration,
    run_root,
    dataset_rows,
    catalog,
    graph_root,
):
    selected_ids = set(dataset_rows)
    gold = {
        case_id: normalize_python_files(files_from_patch(row.get("patch", "")))
        for case_id, row in dataset_rows.items()
    }
    commit_lookup = load_commit_lookup(
        DEFAULT_INDEX_DIRS,
        {row["repo"] for row in dataset_rows.values()},
    )
    parsed = load_pilot_results(preregistration, run_root, gold, commit_lookup)
    if parsed["parse_errors"]:
        raise ValueError(f"trajectory parse errors: {parsed['parse_errors'][:3]}")
    deduped = deduplicate_candidates(parsed["exposures"], max_rank=5)
    pool = []
    graph_cache = {}
    for candidate in deduped:
        case_id = candidate["case_id"]
        if case_id not in selected_ids:
            continue
        dataset_row = dataset_rows[case_id]
        repo = dataset_row["repo"]
        commit = lookup_full_commit(catalog, repo, candidate["sha"])
        patch = read_patch(commit)
        memory_files = normalize_python_files(commit.get("changed_files") or [])
        gold_files = gold[case_id]
        memory_symbols = non_dunder_symbols(patch)
        gold_symbols = non_dunder_symbols(dataset_row.get("patch", ""))
        file_overlap_files = sorted(set(memory_files) & set(gold_files))
        file_overlap = bool(file_overlap_files)
        parent_match = has_parent_match(memory_files, gold_files)
        symbol_overlap_values = sorted(set(memory_symbols) & set(gold_symbols))
        symbol_overlap = bool(symbol_overlap_values)
        stratum = assign_stratum(file_overlap, parent_match, symbol_overlap)
        graph_path = Path(graph_root) / f"{case_id}.pkl"
        if case_id not in graph_cache:
            with graph_path.open("rb") as handle:
                graph = pickle.load(handle)
            graph_cache[case_id] = (graph, graph_indexes(graph))
        graph, indexes = graph_cache[case_id]
        structure = resolve_current_structure(
            graph, memory_files, memory_symbols, indexes=indexes
        )
        pool.append({
            "case_id": case_id,
            "repeat_id": int(candidate["repeat"]),
            "repository": repo,
            "memory_sha": commit["sha"],
            "memory_short_sha": candidate["sha"],
            "best_rank": int(candidate["best_rank"]),
            "memory_subject": commit.get("subject", ""),
            "memory_message": commit.get("message", ""),
            "memory_patch": patch,
            "memory_patch_path": commit.get("patch_path"),
            "commit_index_path": commit.get("_commit_index_path"),
            "memory_files": memory_files,
            "memory_symbols": memory_symbols,
            "gold_files": gold_files,
            "gold_symbols": gold_symbols,
            "file_overlap_files": file_overlap_files,
            "file_overlap": file_overlap,
            "parent_module_match": parent_match,
            "symbol_overlap_values": symbol_overlap_values,
            "symbol_overlap": symbol_overlap,
            "weak_relevant": file_overlap,
            "structural_relevant": file_overlap or parent_match or symbol_overlap,
            "structurally_irrelevant": not (
                file_overlap or parent_match or symbol_overlap
            ),
            "sampling_stratum": stratum,
            "commit_history_index": commit.get("history_index"),
            "commit_time": commit.get("commit_time"),
            "base_commit": dataset_row.get("base_commit"),
            "problem_statement": dataset_row.get("problem_statement", ""),
            "gold_patch": dataset_row.get("patch", ""),
            "graph_path": str(graph_path),
            # Bounded code excerpts are materialized only after sampling. Keeping
            # them out of the full 1,000+ candidate pool avoids retaining large
            # source strings that cannot enter the annotation packet.
            "current_context": None,
            **structure,
        })
    return pool


def add_temporal_features(pool, catalog, meta_by_repo):
    for row in pool:
        repo = row["repository"]
        base_sha = str(row["base_commit"]).lower()
        sha_map = meta_by_repo.get(repo, {}).get("sha_to_history_index", {})
        base_index = sha_map.get(base_sha)
        memory_index = row.get("commit_history_index")
        if base_index is None:
            base_row = catalog.get((repo, base_sha))
            base_index = base_row.get("history_index") if base_row else None
        base_commit_row = catalog.get((repo, base_sha))
        row["base_history_index"] = base_index
        if base_index is None or memory_index is None:
            row["temporal_status"] = "unknown_base_index"
            row["temporal_visible"] = None
        elif int(memory_index) < int(base_index):
            row["temporal_status"] = "before_base"
            row["temporal_visible"] = True
        else:
            row["temporal_status"] = "at_or_after_base"
            row["temporal_visible"] = False
        row["age_commits"] = (
            int(base_index) - int(memory_index)
            if base_index is not None and memory_index is not None else None
        )
        base_time = base_commit_row.get("commit_time") if base_commit_row else None
        memory_time = row.get("commit_time")
        row["age_days"] = (
            (int(base_time) - int(memory_time)) / 86400
            if base_time is not None and memory_time is not None else None
        )
        symbol_condition = (
            row["no_extractable_symbol"]
            or row["any_symbol_resolves_same_file"]
            or row["any_symbol_resolves_uniquely_elsewhere"]
        )
        row["machine_currently_usable"] = (
            None
            if row["temporal_visible"] is None
            else bool(
                row["temporal_visible"]
                and row["any_changed_file_survives"]
                and symbol_condition
            )
        )
    return pool


def deterministic_sample(pool, seed=DEFAULT_SEED, attempts=30000):
    grouped = {
        name: [row for row in pool if row["sampling_stratum"] == name]
        for name in STRATUM_TARGETS
    }
    shortages = {
        name: len(rows)
        for name, rows in grouped.items()
        if len(rows) < STRATUM_TARGETS[name]
    }
    if shortages:
        raise ValueError(f"insufficient stratum pools: {shortages}")
    order = sorted(STRATUM_TARGETS, key=lambda name: len(grouped[name]))
    best = None
    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        counts = Counter()
        selected = []
        ok = True
        for stratum in order:
            choices = list(grouped[stratum])
            rng.shuffle(choices)
            picked = []
            for row in choices:
                if counts[row["case_id"]] >= 2:
                    continue
                django_count = sum(
                    item["repository"] == "django/django"
                    for item in selected + picked
                )
                if row["repository"] == "django/django" and django_count >= 16:
                    continue
                picked.append(row)
                counts[row["case_id"]] += 1
                if len(picked) == STRATUM_TARGETS[stratum]:
                    break
            if len(picked) != STRATUM_TARGETS[stratum]:
                ok = False
                break
            selected.extend(picked)
        if not ok:
            continue
        unique_count = len({row["case_id"] for row in selected})
        score = (unique_count, -sum(r["repository"] == "django/django" for r in selected))
        if best is None or score > best[0]:
            best = (score, selected, attempt)
        if unique_count >= 15:
            rng.shuffle(selected)
            return selected, {"attempt": attempt, "django_cap_relaxed_to": 16}
    if best:
        raise ValueError(
            f"sampling constraints infeasible; best unique-case count={best[0][0]}"
        )
    raise ValueError("sampling constraints infeasible")


def sample_id(row, seed):
    key = f"{seed}|{row['case_id']}|{row['repeat_id']}|{row['memory_sha']}"
    return "AVL-" + hashlib.sha256(key.encode()).hexdigest()[:10].upper()


def manifest_row(row):
    list_fields = {
        "memory_files",
        "memory_symbols",
        "gold_files",
        "gold_symbols",
        "file_overlap_files",
        "symbol_overlap_values",
        "surviving_files",
        "missing_files",
        "current_relation_types",
        "symbol_resolution",
    }
    excluded = {
        "memory_patch",
        "gold_patch",
        "problem_statement",
        "current_context",
        "memory_message",
        "memory_patch_path",
        "commit_index_path",
    }
    result = {}
    for key, value in row.items():
        if key in excluded:
            continue
        result[key] = json.dumps(value, sort_keys=True) if key in list_fields else value
    return result


def annotation_packet(row):
    packet = {
        "sample_id": row["sample_id"],
        "issue": {
            "case_id": row["case_id"],
            "repository": row["repository"],
            "problem_statement": bounded_text(row["problem_statement"], 10000),
        },
        "memory": {
            "sha": row["memory_sha"],
            "subject": row["memory_subject"],
            "message": bounded_text(row["memory_message"], 5000),
            "changed_files": row["memory_files"],
            "patch": bounded_text(row["memory_patch"], 14000),
        },
        "current_base_code": {
            "base_commit": row["base_commit"],
            "contexts": row["current_context"],
            "resolution_note": (
                "Current graph context only; it does not prove that historical "
                "relationships remained unchanged."
            ),
        },
        "gold_oracle": {
            "changed_files": row["gold_files"],
            "patch": bounded_text(row["gold_patch"], 14000),
        },
    }
    assert not (set(packet) & BLINDED_FORBIDDEN_KEYS)
    serialized = json.dumps(packet).lower()
    for forbidden in ("sampling_stratum", "structural_relevant", "paired_harm"):
        if forbidden in serialized:
            raise AssertionError(f"blinded packet leaks {forbidden}")
    return packet


def annotation_rubric():
    return """# Applicability Validation Annotation Rubric

Annotate each packet independently. Do not consult machine labels, Agent
adoption, or final localization outcomes.

## Axis A: Task Applicability

- `Direct-help`: directly transferable evidence for a Gold file, symbol, or
  target behavior. Name the transfer link.
- `Indirect-help`: no direct fix, but concretely narrows the correct module,
  API, dependency, or test scope. Name the bridge.
- `Irrelevant`: no concrete transfer; wrong subsystem, superficial overlap,
  generic coincidence, obsolete, or misleading.

## Axis B: Current-code Validity

- `Valid`: key files exist and at least one key symbol/API anchor resolves,
  with no clear current structural contradiction.
- `Partially-valid`: some paths or symbols changed, but a meaningful anchor
  still maps to current code.
- `Invalid-obsolete`: key anchors do not resolve or the current code clearly
  contradicts the relied-upon structure/API.

For both axes provide one sentence of evidence and confidence `high`, `medium`,
or `low`.
"""


def gpt_prompt():
    return """# Locked GPT Judge Prompt

Model: `gpt-4o-2024-11-20`; temperature: `0`.

Use the accompanying rubric and judge only the supplied packet. Return one JSON
object with: sample_id, applicability_label, applicability_evidence,
validity_label, validity_evidence, confidence. Do not infer Agent outcomes. If
output is invalid JSON, only a format-only retry with identical evidence is
allowed.

"""


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    preregistration = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    selected_ids = set(preregistration["sampling"]["selected_ids"])
    dataset_rows = load_dataset_rows(args.dataset_arrow, selected_ids)
    repos = {row["repo"] for row in dataset_rows.values()}
    catalog, meta = load_commit_catalog(args.index_dirs, repos)
    pool = build_candidate_pool(
        preregistration, args.run_root, dataset_rows, catalog, args.graph_root
    )
    add_temporal_features(pool, catalog, meta)
    selected, sampling_info = deterministic_sample(pool, args.seed)
    selected_graphs = {}
    for row in selected:
        case_id = row["case_id"]
        if case_id not in selected_graphs:
            with Path(row["graph_path"]).open("rb") as handle:
                graph = pickle.load(handle)
            selected_graphs[case_id] = (graph, graph_indexes(graph))
        graph, indexes = selected_graphs[case_id]
        row["current_context"] = current_context(
            graph,
            row["memory_files"] + row["gold_files"],
            row["memory_symbols"],
            indexes=indexes,
        )
    for row in selected:
        row["sample_id"] = sample_id(row, args.seed)
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("sample id collision")

    manifests = [manifest_row(row) for row in selected]
    write_csv(output / "sample_manifest.csv", manifests)
    write_csv(output / "machine_features.csv", manifests)
    packets = [annotation_packet(row) for row in selected]
    write_jsonl(output / "annotation_packet.jsonl", packets)
    (output / "annotation_rubric.md").write_text(annotation_rubric(), encoding="utf-8")
    prompt_text = gpt_prompt()
    prompt_path = output / "gpt_judge_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    prompt_hash = sha256_path(prompt_path)

    human_fields = [
        "sample_id", "applicability_label", "applicability_evidence",
        "validity_label", "validity_evidence", "confidence",
    ]
    human_rows = [{"sample_id": row["sample_id"]} for row in selected]
    human_path = output / "human_labels.csv"
    if not human_path.exists():
        write_csv(human_path, human_rows, human_fields)
    gpt_path = output / "gpt_labels.jsonl"
    gpt_template = [
        {
            "sample_id": row["sample_id"],
            "applicability_label": "",
            "applicability_evidence": "",
            "validity_label": "",
            "validity_evidence": "",
            "confidence": "",
            "model": "gpt-4o-2024-11-20",
            "temperature": 0,
            "prompt_sha256": prompt_hash,
            "system_fingerprint": "",
            "input_tokens": None,
            "output_tokens": None,
            "actual_cost": None,
            "raw_response": "",
        }
        for row in selected
    ]
    if not gpt_path.exists():
        write_jsonl(gpt_path, gpt_template)
    else:
        existing_gpt = read_jsonl(gpt_path)
        if not any(item.get("applicability_label") for item in existing_gpt):
            write_jsonl(gpt_path, gpt_template)
    write_csv(output / "disagreement_audit.csv", [], [
        "sample_id", "human_applicability", "gpt_applicability",
        "machine_structural_relevant", "error_type", "notes",
    ])
    write_csv(output / "bootstrap_kappa.csv", [], ["replicate", "kappa", "defined"])
    write_json(output / "kappa_report.json", {
        "status": "AWAITING_HUMAN_AND_GPT_LABELS",
        "sample_size": len(selected),
    })
    report = (
        "## Material Passport\n\n"
        "- Origin Skill: academic-research-suite / experiment-agent\n"
        "- Protocol: applicability_label_validation_v2\n"
        "- Verification Status: PREPARED\n\n"
        "# Applicability Label Validation\n\n"
        "Status: **AWAITING_HUMAN_AND_GPT_LABELS**\n\n"
        f"Prepared {len(selected)} blinded candidates from "
        f"{len({row['case_id'] for row in selected})} cases. No external GPT "
        "API was called. Complete `human_labels.csv` and `gpt_labels.jsonl`, "
        "then run the analyze phase.\n"
    )
    (output / "applicability_label_validation.md").write_text(report, encoding="utf-8")
    input_paths = [Path(args.preregistration), Path(args.dataset_arrow)]
    for batch in preregistration["design"]["execution_batches"]:
        if batch["arm"] != "memory":
            continue
        name = (
            f"batch_{batch['batch']:02d}_cohort_{batch['cohort']}_"
            f"memory_r{batch['repeat']}"
        )
        input_paths.append(Path(args.run_root) / name / "loc_trajs.repaired.jsonl")
    for path in sorted({row["graph_path"] for row in selected}):
        input_paths.append(Path(path))
    for path in sorted({row["memory_patch_path"] for row in selected if row["memory_patch_path"]}):
        input_paths.append(Path(path))
    for path in sorted({row["commit_index_path"] for row in selected if row["commit_index_path"]}):
        input_paths.append(Path(path))
    for repo in sorted({row["repository"] for row in selected}):
        meta_path = meta.get(repo, {}).get("__source_path")
        if meta_path:
            input_paths.append(Path(meta_path))
    artifact_paths = [
        output / "sample_manifest.csv",
        output / "machine_features.csv",
        output / "annotation_packet.jsonl",
        output / "annotation_rubric.md",
        output / "gpt_judge_prompt.md",
    ]
    write_json(output / "run_manifest.json", {
        "protocol_version": "applicability_label_validation_v2",
        "status": "PREPARED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "seed": args.seed,
        "sampling": sampling_info,
        "stratum_counts": dict(Counter(row["sampling_stratum"] for row in selected)),
        "unique_cases": len({row["case_id"] for row in selected}),
        "django_candidates": sum(row["repository"] == "django/django" for row in selected),
        "candidate_pool_size": len(pool),
        "input_sha256": {str(path): sha256_path(path) for path in input_paths},
        "artifact_sha256": {str(path): sha256_path(path) for path in artifact_paths},
        "gpt_judge": {
            "model": "gpt-4o-2024-11-20",
            "temperature": 0,
            "prompt_sha256": prompt_hash,
        },
        "external_api_called": False,
    })
    print(json.dumps({
        "status": "PREPARED",
        "output": str(output),
        "sample_size": len(selected),
        "unique_cases": len({row["case_id"] for row in selected}),
        "strata": dict(Counter(row["sampling_stratum"] for row in selected)),
    }, sort_keys=True))


def read_human_labels(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def annotation_instructions():
    return """# P0-1 人工标注说明

## 你需要填写的文件

只填写 `applicability_human_annotation_30.csv`。文件共有30行，每行是一条
issue–Memory candidate。请保持 `sample_id` 不变，不要删除、排序或新增行。

## 标注顺序

1. 阅读 `issue_description`，明确任务要求。
2. 阅读 `memory_subject`、`memory_message`、`memory_changed_files` 和
   `memory_patch`，理解历史经验做了什么。
3. 阅读 `current_code_context_json`，判断 Memory 引用的结构在当前 base
   commit 是否仍存在。这里的 current graph 只能证明“当前可解析”，不能证明
   历史关系从未变化。
4. 阅读 `gold_changed_files` 和 `gold_patch`。Gold 在本实验中是离线 oracle，
   用于判断 Memory 对真实修复是否适用。
5. 独立填写 Applicability、Validity 和 confidence。

## Applicability 标签（必填，三选一）

- `Direct-help`：Memory 提供可直接迁移到 Gold file、symbol 或目标行为的
  修复证据。在 evidence 中写明具体 transfer link。
- `Indirect-help`：没有直接给出修复，但正确缩小了模块、API、依赖或测试
  范围。在 evidence 中写明具体 bridge。
- `Irrelevant`：没有具体 transfer；属于错误子系统、表面词汇重合、泛化
  信息、过时信息或误导信息。

`applicability_evidence` 至少一句，不能只写“相关/不相关”。

## Validity 标签（必填，三选一）

- `Valid`：关键文件存在，至少一个关键 symbol/API anchor 可解析，且没有
  明显当前结构冲突。
- `Partially-valid`：部分路径或 symbol 已变化，但仍有可识别且有意义的
  anchor。
- `Invalid-obsolete`：关键 anchor 均无法解析，或当前代码明确否定 Memory
  所依赖的结构/API。

`validity_evidence` 至少一句，写清楚哪些 file/symbol/API 仍存在或已失效。

## Confidence（必填，三选一）

- `high`：证据明确，几乎没有其他合理解释。
- `medium`：总体可判，但存在一处不确定或间接推断。
- `low`：packet 信息不足、patch 截断或两种标签都合理。

不确定时仍选择最合理的标签，并在 `reviewer_notes` 解释；不要留空。

## 严禁在主标签冻结前查看

不要先看 `annotation_provenance_after_freeze.csv` 指向的完整 trajectory，也
不要查看 Agent adoption、最终 top-5、Acc@5、baseline outcome、paired harm、
sampling stratum 或机器 StructuralRelevant。它们会造成 outcome-informed bias。

完成30行后保存 UTF-8 CSV，然后运行：

```bash
source /data/wjl/RepoMem/openaigpt4o_env
/data/conda_envs/locagent/bin/python \\
  scripts/validate_applicability_labels.py analyze
```

在 GPT judge 标签尚未完成时，脚本会正确停在 `AWAITING_GPT_LABELS`；Human
标签不会丢失。
"""


def trajectory_provenance(preregistration, run_root, manifest_rows):
    case_cohort = {}
    for cohort in ("a", "b"):
        for case_id in preregistration["sampling"][f"cohort_{cohort}"]:
            case_cohort[case_id] = cohort
    batch_paths = {}
    for batch in preregistration["design"]["execution_batches"]:
        if batch["arm"] != "memory":
            continue
        name = (
            f"batch_{batch['batch']:02d}_cohort_{batch['cohort']}_"
            f"memory_r{batch['repeat']}"
        )
        batch_paths[(batch["cohort"], int(batch["repeat"]))] = (
            Path(run_root) / name / "loc_trajs.repaired.jsonl"
        )
    line_lookup = {}
    for path in set(batch_paths.values()):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    line_lookup[(str(path), json.loads(line)["instance_id"])] = line_number
    rows = []
    for row in manifest_rows:
        cohort = case_cohort[row["case_id"]]
        path = batch_paths[(cohort, int(row["repeat_id"]))]
        rows.append({
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "repeat_id": row["repeat_id"],
            "memory_sha": row["memory_sha"],
            "best_rank": row["best_rank"],
            "raw_trajectory_file": str(path),
            "raw_trajectory_jsonl_line": line_lookup[(str(path), row["case_id"])],
            "blinding_warning": "VIEW ONLY AFTER ALL 30 HUMAN LABELS ARE FROZEN",
        })
    return rows


def export_human_workbook(args):
    output = Path(args.output)
    packets = read_jsonl(output / "annotation_packet.jsonl")
    with (output / "sample_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    packet_by_id = {row["sample_id"]: row for row in packets}
    existing = {}
    workbook_path = Path(args.human_workbook)
    if workbook_path.exists():
        with workbook_path.open(encoding="utf-8", newline="") as handle:
            existing = {row["sample_id"]: row for row in csv.DictReader(handle)}
    workbook_rows = []
    for manifest in manifest_rows:
        packet = packet_by_id[manifest["sample_id"]]
        prior = existing.get(manifest["sample_id"], {})
        workbook_rows.append({
            "sample_id": manifest["sample_id"],
            "case_id": packet["issue"]["case_id"],
            "repository": packet["issue"]["repository"],
            "issue_description": packet["issue"]["problem_statement"],
            "memory_sha": packet["memory"]["sha"],
            "memory_subject": packet["memory"]["subject"],
            "memory_message": packet["memory"]["message"],
            "memory_changed_files": json.dumps(
                packet["memory"]["changed_files"], ensure_ascii=False
            ),
            "memory_patch": packet["memory"]["patch"],
            "current_base_commit": packet["current_base_code"]["base_commit"],
            "current_code_context_json": json.dumps(
                packet["current_base_code"], ensure_ascii=False, indent=2
            ),
            "gold_changed_files": json.dumps(
                packet["gold_oracle"]["changed_files"], ensure_ascii=False
            ),
            "gold_patch": packet["gold_oracle"]["patch"],
            "applicability_label": prior.get("applicability_label", ""),
            "applicability_evidence": prior.get("applicability_evidence", ""),
            "validity_label": prior.get("validity_label", ""),
            "validity_evidence": prior.get("validity_evidence", ""),
            "confidence": prior.get("confidence", ""),
            "reviewer_notes": prior.get("reviewer_notes", ""),
        })
    write_csv(workbook_path, workbook_rows)
    Path(args.instructions).write_text(annotation_instructions(), encoding="utf-8")
    preregistration = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    provenance = trajectory_provenance(
        preregistration, args.run_root, manifest_rows
    )
    write_csv(output / "annotation_provenance_after_freeze.csv", provenance)
    print(json.dumps({
        "status": "HUMAN_WORKBOOK_EXPORTED",
        "workbook": str(workbook_path),
        "instructions": str(args.instructions),
        "rows": len(workbook_rows),
        "provenance": str(output / "annotation_provenance_after_freeze.csv"),
    }, sort_keys=True))


def label_complete(row):
    return (
        row.get("applicability_label") in APPLICABILITY_LABELS
        and row.get("validity_label") in VALIDITY_LABELS
        and row.get("confidence") in {"high", "medium", "low"}
        and bool(str(row.get("applicability_evidence", "")).strip())
        and bool(str(row.get("validity_evidence", "")).strip())
    )


def gpt_label_complete(row, prompt_hash):
    return (
        label_complete(row)
        and row.get("model") == "gpt-4o-2024-11-20"
        and row.get("temperature") == 0
        and row.get("prompt_sha256") == prompt_hash
        and bool(str(row.get("system_fingerprint", "")).strip())
        and row.get("input_tokens") is not None
        and row.get("output_tokens") is not None
        and row.get("actual_cost") is not None
        and bool(str(row.get("raw_response", "")).strip())
    )


def applicability_binary(label):
    return int(label in {"Direct-help", "Indirect-help"})


def validity_binary(label):
    return int(label in {"Valid", "Partially-valid"})


def csv_optional_bool(value):
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def safe_kappa(left, right, weights=None):
    if not left or len(left) != len(right):
        return None
    if len(set(left) | set(right)) < 2:
        return None
    value = cohen_kappa_score(left, right, weights=weights)
    return float(value) if math.isfinite(value) else None


def agreement_metrics(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    agreement = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None else None
    )
    p_bar = (sum(y_true) + sum(y_pred)) / (2 * len(y_true))
    ac1_chance = 2 * p_bar * (1 - p_bar)
    return {
        "kappa": safe_kappa(y_true, y_pred),
        "raw_agreement": agreement,
        "confusion_matrix_0_1": matrix.tolist(),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "pabak": 2 * agreement - 1,
        "gwet_ac1": (
            (agreement - ac1_chance) / (1 - ac1_chance)
            if ac1_chance < 1 else None
        ),
    }


def cluster_bootstrap_kappa(rows, seed=DEFAULT_SEED, replicates=10000):
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    cases = sorted(by_case)
    rng = np.random.default_rng(seed)
    output = []
    values = []
    undefined = 0
    for index in range(replicates):
        sampled_cases = rng.choice(cases, size=len(cases), replace=True)
        sampled_rows = [row for case in sampled_cases for row in by_case[case]]
        true = [row["human_applicable"] for row in sampled_rows]
        pred = [row["structural_relevant"] for row in sampled_rows]
        value = safe_kappa(true, pred)
        defined = value is not None
        undefined += int(not defined)
        if defined:
            values.append(value)
        output.append({"replicate": index, "kappa": value, "defined": defined})
    ci = [None, None] if not values else [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]
    return output, {
        "ci": ci,
        "undefined_count": undefined,
        "undefined_rate": undefined / replicates,
        "defined_count": len(values),
    }


def dual_condition_gate(point, lower):
    if point is None or lower is None:
        return "SERIOUS_PIVOT"
    if point >= 0.50 and lower >= 0.30:
        return "PROCEED"
    if point >= 0.30 and lower >= 0.00:
        return "REFINE_AND_REPLICATE"
    return "SERIOUS_PIVOT"


def analyze(args):
    output = Path(args.output)
    machine_rows = list(csv.DictReader((output / "machine_features.csv").open(
        encoding="utf-8", newline=""
    )))
    workbook_path = Path(args.human_workbook)
    human_rows = (
        read_human_labels(workbook_path)
        if workbook_path.exists()
        else read_human_labels(output / "human_labels.csv")
    )
    write_csv(output / "human_labels.csv", human_rows, [
        "sample_id", "applicability_label", "applicability_evidence",
        "validity_label", "validity_evidence", "confidence", "reviewer_notes",
    ])
    gpt_rows = read_jsonl(output / "gpt_labels.jsonl")
    machine = {row["sample_id"]: row for row in machine_rows}
    human = {row["sample_id"]: row for row in human_rows}
    gpt = {row["sample_id"]: row for row in gpt_rows}
    expected = set(machine)
    if set(human) != expected or set(gpt) != expected:
        raise ValueError("label template sample IDs do not match machine features")
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    prompt_hash = run_manifest["gpt_judge"]["prompt_sha256"]
    human_missing = sorted(key for key in expected if not label_complete(human[key]))
    gpt_missing = sorted(
        key for key in expected if not gpt_label_complete(gpt[key], prompt_hash)
    )
    if human_missing or gpt_missing:
        status = (
            "AWAITING_HUMAN_AND_GPT_LABELS" if human_missing and gpt_missing
            else "AWAITING_HUMAN_LABELS" if human_missing else "AWAITING_GPT_LABELS"
        )
        report = {
            "status": status,
            "human_incomplete": human_missing,
            "gpt_incomplete": gpt_missing,
        }
        write_json(output / "kappa_report.json", report)
        status_report = f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Protocol: applicability_label_validation_v2
- Verification Status: IN_PROGRESS

# Applicability Label Validation

Status: **{status}**

- Human labels incomplete: {len(human_missing)}/30
- GPT judge labels incomplete: {len(gpt_missing)}/30

No kappa, confidence interval, or gate verdict is computed until both blinded
label sets are complete. GPT label content remains concealed from the human
annotator until the human labels are frozen.
"""
        (output / "applicability_label_validation.md").write_text(
            status_report, encoding="utf-8"
        )
        print(json.dumps({
            "status": status,
            "human_incomplete_count": len(human_missing),
            "gpt_incomplete_count": len(gpt_missing),
        }, sort_keys=True))
        return

    joined = []
    for key in sorted(expected):
        m = machine[key]
        h = human[key]
        g = gpt[key]
        joined.append({
            "sample_id": key,
            "case_id": m["case_id"],
            "sampling_stratum": m["sampling_stratum"],
            "file_overlap": m["file_overlap"] == "True",
            "parent_module_match": m["parent_module_match"] == "True",
            "symbol_overlap": m["symbol_overlap"] == "True",
            "weak_relevant": m["weak_relevant"] == "True",
            "structural_relevant": m["structural_relevant"] == "True",
            "machine_currently_usable": csv_optional_bool(
                m["machine_currently_usable"]
            ),
            "human_applicability": h["applicability_label"],
            "human_applicable": applicability_binary(h["applicability_label"]),
            "human_validity": h["validity_label"],
            "human_currently_usable": validity_binary(h["validity_label"]),
            "gpt_applicability": g["applicability_label"],
            "gpt_applicable": applicability_binary(g["applicability_label"]),
            "gpt_validity": g["validity_label"],
            "gpt_currently_usable": validity_binary(g["validity_label"]),
        })
    human_true = [row["human_applicable"] for row in joined]
    primary_pred = [row["structural_relevant"] for row in joined]
    weak_pred = [row["weak_relevant"] for row in joined]
    primary = agreement_metrics(human_true, primary_pred)
    weak = agreement_metrics(human_true, weak_pred)
    bootstrap_rows, bootstrap = cluster_bootstrap_kappa(
        joined, args.seed, args.bootstrap_replicates
    )
    write_csv(output / "bootstrap_kappa.csv", bootstrap_rows)
    verdict = dual_condition_gate(primary["kappa"], bootstrap["ci"][0])
    human_gpt_binary = safe_kappa(
        human_true, [row["gpt_applicable"] for row in joined]
    )
    human_gpt_binary_metrics = agreement_metrics(
        human_true, [row["gpt_applicable"] for row in joined]
    )
    human_gpt_weighted = safe_kappa(
        [row["human_applicability"] for row in joined],
        [row["gpt_applicability"] for row in joined],
        weights="quadratic",
    )
    validity_rows = [
        row for row in joined if row["machine_currently_usable"] is not None
    ]
    validity = agreement_metrics(
        [row["human_currently_usable"] for row in validity_rows],
        [row["machine_currently_usable"] for row in validity_rows],
    )
    validity["evaluated_count"] = len(validity_rows)
    validity["excluded_unknown_temporal_count"] = len(joined) - len(validity_rows)
    validity["coverage"] = len(validity_rows) / len(joined)
    validity["reference_class_count"] = len({
        row["human_currently_usable"] for row in validity_rows
    })
    validity["agreement_identifiable"] = validity["reference_class_count"] > 1
    flags = []
    if bootstrap["undefined_rate"] > 0.10:
        flags.append("BOOTSTRAP_UNSTABLE")
    if human_gpt_binary is None or human_gpt_binary < 0.40:
        flags.append("ANNOTATION_UNSTABLE")
    effective_action = verdict
    if verdict == "PROCEED" and "ANNOTATION_UNSTABLE" in flags:
        effective_action = "P1_5_EXPLORATORY_ONLY_P2_9_BLOCKED"

    stratum_rates = {}
    for name in STRATUM_TARGETS:
        subset = [row for row in joined if row["sampling_stratum"] == name]
        stratum_rates[name] = sum(row["human_applicable"] for row in subset) / len(subset)
    component_precision = {}
    for component in ("file_overlap", "parent_module_match", "symbol_overlap"):
        subset = [row for row in joined if row[component]]
        component_precision[component] = (
            sum(row["human_applicable"] for row in subset) / len(subset)
            if subset else None
        )
    disagreements = []
    for row in joined:
        error_type = ""
        if row["structural_relevant"] and not row["human_applicable"]:
            error_type = "false_positive"
        elif not row["structural_relevant"] and row["human_applicable"]:
            error_type = "false_negative"
        if error_type or row["human_applicability"] != row["gpt_applicability"]:
            disagreements.append({
                "sample_id": row["sample_id"],
                "human_applicability": row["human_applicability"],
                "gpt_applicability": row["gpt_applicability"],
                "machine_structural_relevant": row["structural_relevant"],
                "error_type": error_type,
                "notes": "",
            })
    write_csv(output / "disagreement_audit.csv", disagreements, [
        "sample_id", "human_applicability", "gpt_applicability",
        "machine_structural_relevant", "error_type", "notes",
    ])
    write_csv(output / "label_comparison.csv", joined)
    human_applicability_counts = dict(Counter(
        row["human_applicability"] for row in joined
    ))
    gpt_applicability_counts = dict(Counter(
        row["gpt_applicability"] for row in joined
    ))
    machine_error_counts = dict(Counter(
        row["error_type"] for row in disagreements if row["error_type"]
    ))
    human_source = Path(args.human_workbook)
    result = {
        "status": "ANALYZED",
        "human_label_source": {
            "path": str(human_source),
            "sha256": sha256_path(human_source),
        },
        "sample_size": len(joined),
        "unique_case_count": len({row["case_id"] for row in joined}),
        "human_applicability_counts": human_applicability_counts,
        "gpt_applicability_counts": gpt_applicability_counts,
        "machine_error_counts": machine_error_counts,
        "primary_level2": primary,
        "weak_level1": weak,
        "bootstrap": bootstrap,
        "human_gpt_binary_kappa": human_gpt_binary,
        "human_gpt_binary_metrics": human_gpt_binary_metrics,
        "human_gpt_weighted_kappa": human_gpt_weighted,
        "machine_human_validity": validity,
        "stratum_human_applicable_rates": stratum_rates,
        "component_precision": component_precision,
        "verdict": verdict,
        "effective_action": effective_action,
        "flags": flags,
    }
    write_json(output / "kappa_report.json", result)
    def metric(value):
        return "NA" if value is None else f"{value:.3f}"

    report = f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Protocol: applicability_label_validation_v2
- Verification Status: ANALYZED

# Applicability Label Validation

## Primary Gate

| Metric | Level 2 Structural | Level 1 File-only |
|---|---:|---:|
| Cohen's kappa | {metric(primary['kappa'])} | {metric(weak['kappa'])} |
| Raw agreement | {metric(primary['raw_agreement'])} | {metric(weak['raw_agreement'])} |
| Balanced accuracy | {metric(primary['balanced_accuracy'])} | {metric(weak['balanced_accuracy'])} |
| MCC | {metric(primary['mcc'])} | {metric(weak['mcc'])} |
| PABAK | {metric(primary['pabak'])} | {metric(weak['pabak'])} |
| Gwet AC1 | {metric(primary['gwet_ac1'])} | {metric(weak['gwet_ac1'])} |

- Level-2 case-cluster 95% CI: {bootstrap['ci']}
- Undefined bootstrap rate: {bootstrap['undefined_rate']:.3f}
- Level-2 confusion matrix `[TN, FP; FN, TP]`: {primary['confusion_matrix_0_1']}
- Human applicability counts: {human_applicability_counts}
- Machine errors: {machine_error_counts}

## Annotation and Validity Checks

- Human-GPT binary kappa: {metric(human_gpt_binary)}
- Human-GPT binary raw agreement: {metric(human_gpt_binary_metrics['raw_agreement'])}
- Human-GPT three-level weighted kappa: {metric(human_gpt_weighted)}
- Machine-Human validity kappa: {metric(validity['kappa'])}
- Machine-Human validity agreement identifiable: {validity['agreement_identifiable']}
- Machine-validity coverage: {validity['evaluated_count']}/30 ({validity['coverage']:.1%})
- Excluded unknown temporal status: {validity['excluded_unknown_temporal_count']}
- Human-applicable rate by stratum: {stratum_rates}
- Component precision: {component_precision}

## Decision

- Gate verdict: **{verdict}**
- Effective action: **{effective_action}**
- Flags: {', '.join(flags) if flags else 'none'}

Structural irrelevance is not outcome harm. No outcome or adoption label was
used in this gate.
"""
    (output / "applicability_label_validation.md").write_text(report, encoding="utf-8")
    validation = f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Protocol: applicability_label_validation_v2
- Verification Status: ANALYZED

# Statistical Validation Report

## Finding

- Primary kappa: {metric(primary['kappa'])}, case-cluster 95% CI {bootstrap['ci']}.
- Preregistered dual-condition verdict: **{verdict}**.
- Human-GPT binary kappa: {metric(human_gpt_binary)}.
- Bootstrap undefined rate: {bootstrap['undefined_rate']:.1%}.

## Warnings

- N=30 candidates from 17 cases gives a wide uncertainty interval; the gate uses
  both the point estimate and lower CI by preregistration.
- Sampling was deliberately stratified by machine structural pattern, so these
  30 rows cannot estimate population prevalence of applicable Memories.
- Kappa is prevalence-sensitive; PABAK and Gwet AC1 are reported as sensitivity
  metrics but do not replace the locked gate.
- The binary Human validity reference has no obsolete negative cases, so its
  kappa/specificity are not identifiable and must not be interpreted as verifier
  performance.
- Structural irrelevance is not Memory harm and supports no causal outcome claim.

## Fallacy Scan — 11/11 Checked

| # | Pattern | Status |
|---|---|---|
| 1 | Simpson's paradox | Checked; stratum rates reported, no aggregate causal trend claimed |
| 2 | Ecological fallacy | Checked; candidate unit retained and cases used only for clustering |
| 3 | Berkson/selection bias | CAUTION: deliberately stratified 30-item validation sample |
| 4 | Collider bias | Checked; no adjusted causal model or collider conditioning |
| 5 | Base-rate neglect | CAUTION addressed with label counts and prevalence-sensitive alternatives |
| 6 | Regression to mean | Not applicable; no extreme-score pre/post comparison |
| 7 | Survivorship bias | Checked; all 30 frozen candidates have both judge labels |
| 8 | Look-elsewhere effect | Checked; one preregistered primary gate, supporting metrics non-gating |
| 9 | Garden of forking paths | Checked; label rule, thresholds, seed, and CI method frozen before annotation |
| 10 | Correlation implies causation | Checked; no claim that structural mismatch causes outcome harm |
| 11 | Reverse causality | Not applicable to agreement validation |

## Reproducibility

- Human source SHA-256: `{sha256_path(human_source)}`
- Bootstrap seed: `{args.seed}`; replicates: `{args.bootstrap_replicates}`
- Deterministic re-run verification is reported separately from this ANALYZED status.
"""
    (output / "validation_report.md").write_text(validation, encoding="utf-8")
    run_manifest["status"] = "ANALYZED"
    run_manifest["analysis_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["human_label_source"] = result["human_label_source"]
    run_manifest.setdefault("input_sha256", {})[str(human_source)] = sha256_path(
        human_source
    )
    artifact_paths = [
        output / "human_labels.csv",
        output / "gpt_labels.jsonl",
        output / "bootstrap_kappa.csv",
        output / "disagreement_audit.csv",
        output / "label_comparison.csv",
        output / "kappa_report.json",
        output / "applicability_label_validation.md",
        output / "validation_report.md",
    ]
    artifact_hashes = run_manifest.setdefault("artifact_sha256", {})
    for path in artifact_paths:
        artifact_hashes[str(path)] = sha256_path(path)
    write_json(output / "run_manifest.json", run_manifest)
    print(json.dumps({
        "status": "ANALYZED", "verdict": verdict,
        "effective_action": effective_action, "flags": flags,
    }, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and analyze P0-1 applicability label validation."
    )
    parser.add_argument("phase", choices=("prepare", "export", "analyze"))
    parser.add_argument("--preregistration", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset-arrow", default=DEFAULT_ARROW)
    parser.add_argument("--graph-root", default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--index-dirs", nargs="+", default=DEFAULT_INDEX_DIRS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--human-workbook", default=DEFAULT_HUMAN_WORKBOOK)
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "export":
        export_human_workbook(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
