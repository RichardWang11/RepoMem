import argparse
import ast
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

DEFAULT_SELECTION = "reproduce_lists/verified_m3a_matched_15.json"
DEFAULT_FLIP_FILE = (
    "results/full500_azure_gpt4o_repaired_compare/check9_gate/"
    "flip_decomposition/flip_decomposition.csv"
)
DEFAULT_V2_ROOT = "results/context_safe_m3a/runs_v2"
DEFAULT_V3_ROOT = "results/context_safe_m3a/replication_v3/runs"
DEFAULT_V4_ROOT = "results/context_safe_m3a/replication_v4/runs"
DEFAULT_OUTPUT = "results/context_safe_m3a/replication_v4/analysis/n2_adoption"
DEFAULT_INDEX_DIRS = [
    "repomem_index/verified_200_episodic_with_issues",
    "repomem_index/verified_remaining_247_episodic_with_issues_v2",
]

MEMORY_TOOLS = {"search_commit", "examine_commit"}
SEARCH_SECTION_RE = re.compile(
    r"##Searching for query `(.*?)`\.\.\.\s*(.*?)(?=##Searching for query `|\Z)",
    re.DOTALL,
)
SEARCH_RESULT_RE = re.compile(
    r"^SHA:\s+(?P<sha>[0-9a-fA-F]+)\s+\|\s+Commit message:\s+"
    r"(?P<subject>.*?)\s+\|\s+Edited files:\s+(?P<files>\[.*?\])"
    r"(?:\s+\|\s+score:\s+(?P<score>[-+]?\d+(?:\.\d+)?))?\s*$"
)
EXAMINE_HEADER_RE = re.compile(
    r"^##Commit for sha `(?P<sha>[0-9a-fA-F]+)`\.\.\.\s*$", re.MULTILINE
)
EDITED_FILES_RE = re.compile(r"^Edited files:\s+(\[.*\])\s*$", re.MULTILINE)
PY_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py)"
)
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEF_CLASS_RE = re.compile(r"^[+\- ]*\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", re.MULTILINE)
HUNK_SYMBOL_RE = re.compile(r"^@@.*?@@\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", re.MULTILINE)


def load_ids(path):
    with open(path, encoding="utf-8") as handle:
        return set(json.load(handle))


def files_from_patch(patch):
    files = []
    seen = set()
    for line in (patch or "").splitlines():
        if not line.startswith("+++ "):
            continue
        path = normalize_path(line[4:].split("\t", 1)[0])
        if path == "/dev/null" or not path.endswith(".py") or path in seen:
            continue
        files.append(path)
        seen.add(path)
    return files


def load_gold(dataset, split, selected_ids, dataset_arrow=None):
    from datasets import Dataset, load_dataset

    result = {}
    rows = Dataset.from_file(dataset_arrow) if dataset_arrow else load_dataset(dataset, split=split)
    for row in rows:
        instance_id = row["instance_id"]
        if instance_id in selected_ids:
            result[instance_id] = files_from_patch(row.get("patch", ""))
    missing = selected_ids - set(result)
    if missing:
        raise ValueError(f"missing gold patches: {sorted(missing)}")
    return result


def normalize_path(value):
    value = str(value or "").strip().strip("`'\" ,:;").replace("\\", "/")
    for prefix in ("./", "a/", "b/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value.lstrip("/")


def normalize_sha(value):
    return str(value or "").strip().lower()


def parse_json_arguments(raw):
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_file_list(raw):
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [normalize_path(item) for item in value if str(item).strip()]


def parse_search_observation(content):
    sections = []
    for query, body in SEARCH_SECTION_RE.findall(content or ""):
        items = []
        for line in body.splitlines():
            match = SEARCH_RESULT_RE.match(line.strip())
            if not match:
                continue
            score = match.group("score")
            items.append(
                {
                    "sha": normalize_sha(match.group("sha")),
                    "subject": match.group("subject").strip(),
                    "visible_files": parse_file_list(match.group("files")),
                    "score": float(score) if score is not None else None,
                    "rank": len(items) + 1,
                }
            )
        sections.append({"query": query, "items": items})
    return sections


def parse_examine_observation(content):
    content = content or ""
    headers = list(EXAMINE_HEADER_RE.finditer(content))
    rows = []
    for index, match in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(content)
        block = content[match.end() : end]
        file_match = EDITED_FILES_RE.search(block)
        patch = block.split("Patch:\n", 1)[1].strip() if "Patch:\n" in block else ""
        rows.append(
            {
                "sha": normalize_sha(match.group("sha")),
                "files": parse_file_list(file_match.group(1)) if file_match else [],
                "patch": patch,
                "patch_present": bool(patch),
                "patch_truncated": "[patch truncated]" in patch,
                "patch_symbols": sorted(extract_patch_symbols(patch)),
            }
        )
    return rows


def extract_patch_symbols(patch):
    return set(DEF_CLASS_RE.findall(patch or "")) | set(HUNK_SYMBOL_RE.findall(patch or ""))


def flatten_string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from flatten_string_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from flatten_string_values(child)


def extract_argument_features(arguments, raw_arguments=""):
    strings = list(flatten_string_values(arguments or {}))
    if not strings and raw_arguments:
        strings = [raw_arguments]
    paths = set()
    identifiers = set()
    for value in strings:
        identifiers.update(IDENTIFIER_RE.findall(value))
        for match in PY_PATH_RE.findall(value):
            paths.add(normalize_path(match.split(":", 1)[0]))
    return sorted(paths), sorted(identifiers)


def flatten_predicted_files(value):
    if not isinstance(value, list):
        return []
    if len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    result = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        path = normalize_path(item)
        if path and path not in seen:
            result.append(path)
            seen.add(path)
    return result


def load_flip_classes(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["case_id"]: row["flip_class"] for row in csv.DictReader(handle)}


def load_commit_lookup(index_dirs, selected_repos=None):
    selected_repos = set(selected_repos or [])
    lookup = {}
    for root in index_dirs:
        if not os.path.isdir(root):
            continue
        for repo_dir in sorted(os.listdir(root)):
            path = Path(root) / repo_dir / "commits.jsonl"
            if not path.exists():
                continue
            repo_from_dir = repo_dir.replace("__", "/")
            if selected_repos and repo_from_dir not in selected_repos:
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    repo = row.get("repo") or repo_from_dir
                    sha = normalize_sha(row.get("sha"))
                    if not sha:
                        continue
                    item = {
                        "repo": repo,
                        "sha": sha,
                        "subject": row.get("subject", ""),
                        "message": row.get("message", ""),
                        "changed_files": [
                            normalize_path(item) for item in row.get("changed_files") or []
                        ],
                        "patch_path": str(Path(root) / repo_dir / row["patch_path"])
                        if row.get("patch_path")
                        else None,
                    }
                    for length in range(7, min(13, len(sha) + 1)):
                        lookup[(repo, sha[:length])] = item
                    lookup[(repo, sha)] = item
    return lookup


def lookup_commit(lookup, repo, sha):
    sha = normalize_sha(sha)
    if (repo, sha) in lookup:
        return lookup[(repo, sha)]
    matches = [item for (candidate_repo, key), item in lookup.items() if candidate_repo == repo and key.startswith(sha)]
    unique = {item["sha"]: item for item in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def latest_event_for_sha(events, sha, before_message_index=None):
    sha = normalize_sha(sha)
    candidates = []
    for event in events:
        observed_at = event.get("observation_message_index")
        if observed_at is None:
            continue
        if before_message_index is not None and observed_at >= before_message_index:
            continue
        if sha in event.get("sha_set", set()):
            candidates.append(event)
    return max(candidates, key=lambda item: (item["observation_message_index"], item["event_index"])) if candidates else None


def same_parent(changed_file, argument_path):
    changed_parent = str(Path(changed_file).parent).replace("\\", "/")
    argument_parent = str(Path(argument_path).parent).replace("\\", "/")
    return changed_parent not in ("", ".") and changed_parent == argument_parent


def analyze_trajectory_row(row, version, repeat, gold_files, flip_class, commit_lookup):
    instance_id = row.get("instance_id")
    repo = (row.get("meta_data") or {}).get("repo", "")
    predicted = flatten_predicted_files(row.get("found_files") or [])
    final_top5 = predicted[:5]
    gold_set = {normalize_path(item) for item in gold_files or []}
    all_exposures = []
    all_examinations = []
    all_tool_rows = []
    all_event_rows = []
    trajectory_rows = []
    parse_errors = []

    trajectories = (row.get("loc_trajs") or {}).get("trajs") or []
    for traj_index, trajectory in enumerate(trajectories):
        messages = trajectory.get("messages") or []
        events = []
        pending = {}
        examine_blocks = []
        tool_calls = []
        call_sequence = 0

        for message_index, message in enumerate(messages):
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                name = function.get("name") or ""
                raw_arguments = function.get("arguments") or "{}"
                arguments = parse_json_arguments(raw_arguments)
                call = {
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": name,
                    "arguments": arguments,
                    "raw_arguments": raw_arguments,
                    "message_index": message_index,
                    "call_sequence": call_sequence,
                }
                call_sequence += 1
                pending[call["tool_call_id"]] = call
                tool_calls.append(call)
                if arguments is None:
                    parse_errors.append(
                        f"{instance_id}: malformed arguments for {name} at message {message_index}"
                    )
                if name == "search_commit":
                    args = arguments or {}
                    queries = args.get("query_list") or []
                    if isinstance(queries, str):
                        queries = [queries]
                    try:
                        requested_top_k = int(args.get("top_k", 20))
                    except (TypeError, ValueError):
                        requested_top_k = 20
                    event = {
                        "event_index": len(events),
                        "tool_call_id": call["tool_call_id"],
                        "call_message_index": message_index,
                        "observation_message_index": None,
                        "call_sequence": call["call_sequence"],
                        "queries_submitted": [str(query) for query in queries],
                        "queries_executed": [str(query) for query in queries[:3]],
                        "requested_top_k": requested_top_k,
                        "items": [],
                        "sha_set": set(),
                    }
                    events.append(event)
                    call["retrieval_event_index"] = event["event_index"]

            if message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id")
            call = pending.get(call_id)
            if not call:
                if message.get("name") in MEMORY_TOOLS:
                    parse_errors.append(f"{instance_id}: unmatched observation {call_id}")
                continue
            if call["tool_name"] == "search_commit":
                event = events[call["retrieval_event_index"]]
                event["observation_message_index"] = message_index
                sections = parse_search_observation(message.get("content") or "")
                expected_queries = event["queries_executed"]
                if len(sections) != len(expected_queries):
                    parse_errors.append(
                        f"{instance_id}: event {event['event_index']} has "
                        f"{len(expected_queries)} executed queries but {len(sections)} sections"
                    )
                used_sections = set()
                for query_index, query in enumerate(expected_queries):
                    section_index = next(
                        (
                            index
                            for index, section in enumerate(sections)
                            if index not in used_sections and section["query"] == query
                        ),
                        query_index if query_index < len(sections) else None,
                    )
                    if section_index is None:
                        continue
                    used_sections.add(section_index)
                    section = sections[section_index]
                    for parsed_item in section["items"]:
                        item = dict(parsed_item)
                        item["query_index"] = query_index
                        item["query"] = query
                        event["items"].append(item)
                        event["sha_set"].add(item["sha"])
            elif call["tool_name"] == "examine_commit":
                for block in parse_examine_observation(message.get("content") or ""):
                    block.update(
                        {
                            "message_index": message_index,
                            "call_message_index": call["message_index"],
                            "call_sequence": call["call_sequence"],
                            "tool_call_id": call_id,
                        }
                    )
                    examine_blocks.append(block)

        event_sha_state = defaultdict(
            lambda: {
                "selected": False,
                "strict": False,
                "medium": False,
                "loose": False,
                "final": False,
                "examine_message_index": None,
                "patch_symbols": set(),
            }
        )

        for block in examine_blocks:
            event = latest_event_for_sha(events, block["sha"], block["message_index"])
            if event is None:
                parse_errors.append(
                    f"{instance_id}: examined SHA {block['sha']} has no prior retrieval event"
                )
                attributed_event = None
            else:
                attributed_event = event["event_index"]
                state = event_sha_state[(attributed_event, block["sha"])]
                state["selected"] = True
                state["examine_message_index"] = block["message_index"]
                state["patch_symbols"].update(block["patch_symbols"])
            examination_row = {
                "version": version,
                "instance_id": instance_id,
                "flip_class": flip_class,
                "repeat": repeat,
                "traj_index": traj_index,
                "retrieval_event_index": attributed_event,
                **{key: value for key, value in block.items() if key != "patch"},
                "patch_chars": len(block["patch"]),
            }
            all_examinations.append(examination_row)

        non_memory_tools = []
        for call in tool_calls:
            if call["tool_name"] in MEMORY_TOOLS or call["tool_name"] == "finish":
                continue
            paths, identifiers = extract_argument_features(
                call["arguments"], call["raw_arguments"]
            )
            prior_events = [
                event
                for event in events
                if event.get("observation_message_index") is not None
                and event["observation_message_index"] < call["message_index"]
            ]
            if not prior_events:
                continue
            tool_row = {
                "version": version,
                "instance_id": instance_id,
                "flip_class": flip_class,
                "repeat": repeat,
                "traj_index": traj_index,
                "message_index": call["message_index"],
                "call_sequence": call["call_sequence"],
                "tool_call_id": call["tool_call_id"],
                "tool_name": call["tool_name"],
                "argument_paths": paths,
                "argument_identifiers": identifiers,
                "raw_arguments": call["raw_arguments"],
            }
            non_memory_tools.append(tool_row)
            all_tool_rows.append(tool_row)

            all_prior_shas = {sha for event in prior_events for sha in event["sha_set"]}
            for sha in all_prior_shas:
                event = latest_event_for_sha(events, sha, call["message_index"])
                if event is None:
                    continue
                commit = lookup_commit(commit_lookup, repo, sha)
                changed_files = set(commit["changed_files"] if commit else [])
                state = event_sha_state[(event["event_index"], sha)]
                if changed_files & set(paths):
                    state["strict"] = True
                if any(same_parent(changed, path) for changed in changed_files for path in paths):
                    state["medium"] = True
                examined_before_call = (
                    state["examine_message_index"] is not None
                    and state["examine_message_index"] < call["message_index"]
                )
                if examined_before_call and state["patch_symbols"] & set(identifiers):
                    state["loose"] = True

        for sha in {sha for event in events for sha in event["sha_set"]}:
            event = latest_event_for_sha(events, sha)
            if event is None:
                continue
            commit = lookup_commit(commit_lookup, repo, sha)
            changed_files = set(commit["changed_files"] if commit else [])
            if changed_files & set(final_top5):
                event_sha_state[(event["event_index"], sha)]["final"] = True

        for event in events:
            event_exposures = []
            unique_shas = set()
            for item in event["items"]:
                sha = item["sha"]
                unique_shas.add(sha)
                commit = lookup_commit(commit_lookup, repo, sha)
                full_files = commit["changed_files"] if commit else item["visible_files"]
                state = event_sha_state[(event["event_index"], sha)]
                exposure = {
                    "version": version,
                    "instance_id": instance_id,
                    "repo": repo,
                    "flip_class": flip_class,
                    "repeat": repeat,
                    "traj_index": traj_index,
                    "retrieval_event_index": event["event_index"],
                    "retrieval_message_index": event["observation_message_index"],
                    "query_index": item["query_index"],
                    "query": item["query"],
                    "requested_top_k": event["requested_top_k"],
                    "result_count_for_query": sum(
                        candidate["query_index"] == item["query_index"]
                        for candidate in event["items"]
                    ),
                    "sha": sha,
                    "rank": item["rank"],
                    "rank_quality": -item["rank"],
                    "score": item["score"],
                    "subject": commit["subject"] if commit else item["subject"],
                    "visible_files": item["visible_files"],
                    "changed_files": full_files,
                    "commit_lookup_found": commit is not None,
                    "gold_file_overlap": bool(set(full_files) & gold_set),
                    "gold_overlap_files": sorted(set(full_files) & gold_set),
                    "selection_adoption": state["selected"],
                    "exploration_strict": state["strict"],
                    "exploration_medium": state["medium"],
                    "exploration_loose": state["loose"],
                    "final_adoption": state["final"],
                }
                event_exposures.append(exposure)
                all_exposures.append(exposure)

            unique_states = [
                event_sha_state[(event["event_index"], sha)] for sha in unique_shas
            ]
            event_row = {
                "version": version,
                "instance_id": instance_id,
                "flip_class": flip_class,
                "repeat": repeat,
                "traj_index": traj_index,
                "retrieval_event_index": event["event_index"],
                "call_message_index": event["call_message_index"],
                "observation_message_index": event["observation_message_index"],
                "queries_submitted": event["queries_submitted"],
                "queries_executed": event["queries_executed"],
                "requested_top_k": event["requested_top_k"],
                "result_exposure_count": len(event_exposures),
                "unique_sha_count": len(unique_shas),
                "any_examined": any(state["selected"] for state in unique_states),
                "any_exploration_strict": any(state["strict"] for state in unique_states),
                "any_exploration_medium": any(state["medium"] for state in unique_states),
                "any_exploration_loose": any(state["loose"] for state in unique_states),
                "any_final_adoption": any(state["final"] for state in unique_states),
            }
            all_event_rows.append(event_row)

        event_rows_for_traj = [
            item
            for item in all_event_rows
            if item["repeat"] == repeat
            and item["instance_id"] == instance_id
            and item["traj_index"] == traj_index
        ]
        submitted_count = sum(len(event["queries_submitted"]) for event in events)
        result_count = sum(len(event["items"]) for event in events)
        selected_states = [state for state in event_sha_state.values() if state["selected"]]
        trajectory_rows.append(
            {
                "version": version,
                "instance_id": instance_id,
                "repo": repo,
                "flip_class": flip_class,
                "repeat": repeat,
                "traj_index": traj_index,
                "retrieval_event_count": len(events),
                "query_submitted_count": submitted_count,
                "result_exposure_count": result_count,
                "any_examined": bool(selected_states),
                "examined_unique_sha_count": sum(
                    1 for state in event_sha_state.values() if state["selected"]
                ),
                "any_exploration_strict": any(
                    item["any_exploration_strict"] for item in event_rows_for_traj
                ),
                "any_exploration_medium": any(
                    item["any_exploration_medium"] for item in event_rows_for_traj
                ),
                "any_exploration_loose": any(
                    item["any_exploration_loose"] for item in event_rows_for_traj
                ),
                "any_final_adoption": any(
                    state["selected"] and state["final"] for state in event_sha_state.values()
                ),
                "final_top5": final_top5,
                "final_file_count_before_cap": len(predicted),
                "parse_error_count": len(parse_errors),
            }
        )

    return {
        "exposures": all_exposures,
        "examinations": all_examinations,
        "tools": all_tool_rows,
        "events": all_event_rows,
        "trajectories": trajectory_rows,
        "parse_errors": parse_errors,
    }


def load_version(root, version, selected_ids, gold, flip_classes, commit_lookup):
    combined = {
        "exposures": [],
        "examinations": [],
        "tools": [],
        "events": [],
        "trajectories": [],
        "parse_errors": [],
    }
    for repeat in range(1, 5):
        path = Path(root) / f"memory_r{repeat}" / "loc_trajs.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        seen = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                instance_id = row.get("instance_id")
                if instance_id not in selected_ids:
                    continue
                seen.add(instance_id)
                result = analyze_trajectory_row(
                    row,
                    version,
                    repeat,
                    gold[instance_id],
                    flip_classes[instance_id],
                    commit_lookup,
                )
                for key in combined:
                    combined[key].extend(result[key])
        missing = selected_ids - seen
        if missing:
            raise ValueError(f"{path}: missing selected cases {sorted(missing)}")
    return combined


def funnel_counts(trajectories):
    counts = Counter()
    for row in trajectories:
        if row["query_submitted_count"] == 0:
            counts["zero_query_submitted"] += 1
        elif row["result_exposure_count"] == 0:
            counts["query_submitted_zero_results"] += 1
        elif not row["any_examined"]:
            counts["results_returned_zero_examined"] += 1
        else:
            counts["results_returned_ge1_examined"] += 1
        if row["any_examined"] and row["any_final_adoption"]:
            counts["examined_ge1_file_in_top5"] += 1
    return dict(counts)


def safe_tau(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    result = kendalltau(x, y, variant="b", nan_policy="omit")
    return None if math.isnan(result.statistic) else float(result.statistic)


def cluster_bootstrap_tau(rows, x_key, y_key, n_boot=10000, seed=42):
    by_case = defaultdict(list)
    for row in rows:
        if row.get(x_key) is not None and row.get(y_key) is not None:
            by_case[row["instance_id"]].append(row)
    cases = sorted(by_case)
    if not cases:
        return [None, None]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        sample_rows = [row for case in sampled for row in by_case[case]]
        tau = safe_tau(
            [row[x_key] for row in sample_rows],
            [int(row[y_key]) for row in sample_rows],
        )
        if tau is not None:
            values.append(tau)
    return np.percentile(values, [2.5, 97.5]).tolist() if values else [None, None]


def within_query_permutation(rows, n_perm=10000, seed=42):
    valid = [row for row in rows if row.get("rank_quality") is not None]
    observed = safe_tau(
        [row["rank_quality"] for row in valid],
        [int(row["gold_file_overlap"]) for row in valid],
    )
    if observed is None:
        return {"observed_tau": None, "p_two_sided": None, "n_perm": n_perm}
    groups = defaultdict(list)
    for index, row in enumerate(valid):
        key = (
            row["instance_id"],
            row["repeat"],
            row["traj_index"],
            row["retrieval_event_index"],
            row["query_index"],
        )
        groups[key].append(index)
    x = [row["rank_quality"] for row in valid]
    base_y = np.asarray([int(row["gold_file_overlap"]) for row in valid])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        permuted = base_y.copy()
        for indexes in groups.values():
            permuted[indexes] = rng.permutation(permuted[indexes])
        tau = safe_tau(x, permuted.tolist())
        if tau is not None and abs(tau) >= abs(observed) - 1e-12:
            extreme += 1
    return {
        "observed_tau": observed,
        "p_two_sided": (extreme + 1) / (n_perm + 1),
        "n_perm": n_perm,
    }


def alignment_label(tau):
    if tau is None:
        return "not_estimable"
    if tau > 0.30:
        return "aligned"
    if abs(tau) <= 0.10:
        return "near_zero"
    if tau < -0.10:
        return "anti_aligned"
    return "weak_or_inconclusive_positive"


def deduplicate_trajectory_sha(rows):
    best = {}
    for row in rows:
        key = (row["instance_id"], row["repeat"], row["traj_index"], row["sha"])
        current = best.get(key)
        if current is None or row["rank"] < current["rank"]:
            best[key] = row
    return list(best.values())


def bm25_alignment(rows, n_boot, n_perm, primary=False):
    rank_tau = safe_tau(
        [row["rank_quality"] for row in rows],
        [int(row["gold_file_overlap"]) for row in rows],
    )
    deduped = deduplicate_trajectory_sha(rows)
    result = {
        "n_exposures": len(rows),
        "n_relevant_exposures": sum(row["gold_file_overlap"] for row in rows),
        "n_unique_trajectory_shas": len(deduped),
        "rank_quality_vs_gold_overlap_tau_b": rank_tau,
        "rank_alignment_label": alignment_label(rank_tau),
        "rank_tau_cluster_bootstrap_95ci": cluster_bootstrap_tau(
            rows, "rank_quality", "gold_file_overlap", n_boot=n_boot
        ),
        "dedup_rank_quality_vs_gold_overlap_tau_b": safe_tau(
            [row["rank_quality"] for row in deduped],
            [int(row["gold_file_overlap"]) for row in deduped],
        ),
        "score_vs_gold_overlap_tau_b": safe_tau(
            [row["score"] for row in rows if row["score"] is not None],
            [
                int(row["gold_file_overlap"])
                for row in rows
                if row["score"] is not None
            ],
        ),
        "rank_quality_vs_selection_tau_b": safe_tau(
            [row["rank_quality"] for row in rows],
            [int(row["selection_adoption"]) for row in rows],
        ),
    }
    result["within_query_permutation"] = (
        within_query_permutation(rows, n_perm=n_perm) if primary else None
    )
    return result


def cluster_bootstrap_rate(rows, field, n_boot=10000, seed=42):
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["instance_id"]].append(row)
    cases = sorted(by_case)
    if not cases:
        return [None, None]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        sample_rows = [row for case in sampled for row in by_case[case]]
        values.append(np.mean([bool(row[field]) for row in sample_rows]))
    return np.percentile(values, [2.5, 97.5]).tolist()


def adoption_summary(trajectories, n_boot=10000):
    fields = [
        "any_examined",
        "any_exploration_strict",
        "any_exploration_medium",
        "any_exploration_loose",
        "any_final_adoption",
    ]
    groups = {"all": trajectories}
    for label in ("memory_flipped", "stable_harm"):
        groups[label] = [row for row in trajectories if row["flip_class"] == label]
    result = {}
    for label, rows in groups.items():
        result[label] = {"n_trajectories": len(rows), "n_cases": len({row["instance_id"] for row in rows})}
        for field in fields:
            count = sum(bool(row[field]) for row in rows)
            result[label][field] = {
                "count": count,
                "rate": count / len(rows) if rows else None,
                "instance_cluster_bootstrap_95ci": cluster_bootstrap_rate(
                    rows, field, n_boot=n_boot
                ),
            }
    return result


def item_adoption_summary(exposures):
    result = {}
    scopes = {
        "all_retrieved": exposures,
        "examined_only": [row for row in exposures if row["selection_adoption"]],
        "gold_relevant": [row for row in exposures if row["gold_file_overlap"]],
        "gold_irrelevant": [row for row in exposures if not row["gold_file_overlap"]],
    }
    for label, rows in scopes.items():
        item = {"n_exposures": len(rows)}
        for field in (
            "selection_adoption",
            "exploration_strict",
            "exploration_medium",
            "exploration_loose",
            "final_adoption",
        ):
            count = sum(bool(row[field]) for row in rows)
            item[field] = {
                "count": count,
                "rate": count / len(rows) if rows else None,
            }
        result[label] = item
    return result


def adoption_strata_rows(exposures, version, stratum_key):
    groups = defaultdict(list)
    for row in exposures:
        groups[row[stratum_key]].append(row)
    result = []
    for stratum, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        result.append(
            {
                "version": version,
                "stratum": stratum_key,
                "value": stratum,
                "n_exposures": len(rows),
                "gold_relevant_count": sum(row["gold_file_overlap"] for row in rows),
                "selection_count": sum(row["selection_adoption"] for row in rows),
                "selection_rate": np.mean([row["selection_adoption"] for row in rows]),
                "exploration_strict_rate": np.mean([row["exploration_strict"] for row in rows]),
                "exploration_medium_rate": np.mean([row["exploration_medium"] for row in rows]),
                "exploration_loose_rate": np.mean([row["exploration_loose"] for row in rows]),
                "final_adoption_rate": np.mean([row["final_adoption"] for row in rows]),
            }
        )
    return result


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(json_ready(value), ensure_ascii=False)
                        if isinstance(value, (list, dict, tuple, set))
                        else value
                        for key, value in row.items()
                    }
                )


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def render_verify_report(version_results):
    v4 = version_results["v4"]
    trajectories = v4["trajectories"]
    exposures = v4["exposures"]
    examine_calls = len(
        {
            (row["instance_id"], row["repeat"], row["tool_call_id"])
            for row in v4["examinations"]
        }
    )
    patches_present = sum(row["patch_present"] for row in v4["examinations"])
    return "\n".join(
        [
            "# V4 Memory trajectory logging verify report",
            "",
            "| Check | Status | Evidence |",
            "|---|---|---|",
            f"| Q1 retrieval query | Present | {sum(row['query_submitted_count'] > 0 for row in trajectories)}/60 trajectories |",
            f"| Q2 Top-K IDs and scores | Present | {len(exposures)} query-item exposures from 60 retrieval calls; requested K is retained and actual result count is parsed |",
            f"| Q3 item content | Present in stages | Search exposes message/files; {examine_calls} ExamineCommit calls returned {len(v4['examinations'])} commit blocks, {patches_present} with patches |",
            f"| Q4 retrieval iteration | Derivable | {len(v4['events'])} ordered retrieval events with message/call indexes; no standalone field |",
            f"| Q5 later tool calls | Sequence present; timestamp absent | {len(v4['tools'])} post-retrieval non-memory calls retain message/call indexes and tool_call_id |",
            f"| Q6 final top-5 | Derivable | {sum(bool(row['final_top5']) for row in trajectories)}/60 trajectories have ordered predictions |",
            "",
            "Decision: sequence-based N2-A is supported without relogging. Wall-clock latency analysis is not supported.",
            "",
        ]
    )


def render_report(summary):
    v4_funnel = summary["versions"]["v4"]["funnel"]
    v4_alignment = summary["versions"]["v4"]["bm25_alignment"]
    v4_adoption = summary["versions"]["v4"]["adoption"]["all"]
    ci = v4_alignment["rank_tau_cluster_bootstrap_95ci"]
    lines = [
        "# N2-A: V4 Memory Retrieval–Adoption Mechanism Analysis",
        "",
        "## Sample scope disclaimer",
        "",
        "All analyses use 15 outcome-selected cases (10 memory-flipped and 5 stable-harm) from the Paper 1 case set. Results are diagnostic mechanism findings conditional on this subset, not general mechanism claims about random SWE-bench Verified cases. Generalization evidence must come from the Paper 2 random pilot (n=20).",
        "",
        "The 60 trajectories are repeated measurements of 15 cases, not 60 independent cases. Instance-cluster bootstrap intervals preserve that distinction.",
        "",
        "## V4 adoption baseline distribution",
        "",
        "| Category | trajectories / 60 |",
        "|---|---:|",
        f"| Zero query submitted | {v4_funnel.get('zero_query_submitted', 0)} |",
        f"| Query submitted, zero results returned | {v4_funnel.get('query_submitted_zero_results', 0)} |",
        f"| Results returned, zero examined | {v4_funnel.get('results_returned_zero_examined', 0)} |",
        f"| Results returned, ≥1 examined | {v4_funnel.get('results_returned_ge1_examined', 0)} |",
        f"| Results examined, ≥1 examined-commit file in top-5 | {v4_funnel.get('examined_ge1_file_in_top5', 0)} |",
        "",
        "The four no-final-adoption trajectories are all repeats of `sympy__sympy-17318`; this is one case-level pattern, not four independent failures.",
        "",
        "## Layered adoption",
        "",
        "| Metric | count | rate | instance-cluster 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for field, label in [
        ("any_examined", "Selection: ≥1 retrieved SHA examined"),
        ("any_exploration_strict", "Exploration L1: exact file path"),
        ("any_exploration_medium", "Exploration L2: immediate parent module"),
        ("any_exploration_loose", "Exploration L3: exposed patch symbol"),
        ("any_final_adoption", "Final: examined commit file in top-5"),
    ]:
        metric = v4_adoption[field]
        metric_ci = metric["instance_cluster_bootstrap_95ci"]
        lines.append(
            f"| {label} | {metric['count']} | {metric['rate']:.3f} | "
            f"[{fmt(metric_ci[0])}, {fmt(metric_ci[1])}] |"
        )
    item_summary = summary["versions"]["v4"]["item_adoption"]
    lines.extend(
        [
            "",
            "Exploration levels are parallel operationalizations and are not collapsed into one adopted flag. Exact final adoption uses complete index changed-files joined by SHA.",
            "",
            "| Exposure scope | N | selected | strict exploration | medium exploration | loose exploration | final overlap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scope in ("all_retrieved", "examined_only", "gold_relevant", "gold_irrelevant"):
        item = item_summary[scope]
        lines.append(
            f"| {scope} | {item['n_exposures']} | "
            f"{fmt(item['selection_adoption']['rate'])} | "
            f"{fmt(item['exploration_strict']['rate'])} | "
            f"{fmt(item['exploration_medium']['rate'])} | "
            f"{fmt(item['exploration_loose']['rate'])} | "
            f"{fmt(item['final_adoption']['rate'])} |"
        )
    permutation = v4_alignment["within_query_permutation"]
    lines.extend(
        [
            "",
            "## BM25 message-ranking versus file relevance",
            "",
            f"BM25 rank quality (`-rank`) versus gold-file overlap: Kendall τ-b = {fmt(v4_alignment['rank_quality_vs_gold_overlap_tau_b'])}, instance-cluster bootstrap 95% CI [{fmt(ci[0])}, {fmt(ci[1])}]. Diagnostic label: **{v4_alignment['rank_alignment_label']}**.",
            "",
            f"There are {v4_alignment['n_exposures']} retrieval exposures, of which {v4_alignment['n_relevant_exposures']} touch at least one gold file. The trajectory-SHA deduplicated sensitivity estimate is τ-b = {fmt(v4_alignment['dedup_rank_quality_vs_gold_overlap_tau_b'])}.",
            "",
            f"Score versus gold overlap τ-b = {fmt(v4_alignment['score_vs_gold_overlap_tau_b'])}; rank quality versus Agent selection τ-b = {fmt(v4_alignment['rank_quality_vs_selection_tau_b'])}.",
            "",
            f"The within-query permutation test gives two-sided p = {fmt(permutation['p_two_sided'])} ({permutation['n_perm']} permutations).",
            "",
            "BM25 indexes `subject + full commit message`; patch, changed files, and issue text do not participate in scoring. Consequently, weak rank–file alignment is evidence of objective mismatch, not by itself proof that BM25 is intrinsically noisy on message retrieval.",
            "",
            "## Replication and sensitivity",
            "",
            "| Version | trajectories | retrieval events | rank–relevance τ-b | label |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for version in ("v4", "v3", "v2"):
        item = summary["versions"][version]
        alignment = item["bm25_alignment"]
        lines.append(
            f"| {version.upper()} | {item['trajectory_count']} | {item['retrieval_event_count']} | "
            f"{fmt(alignment['rank_quality_vs_gold_overlap_tau_b'])} | {alignment['rank_alignment_label']} |"
        )
    lines.extend(
        [
            "",
            "V3 multi-retrieval trajectories are expanded per retrieval event. Its trajectory-level `any-*` summaries are the compatibility view used beside V4. V2 is sensitivity-only because provider response metadata are unavailable.",
            "",
            "## Interpretation limits",
            "",
            "The logs preserve event order but not absolute timestamps, so this report makes no latency claim. Adoption is behavioral overlap, not proof that memory causally produced a tool call or final prediction. All mechanism interpretations remain conditional on the outcome-selected 15-case sample.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze memory retrieval and adoption in V4/V3/V2 trajectories.")
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--flip_file", default=DEFAULT_FLIP_FILE)
    parser.add_argument("--v2_root", default=DEFAULT_V2_ROOT)
    parser.add_argument("--v3_root", default=DEFAULT_V3_ROOT)
    parser.add_argument("--v4_root", default=DEFAULT_V4_ROOT)
    parser.add_argument("--index_dirs", nargs="+", default=DEFAULT_INDEX_DIRS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--dataset_arrow",
        default=None,
        help="Optional read-only Arrow dataset file; bypasses Hugging Face cache locks.",
    )
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected_ids = load_ids(args.selection)
    if len(selected_ids) != 15:
        raise ValueError(f"expected 15 selected cases, got {len(selected_ids)}")
    flip_classes = load_flip_classes(args.flip_file)
    missing_flip = selected_ids - set(flip_classes)
    if missing_flip:
        raise ValueError(f"missing flip classes: {sorted(missing_flip)}")
    gold = load_gold(args.dataset, args.split, selected_ids, args.dataset_arrow)

    selected_repos = {case_id.rsplit("-", 1)[0].replace("__", "/") for case_id in selected_ids}
    commit_lookup = load_commit_lookup(args.index_dirs, selected_repos)
    if not commit_lookup:
        raise ValueError("no commit index rows loaded")

    roots = {"v2": args.v2_root, "v3": args.v3_root, "v4": args.v4_root}
    version_results = {
        version: load_version(
            root, version, selected_ids, gold, flip_classes, commit_lookup
        )
        for version, root in roots.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for version, result in version_results.items():
        write_csv(output_dir / f"retrieval_items_{version}.csv", result["exposures"])
        write_csv(output_dir / f"examinations_{version}.csv", result["examinations"])
        write_csv(output_dir / f"post_retrieval_tools_{version}.csv", result["tools"])
        write_csv(output_dir / f"retrieval_events_{version}.csv", result["events"])
        write_csv(output_dir / f"trajectory_summary_{version}.csv", result["trajectories"])

    strata_rows = []
    for version, result in version_results.items():
        strata_rows.extend(adoption_strata_rows(result["exposures"], version, "rank"))
        strata_rows.extend(
            adoption_strata_rows(result["exposures"], version, "gold_file_overlap")
        )
    write_csv(output_dir / "adoption_by_rank_and_relevance.csv", strata_rows)

    summary = {
        "analysis": "N2-A V4 Memory Retrieval-Adoption Mechanism Analysis",
        "sample": {
            "case_count": 15,
            "memory_flipped_cases": 10,
            "stable_harm_cases": 5,
            "outcome_selected": True,
            "generalization_target": "diagnostic subset only",
        },
        "rules": {
            "analysis_unit": "trajectory x retrieval event x query x retrieved item",
            "multi_retrieval_attribution": "last-mention wins",
            "final_top_k": 5,
            "exploration_strict": "exact normalized repository-relative file path in later tool arguments",
            "exploration_medium": "same immediate parent directory as a later tool-argument path",
            "exploration_loose": "identifier from exposed examined patch in later tool arguments",
            "bm25_document": "commit subject + full commit message",
            "gold_relevance": "complete indexed changed-files intersects gold patch files",
        },
        "versions": {},
    }
    for version, result in version_results.items():
        summary["versions"][version] = {
            "trajectory_count": len(result["trajectories"]),
            "retrieval_event_count": len(result["events"]),
            "retrieval_exposure_count": len(result["exposures"]),
            "examine_call_count": len(
                {
                    (row["instance_id"], row["repeat"], row["tool_call_id"])
                    for row in result["examinations"]
                }
            ),
            "examination_count": len(result["examinations"]),
            "parse_error_count": len(result["parse_errors"]),
            "parse_errors": result["parse_errors"],
            "funnel": funnel_counts(result["trajectories"]),
            "adoption": adoption_summary(result["trajectories"], n_boot=args.n_boot),
            "item_adoption": item_adoption_summary(result["exposures"]),
            "bm25_alignment": bm25_alignment(
                result["exposures"],
                n_boot=args.n_boot,
                n_perm=args.n_perm,
                primary=version == "v4",
            ),
        }

    funnel_rows = []
    for version, item in summary["versions"].items():
        for category, count in item["funnel"].items():
            funnel_rows.append({"version": version, "category": category, "count": count})
    write_csv(output_dir / "adoption_baseline_distribution.csv", funnel_rows)
    write_csv(
        output_dir / "bm25_alignment.csv",
        [
            {"version": version, **item["bm25_alignment"]}
            for version, item in summary["versions"].items()
        ],
    )
    (output_dir / "n2_adoption_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "verify_report.md").write_text(
        render_verify_report(version_results), encoding="utf-8"
    )
    (output_dir / "n2_adoption_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps(json_ready(summary["versions"]["v4"]), ensure_ascii=False, indent=2))
    print(f"wrote N2-A outputs to {output_dir}")


if __name__ == "__main__":
    main()
