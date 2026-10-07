"""Small, deterministic Entity Grounding helpers for Common93 experiments."""

import hashlib
import json
import random
import re
import ast
from collections import Counter, defaultdict


CALLABLE_TYPES = {"function", "class"}


def entity_name(entity_id):
    """Return the terminal callable name from a RepoMem graph node id."""
    return entity_id.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


def entity_file(entity_id):
    return entity_id.split(":", 1)[0]


def entity_type(graph, entity_id):
    return graph.nodes[entity_id].get("type", "")


def ambiguous_callable_index(graph):
    """Only retain callable names with more than one graph location."""
    index = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "function":
            index[entity_name(node_id)].append(node_id)
    return {name: ids for name, ids in index.items() if len(ids) > 1}


def raw_identifiers(problem_statement):
    return sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", problem_statement)))


def _score(problem_statement, graph, entity_id):
    terms = set(raw_identifiers(problem_statement.lower()))
    text = " ".join((entity_id, str(graph.nodes[entity_id].get("code", "")))).lower()
    return sum(term in text for term in terms)


def _rank(problem_statement, graph, candidates, limit):
    return sorted(candidates, key=lambda node: (-_score(problem_statement, graph, node), node))[:limit]


def build_grounding(graph, instance_id, problem_statement, mode="real", ablation="full"):
    """Build bounded graph roots without reading the gold patch or issuing model calls."""
    if mode == "off":
        return {"instance_id": instance_id, "selected_entities": [], "initial_actions": [],
                "ambiguity_events": [], "decomposed_actions": [], "grounded_queries": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0, "tool_calls": 0},
                "model_calls": []}
    ids = raw_identifiers(problem_statement)
    functions = [node for node, data in graph.nodes(data=True) if data.get("type") == "function"]
    classes = [node for node, data in graph.nodes(data=True) if data.get("type") == "class"]
    files = [node for node, data in graph.nodes(data=True) if data.get("type") == "file"]
    by_name = defaultdict(list)
    for node in functions:
        by_name[entity_name(node)].append(node)
    ambiguous = ambiguous_callable_index(graph)
    selected, initial, ambiguity, decomposed = [], [], [], []
    disambiguate = ablation != "no_disambiguation"
    expand = ablation != "no_file_class"
    for name in ids:
        matches = by_name.get(name, [])
        if not matches:
            matches = [node for node in classes + files if entity_name(node) == name or node == name]
        if not matches:
            continue
        initial.append({"query": name, "candidates": matches})
        if name in ambiguous:
            ambiguity.append({"query": name, "candidates": matches})
            if not disambiguate:
                continue
            matches = _rank(problem_statement, graph, matches, 3)
            decomposed.extend({"kind": "disambiguation", "entity": node} for node in matches)
        selected.extend(matches if name in ambiguous else _rank(problem_statement, graph, matches, 1))
    if expand:
        for root in list(selected):
            typ = entity_type(graph, root)
            if typ not in CALLABLE_TYPES and typ != "file":
                continue
            limit = 3 if typ == "class" else 2
            children = [dst for _, dst, data in graph.out_edges(root, data=True)
                        if data.get("type") == "contains" and entity_type(graph, dst) in {"function", "class"}]
            for child in _rank(problem_statement, graph, children, limit):
                decomposed.append({"kind": typ, "entity": child})
                selected.append(child)
    selected = list(dict.fromkeys(selected))[:5]
    if mode == "placebo":
        selected = placebo_entities(graph, instance_id, selected, ids)
    queries = [" ".join([problem_statement.splitlines()[0], entity_file(node), node.rsplit(":", 1)[-1]]) for node in selected]
    return {"instance_id": instance_id, "selected_entities": selected, "initial_actions": initial,
            "ambiguity_events": ambiguity, "decomposed_actions": decomposed, "grounded_queries": queries,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0, "tool_calls": 0},
            "model_calls": []}


def placebo_entities(graph, instance_id, real_entities, identifiers):
    """Sample same-type same-repo roots deterministically; never inspect the gold patch."""
    seed = int(hashlib.sha256((instance_id + json.dumps(real_entities) + "placebo-v1").encode()).hexdigest(), 16)
    rng = random.Random(seed)
    blocked = {term.lower() for term in identifiers}
    excluded = set(real_entities)
    frontier = set(real_entities)
    for _ in range(2):
        frontier = {neighbor for node in frontier for neighbor in (set(graph.predecessors(node)) | set(graph.successors(node)))} - excluded
        excluded.update(frontier)
    roots = []
    for real in real_entities:
        typ = entity_type(graph, real)
        options = [node for node, data in graph.nodes(data=True)
                   if data.get("type") == typ and node not in excluded
                   and not (set(raw_identifiers(node.lower())) & blocked)]
        if not options:
            raise ValueError(f"No placebo candidate for {real}")
        candidate = rng.choice(sorted(options))
        while candidate in roots and len(options) > len(roots):
            candidate = rng.choice(sorted(options))
        roots.append(candidate)
    return roots


def grounding_prompt(memory):
    roots = memory["selected_entities"]
    if not roots:
        return ""
    return "\nRecommended CodeGraph roots (verify before use):\n" + "\n".join(f"- {root}" for root in roots)


def replay_telemetry(messages, memory, retrieved_commits=()):
    """Extract pre-outcome channel usage from saved function-call messages."""
    roots, contents, search_queries = [], [], []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", {})
            name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
            raw = function.get("arguments", "{}") if isinstance(function, dict) else getattr(function, "arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                continue
            if name == "search_commit":
                search_queries.extend(args.get("query_list", []))
            elif name == "explore_tree_structure":
                roots.extend(args.get("start_entities", []))
            elif name == "get_entity_contents":
                contents.extend(args.get("entity_names", []))
    selected = set(memory["selected_entities"])
    ancestors = {entity_file(node) for node in selected}
    loc_active = bool(selected & (set(roots) | set(contents)) or ancestors & (set(roots) | set(contents)))
    retrieved = list(retrieved_commits)
    file_hit = any(ancestors & set(commit.get("changed_files", [])) for commit in retrieved)
    symbol_hit = any(any(entity_name(node) in commit.get("patch", "") for node in selected) for commit in retrieved)
    retrieval_active = file_hit or symbol_hit
    state = "both" if retrieval_active and loc_active else "retrieval_only" if retrieval_active else "loc_only" if loc_active else "neither"
    return {"retrieval_channel": {"query_entities": memory["selected_entities"], "retrieved_commits": retrieved,
                                    "entity_file_in_retrieved_changed_files": file_hit,
                                    "entity_symbol_in_retrieved_patch": symbol_hit, "active": retrieval_active},
            "localization_channel": {"injected_roots": memory["selected_entities"], "graph_root_calls": roots,
                                       "entity_content_calls": contents, "active": loc_active},
            "channel_state": state, "search_queries": search_queries}


def parse_search_report(report):
    """Keep SearchCommit's ranking replayable without changing its text API."""
    rows = []
    for sha, files in re.findall(r"SHA: ([0-9a-f]+) .*?\| Edited files: (\[.*?\]) \| score:", report):
        try:
            changed_files = ast.literal_eval(files)
        except (SyntaxError, ValueError):
            changed_files = []
        rows.append({"short_sha": sha, "changed_files": changed_files, "patch": ""})
    return rows
