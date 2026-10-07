"""Frozen-prefix utilities for the E2 SearchCommit Top5/Top20 experiment."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from util.runtime.e1_supplemental_leakage import (
    find_first_differential_observation,
    treatment_prefix,
)


CONTROL_TOP_K = 5
TREATMENT_TOP_K = 20


def parse_search_result_groups(observation: str) -> list[list[str]]:
    """Extract ordered SearchCommit result lines for each query section."""
    groups: list[list[str]] = []
    current: list[str] | None = None
    for line in str(observation).splitlines():
        if line.startswith("##Searching for query "):
            current = []
            groups.append(current)
        elif line.startswith("SHA: "):
            if current is None:
                raise AssertionError("SearchCommit result appeared before query header")
            current.append(line)
    return groups


def assert_topk_prefix(control: str, treatment: str) -> dict[str, Any]:
    """Prove every Top5 result group is the prefix of its Top20 group."""
    control_groups = parse_search_result_groups(control)
    treatment_groups = parse_search_result_groups(treatment)
    if len(control_groups) != len(treatment_groups):
        raise AssertionError("E2 query-group count differs between arms")
    for index, (top5, top20) in enumerate(
        zip(control_groups, treatment_groups, strict=True)
    ):
        if len(top5) > CONTROL_TOP_K:
            raise AssertionError(f"E2 control query {index} returned more than Top5")
        if len(top20) > TREATMENT_TOP_K:
            raise AssertionError(f"E2 treatment query {index} returned more than Top20")
        if top5 != top20[:len(top5)]:
            raise AssertionError(
                f"E2 ranking-prefix invariant failed for query {index}"
            )
    return {
        "query_count": len(control_groups),
        "control_result_counts": [len(group) for group in control_groups],
        "treatment_result_counts": [len(group) for group in treatment_groups],
        "ranking_prefix_invariant_passed": True,
    }


def execute_search_arm(
    repo_ops,
    name: str,
    arguments: dict[str, Any],
    arm: str,
) -> str:
    """Replay an episodic call with SearchCommit depth assigned by E2 arm."""
    if name not in {"search_commit", "examine_commit"}:
        raise ValueError(f"unsupported E2 memory tool: {name}")
    old_top_k = os.environ.get("REPOMEM_SEARCH_COMMIT_FORCE_TOPK")
    try:
        os.environ["REPOMEM_SEARCH_COMMIT_FORCE_TOPK"] = str(int(arm))
        function = {
            "search_commit": repo_ops.search_commit,
            "examine_commit": repo_ops.examine_commit,
        }[name]
        result = function(**arguments)
        return result if isinstance(result, str) else str(result)
    finally:
        if old_top_k is None:
            os.environ.pop("REPOMEM_SEARCH_COMMIT_FORCE_TOPK", None)
        else:
            os.environ["REPOMEM_SEARCH_COMMIT_FORCE_TOPK"] = old_top_k


def find_first_topk_differential(
    messages: list[dict[str, Any]], repo_ops
) -> dict[str, Any] | None:
    """Find the first Top5/Top20 SearchCommit difference on a shared run."""
    differential = find_first_differential_observation(
        messages,
        lambda name, arguments, arm: execute_search_arm(
            repo_ops,
            name,
            arguments,
            CONTROL_TOP_K if arm == "off" else TREATMENT_TOP_K,
        ),
    )
    if differential is None:
        return None
    if differential["tool_name"] != "search_commit":
        raise AssertionError(
            "E2 arms may first differ only at a SearchCommit observation"
        )
    differential["prefix_audit"] = assert_topk_prefix(
        differential["control_observation"],
        differential["treatment_observation"],
    )
    return differential


def _examine_sequence(
    messages: list[dict[str, Any]],
    after_index: int,
    visible_shas: list[str],
) -> list[str]:
    sequence: list[str] = []
    for message in messages[after_index + 1:]:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "examine_commit":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                sequence.append("<invalid-json>")
                continue
            values = arguments.get("sha_list", arguments.get("sha", []))
            if isinstance(values, str):
                values = [values]
            for value in values or []:
                wanted = str(value).strip().lower()
                matches = [sha for sha in visible_shas if sha.startswith(wanted)]
                if len(matches) == 1:
                    sequence.append(matches[0])
                elif not matches:
                    sequence.append(f"<not-visible>:{wanted}")
                else:
                    sequence.append(f"<ambiguous>:{wanted}")
    return sequence


def adoption_outcome(
    control_messages: list[dict[str, Any]],
    treatment_messages: list[dict[str, Any]] | None,
    differential: dict[str, Any] | None,
    visible_shas: list[str] | None = None,
) -> dict[str, Any]:
    """Compute E2's ITT Memory-Adoption Change endpoint."""
    if differential is None:
        return {
            "differential_exposure": False,
            "control_examine_sequence": [],
            "treatment_examine_sequence": [],
            "memory_adoption_changed": False,
        }
    index = int(differential["message_index"])
    visible_shas = visible_shas or []
    control = _examine_sequence(control_messages, index, visible_shas)
    treatment = _examine_sequence(
        treatment_messages or [], index, visible_shas
    )
    return {
        "differential_exposure": True,
        "control_examine_sequence": control,
        "treatment_examine_sequence": treatment,
        "memory_adoption_changed": control != treatment,
    }


def corpus_audit(repo_ops, memory_data) -> dict[str, Any]:
    """Freeze the exact leakage-filtered strict corpus shared by both arms."""
    rows = list(repo_ops._iter_visible_commits(memory_data))
    shas = [row["sha"] for row in rows]
    payload = "\n".join(shas).encode("ascii")
    return {
        "visible_count": len(shas),
        "visible_sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_base_sha": repo_ops.get_repomem_visibility_audit().get(
            "canonical_base_sha"
        ),
        "base_not_visible": (
            repo_ops.get_repomem_visibility_audit().get("canonical_base_sha")
            not in set(shas)
        ),
        "_visible_shas": shas,
    }


__all__ = [
    "CONTROL_TOP_K",
    "TREATMENT_TOP_K",
    "adoption_outcome",
    "assert_topk_prefix",
    "corpus_audit",
    "execute_search_arm",
    "find_first_topk_differential",
    "parse_search_result_groups",
    "treatment_prefix",
]
