"""Frozen-prefix utilities for the E1 supplemental-leakage experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Callable


MEMORY_TOOLS = {"search_commit", "examine_commit"}


def canonical_action(message: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return E1's tool+arguments or finish+files action signature."""
    if not message or message.get("role") != "assistant":
        return None
    calls = message.get("tool_calls") or []
    if calls:
        call = calls[0]
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": function.get("arguments")}
        name = function.get("name")
        if name == "finish":
            value = next(iter(arguments.values()), "")
            return {"finish": normalize_predicted_files(value)}
        return {"tool": name, "arguments": _canonical_json(arguments)}
    content = message.get("content") or ""
    if "<finish>" in content.lower():
        return {"finish": normalize_predicted_files(content)}
    return {"message": " ".join(str(content).split())}


def _canonical_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def normalize_predicted_files(value: Any) -> list[str]:
    """Extract a stable ordered file list from a finish payload."""
    text = str(value or "").replace("\\", "/")
    candidates = []
    for token in text.replace("`", " ").replace(",", " ").split():
        token = token.strip("[](){}:'\";.")
        if token.endswith(".py") and "/" in token:
            candidates.append(token.lstrip("./"))
    return sorted(set(candidates))


def _tool_call_by_id(messages: list[dict[str, Any]], index: int) -> dict[str, Any]:
    tool_id = messages[index].get("tool_call_id")
    for prior in reversed(messages[:index]):
        if prior.get("role") != "assistant":
            continue
        calls = prior.get("tool_calls") or []
        for call in calls:
            if call.get("id") == tool_id:
                if len(calls) != 1:
                    raise ValueError(
                        "E1 requires one tool call per assistant decision"
                    )
                return call
    raise ValueError(f"no assistant tool call found for tool_call_id={tool_id!r}")


def find_first_differential_observation(
    messages: list[dict[str, Any]],
    execute: Callable[[str, dict[str, Any], str], str],
) -> dict[str, Any] | None:
    """Replay memory calls against both arms and locate the first difference."""
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or message.get("name") not in MEMORY_TOOLS:
            continue
        call = _tool_call_by_id(messages, index)
        function = call.get("function") or {}
        arguments = json.loads(function.get("arguments") or "{}")
        name = function.get("name")
        control_raw = execute(name, arguments, "off")
        treatment_raw = execute(name, arguments, "on")
        recorded = message.get("content") or ""
        expected_raw = "OBSERVATION:\n" + control_raw
        expected_printed = expected_raw + "\n"
        if recorded == expected_raw:
            control = control_raw
            treatment = treatment_raw
        elif recorded == expected_printed:
            # Function-calling execution currently goes through
            # ``print(tool(...))`` in IPython, which contributes one transport
            # newline not present in the direct read-only replay result.
            control = control_raw + "\n"
            treatment = treatment_raw + "\n"
        else:
            raise AssertionError(
                "recorded shared observation does not match the replayed "
                "control arm (including the optional print transport newline)"
            )
        if control != treatment:
            return {
                "message_index": index,
                "tool_call_id": call.get("id"),
                "tool_name": name,
                "arguments": arguments,
                "control_observation": control,
                "treatment_observation": treatment,
                "control_sha256": hashlib.sha256(control.encode()).hexdigest(),
                "treatment_sha256": hashlib.sha256(treatment.encode()).hexdigest(),
            }
    return None


def treatment_prefix(
    messages: list[dict[str, Any]], differential: dict[str, Any]
) -> list[dict[str, Any]]:
    """Freeze the shared prefix and replace only its differential observation."""
    stop = int(differential["message_index"]) + 1
    prefix = copy.deepcopy(messages[:stop])
    prefix[-1]["content"] = (
        "OBSERVATION:\n" + differential["treatment_observation"]
    )
    return prefix


def first_assistant_action_after(
    messages: list[dict[str, Any]], message_index: int
) -> dict[str, Any] | None:
    for message in messages[message_index + 1:]:
        signature = canonical_action(message)
        if signature is not None:
            return signature
    return None


def exposure_outcome(
    control_messages: list[dict[str, Any]],
    treatment_messages: list[dict[str, Any]] | None,
    differential: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute E1 exposure and the ITT next-action primary endpoint."""
    if differential is None:
        return {
            "differential_exposure": False,
            "control_next_action": None,
            "treatment_next_action": None,
            "treatment_induced_next_action_change": False,
        }
    index = int(differential["message_index"])
    control_action = first_assistant_action_after(control_messages, index)
    treatment_action = first_assistant_action_after(
        treatment_messages or [], index
    )
    return {
        "differential_exposure": True,
        "control_next_action": control_action,
        "treatment_next_action": treatment_action,
        "treatment_induced_next_action_change": control_action != treatment_action,
    }


def execute_memory_arm(repo_ops, name: str, arguments: dict[str, Any], arm: str) -> str:
    """Execute one read-only episodic tool call under an explicit E1 arm."""
    old = os.environ.get("REPOMEM_SUPPLEMENTAL_LEAKAGE")
    try:
        os.environ["REPOMEM_SUPPLEMENTAL_LEAKAGE"] = arm
        function = {
            "search_commit": repo_ops.search_commit,
            "examine_commit": repo_ops.examine_commit,
        }[name]
        result = function(**arguments)
        return result if isinstance(result, str) else str(result)
    finally:
        if old is None:
            os.environ.pop("REPOMEM_SUPPLEMENTAL_LEAKAGE", None)
        else:
            os.environ["REPOMEM_SUPPLEMENTAL_LEAKAGE"] = old
