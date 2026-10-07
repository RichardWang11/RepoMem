"""Context-budget helpers for reproducible agent experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import litellm


TRUNCATION_MARKER = "\n\n[... observation truncated by context guard ...]\n\n"
CONTEXT_ERROR_PATTERNS = (
    "context_length_exceeded",
    "context window",
    "context length",
    "maximum context",
    "maximum context length",
    "too many tokens",
    "token limit",
    "message is too long",
    "maximum size",
    "exceeds the limit",
)


def count_text_tokens(text: str, model: str) -> int:
    """Count tokens with LiteLLM's model-aware tokenizer."""
    return len(litellm.encode(model=model, text=text or ""))


def count_context_tokens(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[Any] | None = None,
) -> int:
    """Count prompt tokens, including function schemas when supported."""
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    return int(litellm.token_counter(**kwargs))


def truncate_text_to_tokens(text: str, token_cap: int, model: str) -> str:
    """Keep both ends of text under a token cap and mark the omission."""
    if token_cap <= 0:
        return ""
    tokens = litellm.encode(model=model, text=text or "")
    if len(tokens) <= token_cap:
        return text

    marker_tokens = litellm.encode(model=model, text=TRUNCATION_MARKER)
    if token_cap <= len(marker_tokens):
        return litellm.decode(model=model, tokens=marker_tokens[:token_cap])
    content_cap = max(0, token_cap - len(marker_tokens))
    head_count = (content_cap + 1) // 2
    tail_count = content_cap // 2
    kept = tokens[:head_count]
    if tail_count:
        kept = kept + marker_tokens + tokens[-tail_count:]
    else:
        kept = kept + marker_tokens
    return litellm.decode(model=model, tokens=kept)


def is_context_error(error: BaseException | str) -> bool:
    """Recognize provider context or per-message limit failures."""
    message = str(error).lower()
    return any(pattern in message for pattern in CONTEXT_ERROR_PATTERNS)


@dataclass
class ObservationGuardResult:
    text: str
    audit: dict[str, Any]


def guard_observation(
    observation: str,
    *,
    model: str,
    tool_name: str | None,
    iteration: int,
    messages: list[dict[str, Any]],
    tools: list[Any] | None,
    observation_token_cap: int,
    context_token_budget: int,
) -> ObservationGuardResult:
    """Apply per-observation and remaining-context limits."""
    before_tokens = count_text_tokens(observation, model)
    prompt_tokens_before = count_context_tokens(messages, model, tools)
    caps = []
    reasons = []
    if observation_token_cap > 0:
        caps.append(observation_token_cap)
        if before_tokens > observation_token_cap:
            reasons.append("tool_observation_token_cap")
    if context_token_budget > 0:
        remaining = max(0, context_token_budget - prompt_tokens_before)
        caps.append(remaining)
        if before_tokens > remaining:
            reasons.append("context_token_budget")

    effective_cap = min(caps) if caps else before_tokens
    guarded = truncate_text_to_tokens(observation, effective_cap, model)
    after_tokens = count_text_tokens(guarded, model)
    audit = {
        "event": "tool_observation",
        "iteration": iteration,
        "tool_name": tool_name,
        "truncated": guarded != observation,
        "reasons": reasons,
        "chars_before": len(observation),
        "chars_after": len(guarded),
        "tokens_before": before_tokens,
        "tokens_after": after_tokens,
        "tokens_removed": max(0, before_tokens - after_tokens),
        "prompt_tokens_before": prompt_tokens_before,
        "context_token_budget": context_token_budget,
        "observation_token_cap": observation_token_cap,
    }
    return ObservationGuardResult(text=guarded, audit=audit)


def summarize_context_audit(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize request and observation events for one trajectory."""
    observations = [event for event in events if event.get("event") == "tool_observation"]
    requests = [event for event in events if event.get("event") == "request"]
    truncated = [event for event in observations if event.get("truncated")]
    tokens_before = sum(event.get("tokens_before", 0) for event in observations)
    tokens_removed = sum(event.get("tokens_removed", 0) for event in observations)
    return {
        "request_count": len(requests),
        "observation_count": len(observations),
        "truncated_observation_count": len(truncated),
        "truncated_tools": sorted(
            {event.get("tool_name") for event in truncated if event.get("tool_name")}
        ),
        "observation_tokens_before": tokens_before,
        "observation_tokens_removed": tokens_removed,
        "observation_token_removal_rate": (
            tokens_removed / tokens_before if tokens_before else 0.0
        ),
        "max_prompt_tokens": max(
            (event.get("prompt_tokens", 0) for event in requests), default=0
        ),
    }
