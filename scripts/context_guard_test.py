import unittest
from unittest.mock import patch

from util.runtime.context_guard import (
    TRUNCATION_MARKER,
    guard_observation,
    is_context_error,
    summarize_context_audit,
    truncate_text_to_tokens,
)
from plugins.location_tools.repo_ops.repo_ops import _truncate_tool_lines


class ContextGuardTest(unittest.TestCase):
    def test_short_observation_is_unchanged(self):
        text = "short observation"
        self.assertEqual(truncate_text_to_tokens(text, 100, "gpt-4o"), text)

    def test_truncation_preserves_both_ends(self):
        text = "HEAD " + "middle " * 200 + " TAIL"
        result = truncate_text_to_tokens(text, 30, "gpt-4o")
        self.assertTrue(result.startswith("HEAD"))
        self.assertTrue(result.endswith(" TAIL"))
        self.assertIn(TRUNCATION_MARKER.strip(), result)

    def test_context_remaining_budget_can_be_tighter(self):
        with patch(
            "util.runtime.context_guard.count_context_tokens", return_value=95
        ):
            result = guard_observation(
                "token " * 100,
                model="gpt-4o",
                tool_name="explore_tree_structure",
                iteration=2,
                messages=[],
                tools=None,
                observation_token_cap=20,
                context_token_budget=100,
            )
        self.assertTrue(result.audit["truncated"])
        self.assertIn("context_token_budget", result.audit["reasons"])
        self.assertLessEqual(result.audit["tokens_after"], 5)

    def test_context_error_detection(self):
        self.assertTrue(is_context_error("maximum context length exceeded"))
        self.assertTrue(is_context_error("message is too long"))
        self.assertFalse(is_context_error("invalid API key"))

    def test_audit_summary(self):
        events = [
            {"event": "request", "prompt_tokens": 100},
            {
                "event": "tool_observation",
                "tool_name": "explore_tree_structure",
                "truncated": True,
                "tokens_before": 100,
                "tokens_removed": 75,
            },
        ]
        summary = summarize_context_audit(events)
        self.assertEqual(summary["truncated_observation_count"], 1)
        self.assertEqual(summary["truncated_tools"], ["explore_tree_structure"])
        self.assertEqual(summary["observation_token_removal_rate"], 0.75)

    def test_tool_line_bound_is_explicit(self):
        result = _truncate_tool_lines("a\nb\nc\nd", 2, "test output")
        self.assertTrue(result.startswith("a\nb\n"))
        self.assertIn("truncated after 2 of 4 lines", result)


if __name__ == "__main__":
    unittest.main()
