import unittest

from scripts.analyze_context_guard_pilot import decide


class ContextGuardPilotTest(unittest.TestCase):
    def test_sparse_graph_truncation_selects_guard_a(self):
        result = decide(
            {
                "truncated_observation_count": 1,
                "observation_token_removal_rate": 0.1,
                "truncated_tools": ["explore_tree_structure"],
            },
            [],
        )
        self.assertEqual(result["decision"], "GUARD_A")

    def test_memory_truncation_selects_tool_c(self):
        result = decide(
            {
                "truncated_observation_count": 1,
                "observation_token_removal_rate": 0.1,
                "truncated_tools": ["examine_commit"],
            },
            [],
        )
        self.assertEqual(result["decision"], "TOOL_C")

    def test_frequent_truncation_selects_tool_c(self):
        result = decide(
            {
                "truncated_observation_count": 3,
                "observation_token_removal_rate": 0.1,
                "truncated_tools": ["explore_tree_structure"],
            },
            [],
        )
        self.assertEqual(result["decision"], "TOOL_C")

    def test_clean_bounded_pilot_validates_tool_c(self):
        result = decide(
            {
                "truncated_observation_count": 0,
                "observation_token_removal_rate": 0.0,
                "truncated_tools": [],
            },
            [],
            bounded_tool_output=True,
        )
        self.assertEqual(result["decision"], "TOOL_C_VALIDATED")


if __name__ == "__main__":
    unittest.main()
