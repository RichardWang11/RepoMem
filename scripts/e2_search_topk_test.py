import os
import unittest
from unittest import mock

from plugins.location_tools.repo_ops import repo_ops
from util.runtime.e2_search_topk import (
    adoption_outcome,
    assert_topk_prefix,
    find_first_topk_differential,
)


def search_observation(shas):
    lines = ["##Searching for query `widget`...", "### Search Result:"]
    lines.extend(
        f"SHA: {sha} | Commit message: message {sha} | Edited files: [] | score: 1.000"
        for sha in shas
    )
    return "\n".join(lines)


class E2SearchTopKTest(unittest.TestCase):
    def test_forced_topk_is_validated(self):
        with mock.patch.dict(
            os.environ, {"REPOMEM_SEARCH_COMMIT_FORCE_TOPK": "0"}
        ):
            with self.assertRaises(ValueError):
                repo_ops._repomem_forced_search_commit_topk()

    def test_top5_must_be_exact_top20_prefix(self):
        top5 = search_observation([str(index) for index in range(5)])
        top20 = search_observation([str(index) for index in range(20)])
        audit = assert_topk_prefix(top5, top20)
        self.assertTrue(audit["ranking_prefix_invariant_passed"])
        self.assertEqual(audit["control_result_counts"], [5])
        self.assertEqual(audit["treatment_result_counts"], [20])
        bad = search_observation(["x"] + [str(index) for index in range(19)])
        with self.assertRaises(AssertionError):
            assert_topk_prefix(top5, bad)

    def test_lazy_fork_uses_same_search_query(self):
        top5 = search_observation([str(index) for index in range(5)])
        top20 = search_observation([str(index) for index in range(20)])
        messages = [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "search-1",
                    "function": {
                        "name": "search_commit",
                        "arguments": '{"query_list":["widget"],"top_k":5}',
                    },
                }],
            },
            {
                "role": "tool",
                "name": "search_commit",
                "tool_call_id": "search-1",
                "content": "OBSERVATION:\n" + top5,
            },
        ]

        def execute(name, arguments, arm):
            self.assertEqual(name, "search_commit")
            self.assertEqual(arguments["query_list"], ["widget"])
            return top5 if arm == "5" else top20

        with mock.patch(
            "util.runtime.e2_search_topk.execute_search_arm",
            side_effect=lambda *args, **kwargs: execute(args[1], args[2], str(args[3])),
        ):
            differential = find_first_topk_differential(messages, object())
        self.assertEqual(differential["arguments"]["top_k"], 5)
        self.assertTrue(
            differential["prefix_audit"]["ranking_prefix_invariant_passed"]
        )

    def test_lazy_fork_accepts_ipython_print_transport_newline(self):
        top5 = search_observation([str(index) for index in range(5)])
        top20 = search_observation([str(index) for index in range(20)])
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "search-printed",
                    "function": {
                        "name": "search_commit",
                        "arguments": '{"query_list":["widget"]}',
                    },
                }],
            },
            {
                "role": "tool",
                "name": "search_commit",
                "tool_call_id": "search-printed",
                "content": "OBSERVATION:\n" + top5 + "\n",
            },
        ]
        with mock.patch(
            "util.runtime.e2_search_topk.execute_search_arm",
            side_effect=lambda _repo, _name, _arguments, arm: (
                top5 if arm == 5 else top20
            ),
        ):
            differential = find_first_topk_differential(messages, object())
        self.assertTrue(
            differential["treatment_observation"].endswith("\n")
        )

    def test_adoption_change_is_itt_and_not_exposure_itself(self):
        differential = {"message_index": 2}
        base = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "search"},
            {"role": "tool", "content": "different"},
        ]
        same_action = {
            "role": "assistant",
            "tool_calls": [{
                "id": "examine-1",
                "function": {
                    "name": "examine_commit",
                    "arguments": '{"sha_list":["ABC123"]}',
                },
            }],
        }
        unchanged = adoption_outcome(
            base + [same_action],
            base + [same_action],
            differential,
            visible_shas=["abc123" + "0" * 34],
        )
        self.assertTrue(unchanged["differential_exposure"])
        self.assertFalse(unchanged["memory_adoption_changed"])
        no_exposure = adoption_outcome(base, None, None)
        self.assertFalse(no_exposure["memory_adoption_changed"])


if __name__ == "__main__":
    unittest.main()
