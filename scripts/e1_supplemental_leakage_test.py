import os
import unittest
from unittest import mock

from plugins.location_tools.repo_ops import repo_ops
from util.runtime.e1_supplemental_leakage import (
    exposure_outcome,
    find_first_differential_observation,
    treatment_prefix,
)


class E1SupplementalLeakageTest(unittest.TestCase):
    def setUp(self):
        self.instance = {
            "instance_id": "owner__repo-1",
            "base_commit": "b" * 40,
            "problem_statement": "Regression described by #123",
        }
        self.original = "1" * 40
        self.supplemental = "2" * 40
        self.safe = "3" * 40
        self.memory = {
            "commits": [
                {"sha": self.original, "issue_refs": ["123"]},
            ],
            "visibility_commits": [
                {"sha": self.supplemental, "issue_refs": ["123"]},
                {"sha": self.safe, "issue_refs": []},
            ],
            "original_shas": {self.original},
            "supplemental_shas": {self.supplemental, self.safe},
            "issues": {
                "123": {
                    "issue_number": "123",
                    "issue_summary": "old issue",
                    "issue_description": "old description",
                }
            },
            "_leakage_cache": {},
        }

    def test_production_propagates_but_control_preserves_old_scope(self):
        with mock.patch.dict(
            os.environ, {"REPOMEM_SUPPLEMENTAL_LEAKAGE": "off"}
        ):
            control, _ = repo_ops._leakage_exclusions_for_instance(
                self.memory, self.instance
            )
        with mock.patch.dict(
            os.environ, {"REPOMEM_SUPPLEMENTAL_LEAKAGE": "on"}
        ):
            treatment, _ = repo_ops._leakage_exclusions_for_instance(
                self.memory, self.instance
            )
        self.assertEqual(control, {self.original})
        self.assertEqual(treatment, {self.original, self.supplemental})

    def test_static_case_uses_exact_delta_and_ordered_filter(self):
        rows = [
            {"sha": self.supplemental},
            {"sha": self.safe},
            {"sha": self.original},
        ]
        audit = {"canonical_base_sha": self.instance["base_commit"]}
        with mock.patch.object(
            repo_ops, "_strict_visible_commits", return_value=(rows, audit)
        ):
            case = repo_ops.build_e1_supplemental_leakage_case(
                self.memory, self.instance
            )
        self.assertTrue(case["affected"])
        self.assertEqual(case["delta_shas"], [self.supplemental])
        self.assertEqual(case["control_visible_count"], 2)
        self.assertEqual(case["treatment_visible_count"], 1)
        self.assertTrue(case["core_invariants_passed"])

    def test_lazy_fork_counts_downstream_action_not_mechanical_denial(self):
        messages = [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-1",
                    "function": {
                        "name": "examine_commit",
                        "arguments": '{"sha_list":["222"]}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "examine_commit",
                "content": "OBSERVATION:\npatch",
            },
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-2",
                    "function": {
                        "name": "search_code_snippets",
                        "arguments": '{"query":"Widget"}',
                    },
                }],
            },
        ]

        def execute(name, arguments, arm):
            self.assertEqual(name, "examine_commit")
            return "patch" if arm == "off" else "Commit `222` was not found"

        differential = find_first_differential_observation(messages, execute)
        prefix = treatment_prefix(messages, differential)
        treatment = prefix + [messages[-1]]
        outcome = exposure_outcome(messages, treatment, differential)
        self.assertTrue(outcome["differential_exposure"])
        self.assertFalse(outcome["treatment_induced_next_action_change"])

    def test_no_exposure_is_zero_in_affected_itt_denominator(self):
        outcome = exposure_outcome([], None, None)
        self.assertFalse(outcome["differential_exposure"])
        self.assertFalse(outcome["treatment_induced_next_action_change"])


if __name__ == "__main__":
    unittest.main()
