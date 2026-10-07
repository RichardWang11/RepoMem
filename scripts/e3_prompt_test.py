import unittest

from auto_search_main import EPISODIC_TASK_PROMPT, get_task_instruction
from scripts.analyze_e3_prompt import behavior_metrics
from scripts.prepare_e3_prompt import build_cells


def call(call_id, name):
    return {
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "function": {"name": name, "arguments": "{}"},
        }],
    }


class E3PromptTest(unittest.TestCase):
    def test_manifest_has_two_seeds_and_counterbalanced_arm_order(self):
        cells = build_cells()
        self.assertEqual(len(cells), 40)
        self.assertEqual(
            sum(cell["arm"] == "control" for cell in cells), 20
        )
        self.assertEqual(
            sum(cell["arm"] == "treatment" for cell in cells), 20
        )
        pairs = {}
        for cell in cells:
            pairs.setdefault((cell["instance_id"], cell["seed"]), []).append(
                cell["arm"]
            )
        self.assertEqual(set(map(tuple, pairs.values())), {
            ("control", "treatment"),
            ("treatment", "control"),
        })

    def test_prompt_arms_differ_by_exact_memory_suffix(self):
        instance = {
            "instance_id": "owner__repo-1",
            "problem_statement": "Title\nBody",
        }
        original = get_task_instruction(
            instance, include_pr=True, include_hint=True, repomem_mode=None
        )
        control = get_task_instruction(
            instance, include_pr=True, include_hint=True,
            repomem_mode="episodic",
        )
        self.assertEqual(control, original + EPISODIC_TASK_PROMPT)

    def test_memory_first_precedes_current_code_exploration(self):
        metrics = behavior_metrics([
            call("one", "search_commit"),
            call("two", "get_entity_contents"),
        ])
        self.assertTrue(metrics["memory_first"])
        reversed_metrics = behavior_metrics([
            call("one", "search_code_snippets"),
            call("two", "search_commit"),
        ])
        self.assertFalse(reversed_metrics["memory_first"])

    def test_no_search_is_not_memory_first(self):
        metrics = behavior_metrics([call("one", "get_entity_contents")])
        self.assertFalse(metrics["memory_first"])
        self.assertFalse(metrics["search_commit_ever"])

    def test_successful_examine_requires_commit_observation(self):
        messages = [
            call("examine", "examine_commit"),
            {
                "role": "tool",
                "name": "examine_commit",
                "tool_call_id": "examine",
                "content": "OBSERVATION:\nCommit `bad` was not found",
            },
        ]
        self.assertFalse(
            behavior_metrics(messages)["successful_examine_adoption"]
        )
        messages[-1]["content"] = "OBSERVATION:\n##Commit for sha `abc`..."
        self.assertTrue(
            behavior_metrics(messages)["successful_examine_adoption"]
        )


if __name__ == "__main__":
    unittest.main()
