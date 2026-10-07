import unittest

from check_m2_trajectory_divergence import (
    build_case_rows,
    extract_trajectory,
    holm_adjust,
    jaccard_distance,
    levenshtein_distance,
    normalized_edit_distance,
    rank_illustrative_cases,
    relative_length_delta,
    resolve_file,
    tool_count_breakdown,
    wilcoxon_p,
)


def trajectory(tools, files=()):
    return {
        "tool_sequence": list(tools),
        "locagent_tool_sequence": [
            tool for tool in tools if tool not in {"search_commit", "examine_commit"}
        ],
        "tool_counts": {},
        "explored_files": list(files),
        "finish_called": "finish" in tools,
        "last_tool": tools[-1] if tools else None,
        "final_text_present": False,
        "final_text_chars": 0,
        "malformed_arguments": 0,
        "unknown_tools": {},
    }


class M2TrajectoryDivergenceTest(unittest.TestCase):
    def test_levenshtein_and_normalization(self):
        self.assertEqual(levenshtein_distance(["a", "b"], ["a", "c"]), 1)
        self.assertEqual(normalized_edit_distance([], []), 0.0)
        self.assertEqual(normalized_edit_distance(["a", "b"], ["a"]), 0.5)
        self.assertEqual(relative_length_delta([], []), 0.0)
        self.assertAlmostEqual(relative_length_delta([1, 2, 3], [1]), 1.0)

    def test_jaccard_empty_semantics(self):
        self.assertEqual(jaccard_distance([], []), 0.0)
        self.assertEqual(jaccard_distance(["a"], []), 1.0)
        self.assertAlmostEqual(jaccard_distance(["a", "b"], ["b", "c"]), 2 / 3)

    def test_file_resolution_and_memory_exclusion(self):
        valid = {"django/db/models/query.py", "django/db/models/fields.py"}
        self.assertEqual(
            resolve_file("django/db/models/query.py:QuerySet", valid),
            "django/db/models/query.py",
        )
        self.assertIsNone(resolve_file("django/**/*.py", valid))
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "memory",
                        "function": {
                            "name": "search_commit",
                            "arguments": '{"query_list":["query"]}',
                        },
                    },
                    {
                        "id": "code",
                        "function": {
                            "name": "get_entity_contents",
                            "arguments": '{"entity_names":["django/db/models/query.py:QuerySet"]}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "name": "search_commit",
                "tool_call_id": "memory",
                "content": "Edited files: ['django/db/models/fields.py']",
            },
            {
                "role": "tool",
                "name": "get_entity_contents",
                "tool_call_id": "code",
                "content": "Found class `django/db/models/query.py:QuerySet`.",
            },
        ]
        result = extract_trajectory(messages, valid)
        self.assertEqual(result["explored_files"], ["django/db/models/query.py"])
        self.assertEqual(result["tool_sequence"], ["search_commit", "get_entity_contents"])

    def test_holm_is_monotonic_in_rank_order(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])
        self.assertEqual(adjusted, [0.04, 0.09, 0.09, 0.2])

    def test_wilcoxon_all_zero(self):
        pvalue, statistic = wilcoxon_p([0.0, 0.0, 0.0])
        self.assertEqual(pvalue, 1.0)
        self.assertEqual(statistic, 0.0)

    def test_tool_breakdown_is_group_labeled(self):
        row = {
            "original_memory": trajectory(["search_commit", "finish"]),
            "rerun_memory": trajectory(["search_commit", "finish", "finish"]),
        }
        row["original_memory"]["tool_counts"] = {"search_commit": 1, "finish": 1}
        row["rerun_memory"]["tool_counts"] = {"search_commit": 1, "finish": 2}
        result = tool_count_breakdown([row], "memory_flipped_15")
        finish = next(item for item in result if item["tool"] == "finish")
        self.assertEqual(finish["group"], "memory_flipped_15")
        self.assertEqual(finish["total_rerun_minus_original"], 1)

    def test_four_way_case_build(self):
        selected = {"case": {"flip_class": "memory_flipped", "repo": "repo"}}
        original_baseline = {"case": trajectory(["a", "finish"], ["a.py"])}
        rerun_baseline = {"case": trajectory(["a", "finish"], ["a.py"])}
        original_memory = {"case": trajectory(["search_commit", "a", "finish"], ["a.py"])}
        rerun_memory = {"case": trajectory(["search_commit", "b", "finish"], ["b.py"])}
        rows = build_case_rows(
            selected,
            original_baseline,
            rerun_baseline,
            original_memory,
            rerun_memory,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["baseline_divergence"]["tool_seq_edit_distance"], 0.0)
        self.assertGreater(rows[0]["memory_divergence"]["tool_seq_edit_distance"], 0.0)

    def test_illustrative_ranking_is_deterministic(self):
        rows = []
        for case_id, value in (("b", 0.2), ("a", 0.2), ("c", 0.8)):
            rows.append(
                {
                    "case_id": case_id,
                    "flip_class": "memory_flipped",
                    "baseline_divergence": {metric: 0.0 for metric in (
                        "tool_seq_edit_distance", "length_delta", "tool_set_jaccard", "explored_files_jaccard"
                    )},
                    "memory_divergence": {metric: value for metric in (
                        "tool_seq_edit_distance", "length_delta", "tool_set_jaccard", "explored_files_jaccard"
                    )},
                    "divergence_delta": {metric: value for metric in (
                        "tool_seq_edit_distance", "length_delta", "tool_set_jaccard", "explored_files_jaccard"
                    )},
                }
            )
        primary, top = rank_illustrative_cases(rows)
        self.assertEqual(primary["case_id"], "c")
        self.assertEqual([row["case_id"] for row in top], ["c", "a", "b"])


if __name__ == "__main__":
    unittest.main()
