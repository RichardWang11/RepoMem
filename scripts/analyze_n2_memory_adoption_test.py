import unittest

from scripts.analyze_n2_memory_adoption import (
    alignment_label,
    analyze_trajectory_row,
    extract_argument_features,
    extract_patch_symbols,
    funnel_counts,
    parse_examine_observation,
    parse_search_observation,
    same_parent,
)


def assistant_call(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def tool_observation(name, content, call_id):
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": call_id,
        "content": content,
    }


class N2MemoryAdoptionTest(unittest.TestCase):
    def test_search_and_examine_parsers_preserve_item_content(self):
        search = """
##Searching for query `fix widget`...
### Search Result:
SHA: abcdef123 | Commit message: Fix widget lookup | Edited files: ['pkg/widget.py'] | score: 7.250
SHA: 123456789 | Commit message: Add regression test | Edited files: ['tests/test_widget.py'] | score: 5.000
"""
        sections = parse_search_observation(search)
        self.assertEqual(sections[0]["query"], "fix widget")
        self.assertEqual(sections[0]["items"][0]["sha"], "abcdef123")
        self.assertEqual(sections[0]["items"][0]["rank"], 1)
        self.assertEqual(sections[0]["items"][0]["score"], 7.25)

        examined = parse_examine_observation(
            """##Commit for sha `abcdef123`...

Commit message: Fix widget lookup
Edited files: ['pkg/widget.py']
Patch:
diff --git a/pkg/widget.py b/pkg/widget.py
@@ -1 +1 @@ def locate_widget(self):
+class WidgetLocator:
...[patch truncated]...
"""
        )
        self.assertEqual(examined[0]["files"], ["pkg/widget.py"])
        self.assertTrue(examined[0]["patch_present"])
        self.assertTrue(examined[0]["patch_truncated"])
        self.assertIn("WidgetLocator", examined[0]["patch_symbols"])
        self.assertIn("locate_widget", examined[0]["patch_symbols"])

    def test_argument_features_and_exploration_rules(self):
        paths, identifiers = extract_argument_features(
            {
                "entity_names": ["pkg/widget.py:WidgetLocator.locate_widget"],
                "note": "check WidgetLocator",
            }
        )
        self.assertEqual(paths, ["pkg/widget.py"])
        self.assertIn("WidgetLocator", identifiers)
        self.assertTrue(same_parent("pkg/other.py", "pkg/widget.py"))
        self.assertFalse(same_parent("other/other.py", "pkg/widget.py"))
        self.assertEqual(
            extract_patch_symbols("+def locate_widget(self):\n+class WidgetLocator:"),
            {"locate_widget", "WidgetLocator"},
        )

    def test_last_mention_wins_for_multi_retrieval(self):
        search_one = """##Searching for query `first`...
### Search Result:
SHA: abcdef123 | Commit message: Fix widget lookup | Edited files: ['pkg/widget.py'] | score: 7.000
"""
        search_two = """##Searching for query `second`...
### Search Result:
SHA: abcdef123 | Commit message: Fix widget lookup | Edited files: ['pkg/widget.py'] | score: 8.000
"""
        examine = """##Commit for sha `abcdef123`...

Commit message: Fix widget lookup
Edited files: ['pkg/widget.py']
Patch:
diff --git a/pkg/widget.py b/pkg/widget.py
@@ -1 +1 @@ def locate_widget(self):
+class WidgetLocator:
"""
        messages = [
            assistant_call("search_commit", '{"query_list":["first"],"top_k":5}', "s1"),
            tool_observation("search_commit", search_one, "s1"),
            assistant_call("search_commit", '{"query_list":["second"],"top_k":5}', "s2"),
            tool_observation("search_commit", search_two, "s2"),
            assistant_call("examine_commit", '{"sha_list":["abcdef123"]}', "e1"),
            tool_observation("examine_commit", examine, "e1"),
            assistant_call(
                "get_entity_contents",
                '{"entity_names":["pkg/widget.py:WidgetLocator"]}',
                "c1",
            ),
            tool_observation("get_entity_contents", "source", "c1"),
        ]
        row = {
            "instance_id": "owner__repo-1",
            "found_files": [["pkg/widget.py"]],
            "meta_data": {"repo": "owner/repo"},
            "loc_trajs": {"trajs": [{"messages": messages}]},
        }
        commit = {
            "repo": "owner/repo",
            "sha": "abcdef1230000000000000000000000000000000",
            "subject": "Fix widget lookup",
            "message": "Fix widget lookup",
            "changed_files": ["pkg/widget.py"],
            "patch_path": None,
        }
        lookup = {
            ("owner/repo", "abcdef123"): commit,
            ("owner/repo", commit["sha"]): commit,
        }
        result = analyze_trajectory_row(
            row,
            version="v3",
            repeat=1,
            gold_files=["pkg/widget.py"],
            flip_class="memory_flipped",
            commit_lookup=lookup,
        )
        exposures = sorted(
            result["exposures"], key=lambda item: item["retrieval_event_index"]
        )
        self.assertFalse(exposures[0]["selection_adoption"])
        self.assertFalse(exposures[0]["final_adoption"])
        self.assertTrue(exposures[1]["selection_adoption"])
        self.assertTrue(exposures[1]["exploration_strict"])
        self.assertTrue(exposures[1]["exploration_medium"])
        self.assertTrue(exposures[1]["exploration_loose"])
        self.assertTrue(exposures[1]["final_adoption"])
        self.assertEqual(result["trajectories"][0]["retrieval_event_count"], 2)
        self.assertEqual(
            funnel_counts(result["trajectories"])["examined_ge1_file_in_top5"], 1
        )

    def test_alignment_labels_are_predeclared(self):
        self.assertEqual(alignment_label(0.31), "aligned")
        self.assertEqual(alignment_label(0.05), "near_zero")
        self.assertEqual(alignment_label(-0.11), "anti_aligned")
        self.assertEqual(alignment_label(0.20), "weak_or_inconclusive_positive")


if __name__ == "__main__":
    unittest.main()
