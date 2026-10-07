import json
import os
import tempfile
import unittest
from pathlib import Path

from check_m1_retrieval_determinism import (
    build_case_rows,
    extract_memory_calls,
    jaccard,
    normalize_sha,
    parse_search_observation,
    replay_one_task,
    symmetric_best_query_jaccard,
)


class M1RetrievalDeterminismTest(unittest.TestCase):
    def test_parse_multi_query_observation(self):
        text = """##Searching for query `alpha`...
### Search Result:
SHA: AABBCC123 | Commit message: one | Edited files: ['a.py'] | score: 2.500
##Searching for query `beta`...
### Search Result:
SHA: ddeeff456 | Commit message: two | Edited files: ['b.py'] | score: 1.250
"""
        rows = parse_search_observation(text)
        self.assertEqual([row["query"] for row in rows], ["alpha", "beta"])
        self.assertEqual(rows[0]["commit_ids"], ["aabbcc123"])
        self.assertEqual(rows[1]["scores"], [1.25])

    def test_tool_call_is_paired_by_id(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "search_commit",
                            "arguments": json.dumps({"query_list": ["alpha"], "top_k": 5}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search_commit",
                "tool_call_id": "call-1",
                "content": "##Searching for query `alpha`...\nNo matching historical commits found.",
            },
        ]
        search, examine, errors = extract_memory_calls(messages, "case", "original")
        self.assertEqual(len(search), 1)
        self.assertEqual(search[0]["requested_top_k"], 5)
        self.assertEqual(examine, [])
        self.assertEqual(errors, [])

    def test_malformed_arguments_are_reported(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "bad", "function": {"name": "search_commit", "arguments": "{"}}
                ],
            },
            {
                "role": "tool",
                "name": "search_commit",
                "tool_call_id": "bad",
                "content": "",
            },
        ]
        _, _, errors = extract_memory_calls(messages, "case", "original")
        self.assertTrue(any("malformed arguments" in error for error in errors))

    def test_normalization_and_empty_jaccard(self):
        self.assertEqual(normalize_sha(" AABBCC "), "aabbcc")
        self.assertEqual(jaccard([], []), 1.0)
        self.assertEqual(jaccard(["a"], []), 0.0)
        self.assertEqual(jaccard(["a", "b"], ["b", "c"]), 1 / 3)

    def test_query_similarity_is_symmetric(self):
        left = ["migration import writer", "nested class"]
        right = ["writer migration import"]
        self.assertEqual(
            symmetric_best_query_jaccard(left, right),
            symmetric_best_query_jaccard(right, left),
        )

    def test_case_join_requires_both_sides(self):
        selection = [{"case_id": "x", "repo": "r"}]
        rows = build_case_rows(selection, {}, {}, {})
        self.assertTrue(rows[0]["missing_original"])
        self.assertTrue(rows[0]["missing_rerun"])

    def test_replay_applies_window_and_leakage_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir) / "owner__repo"
            repo_dir.mkdir()
            commits = [
                {
                    "repo": "owner/repo",
                    "sha": "a" * 40,
                    "short_sha": "a" * 9,
                    "subject": "outside old fix",
                    "message": "outside old fix",
                    "changed_files": ["old.py"],
                    "history_index": 0,
                    "issue_refs": [],
                },
                {
                    "repo": "owner/repo",
                    "sha": "b" * 40,
                    "short_sha": "b" * 9,
                    "subject": "target safe fix",
                    "message": "target safe fix",
                    "changed_files": ["safe.py"],
                    "history_index": 1,
                    "issue_refs": [],
                },
                {
                    "repo": "owner/repo",
                    "sha": "c" * 40,
                    "short_sha": "c" * 9,
                    "subject": "target leaked fix",
                    "message": "target leaked fix",
                    "changed_files": ["leaked.py"],
                    "history_index": 2,
                    "issue_refs": [12],
                },
            ]
            with open(repo_dir / "commits.jsonl", "w", encoding="utf-8") as f:
                for row in commits:
                    f.write(json.dumps(row) + "\n")
            with open(repo_dir / "issues.jsonl", "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "issue_number": 12,
                            "issue_summary": "target leaked fix",
                            "issue_description": "same current problem target leaked fix",
                        }
                    )
                    + "\n"
                )
            with open(repo_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"sha_to_history_index": {"d" * 40: 3}, "repo": "owner/repo"}, f
                )

            old_window = os.environ.get("REPOMEM_COMMIT_WINDOW")
            old_policy = os.environ.get("REPOMEM_VISIBILITY_POLICY")
            try:
                os.environ["REPOMEM_COMMIT_WINDOW"] = "2"
                os.environ["REPOMEM_VISIBILITY_POLICY"] = "legacy"
                from plugins.location_tools.repo_ops import repo_ops

                repo_ops.REPOMEM_CACHE.clear()
                task = {
                    "task_id": "test",
                    "instance_id": "owner__repo-1",
                    "query": "target fix",
                    "requested_top_k": 20,
                    "meta_data": {
                        "repo": "owner/repo",
                        "base_commit": "d" * 40,
                        "problem_statement": "same current problem target leaked fix",
                    },
                }
                result = replay_one_task(task, temp_dir)
                self.assertEqual(result["commit_ids"], ["b" * 9])
            finally:
                if old_window is None:
                    os.environ.pop("REPOMEM_COMMIT_WINDOW", None)
                else:
                    os.environ["REPOMEM_COMMIT_WINDOW"] = old_window
                if old_policy is None:
                    os.environ.pop("REPOMEM_VISIBILITY_POLICY", None)
                else:
                    os.environ["REPOMEM_VISIBILITY_POLICY"] = old_policy


if __name__ == "__main__":
    unittest.main()
