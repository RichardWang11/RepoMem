#!/usr/bin/env python3
import os
import unittest
from unittest import mock

from plugins.location_tools.repo_ops import repo_ops
from util.runtime.memory_tools import get_memory_tools


def function_schema(tool):
    return tool["function"] if isinstance(tool, dict) else tool.function


class PaperToolSchemaTest(unittest.TestCase):
    def test_paper_schema_matches_appendix_interfaces(self):
        tools = get_memory_tools("paper-v1")
        search = function_schema(tools["search_commit"])
        examine = function_schema(tools["examine_commit"])
        view = function_schema(tools["view_summary"])
        summary = function_schema(tools["search_summary"])
        self.assertEqual(search["parameters"]["properties"]["top_k"]["default"], 20)
        self.assertEqual(search["parameters"]["required"], ["query_list"])
        self.assertEqual(examine["parameters"]["required"], ["sha_list"])
        self.assertEqual(view["parameters"]["required"], ["file_list"])
        self.assertEqual(summary["parameters"]["properties"]["top_k"]["default"], 5)

    def test_current_view_summary_interface_is_unchanged(self):
        view = function_schema(get_memory_tools("current")["view_summary"])
        self.assertEqual(view["parameters"]["required"], ["file_name"])


class PaperToolRenderingTest(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {"REPOMEM_TOOL_PROFILE": "paper-v1"})
        self.environment.start()
        repo_ops.CURRENT_INSTANCE = {
            "instance_id": "django__django-test",
            "repo": "django/django",
            "base_commit": "base",
        }

    def tearDown(self):
        repo_ops.CURRENT_INSTANCE = None
        self.environment.stop()

    def test_search_commit_hides_scores_and_test_files(self):
        memory = {"meta": {}, "commits": [], "issues": {}}
        row = {
            "sha": "a" * 40,
            "short_sha": "a" * 9,
            "subject": "Fixed migration ordering.",
            "changed_files": ["django/db/migrations/autodetector.py", "tests/migrations/test.py"],
        }
        with mock.patch.object(repo_ops, "_load_repomem_repo", return_value=memory), \
             mock.patch.object(repo_ops, "_iter_visible_commits", return_value=iter([row])), \
             mock.patch.object(repo_ops, "_rank_repomem_rows", return_value=[(2.5, row)]):
            output = repo_ops.search_commit(["migration ordering"], top_k=20)
        self.assertIn("django/db/migrations/autodetector.py", output)
        self.assertNotIn("tests/migrations/test.py", output)
        self.assertNotIn("score:", output)

    def test_view_summary_accepts_multiple_files(self):
        rows = {
            "a.py": {"file": "a.py", "summary_text": "Summary A", "edit_count": 4},
            "b.py": {"file": "b.py", "summary_text": "Summary B", "edit_count": 2},
        }
        with mock.patch.object(
            repo_ops, "_load_repomem_semantic_repo", return_value={"by_file": rows}
        ):
            output = repo_ops.view_summary(file_list=["a.py", "b.py"])
        self.assertIn("##Summary for file `a.py`...", output)
        self.assertIn("##Summary for file `b.py`...", output)
        self.assertNotIn("edit_count", output)


if __name__ == "__main__":
    unittest.main()
