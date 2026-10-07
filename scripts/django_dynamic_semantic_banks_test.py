#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from plugins.location_tools.repo_ops import repo_ops
from scripts import build_django_dynamic_semantic_banks as dynamic
from scripts import build_repomem_semantic_index as builder


class DynamicSemanticBanksTest(unittest.TestCase):
    def test_materialized_instance_bank_is_runtime_search_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [
                {
                    "rank": rank,
                    "file": f"pkg/file_{rank}.py",
                    "edit_count": 201 - rank,
                    "source_blob_sha": f"{rank:040x}",
                }
                for rank in range(1, 201)
            ]
            membership = {
                "schema_version": 2,
                "repo": "django/django",
                "bank_type": "fresh_control",
                "instance_id": "django__django-test",
                "base_commit": "a" * 40,
                "source_commit_sha": "a" * 40,
                "source_tree_sha": "b" * 40,
                "commit_window": 7000,
                "window_commit_count": 7000,
                "window_commits_sha256": "c" * 64,
                "membership_sha256": "d" * 64,
                "selected_count": 200,
                "include_tests": False,
                "selection_order": "edit_count_desc,file_path_asc",
                "files": files,
            }
            recipe = {
                "recipe_sha256": "e" * 64,
                "summary_truncation": {"unit": "whitespace_word", "max_words": 220},
            }
            objects = {}
            for item in files:
                key = dynamic.object_key("django/django", item["file"], item["source_blob_sha"])
                objects[key] = {
                    "object_id": key,
                    "memory_id": key,
                    "repo": "django/django",
                    "file": item["file"],
                    "summary": f"Summary for {item['file']}",
                    "summary_text": f"Summary for {item['file']}",
                    "summary_sha256": "f" * 64,
                    "status": "ok",
                    "source_commit": "a" * 40,
                    "source_tree": "b" * 40,
                    "source_blob_sha": item["source_blob_sha"],
                    "canonical_instance_id": "django__django-test",
                    "response_models": ["gpt-4o-2024-11-20"],
                    "system_fingerprints": ["fp-test"],
                    "trajectory_path": "/tmp/canonical-trajectory.json",
                }
            args = SimpleNamespace(output_root=str(root), model="gpt-4o")
            dynamic.materialize(args, [(root / "membership.json", membership)], objects, recipe, builder)
            with mock.patch.dict(os.environ, {
                "REPOMEM_SEMANTIC_INDEX_DIR": str(root / "banks"),
                "REPOMEM_SEMANTIC_LAYOUT": "instance",
                "REPOMEM_SEMANTIC_MEMBERSHIP": "exclude_tests",
                "REPOMEM_SEMANTIC_DOCUMENT": "summary_only",
                "REPOMEM_BM25_PROFILE": "bm25-code-v2",
            }):
                repo_ops.REPOMEM_SEMANTIC_CACHE.clear()
                repo_ops.CURRENT_INSTANCE = {
                    "instance_id": membership["instance_id"],
                    "repo": membership["repo"],
                    "base_commit": membership["base_commit"],
                }
                data = repo_ops._load_repomem_semantic_repo("django/django")
                self.assertTrue(data["search_ready"])
                self.assertEqual(len(data["summaries"]), 200)
                output = repo_ops.search_summary("Summary file 4", top_k=4)
                self.assertIn("### Search Result:", output)
            repo_ops.CURRENT_INSTANCE = None


if __name__ == "__main__":
    unittest.main()
