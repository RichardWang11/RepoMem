import collections
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("GRAPH_INDEX_DIR", "/tmp")
os.environ.setdefault("BM25_INDEX_DIR", "/tmp")
os.environ.setdefault("REPOMEM_INDEX_DIR", "/tmp")

from plugins.location_tools.repo_ops import repo_ops
from scripts.build_semantic_membership_manifests import (
    build_membership,
    choose_static_cutoff,
)
from scripts.build_repomem_semantic_index import (
    AGENT_USER_PROMPT,
    build_repo_semantic_index,
    preflight_provider,
    summarize_agent,
    write_corpora,
)
from scripts import validate_semantic_memory_build


def bm25_row(text):
    tokens = repo_ops._repomem_tokenize(text)
    return {
        "text": text,
        "_tokens": tokens,
        "_token_counts": collections.Counter(tokens),
        "_doc_len": len(tokens),
    }


def run_git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def commit(repo, message):
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", message)
    return run_git(repo, "rev-parse", "HEAD")


class SemanticMemoryIntegrityTest(unittest.TestCase):
    def tearDown(self):
        repo_ops.REPOMEM_CACHE.clear()
        repo_ops.REPOMEM_SEMANTIC_CACHE.clear()
        repo_ops.CURRENT_INSTANCE = None
        repo_ops.CURRENT_ISSUE_ID = None

    def test_code_tokenizer_does_not_duplicate_plain_tokens(self):
        with mock.patch.dict(os.environ, {"REPOMEM_BM25_PROFILE": "bm25-code-v2"}):
            self.assertEqual(repo_ops._repomem_tokenize("plain"), ["plain"])
            self.assertEqual(
                repo_ops._repomem_tokenize("MigrationWriter migration_writer"),
                [
                    "migrationwriter",
                    "migration",
                    "writer",
                    "migration_writer",
                    "migration",
                    "writer",
                ],
            )

    def test_experepair_profile_matches_reference_scores(self):
        with mock.patch.dict(os.environ, {"REPOMEM_BM25_PROFILE": "bm25-experepair-v1"}):
            self.assertEqual(
                repo_ops._repomem_tokenize("MigrationWriter: cache-registry"),
                ["migrationwriter", "cache", "registry"],
            )
            rows = [
                bm25_row("migration writer"),
                bm25_row("migration serializer"),
                bm25_row("cache registry"),
            ]
            ranked = repo_ops._rank_repomem_rows("migration writer", rows, 3)
            scores = {row["text"]: score for score, row in ranked}
            self.assertAlmostEqual(scores["migration writer"], 0.5874494673308893)
            self.assertAlmostEqual(scores["migration serializer"], 0.07662384356489861)

    def test_semantic_loader_drops_invalid_rows_and_blocks_legacy_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_dir = root / "django__django"
            repo_dir.mkdir()
            (repo_dir / "meta.json").write_text(
                json.dumps({"top_k": 3, "num_ok": 1}),
                encoding="utf-8",
            )
            rows = [
                {"file": "good.py", "summary": "Useful module", "status": "ok"},
                {"file": "missing.py", "summary": "", "status": "missing_file"},
                {"file": "error.py", "summary": "", "status": "error:Bad"},
            ]
            (repo_dir / "summaries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "REPOMEM_SEMANTIC_INDEX_DIR": str(root),
                    "REPOMEM_BM25_PROFILE": "bm25-code-v2",
                },
            ):
                data = repo_ops._load_repomem_semantic_repo("django/django")
                self.assertEqual([row["file"] for row in data["summaries"]], ["good.py"])
                self.assertEqual(data["dropped_rows"], 2)
                self.assertFalse(data["search_ready"])
                repo_ops.CURRENT_INSTANCE = {"repo": "django/django"}
                self.assertIn("schema-v3", repo_ops.search_summary("useful"))
                self.assertIn("Useful module", repo_ops.view_summary("good.py"))

    def test_common_cutoff_and_membership_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_git(repo, "init")
            run_git(repo, "config", "user.email", "test@example.com")
            run_git(repo, "config", "user.name", "Test")
            (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
            (repo / "b.py").write_text("b = 1\n", encoding="utf-8")
            (repo / "deleted.py").write_text("x = 1\n", encoding="utf-8")
            (repo / "tests").mkdir()
            (repo / "tests" / "test_a.py").write_text("pass\n", encoding="utf-8")
            commit(repo, "root")
            (repo / "deleted.py").write_text("x = 2\n", encoding="utf-8")
            commit(repo, "edit deleted")
            (repo / "deleted.py").unlink()
            cutoff = commit(repo, "cutoff")

            run_git(repo, "checkout", "-b", "left")
            (repo / "a.py").write_text("a = 2\n", encoding="utf-8")
            left = commit(repo, "left")
            run_git(repo, "checkout", "-b", "right", cutoff)
            (repo / "b.py").write_text("b = 2\n", encoding="utf-8")
            right = commit(repo, "right")

            tasks = [
                {"instance_id": "left", "base_commit": left},
                {"instance_id": "right", "base_commit": right},
            ]
            selected, method = choose_static_cutoff(repo, tasks)
            self.assertEqual(selected, cutoff)
            self.assertEqual(method, "latest_verified_best_common_ancestor")

            first, _ = build_membership(repo, "example/repo", cutoff, 20, 2)
            second, _ = build_membership(repo, "example/repo", cutoff, 20, 2)
            self.assertEqual(first["membership_sha256"], second["membership_sha256"])
            self.assertEqual(first["window_commits_sha256"], second["window_commits_sha256"])
            paths = [row["file"] for row in first["files"]]
            self.assertEqual(paths, ["a.py", "b.py"])
            self.assertNotIn("deleted.py", paths)
            self.assertNotIn("tests/test_a.py", paths)

            first.update({
                "bank_type": "static",
                "c0_commit_sha": cutoff,
                "c0_tree_sha": first["source_tree_sha"],
            })
            manifest_path = repo / "membership.json"
            manifest_path.write_text(json.dumps(first), encoding="utf-8")
            output = repo / "semantic"
            index_dir = output / "example__repo"
            index_dir.mkdir(parents=True)
            with mock.patch(
                "scripts.build_repomem_semantic_index.summarize_direct",
                side_effect=lambda repo_name, file_path, edit_count, source, model,
                temperature, seed, request_timeout: (
                    "word " * 221,
                    {
                        "requested_model": model,
                        "response_model": "dummy-v1",
                        "system_fingerprint": "fp-test",
                        "temperature": temperature,
                        "seed": seed,
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                    },
                    None,
                ),
            ):
                build_repo_semantic_index(
                    repo_name="example/repo",
                    repo_dir=repo,
                    episodic_index_dir=repo / "unused",
                    output_dir=output,
                    top_k=2,
                    model="dummy",
                    builder_mode="direct",
                    max_source_chars=1000,
                    temperature=0,
                    request_timeout=1,
                    sleep_seconds=0,
                    overwrite=True,
                    include_tests=False,
                    num_workers=1,
                    membership_manifest=manifest_path,
                )
            metadata = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 3)
            self.assertEqual(metadata["membership_sha256"], first["membership_sha256"])
            build_spec = json.loads((index_dir / "build_spec.json").read_text(encoding="utf-8"))
            build_manifest = json.loads((index_dir / "build_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(build_spec["build_id"], build_manifest["build_id"])
            self.assertEqual(build_manifest["status"], "complete")
            rows = load_rows = [
                json.loads(line)
                for line in (index_dir / "summaries.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(row["source_commit"] == cutoff for row in rows))
            self.assertTrue(all(row["source_blob_sha"] for row in rows))
            self.assertTrue(all(row["prompt_instance_sha256"] for row in rows))
            self.assertEqual(len(rows[0]["summary_text"].split()), 221)
            self.assertIsNone(build_spec["summary_truncation"])
            self.assertEqual(
                build_spec["summary_normalization"],
                "strip_surrounding_whitespace_only",
            )
            with mock.patch.object(
                sys,
                "argv",
                ["validate_semantic_memory_build.py", str(index_dir), "--required_model", "dummy-v1"],
            ):
                validate_semantic_memory_build.main()
            with mock.patch.dict(
                os.environ,
                {
                    "REPOMEM_SEMANTIC_INDEX_DIR": str(output),
                    "REPOMEM_BM25_PROFILE": "bm25-code-v2",
                },
            ):
                repo_ops.REPOMEM_SEMANTIC_CACHE.clear()
                semantic = repo_ops._load_repomem_semantic_repo("example/repo")
                self.assertTrue(semantic["search_ready"])
                self.assertEqual(semantic["document_variant"], "summary_only")

                with mock.patch.dict(os.environ, {"REPOMEM_SEMANTIC_DOCUMENT": "path_plus_summary"}):
                    repo_ops.REPOMEM_SEMANTIC_CACHE.clear()
                    semantic = repo_ops._load_repomem_semantic_repo("example/repo")
                    self.assertTrue(semantic["search_ready"])
                    self.assertEqual(semantic["document_variant"], "path_plus_summary")

                # Tampering with the frozen corpus must fail closed.
                corpus = index_dir / "corpora" / "exclude_tests__summary_only.jsonl"
                corpus.write_text(corpus.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
                repo_ops.REPOMEM_SEMANTIC_CACHE.clear()
                self.assertFalse(repo_ops._load_repomem_semantic_repo("example/repo")["search_ready"])

    def test_union_build_writes_four_frozen_corpora_without_duplicate_rows(self):
        rows = [
            {"memory_id": "a", "file": "a.py", "summary_text": "alpha module", "status": "ok"},
            {"memory_id": "t", "file": "tests/test_a.py", "summary_text": "alpha tests", "status": "ok"},
        ]
        manifests = [
            {
                "include_tests": False,
                "membership_sha256": "exclude",
                "files": [{"file": "a.py"}],
            },
            {
                "include_tests": True,
                "membership_sha256": "include",
                "files": [{"file": "a.py"}, {"file": "tests/test_a.py"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            outputs = write_corpora(Path(tmp), rows, "bm25-code-v2", manifests)
            self.assertEqual(set(outputs), {
                "exclude_tests__summary_only",
                "exclude_tests__path_plus_summary",
                "include_tests__summary_only",
                "include_tests__path_plus_summary",
            })
            self.assertEqual(outputs["exclude_tests__summary_only"]["document_count"], 1)
            self.assertEqual(outputs["include_tests__summary_only"]["document_count"], 2)

    def test_agent_summary_requires_exact_target_inspection(self):
        def fake_process(result_queue, messages, **kwargs):
            messages = messages + [{
                "role": "assistant",
                "tool_calls": [{
                    "function": {
                        "name": "get_entity_contents",
                        "arguments": json.dumps({"entity_names": ["a.py"]}),
                    }
                }],
            }]
            result_queue.put((
                "Alpha module.",
                messages,
                {
                    "messages": messages,
                    "usage": {"total_prompt_tokens": 12, "total_completion_tokens": 3},
                    "provider_metadata": {"responses": [{
                        "model": "gpt-4o-2024-11-20",
                        "system_fingerprint": "fp-test",
                    }]},
                },
            ))

        with mock.patch("auto_search_main.auto_search_process", side_effect=fake_process):
            summary, metadata, trajectory = summarize_agent(
                "example/repo", "a.py", "gpt-4o", 0, 44, 120, 20
            )
        self.assertEqual(summary, "Alpha module.")
        self.assertEqual(metadata["response_models"], ["gpt-4o-2024-11-20"])
        self.assertTrue(trajectory["messages"])

    def test_agent_prompt_has_no_length_limit(self):
        self.assertEqual(
            AGENT_USER_PROMPT,
            "Summarize the high-level functionality and core responsibilities of "
            "`{file_path}`. Inspect the target file directly. If its local context is "
            "insufficient, actively explore relevant dependencies in other files. "
            "Return only the file functionality summary.",
        )
        self.assertNotIn("word", AGENT_USER_PROMPT.lower())

    def test_provider_preflight_locks_response_model(self):
        response = mock.Mock()
        response.model = "gpt-4o-2024-11-20"
        with mock.patch("scripts.build_repomem_semantic_index.litellm.completion", return_value=response) as completion:
            preflight_provider("gpt-4o", "gpt-4o-2024-11-20", 44, 120)
        self.assertEqual(completion.call_args.kwargs["max_tokens"], 2)
        response.model = "wrong-deployment"
        with mock.patch("scripts.build_repomem_semantic_index.litellm.completion", return_value=response):
            with self.assertRaises(RuntimeError):
                preflight_provider("gpt-4o", "gpt-4o-2024-11-20", 44, 120)


if __name__ == "__main__":
    unittest.main()
