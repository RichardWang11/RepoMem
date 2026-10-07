import collections
import os
import unittest
from unittest import mock

os.environ.setdefault("GRAPH_INDEX_DIR", "/tmp")
os.environ.setdefault("BM25_INDEX_DIR", "/tmp")
os.environ.setdefault("REPOMEM_INDEX_DIR", "/tmp")
os.environ["REPOMEM_TOKENIZER"] = "code"
os.environ["REPOMEM_COMMIT_WINDOW"] = "3"

from plugins.location_tools.repo_ops import repo_ops
from scripts.audit_future_reuse import build_memory_rows, summarize_cutoff


def bm25_row(sha, text, index=0):
    tokens = repo_ops._repomem_tokenize(text)
    return {
        "sha": sha,
        "history_index": index,
        "commit_time": index,
        "subject": text,
        "issue_refs": [],
        "_tokens": tokens,
        "_token_counts": collections.Counter(tokens),
        "_doc_len": len(tokens),
    }


class FutureReuseAuditTest(unittest.TestCase):
    def tearDown(self):
        repo_ops.CURRENT_ISSUE_ID = None
        repo_ops.CURRENT_INSTANCE = None

    def test_bm25_code_ranking_ties_and_no_match(self):
        rows = [
            bm25_row("first", "MigrationWriter import fix"),
            bm25_row("second", "MigrationWriter import fix"),
            bm25_row("third", "unrelated cache"),
        ]
        ranked = repo_ops._rank_repomem_rows("migration writer", rows, 3)
        self.assertEqual([row["sha"] for _, row in ranked], ["first", "second"])
        self.assertEqual(repo_ops._rank_repomem_rows("absent_token", rows, 3), [])

    def test_production_window_and_leakage_filter(self):
        commits = [bm25_row(str(index), f"message {index}", index) for index in range(5)]
        commits[2]["issue_refs"] = ["123"]
        memory = {
            "commits": commits,
            "issues": {"123": {"issue_number": "123"}},
            "meta": {"sha_to_history_index": {"base": 5}},
            "_leakage_cache": {},
        }
        repo_ops.CURRENT_ISSUE_ID = "django__django-1"
        repo_ops.CURRENT_INSTANCE = {
            "instance_id": "django__django-1",
            "base_commit": "base",
            "problem_statement": "Regression from #123",
        }
        with mock.patch.dict(os.environ, {"REPOMEM_VISIBILITY_POLICY": "legacy"}):
            self.assertEqual(
                [row["sha"] for row in repo_ops._iter_visible_commits(memory)],
                ["3", "4"],
            )
            repo_ops.CURRENT_INSTANCE["base_commit"] = "after-index"
            memory["_leakage_cache"].clear()
            self.assertEqual(
                [row["sha"] for row in repo_ops._iter_visible_commits(memory)],
                ["3", "4"],
            )

    def test_disjoint_ngrams_skip_quadratic_span_scan(self):
        problem_tokens = ["alpha"] * 100
        issue_tokens = ["beta"] * 100
        with mock.patch.object(
            repo_ops,
            "_longest_common_token_run",
            side_effect=AssertionError("span scan should be skipped"),
        ):
            overlaps, reason = repo_ops._issue_overlaps_current(
                {"issue_number": "1"},
                problem_tokens,
                repo_ops._repomem_ngrams(problem_tokens),
                "unrelated problem",
                issue_tokens=issue_tokens,
                issue_ngrams=repo_ops._repomem_ngrams(issue_tokens),
            )
        self.assertFalse(overlaps)
        self.assertEqual(reason["method"], "none")

    def test_top1_no_go_top3_go(self):
        commits = [bm25_row(f"m{index}", f"memory {index}", index) for index in range(40)]
        opportunities = collections.Counter({row["sha"]: 30 for row in commits})
        events = []
        for index in range(30):
            common = {
                "instance_id": f"q{index}",
                "base_time": index,
                "age_days": 10 + index,
                "retrieval_status": "matched",
            }
            events.append({**common, "rank": 1, "memory_sha": f"m{10 + index}"})
            events.append({**common, "rank": 2, "memory_sha": f"m{index // 3}"})
        memory_rows = build_memory_rows(commits, opportunities, events)
        top1 = summarize_cutoff(memory_rows, events, 1)
        top3 = summarize_cutoff(memory_rows, events, 3)
        self.assertEqual(top1["gate"]["classification"], "NO-GO")
        self.assertEqual(top3["gate"]["classification"], "GO")
        self.assertEqual(top3["counts"]["n_ge_3"], 10)
        self.assertEqual(top3["repeated_memory_task_coverage"], 30)


if __name__ == "__main__":
    unittest.main()
