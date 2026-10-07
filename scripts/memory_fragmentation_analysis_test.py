import unittest

from scripts.analyze_memory_fragmentation import (
    assign_terciles,
    compute_fragmentation_metrics,
    parse_search_commit_observation,
)


class MemoryFragmentationAnalysisTest(unittest.TestCase):
    def test_parse_search_commit_observation_deduplicates_sha_rows(self):
        text = """OBSERVATION:
##Searching for query `migration import`...
### Search Result:
SHA: 73a6ab638 | Commit message: Fixed migration writer imports. | Edited files: ['django/db/migrations/writer.py', 'tests/migrations/test_writer.py'] | score: 12.100
SHA: 533b20877 | Commit message: Fixed schema unique constraint removal. | Edited files: ['django/db/backends/base/schema.py'] | score: 8.000
##Searching for query `writer`...
### Search Result:
SHA: 73a6ab638 | Commit message: Fixed migration writer imports. | Edited files: ['django/db/migrations/writer.py'] | score: 10.000
"""
        commits = parse_search_commit_observation(text)

        self.assertEqual([c["short_sha"] for c in commits], ["73a6ab638", "533b20877"])
        self.assertEqual(commits[0]["subject"], "Fixed migration writer imports.")
        self.assertEqual(
            commits[0]["changed_files"],
            ["django/db/migrations/writer.py", "tests/migrations/test_writer.py"],
        )

    def test_compute_fragmentation_metrics(self):
        commits = [
            {
                "sha": "a",
                "subject": "fix migration writer imports",
                "changed_files": ["django/db/migrations/writer.py", "django/db/migrations/serializer.py"],
            },
            {
                "sha": "b",
                "subject": "fix migration import serialization",
                "changed_files": ["django/db/migrations/writer.py"],
            },
            {
                "sha": "c",
                "subject": "fix auth backend permissions",
                "changed_files": ["django/contrib/auth/backends.py"],
            },
        ]

        metrics = compute_fragmentation_metrics(commits)

        self.assertAlmostEqual(metrics["file_overlap_jaccard_mean"], (0.5 + 0.0 + 0.0) / 3)
        self.assertEqual(metrics["distinct_top_level_dir_count"], 1)
        self.assertGreater(metrics["subject_bm25_selfsim_norm"], 0)
        self.assertLess(metrics["dominant_top_level_dir_share"], 1.01)

    def test_assign_terciles_orders_low_mid_high(self):
        rows = [{"id": i, "metric": v} for i, v in enumerate([1, 2, 3, 4, 5, 6])]

        assign_terciles(rows, "metric", "metric_tercile", high_is_fragmented=True)

        self.assertEqual([r["metric_tercile"] for r in rows], ["low", "low", "mid", "mid", "high", "high"])


if __name__ == "__main__":
    unittest.main()
