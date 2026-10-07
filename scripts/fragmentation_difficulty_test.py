import unittest

import pandas as pd

from scripts.analyze_fragmentation_by_difficulty import (
    compute_stratified_tier_stats,
    count_gold_files,
    low_high_fisher,
    normalize_input_schema,
)
from scripts.analyze_fragmentation_significance import assign_file_overlap_tiers


class FragmentationDifficultyTest(unittest.TestCase):
    def test_count_gold_files_uses_unique_diff_headers(self):
        patch = "\n".join(
            [
                "diff --git a/a.py b/a.py",
                "@@ -1 +1 @@",
                "diff --git a/b.txt b/b.txt",
                "@@ -1 +1 @@",
                "diff --git a/a.py b/a.py",
            ]
        )

        self.assertEqual(count_gold_files(patch), 2)

    def test_normalize_input_schema_adds_expected_aliases(self):
        df = pd.DataFrame(
            [
                {
                    "instance_id": "repo__1",
                    "file_overlap_jaccard_mean": 0.1,
                    "baseline_ok_k1": True,
                    "repomem_ok_k1": False,
                    "baseline_ok_k5": True,
                    "repomem_ok_k5": True,
                }
            ]
        )

        out = normalize_input_schema(df)

        self.assertEqual(out.loc[0, "case_id"], "repo__1")
        self.assertEqual(out.loc[0, "file_overlap_score"], 0.1)
        self.assertEqual(out.loc[0, "baseline_correct_at1"], 1)
        self.assertEqual(out.loc[0, "memory_correct_at1"], 0)

    def test_stratified_stats_and_fisher_run(self):
        df = pd.DataFrame(
            [
                {"file_overlap_jaccard_mean": 0.01, "baseline_correct_at1": 0, "memory_correct_at1": 1, "baseline_correct_at5": 0, "memory_correct_at5": 1, "difficulty": "single_file"},
                {"file_overlap_jaccard_mean": 0.02, "baseline_correct_at1": 1, "memory_correct_at1": 0, "baseline_correct_at5": 1, "memory_correct_at5": 0, "difficulty": "single_file"},
                {"file_overlap_jaccard_mean": 0.20, "baseline_correct_at1": 0, "memory_correct_at1": 1, "baseline_correct_at5": 0, "memory_correct_at5": 1, "difficulty": "single_file"},
                {"file_overlap_jaccard_mean": 0.30, "baseline_correct_at1": 1, "memory_correct_at1": 1, "baseline_correct_at5": 1, "memory_correct_at5": 1, "difficulty": "single_file"},
                {"file_overlap_jaccard_mean": 0.01, "baseline_correct_at1": 0, "memory_correct_at1": 0, "baseline_correct_at5": 0, "memory_correct_at5": 0, "difficulty": "multi_file"},
                {"file_overlap_jaccard_mean": 0.20, "baseline_correct_at1": 1, "memory_correct_at1": 1, "baseline_correct_at5": 1, "memory_correct_at5": 1, "difficulty": "multi_file"},
            ]
        )
        df = assign_file_overlap_tiers(df)

        tier_rows = compute_stratified_tier_stats(df)
        fisher = low_high_fisher(df, "single_file", 1)

        self.assertTrue(tier_rows)
        self.assertEqual(fisher["difficulty"], "single_file")
        self.assertIn("fisher_table", fisher)


if __name__ == "__main__":
    unittest.main()
