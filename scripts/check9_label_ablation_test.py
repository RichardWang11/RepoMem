import unittest

import pandas as pd

from scripts.check9_label_ablation import (
    add_coverage_zero_labels,
    summarize_label_ablation,
)


class Check9LabelAblationTest(unittest.TestCase):
    def test_add_coverage_zero_labels_tracks_multiple_topk_values(self):
        df = pd.DataFrame(
            {
                "case_id": ["x"],
                "gold_files": [["target.py"]],
                "top_commits": [[
                    {"changed_files": ["a.py"]},
                    {"changed_files": ["b.py"]},
                    {"changed_files": ["c.py"]},
                    {"changed_files": ["d.py"]},
                    {"changed_files": ["e.py"]},
                    {"changed_files": ["target.py"]},
                ]],
            }
        )

        out = add_coverage_zero_labels(df, ks=[5, 10, 20])

        self.assertEqual(out.loc[0, "coverage_zero_at5"], 1)
        self.assertEqual(out.loc[0, "coverage_zero_at10"], 0)
        self.assertEqual(out.loc[0, "coverage_zero_at20"], 0)

    def test_summarize_label_ablation_keeps_positive_rate_and_auc(self):
        rows = [
            {"k": 5, "n_cases": 4, "positive_count": 2, "auc": 0.75},
            {"k": 20, "n_cases": 4, "positive_count": 1, "auc": 0.60},
        ]

        summary = summarize_label_ablation(rows)

        self.assertEqual(summary[0]["label"], "coverage_zero_at5")
        self.assertAlmostEqual(summary[0]["positive_rate"], 0.5)
        self.assertEqual(summary[1]["label"], "coverage_zero_at20")


if __name__ == "__main__":
    unittest.main()
