import unittest

import numpy as np
import pandas as pd

from scripts.analyze_check4_adaptive_metrics import (
    compute_metrics,
    fit_ols,
    merge_rare_repos,
    normalize_pred_files,
)


class Check4AdaptiveMetricsTest(unittest.TestCase):
    def test_normalize_pred_files_preserves_rank_and_deduplicates(self):
        self.assertEqual(
            normalize_pred_files(["a.py", "./b.py", "a.py", "pkg/c.py"]),
            ["a.py", "b.py", "pkg/c.py"],
        )

    def test_normalize_pred_files_sorts_score_dict(self):
        self.assertEqual(
            normalize_pred_files({"b.py": 0.2, "a.py": 0.8, "c.py": 0.1}),
            ["a.py", "b.py", "c.py"],
        )

    def test_compute_metrics_handles_multifile_adaptive_hit(self):
        pred = ["a.py", "x.py", "b.py", "c.py", "d.py", "e.py"]
        gold = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"]

        metrics = compute_metrics(pred, gold)

        self.assertAlmostEqual(metrics["recall_at5"], 4 / 6)
        self.assertEqual(metrics["precision_at1"], 1)
        self.assertEqual(metrics["adaptive_hit"], 0)
        self.assertEqual(metrics["acc_at5_original"], 0)

    def test_fit_ols_recovers_positive_slope(self):
        df = pd.DataFrame(
            {
                "y": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
                "x": [-1.0, -0.8, -0.6, 0.6, 0.8, 1.0],
            }
        )

        result = fit_ols(df, "y", ["x"])

        row = {r["term"]: r for r in result["terms"]}
        self.assertGreater(row["x"]["coef"], 0)
        self.assertLess(row["x"]["p_value"], 0.01)
        self.assertEqual(result["n"], 6)

    def test_fit_ols_supports_log_gold_count_term(self):
        df = pd.DataFrame(
            {
                "y": [0.1, 0.2, 0.4, 0.8],
                "file_overlap_z": [-1.0, -0.5, 0.5, 1.0],
                "gold_files_count": [1, 2, 3, 4],
            }
        )

        result = fit_ols(df, "y", ["file_overlap_z", "np.log(gold_files_count)"])

        terms = {r["term"] for r in result["terms"]}
        self.assertIn("log_gold_files_count", terms)

    def test_merge_rare_repos_uses_other_bucket(self):
        df = pd.DataFrame({"repo": ["a", "a", "b", "c", "c", "c"]})

        out = merge_rare_repos(df, "repo", min_n=2)

        self.assertEqual(list(out["repo_fe"]), ["a", "a", "other", "c", "c", "c"])


if __name__ == "__main__":
    unittest.main()
