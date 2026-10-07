import math
import unittest

import numpy as np
import pandas as pd

from scripts.check9_gate_predictability import (
    auc_score,
    balanced_class_weights,
    complete_file_acc,
    compute_topk_coverage,
    extract_case_features,
    feature_columns,
    fit_logistic,
    leave_one_repo_out_predictions,
    make_loro_splits,
)


class Check9GatePredictabilityTest(unittest.TestCase):
    def test_compute_topk_coverage_uses_union_of_top_k_changed_files(self):
        commits = [
            {"changed_files": ["a.py", "x.py"]},
            {"changed_files": ["b.py"]},
            {"changed_files": ["c.py"]},
        ]

        self.assertAlmostEqual(
            compute_topk_coverage(commits, ["a.py", "b.py", "z.py"], top_k=2),
            2 / 3,
        )
        self.assertAlmostEqual(
            compute_topk_coverage(commits, ["c.py"], top_k=2),
            0.0,
        )

    def test_complete_file_acc_requires_full_gold_coverage(self):
        self.assertEqual(complete_file_acc(["a.py", "b.py"], ["a.py", "b.py"], 2), 1)
        self.assertEqual(complete_file_acc(["a.py", "x.py"], ["a.py", "b.py"], 2), 0)
        self.assertEqual(complete_file_acc(["a.py", "b.py"], ["a.py", "b.py"], 1), 0)

    def test_extract_case_features_handles_scores_issue_text_and_paths(self):
        row = {
            "issue_text": "Traceback in `MigrationWriter` for django/db/migrations/writer.py\n```python\nx=1\n```",
            "top_commits": [
                {
                    "score": 10.0,
                    "subject": "Fixed migration writer imports",
                    "message": "Body",
                    "changed_files": ["django/db/migrations/writer.py"],
                },
                {
                    "score": 5.0,
                    "subject": "Fixed schema editor",
                    "message": "",
                    "changed_files": ["django/db/backends/base/schema.py"],
                },
            ],
            "repo_commit_pool_size": 123,
        }

        features = extract_case_features(row)

        self.assertEqual(features["bm25_top1_score"], 10.0)
        self.assertAlmostEqual(features["bm25_top1_over_top5_mean"], 10.0 / 7.5)
        self.assertGreater(features["bm25_top20_std"], 0)
        self.assertEqual(features["top20_unique_files_count"], 2)
        self.assertEqual(features["top20_unique_dirs_count"], 1)
        self.assertGreater(features["issue_char_len"], 0)
        self.assertGreater(features["issue_code_ratio"], 0)
        self.assertEqual(features["mentioned_filename_count"], 1)
        self.assertEqual(features["mentioned_symbol_count"], 1)
        self.assertEqual(features["repo_commit_pool_size"], 123)

    def test_balanced_class_weights_gives_minority_class_larger_weight(self):
        weights = balanced_class_weights(np.array([0, 0, 0, 1]))

        self.assertGreater(weights[1], weights[0])

    def test_feature_columns_excludes_gold_and_agent_process_fields(self):
        df = pd.DataFrame(
            {
                "case_id": ["x"],
                "repo": ["django"],
                "gold_files_count": [2],
                "coverage_zero_at5": [1],
                "top_commit_count": [5],
                "search_commit_calls": [1],
                "bm25_top1_score": [1.0],
                "issue_char_len": [100],
            }
        )

        self.assertEqual(feature_columns(df), ["bm25_top1_score", "issue_char_len"])

    def test_fit_logistic_separates_toy_dataset(self):
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])

        model = fit_logistic(x, y, c=10.0)
        prob = model.predict_proba(x)

        self.assertGreater(auc_score(y, prob), 0.99)
        self.assertLess(prob[0], prob[-1])

    def test_leave_one_repo_out_predictions_hold_out_repos(self):
        df = pd.DataFrame(
            {
                "repo": ["a", "a", "b", "b", "c", "c"],
                "feature": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
                "coverage_zero_at5": [0, 0, 0, 1, 1, 1],
            }
        )
        splits = make_loro_splits(df, "repo")
        for train_idx, test_idx, held_repo in splits:
            self.assertFalse((df.iloc[train_idx]["repo"] == held_repo).any())
            self.assertTrue((df.iloc[test_idx]["repo"] == held_repo).all())

        pred = leave_one_repo_out_predictions(
            df,
            feature_cols=["feature"],
            label_col="coverage_zero_at5",
            repo_col="repo",
        )

        self.assertEqual(len(pred), len(df))
        self.assertFalse(pred["risk_probability"].isna().any())
        self.assertTrue(((pred["risk_probability"] >= 0) & (pred["risk_probability"] <= 1)).all())
        self.assertFalse(math.isnan(auc_score(pred["coverage_zero_at5"], pred["risk_probability"])))


if __name__ == "__main__":
    unittest.main()
