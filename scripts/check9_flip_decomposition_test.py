import unittest

from check9_flip_decomposition import (
    bucket_commit_count,
    classify_flip,
    complete_file_acc,
    normalize_selection_row,
    repo_lookup_keys,
)


class Check9FlipDecompositionTest(unittest.TestCase):
    def test_all_rerun_outcomes_are_exhaustively_classified(self):
        expected = {
            (1, 0): "stable_harm",
            (0, 0): "baseline_flipped",
            (1, 1): "memory_flipped",
            (0, 1): "harm_to_rescue",
        }
        for (rerun_b, rerun_m), flip_class in expected.items():
            with self.subTest(rerun=(rerun_b, rerun_m)):
                self.assertEqual(classify_flip(1, 0, rerun_b, rerun_m), flip_class)
                self.assertNotEqual(classify_flip(1, 0, rerun_b, rerun_m), "both_flipped")

    def test_missing_rerun_prediction_is_classified_separately(self):
        self.assertEqual(classify_flip(1, 0, None, 0), "missing_rerun")
        self.assertEqual(classify_flip(1, 0, 1, None), "missing_rerun")

    def test_bucket_commit_count_boundaries(self):
        self.assertEqual(bucket_commit_count(9999), "small")
        self.assertEqual(bucket_commit_count(10000), "mid")
        self.assertEqual(bucket_commit_count(29999), "mid")
        self.assertEqual(bucket_commit_count(30000), "large")

    def test_complete_file_acc_requires_full_gold_coverage_in_top5(self):
        gold = ["pkg/a.py", "pkg/b.py"]
        self.assertEqual(complete_file_acc(["pkg/a.py", "pkg/b.py"], gold, 5), 1)
        self.assertEqual(complete_file_acc(["pkg/a.py", "pkg/c.py"], gold, 5), 0)
        preds = ["x.py", "y.py", "z.py", "u.py", "v.py", "pkg/a.py", "pkg/b.py"]
        self.assertEqual(complete_file_acc(preds, gold, 5), 0)

    def test_normalize_selection_row_accepts_instance_id(self):
        row = {
            "instance_id": "django__django-1",
            "repo": "django",
            "baseline_at5": 1,
            "memory_at5": 0,
        }
        normalized = normalize_selection_row(row)
        self.assertEqual(normalized["case_id"], "django__django-1")

    def test_normalize_selection_row_fails_when_labels_missing(self):
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            normalize_selection_row({"case_id": "x", "repo": "r", "baseline_at5": 1})

    def test_repo_lookup_keys_handles_hyphenated_owner(self):
        keys = repo_lookup_keys({"case_id": "pylint-dev__pylint-7080", "repo": "pylint"})
        self.assertIn("pylint-dev__pylint", keys)
        self.assertIn("pylint-dev/pylint", keys)
        self.assertIn("pylint", keys)


if __name__ == "__main__":
    unittest.main()
