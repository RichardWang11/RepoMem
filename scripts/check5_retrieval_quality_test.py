import unittest

import pandas as pd

from scripts.analyze_check5_retrieval_quality import (
    build_commit_lookup_from_rows,
    compute_retrieval_gold_coverage,
    fit_ols,
    lookup_commit_files,
    repo_for_lookup,
)


class Check5RetrievalQualityTest(unittest.TestCase):
    def test_compute_retrieval_gold_coverage_counts_gold_file_hits(self):
        commits = [
            {"changed_files": ["a.py", "x.py"]},
            {"changed_files": ["b.py", "c.py"]},
        ]

        self.assertAlmostEqual(
            compute_retrieval_gold_coverage(commits, ["a.py", "b.py", "z.py"]),
            2 / 3,
        )

    def test_lookup_commit_files_matches_short_sha_and_repo(self):
        lookup = build_commit_lookup_from_rows(
            [
                {
                    "repo": "django/django",
                    "sha": "abcdef1234567890",
                    "changed_files": ["django/db/models/base.py"],
                }
            ]
        )

        files = lookup_commit_files(lookup, "django/django", "abcdef123")

        self.assertEqual(files, ["django/db/models/base.py"])

    def test_repo_for_lookup_prefers_full_repo(self):
        row = {"repo": "django", "repo_full": "django/django"}

        self.assertEqual(repo_for_lookup(row), "django/django")

    def test_fit_ols_detects_positive_coverage_effect(self):
        df = pd.DataFrame(
            {
                "improvement": [0.0, 0.1, 0.2, 0.7, 0.8, 0.9],
                "retrieval_gold_coverage": [0.0, 0.0, 0.1, 0.8, 0.9, 1.0],
                "gold_files_count": [1, 1, 2, 1, 2, 3],
            }
        )

        model = fit_ols(df, "improvement", ["retrieval_gold_coverage", "gold_files_count"])
        by_term = {row["term"]: row for row in model["terms"]}

        self.assertGreater(by_term["retrieval_gold_coverage"]["coef"], 0)
        self.assertLess(by_term["retrieval_gold_coverage"]["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
