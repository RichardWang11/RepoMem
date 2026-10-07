import math
import unittest

import pandas as pd

from scripts.analyze_check6_7_coverage_decomposition import (
    assign_coverage_bucket,
    compute_gain_by_bucket,
    coverage_distribution_rows,
)


class Check67CoverageDecompositionTest(unittest.TestCase):
    def test_assign_coverage_bucket_separates_zero_and_boundaries(self):
        cases = [
            (0.0, "coverage_eq_0"),
            (0.001, "coverage_0_0.25"),
            (0.25, "coverage_0_0.25"),
            (0.5, "coverage_0.25_0.5"),
            (0.75, "coverage_0.5_0.75"),
            (1.0, "coverage_0.75_1.0"),
        ]

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(assign_coverage_bucket(value), expected)

    def test_coverage_distribution_counts_expected_buckets(self):
        df = pd.DataFrame({"retrieval_gold_coverage": [0.0, 0.1, 0.3, 0.8]})

        rows = coverage_distribution_rows(df)
        by_bucket = {row["bucket"]: row["count"] for row in rows}

        self.assertEqual(by_bucket["coverage_eq_0"], 1)
        self.assertEqual(by_bucket["coverage_0_0.25"], 1)
        self.assertEqual(by_bucket["coverage_0.25_0.5"], 1)
        self.assertEqual(by_bucket["coverage_0.75_1.0"], 1)

    def test_compute_gain_by_bucket_uses_precision_at1_and_original_acc5(self):
        df = pd.DataFrame(
            {
                "retrieval_gold_coverage": [0.0, 0.1, 0.9, 0.9],
                "baseline_precision_at1": [0, 0, 1, 0],
                "memory_precision_at1": [0, 1, 1, 1],
                "baseline_acc_at5_original": [0, 0, 1, 0],
                "memory_acc_at5_original": [0, 1, 1, 1],
            }
        )

        rows = compute_gain_by_bucket(df)
        by_bucket = {row["bucket"]: row for row in rows}

        self.assertEqual(by_bucket["coverage_eq_0"]["N"], 1)
        self.assertAlmostEqual(by_bucket["coverage_0_0.25"]["delta_acc1"], 1.0)
        self.assertAlmostEqual(by_bucket["coverage_0.5_1.0"]["delta_acc5"], 0.5)

        shares = [
            row["acc1_gain_share"]
            for row in rows
            if row["bucket"] in {"coverage_eq_0", "coverage_0_0.25", "coverage_0.25_0.5", "coverage_0.5_1.0"}
            and not math.isnan(row["acc1_gain_share"])
        ]
        self.assertAlmostEqual(sum(shares), 1.0)


if __name__ == "__main__":
    unittest.main()
