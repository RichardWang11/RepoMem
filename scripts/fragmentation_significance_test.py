import unittest

import pandas as pd

from scripts.analyze_fragmentation_significance import (
    assign_file_overlap_tiers,
    bootstrap_delta_gap,
    build_fisher_table,
    tier_stats,
)


class FragmentationSignificanceTest(unittest.TestCase):
    def make_fixture(self):
        return pd.DataFrame(
            [
                {"file_overlap_jaccard_mean": 0.01, "baseline_ok_k1": False, "repomem_ok_k1": True},
                {"file_overlap_jaccard_mean": 0.02, "baseline_ok_k1": True, "repomem_ok_k1": False},
                {"file_overlap_jaccard_mean": 0.03, "baseline_ok_k1": False, "repomem_ok_k1": True},
                {"file_overlap_jaccard_mean": 0.10, "baseline_ok_k1": True, "repomem_ok_k1": True},
                {"file_overlap_jaccard_mean": 0.20, "baseline_ok_k1": False, "repomem_ok_k1": True},
                {"file_overlap_jaccard_mean": 0.30, "baseline_ok_k1": True, "repomem_ok_k1": True},
            ]
        )

    def test_assign_file_overlap_tiers_reverses_fragmentation_direction(self):
        df = assign_file_overlap_tiers(self.make_fixture())

        ordered = df.sort_values("file_overlap_jaccard_mean")
        self.assertEqual(list(ordered["frag_tier"]), ["high_frag", "high_frag", "mid_frag", "mid_frag", "low_frag", "low_frag"])

    def test_tier_stats_counts_rescues_and_regressions(self):
        df = assign_file_overlap_tiers(self.make_fixture())
        stats = tier_stats(df[df["frag_tier"] == "high_frag"], 1)

        self.assertEqual(stats["N"], 2)
        self.assertEqual(stats["rescue"], 1)
        self.assertEqual(stats["regress"], 1)
        self.assertAlmostEqual(stats["delta_acc"], 0.0)

    def test_build_fisher_table_uses_low_then_high_order(self):
        df = assign_file_overlap_tiers(self.make_fixture())

        table = build_fisher_table(df, 1)

        self.assertEqual(table, [[1, 0], [1, 1]])

    def test_bootstrap_delta_gap_is_deterministic_with_seed(self):
        df = assign_file_overlap_tiers(self.make_fixture())

        first = bootstrap_delta_gap(df, 1, n_boot=100, seed=7)
        second = bootstrap_delta_gap(df, 1, n_boot=100, seed=7)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
