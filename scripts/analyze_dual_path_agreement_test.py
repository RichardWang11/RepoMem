import unittest

from scripts.analyze_dual_path_agreement import (
    decide,
    finite_rbo_distance,
    influence_audit,
    intersection_at_k_distance,
    jaccard_distance,
    partial_spearman,
    repository_sensitivity,
    sparse_degradation,
)


class DistanceTest(unittest.TestCase):
    def test_jaccard_empty_and_overlap_rules(self):
        self.assertEqual(0.0, jaccard_distance([], []))
        self.assertEqual(1.0, jaccard_distance(["a"], []))
        self.assertAlmostEqual(2 / 3, jaccard_distance(["a", "b"], ["b", "c"]))

    def test_sensitivity_distances(self):
        self.assertEqual(0.0, intersection_at_k_distance([], []))
        self.assertAlmostEqual(0.8, intersection_at_k_distance(["a"], ["a"]))
        self.assertEqual(0.0, finite_rbo_distance(["a", "b"], ["a", "b"]))
        self.assertEqual(1.0, finite_rbo_distance(["a"], ["b"]))


class StatisticsTest(unittest.TestCase):
    def test_partial_spearman_removes_linear_rank_confound(self):
        control = [1, 2, 3, 4, 5, 6]
        signal = [1, 2, 3, 4, 5, 7]
        outcome = [1, 2, 3, 4, 6, 5]
        value = partial_spearman(signal, outcome, control)
        self.assertTrue(-1 <= value <= 1)

    def test_sparse_degradation_excludes_ties(self):
        rows = [
            {"stability_class": "degradation", "d_cross_arm": 0.9},
            {"stability_class": "tie", "d_cross_arm": 0.5},
            {"stability_class": "improvement", "d_cross_arm": 0.1},
        ]
        result = sparse_degradation(rows)
        self.assertEqual(2, result["effective_n"])
        self.assertEqual(1, result["positive"])
        self.assertEqual(1.0, result["auroc"])

    def test_influence_signed_drop(self):
        rows = [
            {"case_id": f"c{i}", "repository": "r", "case_flip": int(i > 2), "d_cross_arm": i}
            for i in range(6)
        ]
        result, details = influence_audit(rows, 1.0)
        self.assertEqual(6, len(details))
        self.assertGreaterEqual(result["max_drop"], 0)

    def test_repository_sensitivity_reports_django(self):
        rows = [
            {
                "case_id": "d1", "repository": "django__django",
                "case_flip": 1, "d_cross_arm": 0.9,
            },
            {
                "case_id": "d2", "repository": "django__django",
                "case_flip": 0, "d_cross_arm": 0.1,
            },
            {
                "case_id": "s1", "repository": "sympy__sympy",
                "case_flip": 1, "d_cross_arm": 0.8,
            },
            {
                "case_id": "s2", "repository": "sympy__sympy",
                "case_flip": 0, "d_cross_arm": 0.2,
            },
        ]
        summary, details = repository_sensitivity(rows, 1.0)
        self.assertEqual(2, len(details))
        self.assertEqual(1.0, summary["leave_django_out_auroc"])


class DecisionTest(unittest.TestCase):
    def test_strong_requires_all_locked_conditions(self):
        metrics = {
            "d_cross_arm": {
                "auroc_case_flip": 0.80,
                "spearman_stability_gap": 0.50,
            },
            "d_baseline_internal": {"auroc_case_flip": 0.60},
        }
        bootstrap = {
            "auc": {"d_cross_arm": {"ci": [0.60, 0.95]}},
            "rho": {"d_cross_arm": {"ci": [0.10, 0.80]}},
            "partial_rho": {"ci": [0.05, 0.75]},
        }
        partial = {"observed": 0.40, "p_value_two_sided": 0.03}
        influence = {"max_drop": 0.10}
        result = decide(metrics, bootstrap, partial, influence)
        self.assertEqual("STRONG_PROMISING", result["verdict"])

    def test_secondary_cannot_upgrade_weak_primary(self):
        metrics = {
            "d_cross_arm": {
                "auroc_case_flip": 0.58,
                "spearman_stability_gap": 0.20,
            },
            "d_baseline_internal": {"auroc_case_flip": 0.40},
        }
        bootstrap = {
            "auc": {"d_cross_arm": {"ci": [0.30, 0.80]}},
            "rho": {"d_cross_arm": {"ci": [-0.20, 0.60]}},
            "partial_rho": {"ci": [-0.20, 0.60]},
        }
        result = decide(
            metrics,
            bootstrap,
            {"observed": 0.20, "p_value_two_sided": 0.20},
            {"max_drop": 0.10},
        )
        self.assertEqual("WEAK_INCONCLUSIVE", result["verdict"])


if __name__ == "__main__":
    unittest.main()
