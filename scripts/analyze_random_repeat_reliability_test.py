import unittest

from scripts.analyze_random_repeat_reliability import (
    ab_label,
    bootstrap_harm_confirmation,
    enrich_case,
    exact_sign_flip_p,
    fleiss_kappa,
    harm_confirmation,
    pairwise_agreement,
    parse_outcomes,
    wilson_ci,
)


def make_case(case_id, baseline, memory):
    baseline_stability = pairwise_agreement(baseline)
    memory_stability = pairwise_agreement(memory)
    return enrich_case({
        "case_id": case_id,
        "repository": "owner__repo",
        "baseline_at5": baseline,
        "memory_at5": memory,
        "baseline_stability": baseline_stability,
        "memory_stability": memory_stability,
    }, "test")


class OutcomeParsingTest(unittest.TestCase):
    def test_parse_and_pairwise_stability(self):
        self.assertEqual([1, 1, 1, 1], parse_outcomes("1111"))
        self.assertEqual(1.0, pairwise_agreement([1, 1, 1, 1]))
        self.assertEqual(0.5, pairwise_agreement([1, 1, 1, 0]))
        self.assertAlmostEqual(1 / 3, pairwise_agreement([1, 1, 0, 0]))

    def test_ab_label_definitions(self):
        self.assertEqual("harm", ab_label(1, 0))
        self.assertEqual("rescue", ab_label(0, 1))
        self.assertEqual("both_correct", ab_label(1, 1))
        self.assertEqual("both_wrong", ab_label(0, 0))


class ReliabilityMetricTest(unittest.TestCase):
    def test_case_label_metrics(self):
        row = make_case("owner__repo-1", [1, 1, 0, 0], [0, 1, 1, 0])
        self.assertEqual(
            ["harm", "both_correct", "rescue", "both_wrong"], row["labels"]
        )
        self.assertEqual(0.0, row["label_pairwise_agreement"])
        self.assertEqual(1, row["label_flip"])
        self.assertEqual(1, row["harm_count"])
        self.assertEqual(0, row["majority_harm"])

    def test_fleiss_kappa_perfect(self):
        labels = [
            ["harm"] * 4,
            ["rescue"] * 4,
            ["both_correct"] * 4,
        ]
        self.assertEqual(1.0, fleiss_kappa(labels))

    def test_wilson_bounds(self):
        low, high = wilson_ci(3, 20)
        self.assertLess(low, 3 / 20)
        self.assertGreater(high, 3 / 20)

    def test_exact_sign_flip(self):
        self.assertEqual(1.0, exact_sign_flip_p([0, 0, 0]))
        self.assertAlmostEqual(0.5, exact_sign_flip_p([1, 1]))


class HarmConfirmationTest(unittest.TestCase):
    def test_confirmation_excludes_discovery_repeat(self):
        rows = [
            make_case("owner__repo-1", [1, 1, 1, 1], [0, 0, 1, 1]),
            make_case("owner__repo-2", [1, 1, 1, 1], [1, 1, 1, 1]),
        ]
        result = harm_confirmation(rows)
        self.assertEqual(2, result["selected_harm_events"])
        self.assertEqual(6, result["confirmation_opportunities"])
        self.assertEqual(2, result["confirmed_harm_outcomes"])
        self.assertAlmostEqual(1 / 3, result["harm_recurrence"])

    def test_cluster_bootstrap_is_deterministic(self):
        rows = [
            make_case("owner__repo-1", [1, 1, 1, 1], [0, 0, 1, 1]),
            make_case("owner__repo-2", [1, 1, 1, 1], [1, 1, 1, 1]),
        ]
        first = bootstrap_harm_confirmation(rows, n_boot=100, seed=7)
        second = bootstrap_harm_confirmation(rows, n_boot=100, seed=7)
        self.assertEqual(first, second)
        self.assertGreater(first["discard_rate"], 0)


if __name__ == "__main__":
    unittest.main()
