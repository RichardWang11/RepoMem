import unittest

from scripts.analyze_strict_visibility_pilot import (
    bootstrap_delta,
    exact_mcnemar,
    parse_run_name,
)


class StrictVisibilityAnalysisTest(unittest.TestCase):
    def test_exact_mcnemar(self):
        result = exact_mcnemar([True, True, False], [False, False, True])
        self.assertEqual(result["legacy_only"], 2)
        self.assertEqual(result["strict_only"], 1)
        self.assertEqual(result["two_sided_exact_p"], 1.0)

    def test_bootstrap_is_deterministic(self):
        left = [False, False, True, True]
        right = [True, False, True, False]
        self.assertEqual(
            bootstrap_delta(left, right, repetitions=1000, seed=7),
            bootstrap_delta(left, right, repetitions=1000, seed=7),
        )

    def test_parse_run_name_preserves_instance_id(self):
        self.assertEqual(
            parse_run_name("random_01_django__django-123_strict"),
            ("random", 1, "django__django-123", "strict"),
        )


if __name__ == "__main__":
    unittest.main()
