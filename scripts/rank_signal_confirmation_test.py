import unittest

from scripts.rank_signal_confirmation import (
    SEEDS,
    exact_harmful_tail,
    exact_sign_flip,
    harmful_candidate,
    holm_adjust,
)


def memory_rows(memory, benefit):
    return [
        {
            "memory_sha": memory,
            "instance_id": f"{memory}-task-{task}",
            "seed": seed,
            "benefit": benefit,
        }
        for task in range(3)
        for seed in SEEDS
    ]


class RankSignalConfirmationTest(unittest.TestCase):
    def test_harmful_candidate_requires_all_seed_means_negative(self):
        rows = memory_rows("m", -0.1)
        self.assertTrue(harmful_candidate(rows))
        for row in rows:
            if row["seed"] == SEEDS[0]:
                row["benefit"] = 0.1
        self.assertFalse(harmful_candidate(rows))

    def test_harmful_tail_flips_whole_memory_clusters(self):
        rows = []
        for index in range(12):
            rows.extend(memory_rows(str(index), -0.1 if index < 3 else 0.0))
        result = exact_harmful_tail(rows)
        self.assertEqual(4096, result["assignments"])
        self.assertEqual(3, result["observed_harmful_memories"])
        self.assertEqual(0.125, result["exact_p"])
        self.assertFalse(result["passed"])

    def test_six_consistent_harm_clusters_pass_exact_tail(self):
        rows = []
        for index in range(12):
            rows.extend(memory_rows(str(index), -0.1 if index < 6 else 0.0))
        result = exact_harmful_tail(rows)
        self.assertEqual(1 / 64, result["exact_p"])
        self.assertTrue(result["passed"])

    def test_targeted_six_task_exact_test_can_pass_holm(self):
        self.assertEqual(1 / 64, exact_sign_flip([-0.1] * 6, "less"))
        rows = [{"raw_p": 1 / 64}, {"raw_p": 0.5}]
        holm_adjust(rows)
        self.assertEqual(1 / 32, rows[0]["holm_p"])
        self.assertEqual(0.5, rows[1]["holm_p"])


if __name__ == "__main__":
    unittest.main()
