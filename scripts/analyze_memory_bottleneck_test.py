import unittest

from scripts.analyze_memory_bottleneck import (
    build_case_table,
    build_trajectory_table,
    condition_supported,
    decision,
    deduplicate_candidates,
    matrix_category,
    query_count_strata,
    raw_exposure_sensitivity,
    resolve_mixed_direction,
)


class CandidateTest(unittest.TestCase):
    def test_top5_deduplicates_sha_across_queries(self):
        base = {
            "version": "pilot", "instance_id": "owner__repo-1",
            "repo": "owner/repo", "repeat": 1, "traj_index": 0,
            "sha": "abc", "changed_files": ["pkg/a.py"],
            "gold_overlap_files": ["pkg/a.py"], "gold_file_overlap": True,
            "selection_adoption": True, "final_adoption": True,
            "exploration_strict": True, "commit_lookup_found": True,
        }
        rows = [
            {**base, "rank": 2, "query": "one", "query_index": 0},
            {**base, "rank": 1, "query": "two", "query_index": 1},
            {**base, "sha": "late", "rank": 6, "query": "one", "query_index": 0},
        ]
        candidates = deduplicate_candidates(rows)
        self.assertEqual(1, len(candidates))
        self.assertEqual(1, candidates[0]["best_rank"])
        self.assertEqual(2, candidates[0]["query_count"])
        self.assertTrue(candidates[0]["primary_adopted"])

    def test_primary_adoption_requires_selection_and_final(self):
        base = {
            "version": "pilot", "instance_id": "c", "repo": "r",
            "repeat": 1, "traj_index": 0, "rank": 1, "query": "q",
            "query_index": 0, "changed_files": ["a.py"],
            "gold_overlap_files": ["a.py"], "gold_file_overlap": True,
            "exploration_strict": False, "commit_lookup_found": True,
        }
        rows = [
            {**base, "sha": "a", "selection_adoption": True, "final_adoption": False},
            {**base, "sha": "b", "selection_adoption": False, "final_adoption": True},
            {**base, "sha": "c", "selection_adoption": True, "final_adoption": True},
        ]
        adopted = [row["primary_adopted"] for row in deduplicate_candidates(rows)]
        self.assertEqual([False, False, True], adopted)

    def test_raw_exposure_sensitivity_keeps_duplicates(self):
        rows = [
            {
                "rank": 1, "selection_adoption": True,
                "final_adoption": True, "gold_file_overlap": True,
            },
            {
                "rank": 2, "selection_adoption": True,
                "final_adoption": True, "gold_file_overlap": False,
            },
            {
                "rank": 6, "selection_adoption": True,
                "final_adoption": True, "gold_file_overlap": True,
            },
        ]
        result = raw_exposure_sensitivity(rows)
        self.assertEqual(2, result["top5_query_item_exposure_count"])
        self.assertEqual(0.5, result["raw_exposure_trust_precision"])


class FunnelTest(unittest.TestCase):
    def test_three_failure_stages_and_zero_retrieval(self):
        result = {
            "events": [],
            "trajectories": [
                {
                    "version": "pilot", "instance_id": "c", "repo": "r",
                    "repeat": repeat, "traj_index": 0,
                    "retrieval_event_count": int(repeat != 1),
                    "final_top5": [],
                }
                for repeat in range(1, 5)
            ],
        }
        candidates = [
            {
                "case_id": "c", "repeat": 2, "traj_index": 0,
                "relevant": True, "first_query": True,
                "primary_adopted": False, "strict_adopted": False,
                "gold_overlap_files": ["a.py"],
            },
            {
                "case_id": "c", "repeat": 3, "traj_index": 0,
                "relevant": True, "first_query": True,
                "primary_adopted": True, "strict_adopted": True,
                "gold_overlap_files": ["a.py"],
            },
        ]
        outcomes = {("c", repeat): int(repeat == 4) for repeat in range(1, 5)}
        rows = build_trajectory_table(
            result, candidates, outcomes, {"c": ["a.py"]}
        )
        self.assertEqual(
            ["retrieval_miss", "adoption_miss", "post_adoption_failure", "success"],
            [row["failure_stage"] for row in rows],
        )

    def test_case_hit_uses_mean_of_four(self):
        trajectories = [
            {
                "case_id": "c", "repository": "r", "hit_at5": int(repeat <= 3),
                "first_query_hit_at5": 0, "correct_primary_adopted": 0,
                "memory_failure": 1, "failure_stage": "adoption_miss",
            }
            for repeat in range(1, 5)
        ]
        row = build_case_table(trajectories, [])[0]
        self.assertEqual(0.75, row["hit_at5_rate"])
        self.assertEqual(1, row["any_of4_hit"])
        self.assertEqual(0, row["all_of4_hit"])

    def test_query_count_strata(self):
        rows = [
            {
                "case_id": "a", "executed_query_count": 1,
                "hit_at5": 1, "memory_failure": 0,
            },
            {
                "case_id": "b", "executed_query_count": 1,
                "hit_at5": 0, "memory_failure": 1,
            },
        ]
        result = query_count_strata(rows)
        self.assertEqual(0.5, result[0]["hit_at5"])
        self.assertEqual(2, result[0]["case_count"])


class DecisionTest(unittest.TestCase):
    def test_matrix_cells(self):
        self.assertEqual("RETRIEVAL_LIMITED", matrix_category(0.4, 0.6))
        self.assertEqual("BOTH_LIMITED", matrix_category(0.4, 0.4))
        self.assertEqual("TRUST_LIMITED", matrix_category(0.8, 0.4))
        self.assertEqual("BOTH_GOOD", matrix_category(0.8, 0.8))
        self.assertEqual("MIXED_DIRECTION", matrix_category(0.6, 0.8))

    def test_mixed_tie_breakers(self):
        direction, reason = resolve_mixed_direction(
            0.6, 0.6,
            {"retrieval_miss": 0.5, "adoption_miss": 0.3,
             "post_adoption_failure": 0.2},
        )
        self.assertEqual("RETRIEVAL_LIMITED", direction)
        self.assertEqual("largest_failure_funnel_share", reason)
        direction, reason = resolve_mixed_direction(
            0.6, 0.6,
            {"retrieval_miss": 1 / 3, "adoption_miss": 1 / 3,
             "post_adoption_failure": 1 / 3},
        )
        self.assertEqual("JOINT_RETRIEVAL_FIRST", direction)
        self.assertEqual("exact_tie_fallback", reason)

    def test_ci_support_and_point_fallback(self):
        self.assertTrue(condition_supported([0.2, 0.4], "lt", 0.5))
        self.assertFalse(condition_supported([0.2, 0.6], "lt", 0.5))
        metrics = {
            "hit_at5": 0.4, "trust_precision": 0.6,
            "conditional_trust_recall": 0.7,
            "failure_shares": {
                "retrieval_miss": 0.6, "adoption_miss": 0.2,
                "post_adoption_failure": 0.2,
            },
        }
        bootstrap = {
            "hit_at5": {"ci": [0.2, 0.6]},
            "trust_precision": {"ci": [0.3, 0.8]},
        }
        result = decision(metrics, bootstrap)
        self.assertEqual("RETRIEVAL_LIMITED", result["paper2_direction"])
        self.assertEqual("POINT_ESTIMATE_DIRECTION", result["evidence_tier"])
        self.assertTrue(result["replication_required_before_confirmatory_claim"])


if __name__ == "__main__":
    unittest.main()
