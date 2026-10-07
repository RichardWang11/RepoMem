import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_m3a_matched_noise_floor import stability
from scripts.analyze_paper2_pilot import sample_size_target
from scripts.sample_random_verified_pilot import (
    COHORT_SCHEDULES,
    execution_batches,
    select_sample,
    validate_indexes,
)
from scripts.validate_paper2_pilot_batch import EXPECTED_ARGS, validate_batch


class SamplingTest(unittest.TestCase):
    def test_fixed_seed_is_deterministic_and_excludes_paper1(self):
        population = [f"owner__repo-{index}" for index in range(500)]
        exclusions = population[:15]
        pool1, sample1 = select_sample(population, exclusions)
        pool2, sample2 = select_sample(reversed(population), exclusions)
        self.assertEqual(485, len(pool1))
        self.assertEqual(pool1, pool2)
        self.assertEqual(sample1, sample2)
        self.assertEqual(20, len(sample1))
        self.assertFalse(set(sample1) & set(exclusions))

    def test_index_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "graph"
            bm25 = root / "bm25"
            memory = root / "memory"
            instance_id = "owner__repo-1"
            graph.mkdir()
            (graph / f"{instance_id}.pkl").touch()
            (bm25 / instance_id).mkdir(parents=True)
            (bm25 / instance_id / "retriever.json").touch()
            (memory / "owner__repo").mkdir(parents=True)
            (memory / "owner__repo" / "commits.jsonl").touch()
            validate_indexes([instance_id], graph, bm25, memory)


class ScheduleAndStabilityTest(unittest.TestCase):
    def test_mirrored_schedule_and_batch_balance(self):
        self.assertEqual(8, len(COHORT_SCHEDULES["a"]))
        self.assertEqual(8, len(COHORT_SCHEDULES["b"]))
        for position in range(8):
            self.assertNotEqual(
                COHORT_SCHEDULES["a"][position][0],
                COHORT_SCHEDULES["b"][position][0],
            )
        for cohort in ("a", "b"):
            counts = {}
            for arm, repeat in COHORT_SCHEDULES[cohort]:
                counts[(arm, repeat)] = counts.get((arm, repeat), 0) + 1
            self.assertEqual(
                {(arm, repeat): 1 for arm in ("baseline", "memory") for repeat in range(1, 5)},
                counts,
            )
        self.assertEqual(16, len(execution_batches()))

    def test_four_repeat_stability_values(self):
        self.assertEqual(1.0, stability([1, 1, 1, 1])["pairwise_agreement"])
        self.assertEqual(0.5, stability([1, 1, 1, 0])["pairwise_agreement"])
        self.assertAlmostEqual(
            1 / 3, stability([1, 1, 0, 0])["pairwise_agreement"]
        )


class SampleSizeTest(unittest.TestCase):
    def test_locked_fpc_formula_and_floor(self):
        result = sample_size_target(0.2, 0.08)
        n0 = (1.96 * 0.2 / 0.03) ** 2
        nfpc = n0 / (1 + (n0 - 1) / 485)
        self.assertAlmostEqual(n0, result["n_0_uncorrected"])
        self.assertAlmostEqual(nfpc, result["n_fpc"])
        self.assertEqual(127, result["n_target"])
        self.assertAlmostEqual(
            0.08 * (20 / 127) ** 0.5,
            result["projected_target_half_width"],
        )
        self.assertLessEqual(result["normal_fpc_target_half_width"], 0.03)
        self.assertEqual(20, sample_size_target(0, 0)["n_target"])


class ValidatorTest(unittest.TestCase):
    def _write_jsonl(self, path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_fingerprint_drift_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            args = dict(EXPECTED_ARGS)
            args.update({"used_list_file": "cohort.json", "use_repomem": False})
            (run_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
            trajectory = {
                "provider_metadata": {
                    "responses": [{
                        "model": "gpt-4o-2024-11-20",
                        "system_fingerprint": "fp_wrong",
                    }]
                }
            }
            rows = [{
                "instance_id": "owner__repo-1",
                "found_files": [],
                "loc_trajs": {"trajs": [trajectory]},
            }]
            self._write_jsonl(run_dir / "loc_trajs.jsonl", rows)
            self._write_jsonl(run_dir / "loc_trajs.repaired.jsonl", rows)
            self._write_jsonl(
                run_dir / "context_guard_audit.jsonl",
                [{
                    "instance_id": "owner__repo-1",
                    "summary": {"truncated_observation_count": 0},
                    "bounded_graph_output": True,
                    "bounded_entity_output": True,
                    "bounded_search_output": True,
                }],
            )
            summary = validate_batch(
                run_dir, "baseline", 1, ["owner__repo-1"]
            )
            self.assertFalse(summary["valid"])
            self.assertEqual(1, summary["fingerprint_mismatch_count"])


if __name__ == "__main__":
    unittest.main()
