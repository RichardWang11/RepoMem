import json
import math
import tempfile
import unittest
from pathlib import Path

from auto_search_main import cached_prompt_tokens, load_fixed_memory_cells
from scripts.stale_memory_pilot import (
    cohort_gate,
    fingerprint_balance,
    final_verdict,
    partial_spearman,
    prepare,
    ranking_metrics,
    surfaced_files,
    validate_retryable_run,
    validate_run,
)


class StaleMemoryPilotTest(unittest.TestCase):
    def test_prepare_freezes_expected_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare(
                "results/future_reuse_audit/django231_bm25_code_top3/retrieval_events.csv",
                tmp,
            )
            self.assertEqual(65, len(manifest["cells"]))
            self.assertEqual(58, len(json.loads((Path(tmp) / "no_memory_task_ids.json").read_text())))
            self.assertEqual(20, len({row["memory_sha"] for row in manifest["cells"]}))

    def test_fixed_manifest_rejects_missing_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"cells": [{
                "cell_id": "c", "instance_id": "i", "memory_sha": "a" * 40,
            }]}))
            with self.assertRaises(ValueError):
                load_fixed_memory_cells(path)

    def test_search_parser_uses_only_canonical_outputs(self):
        messages = [
            {"role": "assistant", "tool_calls": [{
                "id": "a", "function": {"name": "search_code_snippets", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "a", "name": "search_code_snippets",
             "content": "Found class `pkg/good.py:Good`.\nrandom pkg/noise.py"},
            {"role": "assistant", "tool_calls": [{
                "id": "b", "function": {"name": "explore_tree_structure", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "b", "name": "explore_tree_structure",
             "content": "pkg/good.py:Good\n└── invokes ── pkg/other.py:work\nmissing.py:X"},
        ]
        result = surfaced_files(messages, {"pkg/good.py", "pkg/other.py", "pkg/noise.py"})
        self.assertEqual(["pkg/good.py", "pkg/other.py"], result["candidate_files"])
        self.assertEqual(2, result["search_rounds"])
        self.assertEqual(2, result["search_calls"])

    def test_cached_tokens_supports_mapping_and_object(self):
        class Usage:
            prompt_tokens_details = {"cached_tokens": 7}
        self.assertEqual(7, cached_prompt_tokens(Usage()))

    def test_ranking_metrics_use_order_and_cap_misses(self):
        ranked = ranking_metrics(["noise.py", "gold.py"], ["gold.py"])
        self.assertEqual(2, ranked["first_gold_rank_at5"])
        self.assertEqual(0.5, ranked["mrr_at5"])
        self.assertAlmostEqual(1 / math.log2(3), ranked["ndcg_at5"])
        self.assertEqual(
            6, ranking_metrics(["noise.py"], ["gold.py"])["first_gold_rank_at5"]
        )

    def test_run_validation_locks_sampling_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "args.json").write_text(json.dumps({
                "model": "openai/gpt-4o-2024-05-13",
                "temperature": 0.0,
                "top_p": None,
                "sampling_seed": 42,
                "num_samples": 1,
                "max_attempt_num": 1,
                "max_iteration_num": 20,
                "context_token_budget": 100000,
                "tool_observation_token_cap": 20000,
                "fail_on_context_error": True,
                "bounded_graph_output": True,
                "bounded_entity_output": True,
                "bounded_search_output": True,
                "use_repomem": False,
                "fixed_episodic_memory_manifest": None,
            }))
            lock = root / "lock.json"
            lock.write_text(json.dumps({
                "requested_model": "openai/gpt-4o-2024-05-13",
                "response_model": "gpt-4o-2024-05-13",
                "system_fingerprint": "fp_test",
            }))
            trajectory = {
                "provider_metadata": {
                    "temperature": 0,
                    "seed": 42,
                    "responses": [{
                        "model": "gpt-4o-2024-05-13",
                        "system_fingerprint": "fp_test",
                        "temperature": 0,
                        "seed": 42,
                    }, {
                        "model": "gpt-4o-2024-05-13",
                        "system_fingerprint": "fp_other",
                        "temperature": 0,
                        "seed": 42,
                    }],
                },
                "messages": [],
                "fixed_memory": None,
            }
            record = {
                "instance_id": "django__django-1",
                "loc_trajs": {"trajs": [trajectory]},
                "meta_data": {},
            }
            (root / "loc_trajs.jsonl").write_text(json.dumps(record) + "\n")
            result = validate_run(root, "no_memory", 1, 1, lock)
            self.assertTrue(result["valid"])
            self.assertTrue(result["fingerprint_mixture"])

    def test_incomplete_empty_results_are_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "args.json").write_text(json.dumps({
                "model": "gpt-4o",
                "temperature": 0.0,
                "top_p": None,
                "sampling_seed": 43,
                "num_samples": 1,
                "max_attempt_num": 1,
                "max_iteration_num": 20,
                "context_token_budget": 100000,
                "tool_observation_token_cap": 20000,
                "fail_on_context_error": True,
                "bounded_graph_output": True,
                "bounded_entity_output": True,
                "bounded_search_output": True,
                "use_repomem": False,
                "fixed_episodic_memory_manifest": None,
            }))
            lock = root / "lock.json"
            lock.write_text(json.dumps({
                "requested_model": "gpt-4o",
                "response_model": "gpt-4o-2024-05-13",
            }))
            rows = [
                {"instance_id": "done", "found_files": [["a.py"]]},
                {"instance_id": "retry", "found_files": [[]]},
            ]
            (root / "loc_outputs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (root / "loc_trajs.jsonl").write_text(
                json.dumps(rows[0]) + "\n"
            )
            result = validate_retryable_run(
                root, "no_memory", 2, 2, lock
            )
            self.assertTrue(result["retryable"])
            self.assertEqual(1, result["missing_count"])

    def test_gates_and_verdicts(self):
        rows = []
        for index in range(10):
            harm = index / 10
            rows.append({
                "memory_sha": str(index),
                "cohort": "top1_primary",
                "harm": harm,
                "harm_seed_42": harm,
                "harm_seed_43": harm,
                "harm_seed_44": harm,
            })
        self.assertTrue(cohort_gate(rows, "top1_primary")["passed"])
        self.assertGreater(partial_spearman(range(10), range(10), [0, 1] * 5), 0.9)
        q2 = {"passed": True, "target_drift_only": False}
        self.assertEqual("GO_VALIDITY_UTILITY_STUDY", final_verdict(True, q2, False))
        self.assertEqual("GO_PLACEBO_CONTROL_REQUIRED", final_verdict(True, q2, True))
        self.assertEqual("STOP", final_verdict(False, q2, False))

    def test_fingerprint_balance_reports_mixture_without_rejecting_it(self):
        rows = []
        for repeat in range(1, 4):
            rows.extend([
                {"repeat": repeat, "arm": "fixed", "fingerprint_counts": {"a": 8, "b": 2}},
                {"repeat": repeat, "arm": "no_memory", "fingerprint_counts": {"a": 7, "b": 3}},
            ])
        result = fingerprint_balance(rows)
        self.assertTrue(result["mixed_backend_observed"])
        self.assertFalse(result["imbalance_flag"])


if __name__ == "__main__":
    unittest.main()
