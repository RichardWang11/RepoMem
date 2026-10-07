import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_applicability_labels import (
    annotation_packet,
    assign_stratum,
    cluster_bootstrap_kappa,
    csv_optional_bool,
    deterministic_sample,
    dual_condition_gate,
    has_parent_match,
    immediate_parent,
    gpt_label_complete,
    non_dunder_symbols,
)


class StructuralRuleTest(unittest.TestCase):
    def test_path_normalization_and_immediate_parent(self):
        self.assertEqual("pkg/mod", immediate_parent("a/pkg/mod/a.py"))
        self.assertIsNone(immediate_parent("setup.py"))
        self.assertTrue(has_parent_match(["pkg/a.py"], ["pkg/b.py"]))
        self.assertFalse(has_parent_match(["a.py"], ["b.py"]))

    def test_exclusive_precedence(self):
        self.assertEqual("exact_file", assign_stratum(True, True, True))
        self.assertEqual("parent_only", assign_stratum(False, True, True))
        self.assertEqual("symbol_only", assign_stratum(False, False, True))
        self.assertEqual("none", assign_stratum(False, False, False))

    def test_patch_symbol_extraction_excludes_dunder(self):
        patch = "+def useful_name():\n+    pass\n+def __init__(self):\n+    pass"
        self.assertEqual(["useful_name"], non_dunder_symbols(patch))


class SamplingTest(unittest.TestCase):
    def fixture_pool(self):
        pool = []
        targets = {"exact_file": 8, "parent_only": 7, "symbol_only": 7, "none": 8}
        offset = 0
        for stratum, count in targets.items():
            for index in range(count * 2):
                case = f"repo__repo-{offset + index // 2}"
                pool.append({
                    "case_id": case,
                    "repeat_id": index % 4 + 1,
                    "memory_sha": f"{offset + index:040x}",
                    "repository": "django/django" if index % 5 == 0 else "x/y",
                    "sampling_stratum": stratum,
                })
            offset += count
        return pool

    def test_deterministic_stratified_sample(self):
        first, _ = deterministic_sample(self.fixture_pool(), seed=7)
        second, _ = deterministic_sample(self.fixture_pool(), seed=7)
        key = lambda rows: [(r["case_id"], r["memory_sha"]) for r in rows]
        self.assertEqual(key(first), key(second))
        self.assertEqual(30, len(first))
        counts = {}
        for row in first:
            counts[row["sampling_stratum"]] = counts.get(row["sampling_stratum"], 0) + 1
        self.assertEqual(
            {"exact_file": 8, "parent_only": 7, "symbol_only": 7, "none": 8},
            counts,
        )
        self.assertGreaterEqual(len({row["case_id"] for row in first}), 15)


class BlindPacketTest(unittest.TestCase):
    def test_packet_excludes_machine_and_outcome_labels(self):
        row = {
            "sample_id": "AVL-X", "case_id": "owner__repo-1",
            "repository": "owner/repo", "problem_statement": "issue",
            "memory_sha": "abc", "memory_subject": "subject",
            "memory_message": "message", "memory_files": ["pkg/a.py"],
            "memory_patch": "+pass", "base_commit": "def",
            "current_context": [], "gold_files": ["pkg/b.py"],
            "gold_patch": "+pass", "sampling_stratum": "parent_only",
            "structural_relevant": True, "paired_harm": True,
        }
        packet = annotation_packet(row)
        text = json.dumps(packet).lower()
        self.assertNotIn("sampling_stratum", text)
        self.assertNotIn("structural_relevant", text)
        self.assertNotIn("paired_harm", text)


class StatisticsTest(unittest.TestCase):
    def test_optional_bool_preserves_temporal_unknown(self):
        self.assertTrue(csv_optional_bool("True"))
        self.assertFalse(csv_optional_bool("False"))
        self.assertIsNone(csv_optional_bool(""))

    def test_dual_condition_gate_boundaries(self):
        self.assertEqual("PROCEED", dual_condition_gate(0.50, 0.30))
        self.assertEqual("REFINE_AND_REPLICATE", dual_condition_gate(0.70, 0.29))
        self.assertEqual("REFINE_AND_REPLICATE", dual_condition_gate(0.30, 0.00))
        self.assertEqual("SERIOUS_PIVOT", dual_condition_gate(0.29, 0.20))
        self.assertEqual("SERIOUS_PIVOT", dual_condition_gate(0.60, -0.01))

    def test_cluster_bootstrap_is_deterministic_and_tracks_undefined(self):
        rows = [
            {"case_id": "a", "human_applicable": 1, "structural_relevant": 1},
            {"case_id": "b", "human_applicable": 0, "structural_relevant": 0},
            {"case_id": "c", "human_applicable": 1, "structural_relevant": 0},
        ]
        one, summary_one = cluster_bootstrap_kappa(rows, seed=3, replicates=100)
        two, summary_two = cluster_bootstrap_kappa(rows, seed=3, replicates=100)
        self.assertEqual(one, two)
        self.assertEqual(summary_one, summary_two)
        self.assertEqual(100, summary_one["defined_count"] + summary_one["undefined_count"])

    def test_gpt_completion_requires_locked_metadata(self):
        row = {
            "applicability_label": "Direct-help",
            "applicability_evidence": "direct transfer",
            "validity_label": "Valid",
            "validity_evidence": "symbol resolves",
            "confidence": "high",
            "model": "gpt-4o-2024-11-20",
            "temperature": 0,
            "prompt_sha256": "locked",
            "system_fingerprint": "fp_test",
            "input_tokens": 10,
            "output_tokens": 5,
            "actual_cost": 0.01,
            "raw_response": "{}",
        }
        self.assertTrue(gpt_label_complete(row, "locked"))
        row["model"] = "gpt-4o"
        self.assertFalse(gpt_label_complete(row, "locked"))


if __name__ == "__main__":
    unittest.main()
