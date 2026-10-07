import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auto_search_main import load_fixed_memory_cells
from plugins.location_tools.repo_ops import repo_ops
from scripts.dual_bm25_nomic_experiment import (
    holm_adjust,
    exact_mcnemar_p,
    profile_analysis,
    ranking_metrics,
    wilson,
)


class DualBm25NomicExperimentTest(unittest.TestCase):
    def test_metric_definitions_are_distinct(self):
        ranking = [
            {"changed_files": ["a.py"]},
            {"changed_files": ["unrelated.py"]},
        ]
        metrics = ranking_metrics(ranking, ["a.py", "b.py"], 2)
        self.assertEqual(metrics["hit"], 1)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_profiles_use_different_tokenization_and_idf(self):
        with mock.patch.dict(os.environ, {"REPOMEM_BM25_PROFILE": "bm25-code-v2"}):
            code = repo_ops._repomem_tokenize("parseHTTP_value")
        with mock.patch.dict(os.environ, {"REPOMEM_BM25_PROFILE": "bm25-experepair-v1"}):
            experepair = repo_ops._repomem_tokenize("parseHTTP_value")
        self.assertNotEqual(code, experepair)

    def test_profile_analysis_counts_rescued_and_harmed(self):
        def candidate(path):
            return {"changed_files": [path]}

        rows = [
            {"instance_id": "a", "repo": "r", "base_commit": "1", "gold_files": ["g.py"],
             "candidates": [candidate("x.py")] * 5 + [candidate("g.py")] + [candidate("x.py")] * 14,
             "reranked_candidates": [candidate("g.py")] + [candidate("x.py")] * 19},
            {"instance_id": "b", "repo": "r", "base_commit": "2", "gold_files": ["g.py"],
             "candidates": [candidate("g.py")] + [candidate("x.py")] * 19,
             "reranked_candidates": [candidate("x.py")] * 20},
        ]
        summary, _ = profile_analysis("p", rows)
        self.assertEqual(summary["mcnemar"]["c_rescued"], 1)
        self.assertEqual(summary["mcnemar"]["b_harmed"], 1)
        self.assertEqual(summary["net_gain_count"], 0)

    def test_holm_adjustment_is_monotone(self):
        rows = [{"mcnemar": {"p_raw": 0.01}}, {"mcnemar": {"p_raw": 0.04}}]
        holm_adjust(rows)
        self.assertEqual(rows[0]["mcnemar"]["p_holm"], 0.02)
        self.assertEqual(rows[1]["mcnemar"]["p_holm"], 0.04)

    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(0, 5), 0.0625)
        self.assertEqual(exact_mcnemar_p(1, 1), 1.0)

    def test_wilson_gate_boundary(self):
        self.assertLess(wilson(15, 100)[0], 0.15)
        self.assertGreater(wilson(30, 100)[0], 0.15)

    def test_fixed_manifest_accepts_ordered_top5_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"cells": [
                {"cell_id": "new", "instance_id": "i", "base_commit": "b", "memory_shas": ["1", "2", "3", "4", "5"]},
                {"cell_id": "old", "instance_id": "j", "base_commit": "b", "memory_sha": "1"},
            ]}))
            self.assertEqual(len(load_fixed_memory_cells(path)), 2)


if __name__ == "__main__":
    unittest.main()
