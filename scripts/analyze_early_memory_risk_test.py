import copy
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.analyze_early_memory_risk import (
    T0_FEATURES,
    bca_auc_interval,
    bootstrap_distribution_diagnostics,
    case_auc_bootstrap,
    case_flip_decision,
    case_flip_influence_audit,
    case_flip_influence_decision,
    cluster_bootstrap,
    extract_early_features,
    loco_splits,
    prior_adjust,
    reliability_rows,
    set_jaccard,
)


def call(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def observation(name, content, call_id):
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": call_id,
        "content": content,
    }


def fixture_row():
    search = """##Searching for query `widget fix`...
### Search Result:
SHA: abcdef123 | Commit message: Fix widget | Edited files: ['pkg/widget.py'] | score: 10.000
SHA: 123456789 | Commit message: Widget test | Edited files: ['tests/test_widget.py'] | score: 8.000
##Searching for query `widget lookup`...
### Search Result:
SHA: abcdef123 | Commit message: Fix widget | Edited files: ['pkg/widget.py'] | score: 9.000
SHA: fedcba987 | Commit message: Lookup support | Edited files: ['pkg/lookup.py'] | score: 7.000
"""
    examine = """##Commit for sha `abcdef123`...

Commit message: Fix widget
Edited files: ['pkg/widget.py']
Patch:
diff --git a/pkg/widget.py b/pkg/widget.py
+def fix_widget():
"""
    messages = [
        call("search_commit", '{"query_list":["widget fix","widget lookup"],"top_k":5}', "s"),
        observation("search_commit", search, "s"),
        call("examine_commit", '{"sha_list":["abcdef123"]}', "e"),
        observation("examine_commit", examine, "e"),
        call("get_entity_contents", '{"entity_names":["pkg/widget.py:fix_widget"]}', "c"),
        observation("get_entity_contents", "Found file `pkg/widget.py`", "c"),
        call("finish", "{}", "f"),
        observation("finish", "done", "f"),
    ]
    return {
        "instance_id": "owner__repo-1",
        "meta_data": {"repo": "owner/repo"},
        "loc_trajs": {"trajs": [{"messages": messages}]},
    }


def lookup():
    values = {}
    for sha, files in {
        "abcdef123": ["pkg/widget.py"],
        "123456789": ["tests/test_widget.py"],
        "fedcba987": ["pkg/lookup.py"],
    }.items():
        values[("owner/repo", sha)] = {
            "repo": "owner/repo",
            "sha": sha,
            "subject": "",
            "message": "",
            "changed_files": files,
            "patch_path": None,
        }
    return values


class EarlyMemoryRiskTest(unittest.TestCase):
    def test_t0_and_t1_feature_math(self):
        features = extract_early_features(fixture_row(), lookup())
        self.assertEqual(features["bm25_top1_score"], 10.0)
        self.assertAlmostEqual(features["bm25_relative_margin"], 0.2)
        self.assertEqual(features["retrieval_missing"], 0)
        self.assertEqual(features["multi_query_missing"], 0)
        self.assertAlmostEqual(features["inter_query_sha_jaccard"], 1 / 3)
        self.assertEqual(features["top1_file_overlap"], 1)
        self.assertEqual(features["commits_examined_before_code"], 1)
        self.assertEqual(features["rapid_unverified_top1_adoption"], 1)

    def test_checkpoint_ignores_later_messages(self):
        row = fixture_row()
        original = extract_early_features(row, lookup())
        changed = copy.deepcopy(row)
        changed["loc_trajs"]["trajs"][0]["messages"].extend(
            [
                call("get_entity_contents", '{"entity_names":["unrelated/later.py:X"]}', "late"),
                observation("get_entity_contents", "Found file `unrelated/later.py`", "late"),
            ]
        )
        mutated = extract_early_features(changed, lookup())
        self.assertEqual(
            {key: original[key] for key in T0_FEATURES},
            {key: mutated[key] for key in T0_FEATURES},
        )
        for key in (
            "memory_code_file_overlap",
            "top1_file_overlap",
            "commits_examined_before_code",
            "rapid_unverified_top1_adoption",
        ):
            self.assertEqual(original[key], mutated[key])

    def test_missing_retrieval_is_retained(self):
        row = {
            "meta_data": {"repo": "owner/repo"},
            "loc_trajs": {"trajs": [{"messages": []}]},
        }
        features = extract_early_features(row, {})
        self.assertEqual(features["retrieval_missing"], 1)
        self.assertTrue(np.isnan(features["bm25_top1_score"]))

    def test_prior_adjustment_is_monotone_and_reduces_probability(self):
        raw = np.array([0.1, 0.4, 0.8])
        adjusted = prior_adjust(raw, 61 / 180, 6 / 80)
        self.assertTrue(np.all(np.diff(adjusted) > 0))
        self.assertTrue(np.all(adjusted < raw))
        self.assertTrue(np.all((adjusted > 0) & (adjusted < 1)))

    def test_reliability_bins_and_case_cluster_bootstrap(self):
        frame = pd.DataFrame(
            {
                "case_id": [f"c{i // 4}" for i in range(80)],
                "target": [1 if i in (0, 1, 8, 9, 16, 17) else 0 for i in range(80)],
                "score": np.linspace(0.01, 0.99, 80),
            }
        )
        bins = reliability_rows(frame, "target", "score", "raw")
        self.assertEqual(len(bins), 10)
        self.assertEqual(sum(row["n"] for row in bins), 80)
        boot = cluster_bootstrap(frame, "target", "score", n_boot=100, seed=42)
        self.assertEqual(boot["requested_replicates"], 100)
        self.assertGreater(boot["valid_replicates"], 0)

    def test_set_jaccard(self):
        self.assertAlmostEqual(set_jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_case_auc_bootstrap_methods_and_determinism(self):
        frame = pd.DataFrame(
            {
                "case_id": [f"c{i}" for i in range(20)],
                "target": [1] * 6 + [0] * 14,
                "score": np.linspace(0.95, 0.05, 20),
            }
        )
        ordinary, ordinary_rows = case_auc_bootstrap(
            frame, "target", "score", "ordinary", n_boot=200, seed=42
        )
        repeated, repeated_rows = case_auc_bootstrap(
            frame, "target", "score", "ordinary", n_boot=200, seed=42
        )
        stratified, stratified_rows = case_auc_bootstrap(
            frame, "target", "score", "stratified", n_boot=200, seed=42
        )
        self.assertEqual(ordinary, repeated)
        self.assertEqual(ordinary_rows, repeated_rows)
        self.assertEqual(ordinary["requested_replicates"], 200)
        self.assertGreater(ordinary["valid_replicates"], 0)
        self.assertEqual(stratified["valid_replicates"], 200)
        self.assertTrue(
            all(
                row["positive_count"] == 6 and row["negative_count"] == 14
                for row in stratified_rows
            )
        )

    def test_case_auc_bootstrap_perfect_reversed_and_constant(self):
        frame = pd.DataFrame(
            {
                "case_id": [f"c{i}" for i in range(8)],
                "target": [1] * 4 + [0] * 4,
                "score": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
            }
        )
        perfect, _ = case_auc_bootstrap(
            frame, "target", "score", "stratified", n_boot=50, seed=7
        )
        reversed_frame = frame.copy()
        reversed_frame["score"] = -reversed_frame["score"]
        reversed_result, _ = case_auc_bootstrap(
            reversed_frame, "target", "score", "stratified", n_boot=50, seed=7
        )
        constant_frame = frame.copy()
        constant_frame["score"] = 0.5
        constant, _ = case_auc_bootstrap(
            constant_frame, "target", "score", "stratified", n_boot=50, seed=7
        )
        self.assertEqual(perfect["auroc_percentile_95ci"], [1.0, 1.0])
        self.assertEqual(reversed_result["auroc_percentile_95ci"], [0.0, 0.0])
        self.assertEqual(constant["auroc_percentile_95ci"], [0.5, 0.5])

    def test_bca_and_bimodal_diagnostics(self):
        labels = np.array([1] * 4 + [0] * 4)
        scores = np.array([0.9, 0.7, 0.6, 0.4, 0.8, 0.5, 0.3, 0.1])
        bootstrap = np.linspace(0.2, 0.95, 200)
        bca = bca_auc_interval(labels, scores, bootstrap)
        self.assertEqual(len(bca["auroc_bca_95ci"]), 2)
        self.assertEqual(bca["jackknife_replicates"], 8)
        values = np.concatenate(
            [np.linspace(0.20, 0.30, 100), np.linspace(0.75, 0.85, 100)]
        )
        diagnostics = bootstrap_distribution_diagnostics(values, seed=42)
        self.assertTrue(diagnostics["bimodal"])

    def test_case_flip_directional_gate_precedes_confidence_interval(self):
        ordinary = {
            "auroc_percentile_95ci": [0.60, 0.90],
            "ci_width": 0.30,
            "class_imbalance_fragility": False,
            "signal_fragility": False,
            "distribution_diagnostics": {"bimodal": False},
            "bca": {"auroc_bca_95ci": [0.45, 0.94]},
        }
        stratified = {"auroc_percentile_95ci": [0.65, 0.88]}
        decision = case_flip_decision(0.367, 0.762, ordinary, stratified)
        self.assertEqual(
            decision["status"], "REPLICATION_REQUIRED_DIRECTIONAL_FLIP"
        )
        self.assertIn("DIRECTIONAL_FLIP", decision["flags"])
        self.assertIn("HIGHLY_UNCERTAIN_PILOT_ESTIMATE", decision["flags"])
        self.assertFalse(decision["stage2_gating_authorized"])

    def test_case_flip_counterfactual_ci_boundary(self):
        ordinary = {
            "auroc_percentile_95ci": [0.51, 0.90],
            "ci_width": 0.39,
            "class_imbalance_fragility": False,
            "signal_fragility": False,
            "distribution_diagnostics": {"bimodal": False},
        }
        stratified = {"auroc_percentile_95ci": [0.55, 0.88]}
        passing = case_flip_decision(0.60, 0.72, ordinary, stratified)
        self.assertEqual(passing["status"], "CONTINUE_TO_PAPER2_CASE_LEVEL")
        ordinary["auroc_percentile_95ci"][0] = 0.50
        stopped = case_flip_decision(0.60, 0.72, ordinary, stratified)
        self.assertEqual(stopped["status"], "STOP_PREDICT_AND_GATE")

    def test_case_flip_influence_uses_sklearn_and_preserves_predictions(self):
        frame = pd.DataFrame(
            {
                "case_id": [f"c{i}" for i in range(6)],
                "case_flip": [1, 1, 1, 0, 0, 0],
                "predicted_risk": [0.9, 0.6, 0.5, 0.6, 0.4, 0.2],
            }
        )
        before = frame.to_csv(index=False)
        summary, singles, pairs = case_flip_influence_audit(
            frame, "case_flip", "predicted_risk", dev_auc=0.7, bca_lower=0.6
        )
        self.assertEqual(before, frame.to_csv(index=False))
        self.assertAlmostEqual(
            summary["original_auroc"],
            roc_auc_score(frame["case_flip"], frame["predicted_risk"]),
        )
        self.assertIn("average-rank tie handling", summary["method"])
        self.assertEqual(summary["positive_deletions"]["n"], 3)
        self.assertEqual(summary["negative_deletions"]["n"], 3)
        self.assertEqual(len(singles), 6)
        self.assertEqual(len(pairs), 15)

    def test_case_flip_influence_decision_boundaries_and_all_stops(self):
        hard_stops, reasons, category, status = case_flip_influence_decision(
            0.55, 0.65, 0.50, 0.20, 0.20, -0.15
        )
        self.assertTrue(hard_stops["DIRECTIONAL_FLIP"])
        self.assertTrue(hard_stops["BCA_CROSSES_CHANCE"])
        self.assertFalse(hard_stops["SINGLE_CASE_DOMINATED"])
        self.assertFalse(hard_stops["PAIR_DOMINANCE"])
        self.assertFalse(hard_stops["NOISE_SUPPRESSOR"])
        self.assertEqual(category, "MODERATE_SINGLE_CASE_SENSITIVITY")
        self.assertEqual(status, "STOP_PREDICT_AND_GATE")
        self.assertEqual(reasons, ["DIRECTIONAL_FLIP", "BCA_CROSSES_CHANCE"])

        hard_stops, reasons, category, status = case_flip_influence_decision(
            0.60, 0.70, 0.51, 0.201, 0.201, -0.151
        )
        self.assertEqual(
            reasons,
            ["SINGLE_CASE_DOMINATED", "PAIR_DOMINANCE", "NOISE_SUPPRESSOR"],
        )
        self.assertEqual(category, "SINGLE_CASE_DOMINATED")
        self.assertEqual(status, "STOP_PREDICT_AND_GATE")

    def test_case_flip_influence_nonstop_statuses(self):
        _, reasons, category, status = case_flip_influence_decision(
            0.60, 0.70, 0.51, 0.10, 0.20, -0.15
        )
        self.assertEqual(reasons, [])
        self.assertEqual(category, "MODERATE_SINGLE_CASE_SENSITIVITY")
        self.assertEqual(status, "REPLICATION_ONLY_MODERATE_SENSITIVITY")
        _, reasons, category, status = case_flip_influence_decision(
            0.60, 0.70, 0.51, 0.099, 0.20, -0.15
        )
        self.assertEqual(reasons, [])
        self.assertEqual(category, "NO_SINGLE_CASE_DOMINANCE")
        self.assertEqual(status, "ELIGIBLE_FOR_INDEPENDENT_REPLICATION")

    def test_loco_splits_never_share_cases(self):
        frame = pd.DataFrame(
            {"case_id": ["a", "a", "b", "b", "c", "c"], "value": range(6)}
        )
        for train, test, heldout in loco_splits(frame):
            self.assertEqual(set(test["case_id"]), {heldout})
            self.assertNotIn(heldout, set(train["case_id"]))
            self.assertFalse(set(train["case_id"]) & set(test["case_id"]))


if __name__ == "__main__":
    unittest.main()
