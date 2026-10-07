import math
import os
import tempfile

import numpy as np
import pandas as pd

from scripts.check9_direct_harm_gate import (
    average_precision_score,
    bootstrap_loro_auc,
    coverage_harm_stats,
    gate_percentile_curve,
    make_direct_labels,
    oracle_and_random_gate,
    permutation_auc_test,
    summarize_predictions,
)


def sample_df():
    return pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(8)],
            "repo": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "coverage_zero_at5": [1, 1, 0, 0, 1, 0, 1, 0],
            "baseline_at1": [1, 0, 1, 0, 1, 0, 0, 1],
            "memory_at1": [0, 1, 1, 0, 0, 1, 0, 1],
            "baseline_at3": [1, 0, 1, 0, 1, 0, 0, 1],
            "memory_at3": [0, 1, 1, 0, 0, 1, 0, 1],
            "baseline_at5": [1, 0, 1, 0, 1, 0, 0, 1],
            "memory_at5": [0, 1, 1, 0, 0, 1, 0, 1],
            "x1": [0.9, 0.2, 0.8, 0.1, 0.85, 0.3, 0.4, 0.7],
            "x2": [2, 5, 2, 4, 2, 5, 4, 2],
        }
    )


def test_make_direct_labels_adds_harm_and_rescue_for_each_k():
    df = make_direct_labels(sample_df())
    assert df["harm_at5"].tolist() == [1, 0, 0, 0, 1, 0, 0, 0]
    assert df["rescue_at5"].tolist() == [0, 1, 0, 0, 0, 1, 0, 0]
    assert int(df["harm_at3"].sum()) == 2
    assert int(df["rescue_at1"].sum()) == 2


def test_average_precision_score_matches_manual_ranking():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1])
    # Positive precisions occur at ranks 1 and 3: (1/1 + 2/3) / 2.
    assert abs(average_precision_score(y, score) - ((1.0 + 2 / 3) / 2)) < 1e-9
    assert math.isnan(average_precision_score([0, 0], [0.1, 0.2]))


def test_coverage_harm_stats_reports_full_2x2_and_association():
    df = make_direct_labels(sample_df())
    stats = coverage_harm_stats(df, "harm_at5", "coverage_zero_at5")
    assert stats["cells"]["coverage_zero_1_harm_1"] == 2
    assert stats["cells"]["coverage_zero_0_harm_1"] == 0
    assert stats["cells"]["coverage_zero_1_harm_0"] == 2
    assert 0 <= stats["cramers_v"] <= 1
    assert stats["mutual_information"] >= 0


def test_gate_percentile_curve_uses_high_risk_baseline_fallback():
    df = make_direct_labels(sample_df())
    pred = df[["case_id", "repo", "harm_at5", "baseline_at5", "memory_at5"]].copy()
    pred["risk_probability"] = [0.9, 0.1, 0.2, 0.1, 0.8, 0.1, 0.1, 0.1]
    rows = gate_percentile_curve(pred, "harm_at5", [0.25], ks=(5,))
    assert rows[0]["gate_acc_at5"] == 0.75
    assert rows[0]["gate_delta_vs_memory_at5"] == 0.25
    assert rows[0]["gated_case_count"] == 2


def test_oracle_and_random_gate_sets_upper_and_baselines():
    df = make_direct_labels(sample_df())
    pred = df[["case_id", "repo", "harm_at5", "baseline_at5", "memory_at5"]].copy()
    pred["risk_probability"] = [0.9, 0.1, 0.2, 0.1, 0.8, 0.1, 0.1, 0.1]
    rows = oracle_and_random_gate(pred, "harm_at5", random_rate=0.25, n_random=20, seed=7, ks=(5,))
    by_name = {row["policy"]: row for row in rows}
    assert by_name["always_memory"]["acc_at5"] == 0.5
    assert by_name["always_baseline"]["acc_at5"] == 0.5
    assert by_name["oracle_harm_gate"]["acc_at5"] == 0.75
    assert "random_gate_mean" in by_name


def test_bootstrap_and_permutation_return_valid_shapes():
    df = make_direct_labels(sample_df())
    feature_cols = ["x1", "x2"]
    boot = bootstrap_loro_auc(df, feature_cols, "harm_at5", "repo", n_boot=5, seed=1)
    assert boot["n_bootstrap_valid"] > 0
    assert len(boot["auc_values"]) == boot["n_bootstrap_valid"]
    perm = permutation_auc_test(df, feature_cols, "harm_at5", "repo", n_perm=5, seed=2)
    assert perm["n_permutations_valid"] > 0
    assert 0 <= perm["p_value"] <= 1


def test_summarize_predictions_includes_pr_auc_and_base_rate():
    df = make_direct_labels(sample_df())
    pred = df[["case_id", "repo", "harm_at5", "baseline_at5", "memory_at5"]].copy()
    pred["risk_probability"] = [0.9, 0.1, 0.2, 0.1, 0.8, 0.1, 0.1, 0.1]
    summary = summarize_predictions(pred, "harm_at5")
    assert summary["positive_count"] == 2
    assert summary["base_rate"] == 0.25
    assert summary["pr_auc"] >= summary["base_rate"]


if __name__ == "__main__":
    tests = [
        test_make_direct_labels_adds_harm_and_rescue_for_each_k,
        test_average_precision_score_matches_manual_ranking,
        test_coverage_harm_stats_reports_full_2x2_and_association,
        test_gate_percentile_curve_uses_high_risk_baseline_fallback,
        test_oracle_and_random_gate_sets_upper_and_baselines,
        test_bootstrap_and_permutation_return_valid_shapes,
        test_summarize_predictions_includes_pr_auc_and_base_rate,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        for test in tests:
            test()
    print(f"{len(tests)} tests passed")
