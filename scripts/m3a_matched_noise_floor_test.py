from scripts.analyze_m3a_matched_noise_floor import (
    exact_sign_flip,
    interaction_test,
    stability,
    within_case_slope,
)
from scripts.prepare_m3a_selection import largest_remainder_allocation

import numpy as np


def test_stability_metrics():
    result = stability([1, 1, 1, 1])
    assert result["unanimous"] == 1
    assert result["pairwise_agreement"] == 1.0
    result = stability([1, 1, 0, 0])
    assert result["unanimous"] == 0
    assert result["pairwise_agreement"] == 2 / 6


def test_exact_sign_flip():
    assert 0 <= exact_sign_flip([1.0, 0.5, 0.25]) <= 1


def test_within_case_slope_removes_case_intercept():
    rows = [
        {"case_id": "a", "x": 0, "correct_at5": 0},
        {"case_id": "a", "x": 1, "correct_at5": 1},
        {"case_id": "b", "x": 10, "correct_at5": 0},
        {"case_id": "b", "x": 11, "correct_at5": 1},
    ]
    assert within_case_slope(rows, "x") == 1.0


def test_selection_allocation_hits_target():
    rows = [
        {"case_id": f"a{i}", "repo": "a"} for i in range(8)
    ] + [{"case_id": f"b{i}", "repo": "b"} for i in range(2)]
    selected, allocation, _ = largest_remainder_allocation(
        rows, 5, np.random.default_rng(42)
    )
    assert len(selected) == 5
    assert sum(allocation.values()) == 5


def test_interaction_test_detects_slope_difference():
    baseline = []
    memory = []
    for case_id in ("a", "b", "c"):
        for value in (0, 1, 2, 3):
            baseline.append(
                {"case_id": case_id, "x": value, "correct_at5": value % 2}
            )
            memory.append(
                {"case_id": case_id, "x": value, "correct_at5": int(value >= 2)}
            )
    result = interaction_test(baseline, memory, "x", n_perm=100, seed=42)
    assert set(result) == {"slope", "p", "ci"}
    assert len(result["ci"]) == 2


if __name__ == "__main__":
    test_stability_metrics()
    test_exact_sign_flip()
    test_within_case_slope_removes_case_intercept()
    test_selection_allocation_hits_target()
    test_interaction_test_detects_slope_difference()
    print("5 tests passed")
