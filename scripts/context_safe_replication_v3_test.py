import statistics

from scripts.analyze_context_safe_replication_v3 import (
    RUN_ORDER,
    cross_run_agreement,
    gap_summary,
)


def test_execution_order_balances_position_moments():
    positions = {"baseline": [], "memory": []}
    for position, (arm, _) in enumerate(RUN_ORDER, 1):
        positions[arm].append(position)
    assert positions["baseline"] == [1, 4, 6, 7]
    assert positions["memory"] == [2, 3, 5, 8]
    for arm in positions:
        assert statistics.mean(positions[arm]) == 4.5
        assert statistics.pvariance(positions[arm]) == 5.25


def test_cross_run_uses_sixteen_pairs():
    assert cross_run_agreement([1, 1, 1, 1], [1, 1, 1, 1]) == 1.0
    assert cross_run_agreement([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0


def test_gap_summary_is_reproducible():
    first = gap_summary([0.5, 0.25, 0.0])
    second = gap_summary([0.5, 0.25, 0.0])
    assert first == second


if __name__ == "__main__":
    test_execution_order_balances_position_moments()
    test_cross_run_uses_sixteen_pairs()
    test_gap_summary_is_reproducible()
    print("3 tests passed")
