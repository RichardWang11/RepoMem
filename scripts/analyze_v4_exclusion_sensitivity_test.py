from scripts.analyze_v4_exclusion_sensitivity import leave_one_out, summarize


def row(case_id, v3_gap, v4_gap):
    return {
        "case_id": case_id,
        "flip_class": "test",
        "v3_baseline_stability": 1.0,
        "v3_memory_stability": 1.0 - v3_gap,
        "v3_gap": v3_gap,
        "v4_baseline_stability": 1.0,
        "v4_memory_stability": 1.0 - v4_gap,
        "v4_gap": v4_gap,
        "v4_minus_v3_gap": v4_gap - v3_gap,
    }


def test_summarize_and_leave_one_out():
    rows = [row("a", 0.0, 0.5), row("b", 0.0, 0.0)]
    assert summarize(rows)["v4_gap"]["gap"] == 0.25
    influence = {item["excluded_case"]: item for item in leave_one_out(rows)}
    assert influence["a"]["remaining_v4_gap"] == 0.0
    assert influence["a"]["v4_gap_change"] == -0.25


if __name__ == "__main__":
    test_summarize_and_leave_one_out()
    print("1 test passed")
