import json
import math
import os
import tempfile

import pandas as pd

from scripts.check9_label_stability import (
    build_harm_selection,
    compare_label_stability,
    complete_file_acc,
    wilson_interval,
    write_selected_ids,
)


def test_build_harm_selection_selects_all_harm_at5_cases():
    df = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "d"],
            "repo": ["r1", "r1", "r2", "r2"],
            "baseline_at5": [1, 1, 0, 1],
            "memory_at5": [0, 1, 1, 0],
        }
    )

    selected = build_harm_selection(df, k=5, limit=30)

    assert selected["case_id"].tolist() == ["a", "d"]
    assert selected["harm_at5"].tolist() == [1, 1]


def test_write_selected_ids_uses_reproduce_list_format():
    rows = pd.DataFrame({"case_id": ["x", "y"]})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ids.json")
        write_selected_ids(rows, path)
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == ["x", "y"]


def test_complete_file_acc_requires_full_gold_coverage():
    assert complete_file_acc(["a.py", "b.py"], ["a.py"], 1) == 1
    assert complete_file_acc(["a.py"], ["a.py", "b.py"], 5) == 0
    assert complete_file_acc([], ["a.py"], 5) == 0


def test_wilson_interval_is_bounded_and_reasonable():
    lo, hi = wilson_interval(27, 30)
    assert 0 <= lo <= 0.9 <= hi <= 1
    assert not math.isnan(lo)
    assert not math.isnan(hi)


def test_compare_label_stability_counts_flips():
    original = pd.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "repo": ["r1", "r1", "r2"],
            "baseline_at5": [1, 1, 1],
            "memory_at5": [0, 0, 0],
        }
    )
    baseline_rerun = {
        "a": ["gold.py"],
        "b": [],
        "c": ["gold.py"],
    }
    memory_rerun = {
        "a": [],
        "b": [],
        "c": [],
    }
    gold = {
        "a": ["gold.py"],
        "b": ["gold.py"],
        "c": ["gold.py"],
    }

    summary, rows = compare_label_stability(original, baseline_rerun, memory_rerun, gold, k=5)

    assert summary["n_cases"] == 3
    assert summary["stable_count"] == 2
    assert summary["flip_count"] == 1
    assert rows[1]["rerun_harm_at5"] == 0


if __name__ == "__main__":
    tests = [
        test_build_harm_selection_selects_all_harm_at5_cases,
        test_write_selected_ids_uses_reproduce_list_format,
        test_complete_file_acc_requires_full_gold_coverage,
        test_wilson_interval_is_bounded_and_reasonable,
        test_compare_label_stability_counts_flips,
    ]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
