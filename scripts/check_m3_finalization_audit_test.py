from scripts.check_m3_finalization_audit import (
    exact_label_permutation,
    extract_trajectory,
    flatten_files,
    holm_adjust,
    subtract_metrics,
)


def trajectory_row(messages, found_files=None):
    return {
        "instance_id": "x__x-1",
        "found_files": found_files or [],
        "loc_trajs": {"trajs": [{"messages": messages}]},
    }


def assistant_call(name):
    return {
        "role": "assistant",
        "content": "final" if name == "finish" else None,
        "tool_calls": [{"function": {"name": name, "arguments": "{}"}}],
    }


def test_extracts_finish_and_verification_depth():
    row = trajectory_row(
        [
            assistant_call("search_code_snippets"),
            {"role": "tool", "content": "result"},
            assistant_call("get_entity_contents"),
            assistant_call("finish"),
        ],
        [["django/db/models/query.py"]],
    )
    result = extract_trajectory(row, max_iterations=8)
    assert result["assistant_iterations"] == 3
    assert result["finish_at_iteration"] == 3
    assert result["finished_before_max"] == 1
    assert result["code_verify_call_count"] == 2
    assert result["final_output_parseable"] == 1


def test_forced_prompt_and_no_finish_use_limit():
    row = trajectory_row(
        [
            assistant_call("search_code_snippets"),
            {
                "role": "user",
                "content": "The Maximum number of interation has been reached",
            },
            {"role": "assistant", "content": "answer", "tool_calls": []},
        ]
    )
    result = extract_trajectory(row, max_iterations=2)
    assert result["finish_at_iteration"] is None
    assert result["effective_finish_iteration"] == 2
    assert result["finished_before_max"] == 0
    assert result["forced_finalization_prompt_seen"] == 1


def test_commit_ratio_is_missing_without_search():
    row = trajectory_row([assistant_call("examine_commit"), assistant_call("finish")])
    assert extract_trajectory(row, 8)["commit_examine_ratio"] is None


def test_multiple_tool_calls_are_counted():
    row = trajectory_row(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "search_commit", "arguments": "{}"}},
                    {"function": {"name": "examine_commit", "arguments": "{}"}},
                ],
            }
        ]
    )
    result = extract_trajectory(row, 8)
    assert result["total_tool_calls"] == 2
    assert result["commit_examine_ratio"] == 1.0


def test_helpers_are_deterministic():
    assert flatten_files([["a.py"], "b.py"]) == ["a.py", "b.py"]
    gap, pvalue = exact_label_permutation([3, 4], [0, 1])
    assert gap == 3.0
    assert 0 <= pvalue <= 1
    assert holm_adjust([0.01, 0.04]) == [0.02, 0.04]
    assert subtract_metrics(
        {"total_tool_calls": 3, "finished_before_max": 1},
        {"total_tool_calls": 1, "finished_before_max": 0},
    )["total_tool_calls"] == 2


if __name__ == "__main__":
    test_extracts_finish_and_verification_depth()
    test_forced_prompt_and_no_finish_use_limit()
    test_commit_ratio_is_missing_without_search()
    test_multiple_tool_calls_are_counted()
    test_helpers_are_deterministic()
    print("5 tests passed")
