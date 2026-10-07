import json
import os
import tempfile

from scripts.analyze_q4_guard_off_baseline import classify, compare
from scripts.validate_q4_guard_off_repeat import (
    GUARD_ARGUMENTS,
    MATCHED_ARGUMENTS,
    validate_run,
)


def test_guard_effect_classification():
    guard_on = {"reference_agreement": 0.78, "ci": [0.01, 0.20]}
    original = {"reference_agreement": 0.90}
    assert classify(0.88, guard_on, original) == "CONSISTENT_WITH_GUARD_POLICY_EFFECT"


def test_sampling_or_drift_classification():
    guard_on = {"reference_agreement": 0.79, "ci": [-0.1, 0.1]}
    original = {"reference_agreement": 0.90}
    assert classify(0.80, guard_on, original) == "CONSISTENT_WITH_SAMPLING_OR_ENDPOINT_DRIFT"


def test_comparison_is_candidate_minus_reference():
    reference = [{"case_id": "a", "pairwise_agreement": 0.5}]
    candidate = [{"case_id": "a", "pairwise_agreement": 1.0}]
    result = compare(reference, candidate, "ref")
    assert result["q4_minus_reference"] == 0.5


def test_guard_off_validator():
    with tempfile.TemporaryDirectory() as run_dir:
        args = {**MATCHED_ARGUMENTS, **GUARD_ARGUMENTS, "use_repomem": False}
        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args, f)
        row = {
            "instance_id": "case",
            "loc_trajs": {
                "trajs": [{"context_guard": {"enabled": False, "events": []}}]
            },
        }
        with open(os.path.join(run_dir, "loc_trajs.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        result = validate_run(run_dir, expected_cases=1)
        assert result["guard_enabled"] is False


if __name__ == "__main__":
    test_guard_effect_classification()
    test_sampling_or_drift_classification()
    test_comparison_is_candidate_minus_reference()
    test_guard_off_validator()
    print("4 tests passed")
