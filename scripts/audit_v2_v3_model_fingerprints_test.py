from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.audit_v2_v3_model_fingerprints import (
    classify,
    log_fingerprints,
    summarize_version,
)


def summary(fingerprints, trajectory_count=2, response_count=2):
    rows = []
    for index in range(trajectory_count):
        fingerprint = fingerprints[index] if index < len(fingerprints) else None
        rows.append({
            "configured_model": "gpt-4o",
            "response_count": 1 if response_count else 0,
            "missing_response_fingerprint_count": int(not fingerprint),
            "system_fingerprints": [fingerprint] if fingerprint else [],
            "response_models": ["gpt-4o-2024-11-20"] if response_count else [],
            "requested_model": "gpt-4o",
            "_response_model_values": (
                ["gpt-4o-2024-11-20"] if response_count else []
            ),
            "_response_fingerprint_values": [fingerprint] if fingerprint else [],
        })
    return summarize_version(rows, {})


def test_same_snapshot():
    left = summary(["fp_40854b0bd2", "fp_40854b0bd2"])
    right = summary(["fp_40854b0bd2", "fp_40854b0bd2"])
    assert classify(left, right) == "SAME_SNAPSHOT"


def test_different_snapshot():
    left = summary(["fp_11111111", "fp_11111111"])
    right = summary(["fp_40854b0bd2", "fp_40854b0bd2"])
    assert classify(left, right) == "DIFFERENT_SNAPSHOT"


def test_v2_missing_snapshot():
    left = summary([], response_count=0)
    right = summary(["fp_40854b0bd2", "fp_40854b0bd2"])
    assert classify(left, right) == "V2_FINGERPRINT_NOT_RECORDED"


def test_mixed_snapshot_takes_precedence():
    left = summary(["fp_11111111", "fp_22222222"])
    right = summary(["fp_40854b0bd2", "fp_40854b0bd2"])
    assert classify(left, right) == "MIXED_WITHIN_RUN"


def test_partial_coverage_is_insufficient():
    left = summary(["fp_40854b0bd2"])
    right = summary(["fp_40854b0bd2", "fp_40854b0bd2"])
    assert classify(left, right) == "INSUFFICIENT_COVERAGE"


def test_code_symbol_is_not_a_fingerprint():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "localize.log"
        path.write_text(
            "imports sympy/combinatorics/fp_groups.py\n", encoding="utf-8"
        )
        assert log_fingerprints(path) == []


if __name__ == "__main__":
    test_same_snapshot()
    test_different_snapshot()
    test_v2_missing_snapshot()
    test_mixed_snapshot_takes_precedence()
    test_partial_coverage_is_insufficient()
    test_code_symbol_is_not_a_fingerprint()
    print("6 tests passed")
