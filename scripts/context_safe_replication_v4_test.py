from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.analyze_context_safe_replication_v4 import RUN_ORDER, point_verdict
from scripts.validate_replication_v4 import (
    audit_trajectory,
    validate_preflight,
)


def test_run_order_matches_v3_position_balance():
    positions = {"baseline": [], "memory": []}
    for position, (arm, _) in enumerate(RUN_ORDER, 1):
        positions[arm].append(position)
    assert positions == {"baseline": [1, 4, 6, 7], "memory": [2, 3, 5, 8]}
    for values in positions.values():
        assert sum(values) / len(values) == 4.5


def test_fingerprint_audit_marks_match_and_drift():
    trajectory = {
        "provider_metadata": {
            "responses": [
                {
                    "model": "gpt-4o-2024-11-20",
                    "system_fingerprint": "fp_40854b0bd2",
                },
                {
                    "model": "gpt-4o-2024-11-20",
                    "system_fingerprint": "fp_rotated123",
                },
            ]
        }
    }
    rows = audit_trajectory("case", trajectory, "memory", 1)
    assert rows[0]["fingerprint_matches"]
    assert not rows[1]["fingerprint_matches"]


def test_fingerprint_audit_marks_missing():
    trajectory = {
        "provider_metadata": {
            "responses": [{"model": "gpt-4o-2024-11-20"}]
        }
    }
    row = audit_trajectory("case", trajectory, "baseline", 1)[0]
    assert not row["fingerprint_present"]
    assert not row["fingerprint_matches"]


def test_point_verdict_boundaries():
    assert point_verdict(0.05, 0.03) == "SMALL_GAP_REPLICATED"
    assert point_verdict(0.20, 0.18) == "LARGE_WITHIN_SNAPSHOT_VARIANCE"
    assert point_verdict(0.40, 0.38) == "EXTREME_WITHIN_SNAPSHOT_VARIANCE"
    assert point_verdict(-0.10, -0.12) == "DIRECTION_REVERSAL"
    assert point_verdict(0.12, 0.10) == "INCONCLUSIVE_INTERMEDIATE"


def test_preflight_age_and_metadata():
    with TemporaryDirectory() as directory:
        now = datetime.now(timezone.utc)
        path = Path(directory) / "ping.json"
        path.write_text(
            "{\n"
            '  "verdict": "CURRENT_MATCH",\n'
            f'  "timestamp_utc": "{(now - timedelta(minutes=1)).isoformat()}",\n'
            '  "endpoint_hash_matches_v3": true,\n'
            '  "response": {"response_model": "gpt-4o-2024-11-20", '
            '"system_fingerprint": "fp_40854b0bd2"}\n'
            "}\n",
            encoding="utf-8",
        )
        assert validate_preflight(path, now=now)["valid"]


if __name__ == "__main__":
    test_run_order_matches_v3_position_balance()
    test_fingerprint_audit_marks_match_and_drift()
    test_fingerprint_audit_marks_missing()
    test_point_verdict_boundaries()
    test_preflight_age_and_metadata()
    print("5 tests passed")
