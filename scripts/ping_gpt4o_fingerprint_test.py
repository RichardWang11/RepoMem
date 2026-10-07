from scripts.ping_gpt4o_fingerprint import classify_ping, sanitize_error


ENDPOINT = "endpoint-hash"


def test_current_match():
    assert classify_ping(
        ENDPOINT, ENDPOINT, None, "gpt-4o-2024-11-20", "fp_40854b0bd2"
    ) == "CURRENT_MATCH"


def test_rotated_snapshot():
    assert classify_ping(
        ENDPOINT, ENDPOINT, None, "gpt-4o-2024-11-20", "fp_other123"
    ) == "SNAPSHOT_ROTATED"


def test_missing_fingerprint():
    assert classify_ping(
        ENDPOINT, ENDPOINT, None, "gpt-4o-2024-11-20", None
    ) == "FINGERPRINT_MISSING"


def test_api_error_is_not_missing_metadata():
    assert classify_ping(
        ENDPOINT, ENDPOINT, {"type": "Timeout"}, None, None
    ) == "API_ERROR"


def test_endpoint_mismatch_takes_precedence():
    assert classify_ping(
        "other", ENDPOINT, {"type": "Timeout"}, None, None
    ) == "ENDPOINT_MISMATCH"


def test_model_mismatch():
    assert classify_ping(
        ENDPOINT, ENDPOINT, None, "gpt-4o-unknown", "fp_40854b0bd2"
    ) == "METADATA_INCONSISTENT"


def test_error_redaction():
    value = sanitize_error("failed https://secret.example/v1 with sk-secret123")
    assert "secret.example" not in value
    assert "sk-secret123" not in value


if __name__ == "__main__":
    test_current_match()
    test_rotated_snapshot()
    test_missing_fingerprint()
    test_api_error_is_not_missing_metadata()
    test_endpoint_mismatch_takes_precedence()
    test_model_mismatch()
    test_error_redaction()
    print("7 tests passed")
