import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm


DEFAULT_MANIFEST = "results/context_safe_m3a/replication_v3/experiment_manifest.json"
DEFAULT_OUTPUT = "results/context_safe_m3a/fingerprint_ping"
EXPECTED_MODEL = "gpt-4o-2024-11-20"
EXPECTED_FINGERPRINT = "fp_40854b0bd2"
URL_RE = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)
SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9_-]+\b")


def endpoint_hash():
    endpoint = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
        "OPENAI_API_BASE"
    )
    return hashlib.sha256((endpoint or "").encode()).hexdigest() if endpoint else None


def sanitize_error(value):
    text = URL_RE.sub("<redacted-url>", str(value))
    return SECRET_RE.sub("<redacted-secret>", text)[:1000]


def classify_ping(
    current_endpoint_hash,
    expected_endpoint_hash,
    api_error,
    response_model,
    fingerprint,
):
    if current_endpoint_hash != expected_endpoint_hash:
        return "ENDPOINT_MISMATCH"
    if api_error:
        return "API_ERROR"
    if not fingerprint:
        return "FINGERPRINT_MISSING"
    if fingerprint != EXPECTED_FINGERPRINT:
        return "SNAPSHOT_ROTATED"
    if response_model != EXPECTED_MODEL:
        return "METADATA_INCONSISTENT"
    return "CURRENT_MATCH"


def get_usage(response):
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    return {
        key: getattr(usage, key, None)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def ping(model, request_timeout):
    started = time.monotonic()
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Reply exactly: OK"}],
        max_tokens=1,
        temperature=0,
        request_timeout=request_timeout,
        num_retries=0,
    )
    elapsed = time.monotonic() - started
    choice = response.choices[0]
    return {
        "response_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": get_usage(response),
        "latency_seconds": elapsed,
    }


def report_text(result):
    verdict = result["verdict"]
    interpretations = {
        "CURRENT_MATCH": (
            "The current endpoint matched V3 at preflight time. This is not "
            "evidence of long-term snapshot stability. V4 may proceed only "
            "with response-level fingerprint locking."
        ),
        "SNAPSHOT_ROTATED": (
            "The current response fingerprint differs from V3. Azure snapshot "
            "rotation is observed; do not start a same-snapshot V4."
        ),
        "FINGERPRINT_MISSING": (
            "The request succeeded, but the response did not expose a system "
            "fingerprint. Snapshot parity is not observable; do not start V4."
        ),
        "ENDPOINT_MISMATCH": (
            "The configured endpoint hash differs from the V3 endpoint hash. "
            "The ping is not comparable to V3."
        ),
        "METADATA_INCONSISTENT": (
            "The fingerprint matches V3, but the response model identifier "
            "does not. Treat snapshot parity as unresolved."
        ),
        "API_ERROR": (
            "The API request failed. This is not evidence that fingerprint "
            "metadata was removed."
        ),
    }[verdict]
    response = result.get("response", {})
    return f"""# Azure GPT-4o Fingerprint Sanity Ping

## Verdict

**{verdict}**

{interpretations}

## Evidence

| Field | Value |
|---|---|
| UTC timestamp | {result['timestamp_utc']} |
| Requested model | {result['requested_model']} |
| Response model | {response.get('response_model') or 'not returned'} |
| System fingerprint | {response.get('system_fingerprint') or 'not returned'} |
| Endpoint hash matches V3 | {result['endpoint_hash_matches_v3']} |
| Latency seconds | {format_latency(response.get('latency_seconds'))} |
| Finish reason | {response.get('finish_reason') or 'not returned'} |

## Boundary

This single request establishes only whether the endpoint matched V3 at the
recorded preflight time. It does not establish temporal stability. A V4 run
must record and enforce `fp_40854b0bd2` for every response.
"""


def format_latency(value):
    return "not available" if value is None else f"{value:.3f}"


def write_outputs(output_dir, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp_utc"].replace(":", "").replace("-", "")
    stamp = stamp.replace("+0000", "Z").replace("+00:00", "Z")
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    timestamped = output_dir / f"fingerprint_ping_{stamp}.json"
    timestamped.write_text(payload, encoding="utf-8")
    (output_dir / "fingerprint_ping_latest.json").write_text(
        payload, encoding="utf-8"
    )
    (output_dir / "fingerprint_ping_report.md").write_text(
        report_text(result), encoding="utf-8"
    )
    return timestamped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--request_timeout", type=int, default=60)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected_endpoint_hash = manifest["request"]["endpoint_sha256"]
    current_endpoint_hash = endpoint_hash()
    timestamp = datetime.now(timezone.utc).isoformat()
    response = {}
    error = None
    if current_endpoint_hash == expected_endpoint_hash:
        try:
            response = ping(args.model, args.request_timeout)
        except Exception as exc:
            error = {
                "type": exc.__class__.__name__,
                "message": sanitize_error(exc),
                "status_code": getattr(exc, "status_code", None),
            }

    verdict = classify_ping(
        current_endpoint_hash,
        expected_endpoint_hash,
        error,
        response.get("response_model"),
        response.get("system_fingerprint"),
    )
    result = {
        "verdict": verdict,
        "timestamp_utc": timestamp,
        "requested_model": args.model,
        "expected": {
            "response_model": EXPECTED_MODEL,
            "system_fingerprint": EXPECTED_FINGERPRINT,
            "endpoint_sha256": expected_endpoint_hash,
        },
        "current_endpoint_sha256": current_endpoint_hash,
        "endpoint_hash_matches_v3": current_endpoint_hash == expected_endpoint_hash,
        "response": response,
        "error": error,
        "request": {
            "message": "Reply exactly: OK",
            "max_tokens": 1,
            "temperature": 0,
            "request_timeout": args.request_timeout,
            "automatic_retries": 0,
        },
        "credentials_recorded": False,
    }
    path = write_outputs(args.output_dir, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {path}")
    if verdict in {"API_ERROR", "ENDPOINT_MISMATCH"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
