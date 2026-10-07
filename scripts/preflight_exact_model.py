#!/usr/bin/env python3
"""Fail before an experiment if a provider aliases the requested model."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import litellm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--required_response_model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    response = litellm.completion(
        model=args.model,
        messages=[{"role": "user", "content": "Reply exactly: OK"}],
        temperature=0,
        max_tokens=2,
        request_timeout=60,
        num_retries=0,
    )
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": args.model,
        "required_response_model": args.required_response_model,
        "response_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "response_id": getattr(response, "id", None),
        "ok": getattr(response, "model", None) == args.required_response_model,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))
    if not row["ok"]:
        raise RuntimeError(
            f"provider returned {row['response_model']!r}, required "
            f"{args.required_response_model!r}"
        )


if __name__ == "__main__":
    main()
