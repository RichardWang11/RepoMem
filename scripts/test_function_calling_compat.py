#!/usr/bin/env python
import argparse
import json
import os
import re
import time

import litellm


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_code_snippets",
            "description": "Search code snippets in a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["search_terms"],
            },
        },
    }
]

PROMPTS = [
    "Use the tool to search for django migration writer.",
    "Use the tool to search for astropy FITS header card parsing.",
    "Use the tool to search for model field deconstruction in Django.",
]


TEXT_TOOL_RE = re.compile(
    r"(<execute_ipython>|search_code_snippets\s*\(|\"name\"\s*:\s*\"search_code_snippets\")",
    re.I,
)


def message_to_dict(msg):
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "dict"):
        return msg.dict()
    return dict(msg)


def run_one(model, prompt, timeout):
    start = time.time()
    resp = litellm.completion(
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
        api_base=os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"),
        messages=[{"role": "user", "content": prompt}],
        tools=TOOLS,
        temperature=0,
        timeout=timeout,
    )
    elapsed = time.time() - start
    msg = resp.choices[0].message
    data = message_to_dict(msg)
    tool_calls = data.get("tool_calls") or []
    content = data.get("content") or ""
    return {
        "ok": True,
        "elapsed_sec": round(elapsed, 2),
        "has_tool_calls": bool(tool_calls),
        "tool_call_count": len(tool_calls),
        "tool_names": [
            tc.get("function", {}).get("name")
            for tc in tool_calls
            if isinstance(tc, dict)
        ],
        "has_text_tool_call": bool(TEXT_TOOL_RE.search(content)),
        "content_preview": content[:240].replace("\n", "\\n"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    for model in args.models:
        model_rows = []
        for prompt in PROMPTS:
            try:
                row = run_one(model, prompt, args.timeout)
            except Exception as exc:
                row = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                }
            row["model"] = model
            row["prompt"] = prompt
            model_rows.append(row)
            rows.append(row)

        total = len(model_rows)
        ok = sum(r["ok"] for r in model_rows)
        tool = sum(r.get("has_tool_calls", False) for r in model_rows)
        text_tool = sum(r.get("has_text_tool_call", False) for r in model_rows)
        fail = total - ok
        avg_latency = (
            sum(r.get("elapsed_sec", 0) for r in model_rows if r["ok"]) / ok
            if ok
            else 0
        )
        print(
            f"{model}\ttool_calls={tool}/{total}\ttext_tool={text_tool}/{total}"
            f"\tfail={fail}/{total}\tavg_latency={avg_latency:.2f}s"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved detail rows to {args.out}")


if __name__ == "__main__":
    main()
