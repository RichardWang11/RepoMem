import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import litellm


SUMMARY_SYSTEM_PROMPT = (
    "You are building semantic repository memory for code localization. "
    "Summarize what a source file is responsible for in concise, concrete terms."
)


SUMMARY_USER_PROMPT = """Repository: {repo_name}
File: {file_path}
Recent edit count: {edit_count}

Write a high-level functionality summary for this file. Focus on:
- the file's main responsibility in the repository
- important classes, functions, and workflows
- inputs, outputs, side effects, or integration points
- why this file may be relevant for future bug localization

Do not propose code changes. Keep the summary under 220 words.

Source code:
```python
{source}
```
"""


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_key(repo_name):
    return repo_name.replace("/", "__")


def load_hotspot_files(episodic_index_dir, repo_name, top_k):
    meta_path = Path(episodic_index_dir) / repo_key(repo_name) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing episodic meta file: {meta_path}")
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    files = []
    for item in meta.get("top_python_files_by_commits", []):
        if not item:
            continue
        file_path = item[0]
        edit_count = item[1] if len(item) > 1 else 0
        if file_path.endswith(".py"):
            files.append((file_path, edit_count))
    return files, meta


def is_test_path(file_path):
    parts = file_path.replace("\\", "/").split("/")
    name = parts[-1] if parts else file_path
    return (
        "tests" in parts
        or "test" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "tests.py"
    )


def read_file(repo_dir, file_path, max_chars):
    path = Path(repo_dir) / file_path
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[source truncated]..."
    return text


def summarize_file(repo_name, file_path, edit_count, source, model, temperature, request_timeout):
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SUMMARY_USER_PROMPT.format(
                    repo_name=repo_name,
                    file_path=file_path,
                    edit_count=edit_count,
                    source=source,
                ),
            },
        ],
        temperature=temperature,
        request_timeout=request_timeout,
    )
    return response.choices[0].message.content.strip()


def summarize_hotspot_file(
    repo_name,
    repo_dir,
    file_path,
    edit_count,
    model,
    builder_mode,
    max_source_chars,
    temperature,
    request_timeout,
):
    source = read_file(repo_dir, file_path, max_source_chars)
    if source is None:
        return {
            "repo": repo_name,
            "file": file_path,
            "summary": "",
            "edit_count": edit_count,
            "summary_model": model,
            "builder_mode": builder_mode,
            "status": "missing_file",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        summary = summarize_file(
            repo_name=repo_name,
            file_path=file_path,
            edit_count=edit_count,
            source=source,
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
        )
        status = "ok"
    except Exception as exc:
        summary = ""
        status = f"error:{type(exc).__name__}:{str(exc)[:200]}"

    return {
        "repo": repo_name,
        "file": file_path,
        "summary": summary,
        "edit_count": edit_count,
        "summary_model": model,
        "builder_mode": builder_mode,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_repo_semantic_index(
    repo_name,
    repo_dir,
    episodic_index_dir,
    output_dir,
    top_k,
    model,
    builder_mode,
    max_source_chars,
    temperature,
    request_timeout,
    sleep_seconds,
    overwrite,
    include_tests,
    num_workers,
):
    out_dir = Path(output_dir) / repo_key(repo_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries_path = out_dir / "summaries.jsonl"
    meta_path = out_dir / "meta.json"

    existing_rows = [] if overwrite else load_jsonl(summaries_path)
    existing_by_file = {row.get("file"): row for row in existing_rows if row.get("file")}
    hotspot_files, episodic_meta = load_hotspot_files(episodic_index_dir, repo_name, top_k)
    if not include_tests:
        hotspot_files = [
            (file_path, edit_count)
            for file_path, edit_count in hotspot_files
            if not is_test_path(file_path)
        ]
    hotspot_files = hotspot_files[:top_k]

    rows_by_file = {
        file_path: existing_by_file[file_path]
        for file_path, _ in hotspot_files
        if file_path in existing_by_file
    }

    def flush_rows():
        ordered_rows = [
            rows_by_file[file_path]
            for file_path, _ in hotspot_files
            if file_path in rows_by_file
        ]
        write_jsonl(summaries_path, ordered_rows)
        return ordered_rows

    pending = [
        (file_path, edit_count)
        for file_path, edit_count in hotspot_files
        if file_path not in rows_by_file
    ]
    if rows_by_file:
        flush_rows()
        print(
            f"{repo_name}: resuming semantic build with "
            f"{len(rows_by_file)}/{len(hotspot_files)} summaries already done",
            flush=True,
        )

    workers = max(1, int(num_workers or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                summarize_hotspot_file,
                repo_name,
                repo_dir,
                file_path,
                edit_count,
                model,
                builder_mode,
                max_source_chars,
                temperature,
                request_timeout,
            ): (file_path, edit_count)
            for file_path, edit_count in pending
        }

        for future in as_completed(futures):
            file_path, _ = futures[future]
            rows_by_file[file_path] = future.result()
            rows = flush_rows()
            row = rows_by_file[file_path]
            print(
                f"{repo_name}: summarized {len(rows)}/{len(hotspot_files)} "
                f"{file_path} [{row.get('status')}]",
                flush=True,
            )
            if sleep_seconds:
                time.sleep(sleep_seconds)

    rows = flush_rows()

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "repo": repo_name,
                "repo_dir": str(repo_dir),
                "episodic_index_dir": str(episodic_index_dir),
                "source_num_commits": episodic_meta.get("num_commits"),
                "top_k": top_k,
                "num_summaries": len(rows),
                "num_ok": sum(1 for row in rows if row.get("status") == "ok"),
                "summary_model": model,
                "builder_mode": builder_mode,
                "max_source_chars": max_source_chars,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    print(f"{repo_name}: wrote {len(rows)} summaries to {summaries_path}")


def parse_repo_map(items):
    repo_map = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected REPO=PATH, got {item}")
        repo, path = item.split("=", 1)
        if not os.path.isdir(path):
            raise FileNotFoundError(path)
        repo_map[repo] = path
    return repo_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--episodic_index_dir", required=True)
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        help="Repository mapping in the form owner/name=/path/to/clone.",
    )
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--model", default=os.environ.get("REPOMEM_SUMMARY_MODEL", "openai/glm-5.1"))
    parser.add_argument(
        "--builder_mode",
        choices=["agent", "direct"],
        default="agent",
        help="Recorded construction mode. The current implementation summarizes each hotspot file directly.",
    )
    parser.add_argument("--max_source_chars", type=int, default=24000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request_timeout", type=int, default=120)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include_tests",
        action="store_true",
        help="Include test files in semantic memory. By default, test files are filtered out.",
    )
    args = parser.parse_args()

    repo_map = parse_repo_map(args.repo)
    for repo_name, repo_dir in repo_map.items():
        build_repo_semantic_index(
            repo_name=repo_name,
            repo_dir=repo_dir,
            episodic_index_dir=args.episodic_index_dir,
            output_dir=args.output_dir,
            top_k=args.top_k,
            model=args.model,
            builder_mode=args.builder_mode,
            max_source_chars=args.max_source_chars,
            temperature=args.temperature,
            request_timeout=args.request_timeout,
            sleep_seconds=args.sleep_seconds,
            overwrite=args.overwrite,
            include_tests=args.include_tests,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
