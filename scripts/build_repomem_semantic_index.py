#!/usr/bin/env python3
"""Build a provenance-locked RepoMem-compatible semantic-memory bank."""

import argparse
import hashlib
import json
import os
import queue
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import litellm


SCHEMA_VERSION = 3
SUMMARY_SYSTEM_PROMPT = (
    "You are building semantic repository memory for code localization. "
    "Summarize what a source file is responsible for in concise, concrete terms."
)
DIRECT_USER_PROMPT = """Repository: {repo_name}
File: {file_path}
Recent edit count: {edit_count}

Write a high-level functionality summary for this file. Focus on:
- the file's main responsibility in the repository
- important classes, functions, and workflows
- inputs, outputs, side effects, or integration points
- why this file may be relevant for future bug localization

Do not propose code changes. Keep the summary concise and high-level.

Source code:
```python
{source}
```
"""
AGENT_USER_PROMPT = (
    "Summarize the high-level functionality and core responsibilities of `{file_path}`. "
    "Inspect the target file directly. If its local context is insufficient, actively "
    "explore relevant dependencies in other files. Return only the file functionality "
    "summary."
)
AGENT_FAKE_USER = "Continue inspecting the target or its dependencies, then finish with only the requested summary."
# The Django pilot reached 24,055 tokens while inspecting a large core file.
# Keep a modest guard below the provider limit without rejecting valid agentic
# dependency exploration.
AGENT_CONTEXT_TOKEN_BUDGET = 32000
AGENT_OBSERVATION_TOKEN_CAP = 4000
# Summaries are stored exactly as returned by the builder after surrounding
# whitespace is stripped.  Keep this compatibility name for older helper
# scripts, but a null value explicitly means that no word cap is applied.
SUMMARY_MAX_WORDS = None


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path):
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    rows = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append([str(child.relative_to(path)), sha256_file(child)])
    return sha256_text(canonical_json(rows))


def truncate_summary_words(text, max_words=SUMMARY_MAX_WORDS):
    """Compatibility helper; the Verified static replication does not truncate."""
    if max_words is None:
        return (text or "").strip()
    words = (text or "").split()
    return " ".join(words[:max_words])


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_key(repo_name):
    return repo_name.replace("/", "__")


def load_hotspot_files(episodic_index_dir, repo_name, top_k):
    meta_path = Path(episodic_index_dir) / repo_key(repo_name) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing episodic meta file: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = []
    for item in meta.get("top_python_files_by_commits", []):
        if item and item[0].endswith(".py"):
            files.append((item[0], item[1] if len(item) > 1 else 0))
    return files[:top_k], meta


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


def read_file(repo_dir, file_path, max_chars, source_commit=None):
    if source_commit:
        process = subprocess.run(
            ["git", "-C", str(repo_dir), "show", f"{source_commit}:{file_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        if process.returncode:
            return None
        text = process.stdout
    else:
        path = Path(repo_dir) / file_path
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + "\n...[source truncated]..." if len(text) > max_chars else text


def response_metadata(response, requested_model, temperature, seed):
    usage = getattr(response, "usage", None)
    return {
        "requested_model": requested_model,
        "response_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "temperature": temperature,
        "seed": seed,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def preflight_provider(model, required_response_model, seed, request_timeout):
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Reply exactly: OK"}],
        temperature=0,
        seed=seed,
        max_tokens=2,
        request_timeout=min(request_timeout, 60),
        num_retries=0,
    )
    response_model = getattr(response, "model", None)
    if required_response_model and response_model != required_response_model:
        raise RuntimeError(
            f"preflight response model {response_model!r} != required {required_response_model!r}"
        )
    print(f"provider preflight ok: requested={model} response={response_model}", flush=True)


def summarize_direct(repo_name, file_path, edit_count, source, model, temperature, seed, request_timeout):
    prompt = DIRECT_USER_PROMPT.format(
        repo_name=repo_name, file_path=file_path, edit_count=edit_count, source=source
    )
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "request_timeout": request_timeout,
    }
    if seed is not None:
        kwargs["seed"] = seed
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content.strip(), response_metadata(
        response, model, temperature, seed
    ), None


def inspected_target(messages, file_path):
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            function = call.get("function", {})
            if function.get("name") != "get_entity_contents":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if any(
                entity == file_path or entity.startswith(file_path + ":")
                for entity in arguments.get("entity_names", [])
            ):
                return True
    return False


def summarize_agent(repo_name, file_path, model, temperature, seed, request_timeout, max_iterations):
    from auto_search_main import auto_search_process
    from util.runtime import function_calling

    prompt = AGENT_USER_PROMPT.format(repo_name=repo_name, file_path=file_path)
    result_queue = queue.Queue()
    auto_search_process(
        result_queue=result_queue,
        model_name=model,
        messages=[
            {"role": "system", "content": function_calling.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        fake_user_msg=AGENT_FAKE_USER,
        tools=function_calling.get_tools(
            codeact_enable_search_keyword=True,
            codeact_enable_search_entity=True,
            codeact_enable_tree_structure_traverser=True,
        ),
        temp=temperature,
        sampling_seed=seed,
        max_iteration_num=max_iterations,
        request_timeout=request_timeout,
        use_function_calling=True,
        context_token_budget=AGENT_CONTEXT_TOKEN_BUDGET,
        tool_observation_token_cap=AGENT_OBSERVATION_TOKEN_CAP,
        fail_on_context_error=True,
        bounded_graph_output=True,
        bounded_entity_output=True,
        bounded_search_output=True,
    )
    result = result_queue.get_nowait()
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(f"{result.get('type')}: {result['error']}")
    summary, _, trajectory = result
    if not inspected_target(trajectory["messages"], file_path):
        raise RuntimeError("agent did not inspect the exact target file")
    provider = trajectory.get("provider_metadata", {})
    responses = provider.get("responses", [])
    response_models = sorted({row.get("model") for row in responses if row.get("model")})
    fingerprints = sorted({row.get("system_fingerprint") for row in responses if row.get("system_fingerprint")})
    metadata = {
        "requested_model": model,
        "response_model": responses[-1].get("model") if responses else None,
        "system_fingerprint": responses[-1].get("system_fingerprint") if responses else None,
        "response_models": response_models,
        "system_fingerprints": fingerprints,
        "temperature": temperature,
        "seed": seed,
        "prompt_tokens": trajectory["usage"].get("total_prompt_tokens", 0),
        "completion_tokens": trajectory["usage"].get("total_completion_tokens", 0),
    }
    return summary.strip(), metadata, trajectory


def make_build_spec(
    repo_name,
    repo_dir,
    manifest,
    model,
    builder_mode,
    max_source_chars,
    temperature,
    seed,
    request_timeout,
    max_iterations,
    retrieval_profile,
    graph_index,
    code_bm25_index,
    required_response_model,
):
    prompt = AGENT_USER_PROMPT if builder_mode == "agent" else DIRECT_USER_PROMPT
    repo_root = Path(__file__).resolve().parents[1]
    source_files = [
        Path(__file__),
        repo_root / "auto_search_main.py",
        repo_root / "plugins/location_tools/repo_ops/repo_ops.py",
    ]
    source_hashes = {
        str(path.relative_to(repo_root)): sha256_file(path)
        for path in source_files if path.exists()
    }
    index_manifest_path = (
        Path(graph_index).parent.parent / "index_manifest.json" if graph_index else None
    )
    agent_tools = []
    system_prompt = SUMMARY_SYSTEM_PROMPT
    if builder_mode == "agent":
        from util.runtime import function_calling
        agent_tools = function_calling.get_tools(
            codeact_enable_search_keyword=True,
            codeact_enable_search_entity=True,
            codeact_enable_tree_structure_traverser=True,
        )
        system_prompt = function_calling.SYSTEM_PROMPT
    spec = {
        "schema_version": SCHEMA_VERSION,
        "reproduction_label": "RepoMem-compatible Semantic Memory replication",
        "repo": repo_name,
        "repo_dir": str(Path(repo_dir).resolve()),
        "bank_type": "static",
        "source_commit": manifest["source_commit_sha"],
        "source_tree": manifest["source_tree_sha"],
        "commit_window": manifest["commit_window"],
        "window_commit_count": manifest["window_commit_count"],
        "window_commits_sha256": manifest["window_commits_sha256"],
        "membership_sha256": manifest["membership_sha256"],
        "union_membership_sha256": sha256_text(canonical_json(manifest["files"])),
        "selected_count": manifest["selected_count"],
        "include_tests": manifest["include_tests"],
        "selection_order": manifest["selection_order"],
        "membership_files": manifest["files"],
        "builder_mode": builder_mode,
        "requested_model": model,
        "required_response_model": required_response_model,
        "temperature": temperature,
        "seed": seed,
        "top_p": None,
        "request_timeout": request_timeout,
        "max_iterations": max_iterations if builder_mode == "agent" else None,
        "max_source_chars": max_source_chars if builder_mode == "direct" else None,
        "system_prompt": system_prompt,
        "user_prompt_template": prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(prompt),
        "summary_normalization": "strip_surrounding_whitespace_only",
        "summary_truncation": None,
        "agent_tools": agent_tools,
        "agent_tools_sha256": sha256_text(canonical_json(agent_tools)),
        "agent_runtime": {
            "context_token_budget": AGENT_CONTEXT_TOKEN_BUDGET,
            "tool_observation_token_cap": AGENT_OBSERVATION_TOKEN_CAP,
            "bounded_graph_output": True,
            "bounded_entity_output": True,
            "bounded_search_output": True,
            "fail_on_context_error": True,
        } if builder_mode == "agent" else None,
        "retrieval": {
            "profile": retrieval_profile,
            "primary_document": "summary_only",
            "sensitivity_document": "path_plus_summary",
            "scorer": "RepoMem BM25",
        },
        "graph_index": str(Path(graph_index).resolve()) if graph_index else None,
        "graph_index_sha256": sha256_path(graph_index) if graph_index else None,
        "code_bm25_index": str(Path(code_bm25_index).resolve()) if code_bm25_index else None,
        "code_bm25_index_sha256": sha256_path(code_bm25_index) if code_bm25_index else None,
        "index_manifest_sha256": sha256_file(index_manifest_path) if index_manifest_path else None,
        "source_code_sha256": source_hashes,
    }
    spec["build_id"] = sha256_text(canonical_json(spec))
    return spec


def prompt_instance_hash(build_id, file_path, source_blob_sha):
    return sha256_text(canonical_json([build_id, file_path, source_blob_sha]))


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
    source_commit=None,
    source_blob_sha=None,
    build_id=None,
    seed=44,
    max_iterations=4,
    trajectory_dir=None,
    required_response_model=None,
):
    created_at = datetime.now(timezone.utc).isoformat()
    memory_id = sha256_text(f"{repo_name}\0{file_path}\0{source_blob_sha}")
    instance_hash = prompt_instance_hash(build_id, file_path, source_blob_sha)
    base = {
        "memory_id": memory_id,
        "repo": repo_name,
        "file": file_path,
        "edit_count": edit_count,
        "summary": "",
        "summary_text": "",
        "summary_sha256": None,
        "summary_model": model,
        "requested_model": model,
        "response_model": None,
        "system_fingerprint": None,
        "temperature": temperature,
        "seed": seed,
        "builder_mode": builder_mode,
        "status": "missing_file",
        "source_commit": source_commit,
        "source_tree": None,
        "source_blob_sha": source_blob_sha,
        "prompt_instance_sha256": instance_hash,
        "build_id": build_id,
        "trajectory_path": None,
        "trajectory_sha256": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "created_at": created_at,
    }
    source = None
    if builder_mode == "direct":
        source = read_file(repo_dir, file_path, max_source_chars, source_commit)
        if source is None:
            return base
    try:
        if builder_mode == "agent":
            summary, metadata, trajectory = summarize_agent(
                repo_name, file_path, model, temperature, seed, request_timeout, max_iterations
            )
        else:
            summary, metadata, trajectory = summarize_direct(
                repo_name, file_path, edit_count, source, model, temperature, seed, request_timeout
            )
        observed_models = metadata.get("response_models") or [metadata.get("response_model")]
        if required_response_model and set(observed_models) != {required_response_model}:
            raise RuntimeError(
                f"response models {observed_models!r} != required {required_response_model!r}"
            )
        summary = truncate_summary_words(summary)
        base.update(metadata)
        base.update({
            "summary": summary,
            "summary_text": summary,
            "summary_sha256": sha256_text(summary),
            "status": "ok",
        })
        if trajectory is not None:
            trajectory_path = Path(trajectory_dir) / f"{memory_id}.json"
            write_json(trajectory_path, trajectory)
            base["trajectory_path"] = str(Path("trajectories") / trajectory_path.name)
            base["trajectory_sha256"] = sha256_file(trajectory_path)
    except Exception as exc:
        base["status"] = f"error:{type(exc).__name__}:{str(exc)[:300]}"
    return base


def membership_label(manifest):
    return "include_tests" if manifest.get("include_tests") else "exclude_tests"


def write_corpora(out_dir, rows, retrieval_profile, manifests):
    from plugins.location_tools.repo_ops import repo_ops

    corpora_dir = Path(out_dir) / "corpora"
    corpora_dir.mkdir(exist_ok=True)
    outputs = {}
    old_profile = os.environ.get("REPOMEM_BM25_PROFILE")
    os.environ["REPOMEM_BM25_PROFILE"] = retrieval_profile
    try:
        rows_by_file = {row["file"]: row for row in rows}
        for manifest in manifests:
            label = membership_label(manifest)
            for variant in ("summary_only", "path_plus_summary"):
                corpus = []
                for member in manifest["files"]:
                    row = rows_by_file.get(member["file"])
                    if not row or row.get("status") != "ok" or not row.get("summary_text", "").strip():
                        continue
                    text = row["summary_text"] if variant == "summary_only" else f"{row['file']} {row['summary_text']}"
                    corpus.append({
                        "memory_id": row["memory_id"],
                        "file": row["file"],
                        "document": text,
                        "tokens": repo_ops._repomem_tokenize(text),
                    })
                key = f"{label}__{variant}"
                path = corpora_dir / f"{key}.jsonl"
                write_jsonl(path, corpus)
                outputs[key] = {
                    "path": str(Path("corpora") / path.name),
                    "sha256": sha256_file(path),
                    "document_count": len(corpus),
                    "membership_sha256": manifest["membership_sha256"],
                    "include_tests": manifest["include_tests"],
                }
    finally:
        if old_profile is None:
            os.environ.pop("REPOMEM_BM25_PROFILE", None)
        else:
            os.environ["REPOMEM_BM25_PROFILE"] = old_profile
    return outputs


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
    membership_manifest=None,
    seed=44,
    max_iterations=4,
    required_response_model=None,
    retrieval_profile="bm25-code-v2",
    graph_index=None,
    code_bm25_index=None,
    pilot_limit=0,
    retry_errors=False,
):
    if not membership_manifest:
        raise ValueError("--membership_manifest is required for schema-v3 reproducible builds")
    manifest_paths = membership_manifest if isinstance(membership_manifest, (list, tuple)) else [membership_manifest]
    manifests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in manifest_paths]
    if any(item.get("repo") != repo_name or item.get("bank_type") != "static" for item in manifests):
        raise ValueError("each membership manifest must describe this repo's static bank")
    if len({membership_label(item) for item in manifests}) != len(manifests):
        raise ValueError("membership manifests must have distinct include_tests policies")
    common_snapshot = {(item["source_commit_sha"], item["source_tree_sha"]) for item in manifests}
    if len(common_snapshot) != 1:
        raise ValueError("all membership manifests must use the same source snapshot")
    if len(manifests) == 1 and include_tests != bool(manifests[0].get("include_tests")):
        raise ValueError("--include_tests must match membership manifest include_tests")
    manifest = dict(manifests[0])
    union_by_file = {}
    for item in manifests:
        for row in item["files"]:
            prior = union_by_file.setdefault(row["file"], row)
            if prior["source_blob_sha"] != row["source_blob_sha"]:
                raise ValueError(f"membership blob disagreement for {row['file']}")
    manifest["files"] = list(union_by_file.values())
    manifest["selected_count"] = len(manifest["files"])
    manifest["include_tests"] = None
    manifest["membership_variants"] = {
        membership_label(item): {
            "membership_sha256": item["membership_sha256"],
            "selected_count": item["selected_count"],
            "include_tests": item["include_tests"],
        }
        for item in manifests
    }
    if builder_mode == "agent":
        if not graph_index or not code_bm25_index:
            raise ValueError("agent builder requires --graph_index and --code_bm25_index")
        if not Path(graph_index).is_file() or not Path(code_bm25_index).is_dir():
            raise FileNotFoundError("agent graph/code-BM25 indexes do not exist")
        if Path(graph_index).stem != Path(code_bm25_index).name:
            raise ValueError("graph and code-BM25 indexes must use the same instance id")
        index_root = Path(graph_index).parent.parent
        if index_root != Path(code_bm25_index).parent.parent:
            raise ValueError("graph and code-BM25 indexes must share one immutable index root")
        index_manifest_path = index_root / "index_manifest.json"
        if not index_manifest_path.is_file():
            raise FileNotFoundError("agent indexes require index_manifest.json")
        index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        if (
            index_manifest.get("source_commit") != manifest["source_commit_sha"]
            or index_manifest.get("source_tree") != manifest["source_tree_sha"]
            or index_manifest.get("graph_index_sha256") != sha256_file(graph_index)
            or index_manifest.get("code_bm25_index_sha256") != sha256_path(code_bm25_index)
        ):
            raise ValueError("agent indexes do not match the semantic source snapshot")
        if num_workers != 1:
            raise ValueError("agent builder requires --num_workers 1 because LocAgent state is process-global")
        os.environ["GRAPH_INDEX_DIR"] = str(Path(graph_index).parent)
        os.environ["BM25_INDEX_DIR"] = str(Path(code_bm25_index).parent)

    out_dir = Path(output_dir) / repo_key(repo_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = out_dir / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    spec = make_build_spec(
        repo_name, repo_dir, manifest, model, builder_mode, max_source_chars,
        temperature, seed, request_timeout, max_iterations, retrieval_profile,
        graph_index, code_bm25_index, required_response_model,
    )
    spec_path = out_dir / "build_spec.json"
    existing = [] if overwrite else load_jsonl(out_dir / "summaries.jsonl")
    if spec_path.exists() and not overwrite:
        existing_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if existing_spec.get("build_id") != spec["build_id"]:
            if retry_errors and existing and not any(row.get("status") == "ok" for row in existing):
                existing = []
            else:
                raise ValueError("existing build_spec.json differs; use a new output directory or --overwrite")

    hotspot_files = [(row["file"], row["edit_count"]) for row in manifest["files"]]
    if pilot_limit:
        hotspot_files = hotspot_files[:pilot_limit]
    blob_by_file = {row["file"]: row["source_blob_sha"] for row in manifest["files"]}
    rows_by_file = {
        row["file"]: row for row in existing
        if row.get("file") in blob_by_file
        and row.get("build_id") == spec["build_id"]
        and row.get("source_blob_sha") == blob_by_file[row["file"]]
        and row.get("prompt_instance_sha256") == prompt_instance_hash(
            spec["build_id"], row["file"], row.get("source_blob_sha")
        )
        and (not retry_errors or row.get("status") == "ok")
    }

    def flush():
        ordered = [rows_by_file[path] for path, _ in hotspot_files if path in rows_by_file]
        write_jsonl(out_dir / "summaries.jsonl", ordered)
        return ordered

    pending = [(path, count) for path, count in hotspot_files if path not in rows_by_file]
    if builder_mode == "agent" and pending:
        preflight_provider(model, required_response_model, seed, request_timeout)
    write_json(spec_path, spec)
    source_commit = manifest["source_commit_sha"]
    agent_ready = False
    if builder_mode == "agent" and pending:
        from plugins.location_tools.repo_ops.repo_ops import set_current_issue
        set_current_issue(instance_data={
            "instance_id": Path(graph_index).stem,
            "repo": repo_name,
            "base_commit": source_commit,
            "problem_statement": "Build repository semantic memory.",
            "patch": "",
        })
        agent_ready = True

    try:
        workers = 1 if builder_mode == "agent" else max(1, int(num_workers or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    summarize_hotspot_file,
                    repo_name, repo_dir, path, count, model, builder_mode,
                    max_source_chars, temperature, request_timeout, source_commit,
                    blob_by_file[path], spec["build_id"], seed, max_iterations,
                    trajectory_dir, required_response_model,
                ): path
                for path, count in pending
            }
            for future in as_completed(futures):
                path = futures[future]
                rows_by_file[path] = future.result()
                rows = flush()
                print(f"{repo_name}: summarized {len(rows)}/{len(hotspot_files)} {path} [{rows_by_file[path]['status']}]", flush=True)
                if sleep_seconds:
                    time.sleep(sleep_seconds)
    finally:
        if agent_ready:
            from plugins.location_tools.repo_ops.repo_ops import reset_current_issue
            reset_current_issue()

    rows = flush()
    for row in rows:
        row["source_tree"] = manifest["source_tree_sha"]
    write_jsonl(out_dir / "summaries.jsonl", rows)
    corpora = write_corpora(out_dir, rows, retrieval_profile, manifests)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    provenance_complete = all(
        row.get("source_commit")
        and row.get("source_tree")
        and row.get("source_blob_sha")
        and row.get("prompt_instance_sha256")
        and row.get("build_id") == spec["build_id"]
        for row in ok_rows
    )
    from util.cost_analysis import calc_cost
    prompt_tokens = sum(row.get("prompt_tokens", 0) for row in rows)
    completion_tokens = sum(row.get("completion_tokens", 0) for row in rows)
    build_manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": spec["build_id"],
        "repo": repo_name,
        "status": "complete" if len(ok_rows) == len(hotspot_files) and provenance_complete else "incomplete",
        "expected_count": len(hotspot_files),
        "num_rows": len(rows),
        "num_ok": len(ok_rows),
        "requested_model": model,
        "response_models": sorted({model for row in rows for model in (row.get("response_models") or [row.get("response_model")]) if model}),
        "system_fingerprints": sorted({fingerprint for row in rows for fingerprint in (row.get("system_fingerprints") or [row.get("system_fingerprint")]) if fingerprint}),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": calc_cost(model, prompt_tokens, completion_tokens),
        "build_spec_sha256": sha256_file(spec_path),
        "summaries_sha256": sha256_file(out_dir / "summaries.jsonl"),
        "corpora": corpora,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "build_manifest.json", build_manifest)
    # Keep meta.json for the existing runtime API; schema-v3 fields are authoritative.
    write_json(out_dir / "meta.json", {
        "schema_version": SCHEMA_VERSION,
        "build_id": spec["build_id"],
        "repo": repo_name,
        "top_k": len(hotspot_files),
        "selected_count": len(hotspot_files),
        "num_summaries": len(rows),
        "num_ok": len(ok_rows),
        "summary_model": model,
        "builder_mode": builder_mode,
        "c0_commit_sha": manifest["c0_commit_sha"],
        "c0_tree_sha": manifest["c0_tree_sha"],
        "window_commits_sha256": manifest["window_commits_sha256"],
        "membership_variants": manifest["membership_variants"],
        "membership_sha256": manifests[0]["membership_sha256"],
        "retrieval_profile": retrieval_profile,
        "primary_document": "summary_only",
        "build_manifest_sha256": sha256_file(out_dir / "build_manifest.json"),
    })
    print(f"{repo_name}: wrote {len(rows)} summaries to {out_dir / 'summaries.jsonl'}")
    return build_manifest


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
    parser.add_argument("--episodic_index_dir", default="")
    parser.add_argument("--repo", action="append", required=True, help="owner/name=/path/to/clone")
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--model", default=os.environ.get("REPOMEM_SUMMARY_MODEL", "gpt-4o"))
    parser.add_argument("--builder_mode", choices=["agent", "direct"], default="agent")
    parser.add_argument("--max_source_chars", type=int, default=24000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--request_timeout", type=int, default=120)
    parser.add_argument("--max_iterations", type=int, default=4)
    parser.add_argument("--required_response_model", default="gpt-4o-2024-11-20")
    parser.add_argument("--retrieval_profile", default="bm25-code-v2")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--membership_manifest", action="append", required=True)
    parser.add_argument("--include_tests", action="store_true")
    parser.add_argument("--graph_index", help="Exact C0 .pkl used by the agent builder")
    parser.add_argument("--code_bm25_index", help="Exact C0 code-BM25 index directory")
    parser.add_argument("--pilot_limit", type=int, default=0, help="Build only the first N membership files")
    parser.add_argument("--retry_errors", action="store_true", help="Explicitly retry cached non-ok rows")
    args = parser.parse_args()
    if args.builder_mode == "direct" and args.required_response_model == "gpt-4o-2024-11-20":
        args.required_response_model = None
    for repo_name, repo_dir in parse_repo_map(args.repo).items():
        build_repo_semantic_index(
            repo_name, repo_dir, args.episodic_index_dir, args.output_dir, args.top_k,
            args.model, args.builder_mode, args.max_source_chars, args.temperature,
            args.request_timeout, args.sleep_seconds, args.overwrite, args.include_tests,
            args.num_workers, args.membership_manifest, args.seed, args.max_iterations,
            args.required_response_model, args.retrieval_profile, args.graph_index,
            args.code_bm25_index, args.pilot_limit,
            args.retry_errors,
        )


if __name__ == "__main__":
    main()
