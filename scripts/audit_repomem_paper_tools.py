#!/usr/bin/env python3
"""Replay the RepoMem Appendix queries without hard-coding paper rankings."""

import argparse
import json
import os
from pathlib import Path

from plugins.location_tools.repo_ops import repo_ops
from util.runtime.memory_tools import get_memory_tools


SEARCH_COMMIT_QUERY = (
    "Fix Cannot drop unique_together constraint on a single field with its own "
    "unique=True constraint"
)
SEARCH_SUMMARY_QUERY = "generates a migration file that is missing an import statement"
PAPER_SHAS = ["73a6ab638", "533b20877", "f5a324391"]
EXAMINE_SHA = "50931dfa5"
SEARCH_REPLAY_BASE = "ce69e34bd646558bb44ea92cecfd98b345a0b3e0"
PAPER_SUMMARY_FILES = [
    "django/db/migrations/operations/special.py",
    "django/db/migrations/serializer.py",
    "django/core/management/commands/makemigrations.py",
    "django/core/management/sql.py",
]


def schema_dict(tool):
    function = tool["function"] if isinstance(tool, dict) else tool.function
    return dict(function)


def find_commit(memory, short_sha):
    return next(row for row in memory["commits"] if row["sha"].startswith(short_sha))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodic-index",
        default="repomem_index/verified_full500_episodic_with_issues_combined",
    )
    parser.add_argument(
        "--semantic-index",
        default="repomem_index/verified_django231_semantic_agentic_pilot_v3",
    )
    parser.add_argument("--output-dir", default="results/repomem_paper_tools_audit")
    args = parser.parse_args()

    os.environ.update({
        "REPOMEM_TOOL_PROFILE": "paper-v1",
        "REPOMEM_INDEX_DIR": args.episodic_index,
        "REPOMEM_SEMANTIC_INDEX_DIR": args.semantic_index,
        "REPOMEM_SEMANTIC_LAYOUT": "repo",
        "REPOMEM_BM25_PROFILE": "bm25-code-v2",
        "REPOMEM_SEARCH_COMMIT_TOPK_CAP": "20",
    })
    repo_ops.REPOMEM_CACHE.clear()
    repo_ops.REPOMEM_SEMANTIC_CACHE.clear()
    repo_ops.CURRENT_INSTANCE = {
        "instance_id": "repomem_appendix_replay",
        "repo": "django/django",
        "problem_statement": SEARCH_COMMIT_QUERY,
        "base_commit": SEARCH_REPLAY_BASE,
    }
    memory = repo_ops._load_repomem_repo("django/django")
    visible = list(repo_ops._iter_visible_commits(memory))
    ranked = repo_ops._rank_repomem_rows(SEARCH_COMMIT_QUERY, visible, 20)
    actual_ranks = {
        short: next((rank for rank, (_, row) in enumerate(ranked, 1) if row["sha"].startswith(short)), None)
        for short in PAPER_SHAS
    }
    search_commit_output = repo_ops.search_commit([SEARCH_COMMIT_QUERY], top_k=20)

    target = find_commit(memory, EXAMINE_SHA)
    replay_base = memory["commits"][target["history_index"] + 1]["sha"]
    repo_ops.CURRENT_INSTANCE["base_commit"] = replay_base
    examine_output = repo_ops.examine_commit([EXAMINE_SHA], display_issue=True)

    repo_ops.CURRENT_INSTANCE["base_commit"] = SEARCH_REPLAY_BASE
    search_summary_output = repo_ops.search_summary(SEARCH_SUMMARY_QUERY, top_k=4)
    semantic = repo_ops._load_repomem_semantic_repo("django/django")
    view_presence = {
        path: path in (semantic or {}).get("by_file", {}) for path in PAPER_SUMMARY_FILES
    }
    view_output = repo_ops.view_summary(
        file_list=[path for path, present in view_presence.items() if present]
    ) if any(view_presence.values()) else "No Figure 8 files are present in this bank."

    tools = get_memory_tools("paper-v1")
    report = {
        "reproduction_scope": "method reproduction; no official implementation exists",
        "tool_profile": "paper-v1",
        "retrieval_profile": "bm25-code-v2 (not the paper's unavailable LLM tokenizer)",
        "episodic_index": str(Path(args.episodic_index).resolve()),
        "semantic_index": str(Path(args.semantic_index).resolve()),
        "search_commit": {
            "query": SEARCH_COMMIT_QUERY,
            "top_k": 20,
            "benchmark_replay_base": SEARCH_REPLAY_BASE,
            "visible_commit_count": len(visible),
            "paper_example_shas": PAPER_SHAS,
            "actual_top20_ranks": actual_ranks,
            "observation": search_commit_output,
        },
        "examine_commit": {
            "sha": EXAMINE_SHA,
            "documentation_replay_base": replay_base,
            "note": "Replay context is used only for Appendix rendering; benchmark runs retain task visibility.",
            "observation": examine_output,
        },
        "search_summary": {
            "query": SEARCH_SUMMARY_QUERY,
            "top_k": 4,
            "search_ready": bool((semantic or {}).get("search_ready")),
            "bank_document_count": len((semantic or {}).get("summaries", [])),
            "figure_file_presence": view_presence,
            "observation": search_summary_output,
        },
        "view_summary": {"observation": view_output},
        "schemas": {name: schema_dict(tool) for name, tool in tools.items()},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    markdown = [
        "# RepoMem Figure 5–8 Tool Audit", "",
        "This is a method reproduction. The paper has no public official implementation, and the current ranking tokenizer is `bm25-code-v2`, not the unavailable BM25(LLM) tokenizer.", "",
        "## SearchCommit", "", "```text", search_commit_output, "```", "",
        f"Paper SHA ranks in the real Top-20: `{actual_ranks}`.", "",
        "## ExamineCommit", "", "```text", examine_output, "```", "",
        "## SearchSummary", "", "```text", search_summary_output, "```", "",
        f"Figure-file presence: `{view_presence}`.", "",
        "## ViewSummary", "", "```text", view_output, "```", "",
    ]
    (output / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output),
        "search_commit_ranks": actual_ranks,
        "semantic_search_ready": report["search_summary"]["search_ready"],
        "semantic_documents": report["search_summary"]["bank_document_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
