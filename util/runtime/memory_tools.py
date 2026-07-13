import os

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


SearchCommitTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="search_commit",
        description=(
            "Search the repository's commit history to find commits similar to "
            "one or more queries, typically hypothetical commit messages. It uses "
            "BM25 over historical commit messages. Use this tool early in the investigation is recommended"
            "to identify related past issues or changes. Keep top_k small; the runtime "
            "may cap large requests to keep observations concise."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "A list of queries. Each query can be a hypothetical commit "
                        "message matched against historical commit messages."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Maximum number of matching commits to return. Prefer 3-5 unless "
                        "the experiment configuration requests a larger historical search."
                    ),
                    "default": _env_int("REPOMEM_SEARCH_COMMIT_SCHEMA_DEFAULT", 5),
                },
            },
            "required": ["query_list"],
        },
    ),
)


ExamineCommitTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="examine_commit",
        description=(
            "Examine details for historical commits returned by search_commit, "
            "including the commit patch and optionally the issue description. "
            "Patch line numbers belong to historical commits, not the current checkout. "
            "Examine only the most promising commits; the runtime may cap long requests."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sha_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of short 9-digit or full commit SHAs returned by search_commit.",
                },
                "display_issue": {
                    "type": "boolean",
                    "description": (
                        "Set to true only for highly relevant commits to include a truncated "
                        "linked issue excerpt when available."
                    ),
                    "default": False,
                },
            },
            "required": ["sha_list"],
        },
    ),
)


ViewSummaryTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="view_summary",
        description=(
            "View the repository's cached high-level functionality summary for "
            "a specific file. Use this tool when a candidate file appears "
            "relevant and you want a compact overview before reading source code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": (
                        "The repository-relative file path to summarize, for "
                        "example `django/db/migrations/serializer.py`."
                    ),
                },
            },
            "required": ["file_name"],
        },
    ),
)


SearchSummaryTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="search_summary",
        description=(
            "Search the repository's cached file summaries to find files similar "
            "to a query describing functionality or keywords. It uses BM25 over "
            "file summaries and returns top-k relevant file-summary pairs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A query describing desired functionality or keywords, "
                        "matched against cached file summaries."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "The top-k files to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
)
