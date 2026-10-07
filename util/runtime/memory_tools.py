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


CURRENT_TOOL_PROFILE = "current"
PAPER_TOOL_PROFILE = "paper-v1"


def normalize_tool_profile(profile=None):
    value = (profile or os.environ.get("REPOMEM_TOOL_PROFILE") or CURRENT_TOOL_PROFILE).strip().lower()
    if value not in {CURRENT_TOOL_PROFILE, PAPER_TOOL_PROFILE}:
        raise ValueError("REPOMEM_TOOL_PROFILE must be current or paper-v1")
    return value


def _search_commit_tool(profile):
    paper = normalize_tool_profile(profile) == PAPER_TOOL_PROFILE
    description = (
        "This tool searches the repository's commit history to find commits similar to a given query - "
        "typically a hypothetical commit message. It uses the BM25 ranking algorithm to match the input "
        "against historical commit messages, surfacing relevant past changes. Using this tool early in the "
        "investigation is recommended, as it can help identify related issues or changes from the past, "
        "allowing you to quickly locate potential areas in the codebase to focus on."
        if paper else
        "Search the repository's commit history to find commits similar to one or more queries, typically "
        "hypothetical commit messages. It uses BM25 over historical commit messages. Using this tool early "
        "in the investigation is recommended to identify related past issues or changes. Keep top_k small; "
        "the runtime may cap large requests to keep observations concise."
    )
    top_k_description = (
        "top_k commits will be returned."
        if paper else
        "Maximum number of matching commits to return. Prefer 3-5 unless the experiment configuration "
        "requests a larger historical search."
    )
    return ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="search_commit",
        description=description,
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
                    "description": top_k_description,
                    "default": 20 if paper else _env_int("REPOMEM_SEARCH_COMMIT_SCHEMA_DEFAULT", 5),
                },
            },
            "required": ["query_list"],
        },
    ),
    )


def _examine_commit_tool(profile):
    paper = normalize_tool_profile(profile) == PAPER_TOOL_PROFILE
    return ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="examine_commit",
        description=(
            "This tool examines the details of a commit based on the short (9-digit) commit sha, including "
            "the commit patch and optionally the issue description. Note that the line numbers displayed in "
            "the patch are from those history commits, not the current repo you are working on."
            if paper else
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
                    "description": (
                        "A list of short 9-digit commit sha that you wish to examine."
                        if paper else
                        "A list of short 9-digit or full commit SHAs returned by search_commit."
                    ),
                },
                "display_issue": {
                    "type": "boolean",
                    "description": (
                        "Set to True if you wish to include the issue description in the output."
                        if paper else
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


def _view_summary_tool(profile):
    paper = normalize_tool_profile(profile) == PAPER_TOOL_PROFILE
    properties = {
        "file_list": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of filenames that you wish to view the summaries.",
        }
    } if paper else {
        "file_name": {
            "type": "string",
            "description": (
                "The repository-relative file path to summarize, for example "
                "`django/db/migrations/serializer.py`."
            ),
        }
    }
    return ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="view_summary",
        description=(
            "This tool views summary of python files in the repo. You can input a list of file names you wish to examine."
            if paper else
            "View the repository's cached high-level functionality summary for "
            "a specific file. Use this tool when a candidate file appears "
            "relevant and you want a compact overview before reading source code."
        ),
        parameters={
            "type": "object",
            "properties": properties,
            "required": ["file_list" if paper else "file_name"],
        },
    ),
    )


def _search_summary_tool(profile):
    paper = normalize_tool_profile(profile) == PAPER_TOOL_PROFILE
    return ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="search_summary",
        description=(
            "This tool searches the repository's cached file summaries to find files similar to a given query - "
            "typically a message describing some functionalities or keywords. It uses BM25 to match the input "
            "query against file summaries and returns top-k ones."
            if paper else
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
                        "A query, which can be a message describing some functionalities or keywords, to be "
                        "matched with the available file summaries to retrieve similar ones."
                        if paper else
                        "A query describing desired functionality or keywords, "
                        "matched against cached file summaries."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "the top_k files to return." if paper else "The top-k files to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    )


def get_memory_tools(profile=None):
    """Return memory-tool schemas for the selected reproducibility profile."""
    profile = normalize_tool_profile(profile)
    return {
        "search_commit": _search_commit_tool(profile),
        "examine_commit": _examine_commit_tool(profile),
        "view_summary": _view_summary_tool(profile),
        "search_summary": _search_summary_tool(profile),
    }


_CURRENT_TOOLS = get_memory_tools(CURRENT_TOOL_PROFILE)
SearchCommitTool = _CURRENT_TOOLS["search_commit"]
ExamineCommitTool = _CURRENT_TOOLS["examine_commit"]
ViewSummaryTool = _CURRENT_TOOLS["view_summary"]
SearchSummaryTool = _CURRENT_TOOLS["search_summary"]
