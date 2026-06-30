from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)


SearchCommitTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="search_commit",
        description=(
            "Search the repository's commit history to find commits similar to "
            "one or more queries, typically hypothetical commit messages. It uses "
            "BM25 over historical commit messages. Use this tool early in the investigation is recommended"
            "to identify related past issues or changes."
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
                    "description": "Maximum number of matching commits to return.",
                    "default": 20,
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
            "Patch line numbers belong to historical commits, not the current checkout."
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
                    "description": "Set to true to include linked issue information when available.",
                    "default": False,
                },
            },
            "required": ["sha_list"],
        },
    ),
)
