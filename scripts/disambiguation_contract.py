#!/usr/bin/env python
"""No-network contract checks for the OrcaLoca ASQ migration."""

import pickle
from pathlib import Path
from queue import Queue

import networkx as nx
from litellm import ModelResponse

import auto_search_main
from dependency_graph.traverse_graph import RepoEntitySearcher
from plugins.location_tools.repo_ops import repo_ops
from util.runtime.search_queue import SearchActionHistory, SearchActionStep, SearchQueue, atomize_tool_call


def install_graph(graph, problem="Fix foo."):
    repo_ops.DP_GRAPH = graph
    repo_ops.DP_GRAPH_ENTITY_SEARCHER = RepoEntitySearcher(graph)
    repo_ops.CURRENT_INSTANCE = {"problem_statement": problem}
    repo_ops.ALL_FILE = []


def tiny_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("a.py:A.foo", type="function", code="def foo(): pass", start_line=1, end_line=1)
    graph.add_node("b.py:B.foo", type="function", code="def foo(): pass", start_line=2, end_line=2)
    for index, path in enumerate(("models.py", "forms.py", "gis.py"), 3):
        graph.add_node(f"{path}:Field", type="class", code="class Field: pass", start_line=index, end_line=index)
    graph.add_node("c.py:bar", type="function", code="def bar(): pass", start_line=6, end_line=6)
    return graph


def queue_contract():
    first = SearchActionStep.create("search_code_snippets", {"search_terms": ["a"]}, "1")
    second = SearchActionStep.create("search_code_snippets", {"search_terms": ["b"]}, "2")
    queue, history = SearchQueue(), SearchActionHistory()
    queue.append(first)
    queue.append(second)
    assert queue.pop() == first

    queue = SearchQueue()
    queue.append(first)
    queue.append(second)
    history.add_action(second)
    history.add_action(second)
    queue.resort(history)
    assert queue.pop() == second
    assert atomize_tool_call("get_entity_contents", {"entity_names": ["a", "b"]}) == [
        {"entity_names": ["a"]}, {"entity_names": ["b"]}
    ]


def message_pairing_contract():
    responses = [
        ModelResponse(choices=[{"message": {"role": "assistant", "content": "search", "tool_calls": [
            {"id": "original_1", "type": "function", "function": {"name": "search_code_snippets", "arguments": '{"search_terms":["a","b"]}'}},
        ]}}], usage={"prompt_tokens": 1, "completion_tokens": 1}),
        ModelResponse(choices=[{"message": {"role": "assistant", "tool_calls": [
            {"id": "finish_1", "type": "function", "function": {"name": "finish", "arguments": '{"final_output":"done"}'}},
        ]}}], usage={"prompt_tokens": 1, "completion_tokens": 1}),
        ModelResponse(choices=[{"message": {"role": "assistant", "tool_calls": [
            {"id": "finish_2", "type": "function", "function": {"name": "finish", "arguments": '{"final_output":"done"}'}},
        ]}}], usage={"prompt_tokens": 1, "completion_tokens": 1}),
    ]
    original_completion = auto_search_main.litellm.completion
    original_execute = auto_search_main.execute_ipython
    auto_search_main.litellm.completion = lambda **_: responses.pop(0)
    auto_search_main.execute_ipython = lambda _: "result"
    result_queue = Queue()
    try:
        auto_search_main.auto_search_process(
            result_queue=result_queue,
            model_name="fake",
            messages=[{"role": "user", "content": "issue"}],
            fake_user_msg="continue",
            tools=[{}],
            max_iteration_num=5,
            enable_asq=True,
        )
    finally:
        auto_search_main.litellm.completion = original_completion
        auto_search_main.execute_ipython = original_execute
    output, messages, trajectory = result_queue.get_nowait()
    assert output == "done"
    assert len([event for event in trajectory["asq_events"] if event["event"] == "pop"]) == 2
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            assert messages[index + 1]["role"] == "tool"
            assert messages[index + 1]["tool_call_id"] == message["tool_calls"][0]["id"]


def main():
    queue_contract()
    message_pairing_contract()
    install_graph(tiny_graph())
    calls = []

    def scorer(_issue, candidate, _code):
        calls.append(candidate["uid"])
        return 90 if candidate["uid"].startswith("a.py") else 50

    repo_ops.configure_disambiguation(True, scorer=scorer)
    event = repo_ops._detect_ambiguity("Field", "test")
    assert event["detected"] and not event["decomposed"]
    assert repo_ops.decompose_disambiguation_event(event) == [
        "models.py:Field", "forms.py:Field", "gis.py:Field"
    ]
    assert calls == []  # Orca class disambiguation never invokes CodeScorer.

    event = repo_ops._detect_ambiguity("foo", "test")
    assert repo_ops.decompose_disambiguation_event(event) == ["a.py:A.foo"]
    assert set(calls) == {"a.py:A.foo", "b.py:B.foo"}
    assert repo_ops._detect_ambiguity("bar", "test") is None
    assert repo_ops._detect_ambiguity("a.py:A.foo", "test") is None

    repo_ops.configure_disambiguation(True, scorer=scorer)
    output = repo_ops.search_code_snippets(search_terms=["Field"])
    event = repo_ops.get_disambiguation_events()[0]
    assert output.startswith('<Disambiguation symbol="Field" priority="2">')
    assert event["detected"] and not event["decomposed"] and not event["selected_uids"]
    assert "models.py:Field" in output and "forms.py:Field" in output

    graph_path = Path(__file__).parents[1] / "index_data/SWE-bench_Verified/graph_index_v2.3/django__django-14580.pkl"
    if graph_path.exists():
        with graph_path.open("rb") as handle:
            install_graph(pickle.load(handle), "Missing models import in migration serialization.")
        repo_ops.configure_disambiguation(True, scorer=lambda *_: 90)
        output = repo_ops.search_code_snippets(["ModelState"])
        event = repo_ops.get_disambiguation_events()[0]
        assert "django/db/migrations/state.py:ModelState" in output
        assert "django/db/models/base.py:ModelState" in output
        assert len(repo_ops.decompose_disambiguation_event(event)) == 2

    print("disambiguation contract: PASS")


if __name__ == "__main__":
    main()
