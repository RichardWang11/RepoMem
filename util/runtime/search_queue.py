"""OrcaLoca-compatible search action queue for function-calling agents."""

import heapq
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchActionStep:
    name: str
    arguments_json: str
    tool_call_id: str = field(compare=False)
    thought: str = field(default="", compare=False)
    origin: str = field(default="agent", compare=False)

    @classmethod
    def create(cls, name, arguments, tool_call_id, thought="", origin="agent"):
        return cls(name, json.dumps(arguments, sort_keys=True), tool_call_id, thought, origin)

    @property
    def arguments(self):
        return json.loads(self.arguments_json)


class SearchActionHistory:
    def __init__(self):
        self.action_history = {}

    def add_action(self, action):
        self.action_history[action] = self.action_history.get(action, 0) + 1

    def get_action_count(self, action):
        return self.action_history.get(action, 0)

    def check_action(self, action):
        return action in self.action_history


class SearchQueue:
    def __init__(self, basic=1):
        self.queue = []
        self.counter = 0
        self.basic = basic

    def append(self, action):
        self.append_with_priority(action, self.basic)

    def append_with_priority(self, action, priority):
        heapq.heappush(self.queue, (-priority, self.counter, action))
        self.counter += 1

    def resort(self, history):
        self.queue = [
            (-max(-priority, history.get_action_count(action)), order, action)
            for priority, order, action in self.queue
        ]
        heapq.heapify(self.queue)

    def pop(self):
        return heapq.heappop(self.queue)[2]

    def __len__(self):
        return len(self.queue)


def atomize_tool_call(name, arguments):
    """Split LocAgent's list-valued searches into Orca-style single searches."""
    split_keys = {
        "search_code_snippets": "search_terms",
        "get_entity_contents": "entity_names",
        "explore_tree_structure": "start_entities",
    }
    key = split_keys.get(name)
    values = arguments.get(key) if key else None
    if not isinstance(values, list) or len(values) <= 1:
        return [arguments]
    return [{**arguments, key: [value]} for value in values]
