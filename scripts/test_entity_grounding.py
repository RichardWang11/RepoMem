#!/usr/bin/env python
"""No-network checks for the Common93 Entity Grounding integration."""

import networkx as nx

from util.entity_grounding import (
    ambiguous_callable_index,
    build_grounding,
    entity_type,
    replay_telemetry,
)


def graph():
    value = nx.MultiDiGraph()
    for node in ('a.py:One.foo', 'b.py:Two.foo', 'z.py:Other.bar', 'q.py:Else.baz'):
        value.add_node(node, type='function', code='def ' + node.rsplit('.', 1)[-1])
    value.add_node('a.py:One', type='class', code='class One')
    value.add_node('a.py', type='file', code='')
    value.add_edge('a.py', 'a.py:One', type='contains')
    value.add_edge('a.py:One', 'a.py:One.foo', type='contains')
    return value


def call(name, arguments):
    return {'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]}


def main():
    value = graph()
    assert set(ambiguous_callable_index(value)) == {'foo'}
    real = build_grounding(value, 'repo-1', 'Fix foo behavior.', 'real')
    placebo = build_grounding(value, 'repo-1', 'Fix foo behavior.', 'placebo')
    assert len(real['selected_entities']) == len(placebo['selected_entities'])
    assert [entity_type(value, node) for node in real['selected_entities']] == [entity_type(value, node) for node in placebo['selected_entities']]
    assert not set(real['selected_entities']) & set(placebo['selected_entities'])
    no_disambiguation = build_grounding(value, 'repo-1', 'Fix foo behavior.', 'real', 'no_disambiguation')
    assert no_disambiguation['ambiguity_events'] and not no_disambiguation['selected_entities']
    telemetry = replay_telemetry(
        [call('search_commit', '{"query_list":["fix foo"]}'),
         call('explore_tree_structure', '{"start_entities":["a.py:One.foo"]}'),
         call('get_entity_contents', '{"entity_names":["a.py:One.foo"]}')],
        real,
        [{'changed_files': ['a.py'], 'patch': ''}],
    )
    assert telemetry['channel_state'] == 'both'
    assert telemetry['search_queries'] == ['fix foo']
    print('entity grounding contract: PASS')


if __name__ == '__main__':
    main()
