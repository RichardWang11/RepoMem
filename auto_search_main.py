import argparse
import ast
import hashlib
import os
import json
import logging
import logging.handlers
import time
import toml
from pathlib import Path
from queue import Empty
from typing import List
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset

from util.runtime.execute_ipython import execute_ipython
from util.runtime import function_calling
from util.actions.action_parser import ResponseParser
from util.actions.action import ActionType
from util.prompts.prompt import PromptManager
from util.prompts import general_prompt
from util.prompts.pipelines import (
    simple_localize_pipeline as simple_loc,
    auto_search_prompt as auto_search,
)
from util.cost_analysis import calc_cost
from util.utils import *
from util.process_output import (
    parse_raw_loc_output,
    get_loc_results_from_raw_outputs,
    merge_sample_locations,
)
from plugins import LocationToolsRequirement
from plugins.location_tools.locationtools import get_documentation as get_location_tools_documentation
from plugins.location_tools.repo_ops.repo_ops import (
    build_e1_supplemental_leakage_case,
    configure_tool_output_bounds,
    configure_disambiguation,
    decompose_disambiguation_event,
    get_disambiguation_events,
    examine_commit,
    EpisodicMemoryUnavailable,
    get_repomem_visibility_audit,
    set_current_issue,
    reset_current_issue,
    get_graph,
    validate_episodic_visibility,
)
from plugins.location_tools.repo_ops import repo_ops
from util.entity_grounding import build_grounding, grounding_prompt, replay_telemetry
import litellm
from litellm import Message as LiteLLMMessage
from openai import APITimeoutError
from evaluation.eval_metric import filtered_instances


from time import sleep
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import torch.multiprocessing as mp
from util.runtime.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
    STOP_WORDS as NON_FNCALL_STOP_WORDS
)
from util.runtime.context_guard import (
    count_context_tokens,
    guard_observation,
    is_context_error,
    summarize_context_audit,
)
from util.runtime.search_queue import (
    SearchActionHistory,
    SearchActionStep,
    SearchQueue,
    atomize_tool_call,
)
from util.runtime.e1_supplemental_leakage import (
    execute_memory_arm,
    exposure_outcome,
    find_first_differential_observation,
    treatment_prefix,
)
from util.runtime.e2_search_topk import (
    CONTROL_TOP_K as E2_CONTROL_TOP_K,
    TREATMENT_TOP_K as E2_TREATMENT_TOP_K,
    adoption_outcome as e2_adoption_outcome,
    corpus_audit as e2_corpus_audit,
    find_first_topk_differential,
)
# litellm.set_verbose=True
# os.environ['LITELLM_LOG'] = 'DEBUG


EPISODIC_TASK_PROMPT = (
    '\nRepository memory is available through SearchCommit and ExamineCommit tools. '
    'Use SearchCommit early with issue-specific queries to find related historical fixes, '
    'then use ExamineCommit on promising commits before deciding final locations. '
    'Historical patches are hints only; verify candidate files in the current codebase with the normal code tools.\n'
)


def normalize_function_response(function_response):
    if function_response is None:
        return ""

    if isinstance(function_response, str):
        try:
            parsed_response = ast.literal_eval(function_response)
        except (SyntaxError, ValueError, TypeError):
            return function_response
    else:
        parsed_response = function_response

    if isinstance(parsed_response, str):
        return parsed_response
    return str(parsed_response)


def filter_dataset(dataset, filter_column: str, used_list: str, used_list_file: str = None):
    if used_list_file:
        with open(used_list_file, 'r') as file:
            selected_ids = json.load(file)
        selected_ids = set(selected_ids)
        logging.info(f'Filtering {len(selected_ids)} tasks from "{used_list_file}"...')

        def filter_function(example):
            return example[filter_column] in selected_ids

        filtered_dataset = dataset.filter(filter_function) if hasattr(dataset, 'filter') else [row for row in dataset if filter_function(row)]
        logging.info(f'Retained {len(filtered_dataset)} tasks after filtering')
        return filtered_dataset

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if used_list in data:
                selected_ids = data[used_list]
                logging.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                def filter_function(example):
                    return example[filter_column] in selected_ids  # Replace 'id' with the actual field name in the dataset
                filtered_dataset = dataset.filter(filter_function) if hasattr(dataset, 'filter') else [row for row in dataset if filter_function(row)]
                # subset = dataset[dataset[filter_column].isin(selected_ids)]
                logging.info(f'Retained {len(filtered_dataset)} tasks after filtering')
                return filtered_dataset
    return dataset


def load_bench_dataset(dataset: str, split: str):
    """Load a Hub dataset or a local JSONL file such as Common93."""
    path = Path(dataset)
    if path.is_file():
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return load_dataset(dataset, split=split)


def configure_repomem_index():
    """Use the checked-in episodic index unless the caller supplied one."""
    if os.environ.get('REPOMEM_INDEX_DIR'):
        return
    default = Path(__file__).resolve().parent / 'repomem_index' / 'verified_200_episodic_with_issues'
    if default.is_dir():
        os.environ['REPOMEM_INDEX_DIR'] = str(default)


def load_fixed_memory_cells(path):
    """Load frozen (task, memory) cells without changing benchmark rows."""
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    cells = data.get('cells') if isinstance(data, dict) else data
    if not isinstance(cells, list) or not cells:
        raise ValueError('fixed episodic memory manifest must contain non-empty cells')
    required = {'cell_id', 'instance_id', 'base_commit'}
    if any(not required.issubset(cell) for cell in cells):
        raise ValueError(f'each fixed memory cell requires {sorted(required)}')
    for cell in cells:
        shas = cell.get('memory_shas')
        if shas is None:
            shas = [cell.get('memory_sha')] if cell.get('memory_sha') else []
        if not isinstance(shas, list) or not shas or any(not isinstance(sha, str) or not sha for sha in shas):
            raise ValueError('each fixed memory cell requires memory_sha or non-empty memory_shas')
        if len(shas) > 5 or len(set(shas)) != len(shas):
            raise ValueError('fixed memory_shas must contain 1-5 unique SHAs')
    if len({cell['cell_id'] for cell in cells}) != len(cells):
        raise ValueError('fixed episodic memory cell_id values must be unique')
    return cells


def cached_prompt_tokens(usage):
    details = getattr(usage, 'prompt_tokens_details', None)
    if isinstance(details, dict):
        return int(details.get('cached_tokens') or 0)
    return int(getattr(details, 'cached_tokens', 0) or 0)


def get_task_instruction(
        instance: dict,
        task: str = 'auto_search',
        include_pr=False,
        include_hint=False,
        repomem_mode: str = None,
):
    output_format = None
    instruction = ""
    
    # for auto-search pipeline
    if task.strip() == 'auto_search':
        task_description = auto_search.TASK_INSTRUECTION.format(
            package_name=instance['instance_id'].split('_')[0]
        )
    
    elif task.strip() == 'simple_localize':
        task_description = simple_loc.SEARCH_LOC_TASK_INSTRUCTION
        output_format = simple_loc.OUTPUT_FORMAT_LOC
        
    else:
        return None

    instruction += task_description
        
    if include_pr:
        problem_statement = instance['problem_statement']
        instruction += general_prompt.PR_TEMPLATE.format(
            title=problem_statement.strip().split('\n')[0],
            description = '\n'.join(problem_statement.strip().split('\n')[1:]).strip()
        )
    
    if output_format:
        instruction += output_format
    
    if include_hint:
        instruction += (
            'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
            'Don\'t include any lambda functions!\n'
            'You should NOT modify any files!\n'
        )

    if repomem_mode in ('episodic', 'both'):
        instruction += EPISODIC_TASK_PROMPT
    if repomem_mode in ('semantic', 'both'):
        instruction += (
            '\nRepository semantic memory is available through SearchSummary and ViewSummary tools. '
            'Use SearchSummary with issue-specific functionality queries to find actively edited files with relevant cached summaries, '
            'then use ViewSummary for candidate files when a compact functionality overview is useful. '
            'File summaries are hints only; verify final candidate files in the current codebase with the normal code tools.\n'
        )

    # NOTE: You can actually set slightly different instruction for different task
    # instruction += AGENT_CLS_TO_INST_SUFFIX
    return instruction


def auto_search_process(result_queue,
                        model_name, messages, fake_user_msg,
                        tools = None,
                        traj_data=None,
                        temp=1.0,
                        top_p=None,
                        sampling_seed=None,
                        max_iteration_num=20,
                        request_timeout=120,
                        use_function_calling=True,
                        context_token_budget=0,
                        tool_observation_token_cap=0,
                        fail_on_context_error=False,
                        bounded_graph_output=False,
                        bounded_entity_output=False,
                        bounded_search_output=False,
                        enable_asq=False,
                        enable_disambiguation=False,
                        required_response_model=None,
                        iteration_offset=0):
    configure_tool_output_bounds(
        graph_max_lines=800 if bounded_graph_output else 0,
        entity_max_lines=400 if bounded_entity_output else 0,
        search_max_lines=600 if bounded_search_output else 0,
        graph_max_depth=3,
    )
    configure_disambiguation(enable_asq, model_name)
    if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower() 
    #             #   or model_name=='azure/gpt-4o' 
    #             #   or model_name == 'litellm_proxy/o3-mini-2025-01-31'
                ):
        use_function_calling = False
        
    # for LLM which do not support function calling
    if tools and not use_function_calling:
        # 转换message
        messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
            
    # code_history = []
    parser = ResponseParser()
    if not traj_data:
        traj_msgs = messages.copy()
        prompt_tokens = 0
        completion_tokens = 0
    else:
        # continue from last traj
        traj_msgs = traj_data['messages']
        prompt_tokens = traj_data['usage']['prompt_tokens']
        completion_tokens = traj_data['usage']['completion_tokens']
    provider_responses = []
        
    cur_interation_num = int(iteration_offset or 0)
    last_message = None
    finish = False
    context_audit = []
    search_queue = SearchQueue()
    action_history = SearchActionHistory()
    asq_events = []
    disambiguation_cursor = 0
    synthetic_call_counter = 0
    while not finish:
        cur_interation_num += 1
        if cur_interation_num == max_iteration_num:
            messages.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })
            traj_msgs.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })

        try:
            if context_token_budget > 0:
                current_tokens = count_context_tokens(messages, model_name, tools)
                context_audit.append({
                    "event": "request",
                    "iteration": cur_interation_num,
                    "prompt_tokens": current_tokens,
                    "context_token_budget": context_token_budget,
                })
                if current_tokens > context_token_budget:
                    result_queue.put({
                        'error': (
                            f'Prompt context budget exceeded before request: '
                            f'{current_tokens}>{context_token_budget}'
                        ),
                        'type': 'ContextBudgetExceeded',
                        'context_error': True,
                        'context_audit': context_audit,
                    })
                    return
            # new conversation
            if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower()):
                messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
                response = litellm.completion(
                    model=model_name,
                    temperature=temp, top_p=0.8, repetition_penalty=1.05, 
                    seed=sampling_seed,
                    messages=messages,
                    request_timeout=request_timeout,
                    stop=NON_FNCALL_STOP_WORDS
                )
            elif tools:
                completion_kwargs = {
                    'model': model_name,
                    'tools': tools,
                    'messages': messages,
                    'temperature': temp,
                    'request_timeout': request_timeout,
                }
                if top_p is not None:
                    completion_kwargs['top_p'] = top_p
                if sampling_seed is not None:
                    completion_kwargs['seed'] = sampling_seed
                response = litellm.completion(
                    **completion_kwargs,
                    # stop=['</execute_ipython>'], #</finish>',
                )
            else:
                completion_kwargs = {
                    'model': model_name,
                    'messages': messages,
                    'temperature': temp,
                    'request_timeout': request_timeout,
                    'stop': ['</execute_ipython>'],
                }
                if top_p is not None:
                    completion_kwargs['top_p'] = top_p
                if sampling_seed is not None:
                    completion_kwargs['seed'] = sampling_seed
                response = litellm.completion(**completion_kwargs)
        except litellm.BadRequestError as e:
            # If there's an error, send the error info back to the parent process
            context_error = is_context_error(e)
            result_queue.put({
                'error': str(e),
                'type': 'ContextWindowExceededError' if context_error else 'BadRequestError',
                'context_error': context_error,
                'context_audit': context_audit,
            })
            return
        except Exception as e:
            context_error = is_context_error(e)
            result_queue.put({
                'error': str(e),
                'type': 'ContextWindowExceededError' if context_error else e.__class__.__name__,
                'context_error': context_error,
                'context_audit': context_audit,
            })
            return
        
        if last_message and response.choices[0].message.content == last_message:
            messages.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            traj_msgs.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            continue
        
        raw_response = deepcopy(response)
        response_model = getattr(raw_response, 'model', None)
        if required_response_model and response_model != required_response_model:
            result_queue.put({
                'error': (
                    f'Provider model drift: observed {response_model!r}, '
                    f'required {required_response_model!r}'
                ),
                'type': 'ProviderModelDrift',
                'context_error': False,
            })
            return
        provider_responses.append({
            'stage': 'localization',
            'call_index': len(provider_responses),
            'requested_model': model_name,
            'temperature': temp,
            'top_p': top_p,
            'seed': sampling_seed,
            'model': getattr(raw_response, 'model', None),
            'system_fingerprint': getattr(raw_response, 'system_fingerprint', None),
            'response_id': getattr(raw_response, 'id', None),
            'created': getattr(raw_response, 'created', None),
            'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
            'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
            'cached_prompt_tokens': cached_prompt_tokens(response.usage),
        })
        # logging.info('response.choices[0].message')
        if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower()
                      or 'deepseek' in model_name
                      ):
            try:
                non_fncall_response_message = response.choices[0].message
                fn_call_messages_with_response = (
                    convert_non_fncall_messages_to_fncall_messages(
                        [non_fncall_response_message], tools # messages + 
                    )
                )
                fn_call_response_message = fn_call_messages_with_response[-1]
                if not isinstance(fn_call_response_message, LiteLLMMessage):
                    fn_call_response_message = LiteLLMMessage(
                        **fn_call_response_message
                    )
                response.choices[0].message = fn_call_response_message
            except:
                logging.info('convert none fncall messages failed.')
                continue 
                
        last_message = response.choices[0].message.content
        logging.debug(response.choices[0].message)
        prompt_tokens += response.usage.prompt_tokens
        completion_tokens += response.usage.completion_tokens  

        if enable_asq and tools and use_function_calling:
            tool_calls = response.choices[0].message.tool_calls or []
            requested_finish = None
            thought = response.choices[0].message.content or ""
            for tool_call in tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "finish":
                    requested_finish = next(iter(arguments.values()), thought)
                    continue
                for atomic_arguments in atomize_tool_call(tool_call.function.name, arguments):
                    synthetic_call_counter += 1
                    queued_action = SearchActionStep.create(
                        tool_call.function.name,
                        atomic_arguments,
                        f"asq_{synthetic_call_counter}",
                        thought=thought,
                    )
                    thought = ""
                    already_seen = action_history.check_action(queued_action)
                    if not already_seen:
                        search_queue.append(queued_action)
                    action_history.add_action(queued_action)
                    asq_events.append({
                        "event": "recommend",
                        "iteration": cur_interation_num,
                        "action": queued_action.name,
                        "arguments": atomic_arguments,
                        "enqueued": not already_seen,
                        "priority": 1,
                    })

            if cur_interation_num >= max_iteration_num:
                final_output = requested_finish or response.choices[0].message.content or ""
                asq_events.append({
                    "event": "drop_at_iteration_limit",
                    "iteration": cur_interation_num,
                    "queue_size": len(search_queue),
                })
                finish = True
                continue

            if requested_finish is not None and not len(search_queue):
                final_output = requested_finish
                logging.info("\nFinal Response:=\n" + final_output)
                finish = True
                continue
            if requested_finish is not None:
                asq_events.append({
                    "event": "defer_finish",
                    "iteration": cur_interation_num,
                    "queue_size": len(search_queue),
                })

            if not len(search_queue):
                assistant_message = convert_to_json(response.choices[0].message)
                messages.append(assistant_message)
                traj_msgs.append(assistant_message)
                messages.append({"role": "user", "content": fake_user_msg})
                traj_msgs.append({"role": "user", "content": fake_user_msg})
                continue

            search_queue.resort(action_history)
            queued_action = search_queue.pop()
            arguments = queued_action.arguments
            assistant_message = {
                "role": "assistant",
                "content": queued_action.thought or None,
                "tool_calls": [{
                    "id": queued_action.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": queued_action.name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            }
            messages.append(assistant_message)
            traj_msgs.append(deepcopy(assistant_message))
            asq_events.append({
                "event": "pop",
                "iteration": cur_interation_num,
                "action": queued_action.name,
                "arguments": arguments,
                "origin": queued_action.origin,
                "queue_size_after": len(search_queue),
            })

            ipython_code = f"print({queued_action.name}(**{arguments!r}))"
            logging.info(f"Executing ASQ action:\n```\n{ipython_code}\n```")
            function_response = normalize_function_response(execute_ipython(ipython_code))

            new_events = get_disambiguation_events()[disambiguation_cursor:]
            disambiguation_cursor += len(new_events)
            if enable_disambiguation:
                for event in new_events:
                    generated_actions = []
                    for uid in decompose_disambiguation_event(event):
                        synthetic_call_counter += 1
                        exact_action = SearchActionStep.create(
                            "get_entity_contents",
                            {"entity_names": [uid]},
                            f"asq_{synthetic_call_counter}",
                            origin="disambiguation",
                        )
                        if action_history.check_action(exact_action):
                            continue
                        search_queue.append_with_priority(exact_action, event["priority"])
                        action_history.add_action(exact_action)
                        generated_actions.append({"action": exact_action.name, "arguments": exact_action.arguments})
                    event["generated_actions"] = generated_actions
                    asq_events.append({
                        "event": "disambiguation_decomposition",
                        "iteration": cur_interation_num,
                        "symbol": event["symbol"],
                        "generated_actions": generated_actions,
                        "priority": event["priority"],
                    })

            if context_token_budget > 0 or tool_observation_token_cap > 0:
                guarded = guard_observation(
                    function_response,
                    model=model_name,
                    tool_name=queued_action.name,
                    iteration=cur_interation_num,
                    messages=messages,
                    tools=tools,
                    observation_token_cap=tool_observation_token_cap,
                    context_token_budget=context_token_budget,
                )
                function_response = guarded.text
                context_audit.append(guarded.audit)
            logging.info("OBSERVATION:\n" + function_response)
            tool_message = {
                "role": "tool",
                "tool_call_id": queued_action.tool_call_id,
                "name": queued_action.name,
                "content": "OBSERVATION:\n" + function_response,
            }
            messages.append(tool_message)
            traj_msgs.append(deepcopy(tool_message))
            continue

        messages.append(convert_to_json(raw_response.choices[0].message))
        traj_msgs.append(convert_to_json(raw_response.choices[0].message))
        actions = parser.parse(response)
        if not isinstance(actions, List):
            actions = [actions]
        if (
            cur_interation_num >= max_iteration_num
            and not any(action.action_type == ActionType.FINISH for action in actions)
        ):
            final_output = response.choices[0].message.content or ''
            logging.info("Maximum iterations reached; using the latest response as final output.")
            finish = True
            continue
        for action in actions:
            logging.debug(action.action_type)
            if action.action_type == ActionType.FINISH:
                final_output = action.thought
                logging.info('='*15)
                logging.info("\nFinal Response:=\n" + final_output)
                finish = True # break
            elif action.action_type == ActionType.MESSAGE:
                logging.debug("thought:\n" + action.content)
                # check if enough
                messages.append({"role": "user", "content": fake_user_msg})
                traj_msgs.append({"role": "user", "content": fake_user_msg})
                # continue
            elif action.action_type == ActionType.RUN_IPYTHON:
                ipython_code = action.code.strip('`')
                logging.info(f"Executing code:\n```\n{ipython_code}\n```")
                function_response = execute_ipython(ipython_code)
                function_response = normalize_function_response(function_response)
                if context_token_budget > 0 or tool_observation_token_cap > 0:
                    guarded = guard_observation(
                        function_response,
                        model=model_name,
                        tool_name=action.function_name,
                        iteration=cur_interation_num,
                        messages=messages,
                        tools=tools,
                        observation_token_cap=tool_observation_token_cap,
                        context_token_budget=context_token_budget,
                    )
                    function_response = guarded.text
                    context_audit.append(guarded.audit)
                
                logging.info("OBSERVATION:\n" + function_response)
                if not tools:
                    messages.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
            else:
                logging.warning('Error Action!')
                # return

    # save traj
    disambiguation_events = get_disambiguation_events()
    scorer_prompt_tokens = sum(
        candidate.get("scorer_usage", {}).get("prompt_tokens", 0)
        for event in disambiguation_events for candidate in event["candidates"]
    )
    scorer_completion_tokens = sum(
        candidate.get("scorer_usage", {}).get("completion_tokens", 0)
        for event in disambiguation_events for candidate in event["candidates"]
    )
    traj_data = {
        'messages': traj_msgs,
        'tools': tools,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'scorer_prompt_tokens': scorer_prompt_tokens,
            'scorer_completion_tokens': scorer_completion_tokens,
            'total_prompt_tokens': prompt_tokens + scorer_prompt_tokens,
            'total_completion_tokens': completion_tokens + scorer_completion_tokens,
        },
        'provider_metadata': {
            'requested_model': model_name,
            'temperature': temp,
            'top_p': top_p,
            'seed': sampling_seed,
            'responses': provider_responses,
        },
        'context_guard': {
            'enabled': bool(context_token_budget or tool_observation_token_cap),
            'context_token_budget': context_token_budget,
            'tool_observation_token_cap': tool_observation_token_cap,
            'fail_on_context_error': fail_on_context_error,
            'bounded_graph_output': bounded_graph_output,
            'bounded_entity_output': bounded_entity_output,
            'bounded_search_output': bounded_search_output,
            'events': context_audit,
        },
        'asq_events': asq_events,
        'disambiguation_events': disambiguation_events,
        'episodic_visibility': get_repomem_visibility_audit(),
    }
    # return final_output, messages, traj_data
    result_queue.put((final_output, messages, traj_data))


def run_e1_lazy_fork(
    args,
    ctx,
    tools,
    control_output,
    control_messages,
    control_traj,
):
    """Fork Treatment only after the first differing memory observation."""
    differential = find_first_differential_observation(
        control_messages,
        lambda name, arguments, arm: execute_memory_arm(
            repo_ops, name, arguments, arm
        ),
    )
    if differential is None:
        outcome = exposure_outcome(control_messages, None, None)
        return {
            "differential": None,
            "outcome": outcome,
            "treatment_output": control_output,
            "treatment_messages": control_messages,
            "treatment_continuation": None,
            "incremental_usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    prefix = treatment_prefix(control_messages, differential)
    iteration_offset = sum(
        message.get("role") == "assistant"
        for message in prefix
    )
    prefix_traj = {
        "messages": deepcopy(prefix),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }
    manager = ctx.Manager()
    result_queue = manager.Queue()
    old_mode = os.environ.get('REPOMEM_SUPPLEMENTAL_LEAKAGE')
    try:
        os.environ['REPOMEM_SUPPLEMENTAL_LEAKAGE'] = 'on'
        process = ctx.Process(target=auto_search_process, kwargs={
            'result_queue': result_queue,
            'model_name': args.model,
            'messages': deepcopy(prefix),
            'fake_user_msg': auto_search.FAKE_USER_MSG_FOR_LOC,
            'temp': args.temperature,
            'top_p': args.top_p,
            'sampling_seed': args.sampling_seed,
            'tools': tools,
            'traj_data': prefix_traj,
            'max_iteration_num': args.max_iteration_num,
            'iteration_offset': iteration_offset,
            'request_timeout': args.request_timeout,
            'use_function_calling': args.use_function_calling,
            'context_token_budget': args.context_token_budget,
            'tool_observation_token_cap': args.tool_observation_token_cap,
            'fail_on_context_error': args.fail_on_context_error,
            'bounded_graph_output': args.bounded_graph_output,
            'bounded_entity_output': args.bounded_entity_output,
            'bounded_search_output': args.bounded_search_output,
            'enable_asq': args.enable_asq,
            'enable_disambiguation': args.enable_disambiguation,
            'required_response_model': args.required_response_model,
        })
        process.start()
        process.join(timeout=args.timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError("E1 treatment continuation exceeded timeout")
        try:
            result = result_queue.get_nowait()
        except Empty as exc:
            raise RuntimeError("E1 treatment continuation returned no result") from exc
    finally:
        manager.shutdown()
        if old_mode is None:
            os.environ.pop('REPOMEM_SUPPLEMENTAL_LEAKAGE', None)
        else:
            os.environ['REPOMEM_SUPPLEMENTAL_LEAKAGE'] = old_mode

    if isinstance(result, dict):
        raise RuntimeError(
            f"E1 treatment continuation failed: {result.get('type')}: "
            f"{result.get('error')}"
        )
    treatment_output, treatment_messages, treatment_traj = result
    outcome = exposure_outcome(
        control_messages, treatment_messages, differential
    )
    return {
        "differential": differential,
        "outcome": outcome,
        "treatment_output": treatment_output,
        "treatment_messages": treatment_messages,
        "treatment_continuation": treatment_traj,
        "incremental_usage": treatment_traj["usage"],
    }


def run_e2_lazy_fork(
    args,
    ctx,
    tools,
    control_output,
    control_messages,
    visible_shas,
):
    """Fork Top20 only after the first differing SearchCommit observation."""
    differential = find_first_topk_differential(control_messages, repo_ops)
    if differential is None:
        outcome = e2_adoption_outcome(
            control_messages, None, None, visible_shas=visible_shas
        )
        return {
            "differential": None,
            "outcome": outcome,
            "treatment_output": control_output,
            "treatment_messages": control_messages,
            "treatment_continuation": None,
            "incremental_usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    prefix = treatment_prefix(control_messages, differential)
    iteration_offset = sum(
        message.get("role") == "assistant"
        for message in prefix
    )
    prefix_traj = {
        "messages": deepcopy(prefix),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }
    manager = ctx.Manager()
    result_queue = manager.Queue()
    old_top_k = os.environ.get("REPOMEM_SEARCH_COMMIT_FORCE_TOPK")
    try:
        os.environ["REPOMEM_SEARCH_COMMIT_FORCE_TOPK"] = str(
            E2_TREATMENT_TOP_K
        )
        process = ctx.Process(target=auto_search_process, kwargs={
            "result_queue": result_queue,
            "model_name": args.model,
            "messages": deepcopy(prefix),
            "fake_user_msg": auto_search.FAKE_USER_MSG_FOR_LOC,
            "temp": args.temperature,
            "top_p": args.top_p,
            "sampling_seed": args.sampling_seed,
            "tools": tools,
            "traj_data": prefix_traj,
            "max_iteration_num": args.max_iteration_num,
            "iteration_offset": iteration_offset,
            "request_timeout": args.request_timeout,
            "use_function_calling": args.use_function_calling,
            "context_token_budget": args.context_token_budget,
            "tool_observation_token_cap": args.tool_observation_token_cap,
            "fail_on_context_error": args.fail_on_context_error,
            "bounded_graph_output": args.bounded_graph_output,
            "bounded_entity_output": args.bounded_entity_output,
            "bounded_search_output": args.bounded_search_output,
            "enable_asq": args.enable_asq,
            "enable_disambiguation": args.enable_disambiguation,
            "required_response_model": args.required_response_model,
        })
        process.start()
        process.join(timeout=args.timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError("E2 Top20 continuation exceeded timeout")
        try:
            result = result_queue.get_nowait()
        except Empty as exc:
            raise RuntimeError("E2 Top20 continuation returned no result") from exc
    finally:
        manager.shutdown()
        if old_top_k is None:
            os.environ.pop("REPOMEM_SEARCH_COMMIT_FORCE_TOPK", None)
        else:
            os.environ["REPOMEM_SEARCH_COMMIT_FORCE_TOPK"] = old_top_k

    if isinstance(result, dict):
        raise RuntimeError(
            f"E2 Top20 continuation failed: {result.get('type')}: "
            f"{result.get('error')}"
        )
    treatment_output, treatment_messages, treatment_traj = result
    outcome = e2_adoption_outcome(
        control_messages,
        treatment_messages,
        differential,
        visible_shas=visible_shas,
    )
    return {
        "differential": differential,
        "outcome": outcome,
        "treatment_output": treatment_output,
        "treatment_messages": treatment_messages,
        "treatment_continuation": treatment_traj,
        "incremental_usage": treatment_traj["usage"],
    }


def run_localize(rank, args, bug_queue, log_queue, output_file_lock, traj_file_lock):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.log_level))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")

    while True:
        try:
            bug = bug_queue.get_nowait()
        except Empty:
            break

        instance_id = bug["instance_id"]
        prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'util/prompts'),
            agent_skills_docs=get_location_tools_documentation(args.repomem_tool_profile),
        )

        logger.info("=" * 60)
        logger.info(f"==== rank {rank} setup localize {instance_id} ====")
        set_current_issue(instance_data=bug, rank=rank)
        e1_case = None
        e2_case = None
        e3_case = None
        if args.e1_supplemental_leakage:
            if not (
                args.use_repomem
                and args.repomem_mode == 'episodic'
                and args.repomem_visibility_policy == 'strict'
            ):
                raise ValueError(
                    "E1 requires episodic RepoMem with strict visibility"
                )
            memory_data = repo_ops._load_repomem_repo(bug['repo'])
            if not memory_data:
                raise EpisodicMemoryUnavailable(
                    f"episodic index is unavailable for {bug['repo']}"
                )
            e1_case = build_e1_supplemental_leakage_case(memory_data, bug)
            if not e1_case['affected']:
                raise ValueError(
                    f"{instance_id} is not E1-affected (D_i is empty)"
                )
            # The one paid shared trajectory uses the historical behavior.
            # Production remains fail-safe ON whenever E1 is not requested.
            os.environ['REPOMEM_SUPPLEMENTAL_LEAKAGE'] = 'off'
        elif args.e2_search_topk:
            if not (
                args.use_repomem
                and args.repomem_mode == 'episodic'
                and args.repomem_visibility_policy == 'strict'
            ):
                raise ValueError(
                    "E2 requires episodic RepoMem with strict visibility"
                )
            memory_data = repo_ops._load_repomem_repo(bug['repo'])
            if not memory_data:
                raise EpisodicMemoryUnavailable(
                    f"episodic index is unavailable for {bug['repo']}"
                )
            os.environ['REPOMEM_SUPPLEMENTAL_LEAKAGE'] = 'on'
            os.environ['REPOMEM_SEARCH_COMMIT_FORCE_TOPK'] = str(
                E2_CONTROL_TOP_K
            )
            e2_case = e2_corpus_audit(repo_ops, memory_data)
            e2_visible_shas = e2_case.pop('_visible_shas')
            if not e2_case['base_not_visible']:
                raise AssertionError(
                    f"{instance_id}: strict base commit is visible in E2 corpus"
                )
        elif args.e3_prompt_arm:
            memory_data = repo_ops._load_repomem_repo(bug['repo'])
            if not memory_data:
                raise EpisodicMemoryUnavailable(
                    f"episodic index is unavailable for {bug['repo']}"
                )
            os.environ['REPOMEM_SUPPLEMENTAL_LEAKAGE'] = 'on'
            os.environ['REPOMEM_SEARCH_COMMIT_FORCE_TOPK'] = str(
                E2_TREATMENT_TOP_K
            )
            e3_case = e2_corpus_audit(repo_ops, memory_data)
            e3_case.pop('_visible_shas')
            if not e3_case['base_not_visible']:
                raise AssertionError(
                    f"{instance_id}: strict base commit is visible in E3 corpus"
                )
        if (
            (args.use_repomem and args.repomem_mode in ('episodic', 'both'))
            or bug.get('_fixed_memory_shas')
        ):
            validate_episodic_visibility(bug)
        fixed_memory = None
        if bug.get('_fixed_memory_shas'):
            if bug['base_commit'] != bug['_fixed_memory_base_commit']:
                raise ValueError(
                    f"{instance_id}: manifest base {bug['_fixed_memory_base_commit']} "
                    f"does not match dataset base {bug['base_commit']}"
                )
            packet = examine_commit(
                bug['_fixed_memory_shas'], display_issue=True, display_patch=True,
                max_commits=5,
            )
            if 'was not found in visible history' in packet:
                raise ValueError(
                    f"{instance_id}: one or more fixed memories are not visible"
                )
            fixed_memory = {
                'cell_id': bug['_experiment_cell_id'],
                'memory_shas': bug['_fixed_memory_shas'],
                'memory_sha': bug['_fixed_memory_shas'][0],
                'cohort': bug.get('_fixed_memory_cohort'),
                'audit_rank': bug.get('_fixed_memory_audit_rank'),
                'payload_sha256': hashlib.sha256(packet.encode()).hexdigest(),
                'payload_chars': len(packet),
                'payload_tokens': count_context_tokens(
                    [{'role': 'user', 'content': packet}], args.model, None
                ),
            }
        grounding_memory = build_grounding(
            get_graph(), instance_id, bug['problem_statement'],
            mode=args.entity_grounding, ablation=args.grounding_ablation,
        )

        # loc result
        raw_output_loc = []
        loc_trajs = {'trajs': []}
        total_prompt_tokens, total_completion_tokens = 0, 0

        for sample_index in range(args.num_samples):
            logger.info("=" * 60)
            logger.info(f"==== rank {rank} begin localizing {instance_id} ====")
            max_attempt_num = args.max_attempt_num
            while max_attempt_num:
                logger.info("=" * 60)
                logger.info(f"==== {instance_id} Count down: attempt {max_attempt_num} ====")
                loc_start_time = time.time()
                try:
                    """
                    Basic instructions:
                        - CodeAct instruction
                        - Few-shot Examples
                    """
                    if args.use_function_calling:
                        system_prompt = function_calling.SYSTEM_PROMPT
                        # system_prompt = CLAUDE_THINKING_INSTRUCTION
                    else:
                        system_prompt = prompt_manager.system_message
                        
                    messages: list[dict] = [{
                        "role": "system",
                        "content": system_prompt
                    }]
                        
                    if args.use_example:
                        messages.append({
                            "role": "user",
                            "content": prompt_manager.initial_user_message
                        })

                    logger.info(f"==== {instance_id} start auto search ====")
                    prompt_repomem_mode = (
                        args.repomem_mode
                        if args.repomem_task_prompt == 'forced-early'
                        else None
                    )
                    instruction = get_task_instruction(
                        bug,
                        include_pr=True,
                        include_hint=True,
                        repomem_mode=(
                            prompt_repomem_mode if args.use_repomem else None
                        ),
                    ) + grounding_prompt(grounding_memory)
                    if fixed_memory:
                        instruction += (
                            '\n\nA single frozen repository memory was retrieved before execution. '
                            'Treat it as a historical hint and verify it against the current codebase.\n'
                            '<fixed_episodic_memory>\n'
                            + packet
                            + '\n</fixed_episodic_memory>\n'
                        )
                    messages.append({
                        "role": "user",
                        "content": instruction,
                    })
                    
                    ctx = mp.get_context('fork')  # use fork to inherit context!!
                    result_queue = ctx.Manager().Queue()
                    tools = None
                    if args.use_function_calling:
                        tools = function_calling.get_tools(
                            codeact_enable_search_keyword=True,
                            codeact_enable_search_entity=True,
                            codeact_enable_tree_structure_traverser=True,
                            codeact_enable_repomem_episodic=args.use_repomem and args.repomem_mode in ('episodic', 'both'),
                            codeact_enable_repomem_semantic=args.use_repomem and args.repomem_mode in ('semantic', 'both'),
                            simple_desc = args.simple_desc,
                            repomem_tool_profile=args.repomem_tool_profile,
                        )
                    process = ctx.Process(target=auto_search_process, kwargs={
                        'result_queue': result_queue,
                        'model_name': args.model,
                        'messages': messages,
                        'fake_user_msg': auto_search.FAKE_USER_MSG_FOR_LOC,
                        'temp': args.temperature,
                        'top_p': args.top_p,
                        'sampling_seed': args.sampling_seed,
                        'tools': tools,
                        'max_iteration_num': args.max_iteration_num,
                        'request_timeout': args.request_timeout,
                        'use_function_calling': args.use_function_calling,
                        'context_token_budget': args.context_token_budget,
                        'tool_observation_token_cap': args.tool_observation_token_cap,
                        'fail_on_context_error': args.fail_on_context_error,
                        'bounded_graph_output': args.bounded_graph_output,
                        'bounded_entity_output': args.bounded_entity_output,
                        'bounded_search_output': args.bounded_search_output,
                        'enable_asq': args.enable_asq,
                        'enable_disambiguation': args.enable_disambiguation,
                        'required_response_model': args.required_response_model,
                    })
                    process.start()
                    process.join(timeout=args.timeout)
                    if process.is_alive():
                        logger.warning(f"{instance_id} attempt {max_attempt_num} execution flow "
                                        f"reconstruction exceeded timeout. Terminating.")
                        process.terminate()
                        process.join()
                        raise TimeoutError
                    
                    # loc_result, messages, traj_data = result_queue.get()
                    try:
                        result = result_queue.get_nowait()
                    except Empty:
                        logger.warning(f"{instance_id} attempt {max_attempt_num} returned no result. Try again.")
                        max_attempt_num = max_attempt_num - 1
                        continue
                    if isinstance(result, dict) and result.get('context_error'):
                        context_error = {
                            'instance_id': instance_id,
                            'attempt': max_attempt_num,
                            'error': result.get('error'),
                            'type': result.get('type'),
                            'context_audit': result.get('context_audit', []),
                        }
                        with output_file_lock:
                            append_to_jsonl(
                                context_error,
                                os.path.join(args.output_folder, 'context_errors.jsonl'),
                            )
                        if args.fail_on_context_error:
                            raise RuntimeError(
                                f"Fail-loud context error for {instance_id}: "
                                f"{result.get('error')}"
                            )
                        max_attempt_num = max_attempt_num - 1
                        continue
                    if isinstance(result, dict) and result.get('type') == 'ProviderModelDrift':
                        raise RuntimeError(result['error'])
                    if isinstance(result, dict) and 'error' in result and result['type'] == 'BadRequestError':
                        raise litellm.BadRequestError(result['error'], args.model, args.model.split('/')[0])
                        # print(f"Error occurred in subprocess: {result['error']}")
                    elif isinstance(result, dict) and 'error' in result:
                        logger.warning(f"{result['type']}: {result['error']}. Try again.")
                        max_attempt_num = max_attempt_num - 1
                        continue
                    else:
                        loc_result, messages, traj_data = result
                        if args.e1_supplemental_leakage:
                            e1_fork = run_e1_lazy_fork(
                                args=args,
                                ctx=ctx,
                                tools=tools,
                                control_output=loc_result,
                                control_messages=messages,
                                control_traj=traj_data,
                            )
                            arm_files, _, _ = get_loc_results_from_raw_outputs(
                                instance_id,
                                [loc_result, e1_fork['treatment_output']],
                            )
                            traj_data['e1_supplemental_leakage'] = {
                                'case': e1_case,
                                'frozen_config': {
                                    'model': args.model,
                                    'required_response_model': (
                                        args.required_response_model
                                    ),
                                    'temperature': args.temperature,
                                    'top_p': args.top_p,
                                    'sampling_seed': args.sampling_seed,
                                    'max_iteration_num': args.max_iteration_num,
                                    'tool_profile': args.repomem_tool_profile,
                                    'visibility_policy': (
                                        args.repomem_visibility_policy
                                    ),
                                    'bm25_profile': os.environ.get(
                                        'REPOMEM_BM25_PROFILE', 'bm25-code-v2'
                                    ),
                                    'search_commit_topk_cap': int(
                                        os.environ.get(
                                            'REPOMEM_SEARCH_COMMIT_TOPK_CAP', '5'
                                        )
                                    ),
                                },
                                'control_output': loc_result,
                                'control_found_files': arm_files[0],
                                'treatment_found_files': arm_files[1],
                                **e1_fork,
                            }
                        elif args.e2_search_topk:
                            e2_fork = run_e2_lazy_fork(
                                args=args,
                                ctx=ctx,
                                tools=tools,
                                control_output=loc_result,
                                control_messages=messages,
                                visible_shas=e2_visible_shas,
                            )
                            arm_files, _, _ = get_loc_results_from_raw_outputs(
                                instance_id,
                                [loc_result, e2_fork['treatment_output']],
                            )
                            traj_data['e2_search_topk'] = {
                                'case': e2_case,
                                'frozen_config': {
                                    'model': args.model,
                                    'required_response_model': (
                                        args.required_response_model
                                    ),
                                    'temperature': args.temperature,
                                    'top_p': args.top_p,
                                    'sampling_seed': args.sampling_seed,
                                    'max_iteration_num': args.max_iteration_num,
                                    'tool_profile': args.repomem_tool_profile,
                                    'visibility_policy': (
                                        args.repomem_visibility_policy
                                    ),
                                    'bm25_profile': os.environ.get(
                                        'REPOMEM_BM25_PROFILE', 'bm25-code-v2'
                                    ),
                                    'control_effective_top_k': E2_CONTROL_TOP_K,
                                    'treatment_effective_top_k': (
                                        E2_TREATMENT_TOP_K
                                    ),
                                    'top_k_semantics': 'per_query',
                                    'supplemental_leakage_propagation': 'on',
                                },
                                'control_output': loc_result,
                                'control_found_files': arm_files[0],
                                'treatment_found_files': arm_files[1],
                                **e2_fork,
                            }
                        elif args.e3_prompt_arm:
                            traj_data['e3_prompt'] = {
                                'arm': args.e3_prompt_arm,
                                'case': e3_case,
                                'frozen_config': {
                                    'model': args.model,
                                    'required_response_model': (
                                        args.required_response_model
                                    ),
                                    'temperature': args.temperature,
                                    'top_p': args.top_p,
                                    'sampling_seed': args.sampling_seed,
                                    'max_iteration_num': args.max_iteration_num,
                                    'tool_profile': args.repomem_tool_profile,
                                    'visibility_policy': (
                                        args.repomem_visibility_policy
                                    ),
                                    'bm25_profile': os.environ.get(
                                        'REPOMEM_BM25_PROFILE', 'bm25-code-v2'
                                    ),
                                    'effective_top_k': E2_TREATMENT_TOP_K,
                                    'top_k_semantics': 'per_query',
                                    'supplemental_leakage_propagation': 'on',
                                    'task_prompt': args.repomem_task_prompt,
                                },
                                'system_prompt_sha256': hashlib.sha256(
                                    system_prompt.encode()
                                ).hexdigest(),
                                'task_prompt_sha256': hashlib.sha256(
                                    instruction.encode()
                                ).hexdigest(),
                                'tool_schema_sha256': hashlib.sha256(
                                    json.dumps(
                                        tools, sort_keys=True, default=str
                                    ).encode()
                                ).hexdigest(),
                            }
                        
                except litellm.BadRequestError as e:
                    logger.warning(f'{e}. Try again.')
                    continue
                except APITimeoutError:
                    logger.warning(f"APITimeoutError. Try again.")
                    sleep(10)
                    continue
                except TimeoutError:
                    logger.warning(f"Processing time exceeded 15 minutes. Try again.")
                    max_attempt_num = max_attempt_num - 1
                    continue
                except litellm.exceptions.ContextWindowExceededError as e:
                    logger.warning(f'{e}. Try again.')
                    max_attempt_num = max_attempt_num - 1
                    continue

                loc_end_time = time.time()
                if not loc_result:
                    continue # empty result

                total_prompt_tokens += traj_data['usage'].get(
                    'total_prompt_tokens', traj_data['usage']['prompt_tokens']
                )
                total_completion_tokens += traj_data['usage'].get(
                    'total_completion_tokens', traj_data['usage']['completion_tokens']
                )
                if args.e1_supplemental_leakage:
                    incremental = traj_data['e1_supplemental_leakage'][
                        'incremental_usage'
                    ]
                    total_prompt_tokens += incremental.get(
                        'total_prompt_tokens', incremental.get('prompt_tokens', 0)
                    )
                    total_completion_tokens += incremental.get(
                        'total_completion_tokens',
                        incremental.get('completion_tokens', 0),
                    )
                elif args.e2_search_topk:
                    incremental = traj_data['e2_search_topk'][
                        'incremental_usage'
                    ]
                    total_prompt_tokens += incremental.get(
                        'total_prompt_tokens', incremental.get('prompt_tokens', 0)
                    )
                    total_completion_tokens += incremental.get(
                        'total_completion_tokens',
                        incremental.get('completion_tokens', 0),
                    )
                traj_data['time'] = loc_end_time - loc_start_time
                traj_data['fixed_memory'] = fixed_memory
                loc_trajs['trajs'].append(traj_data)
                if traj_data.get('context_guard', {}).get('enabled'):
                    audit_events = traj_data['context_guard'].get('events', [])
                    audit_row = {
                        'instance_id': instance_id,
                        'sample_index': sample_index,
                        'use_repomem': args.use_repomem,
                        'repomem_mode': args.repomem_mode if args.use_repomem else None,
                        'bounded_graph_output': args.bounded_graph_output,
                        'bounded_entity_output': args.bounded_entity_output,
                        'bounded_search_output': args.bounded_search_output,
                        **traj_data['context_guard'],
                        'summary': summarize_context_audit(audit_events),
                    }
                    with traj_file_lock:
                        append_to_jsonl(
                            audit_row,
                            os.path.join(args.output_folder, 'context_guard_audit.jsonl'),
                        )

                # generate correct output or finish last attempt
                raw_output_loc.append(loc_result)
                break

        if not raw_output_loc:
            # loc generalization failed
            logger.info(f"==== localizing {instance_id} failed, save empty outputs ====")
            loc_res = {
                    "instance_id": instance_id,
                    "experiment_cell_id": bug.get('_experiment_cell_id', instance_id),
                    "found_files": [[]],
                    "found_modules": [[]],
                    "found_entities": [[]],
                    "raw_output_loc": raw_output_loc,
                    "meta_data": {
                        'repo': bug['repo'],
                        'base_commit': bug['base_commit'],
                        'problem_statement': bug['problem_statement'],
                        'patch': bug['patch'],
                        'fixed_memory': fixed_memory,
                        # 'gt_file_changes': gt_file_changes
                    }
                }
            with output_file_lock:
                append_to_jsonl(loc_res, args.output_file)
        else:
            # process multiple loc outputs
            logger.info(f"==== localizing {instance_id} succeed, process multiple loc outputs ====")

            # all_valid_files = get_all_valid_files()
            all_found_files, all_found_modules, all_found_entities = get_loc_results_from_raw_outputs(
                instance_id, raw_output_loc
            )
            
            loc_res = {
                "instance_id": instance_id,
                "experiment_cell_id": bug.get('_experiment_cell_id', instance_id),
                "found_files": all_found_files,
                "found_modules": all_found_modules,
                "found_entities": all_found_entities,
                "raw_output_loc": raw_output_loc,
                "meta_data": {
                    'repo': bug['repo'],
                    'base_commit': bug['base_commit'],
                    'problem_statement': bug['problem_statement'],
                    'patch': bug['patch'],
                    'fixed_memory': fixed_memory,
                    # 'gt_file_changes': gt_file_changes
                }
            }
            loc_res['entity_grounding_memory'] = grounding_memory
            loc_res['channel_telemetry'] = [
                replay_telemetry(traj['messages'], grounding_memory)
                for traj in loc_trajs['trajs']
            ]
            
            with output_file_lock:
                append_to_jsonl(loc_res, args.output_file)

            cost = calc_cost(args.model, total_prompt_tokens, total_completion_tokens)
            loc_res['usage'] = {'cost($)': f'{round(cost, 5)}', 'prompt_tokens': total_prompt_tokens,
                                'completion_tokens': total_completion_tokens}
            loc_res['loc_trajs'] = loc_trajs
            traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
            with traj_file_lock:
                append_to_jsonl(loc_res, traj_file)

        reset_current_issue()


def localize(args):
    if args.use_repomem or args.fixed_episodic_memory_manifest:
        configure_repomem_index()
    bench_data = load_bench_dataset(args.dataset, args.split)
    bench_tests = filter_dataset(bench_data, 'instance_id', args.used_list, args.used_list_file)
    fixed_cells = load_fixed_memory_cells(args.fixed_episodic_memory_manifest)
    if fixed_cells:
        by_id = {row['instance_id']: row for row in bench_tests}
        missing = sorted({cell['instance_id'] for cell in fixed_cells} - set(by_id))
        if missing:
            raise ValueError(f'fixed memory manifest tasks missing from dataset: {missing}')
        expanded = []
        for cell in fixed_cells:
            bug = deepcopy(by_id[cell['instance_id']])
            bug['_experiment_cell_id'] = cell['cell_id']
            bug['_fixed_memory_shas'] = cell.get('memory_shas') or [cell['memory_sha']]
            bug['_fixed_memory_cohort'] = cell.get('cohort')
            bug['_fixed_memory_base_commit'] = cell['base_commit']
            bug['_fixed_memory_audit_rank'] = cell.get('audit_rank')
            expanded.append(bug)
        bench_tests = expanded
    if (
        (args.use_repomem and args.repomem_mode in ('episodic', 'both'))
        or fixed_cells
    ):
        visibility_audit = []
        unavailable = []
        for bug in bench_tests:
            try:
                audit, visible_count = validate_episodic_visibility(bug)
                visibility_audit.append({
                    'instance_id': bug['instance_id'],
                    'visible_count': visible_count,
                    **audit,
                })
            except EpisodicMemoryUnavailable as exc:
                unavailable.append({
                    'instance_id': bug.get('instance_id'),
                    'repo': bug.get('repo'),
                    'base_commit': bug.get('base_commit'),
                    'error': str(exc),
                })
        audit_path = os.path.join(args.output_folder, 'episodic_visibility_preflight.json')
        with open(audit_path, 'w', encoding='utf-8') as handle:
            json.dump({
                'policy': args.repomem_visibility_policy,
                'available': visibility_audit,
                'unavailable': unavailable,
            }, handle, indent=2)
        if unavailable:
            raise EpisodicMemoryUnavailable(
                f"{len(unavailable)} task(s) failed episodic visibility preflight; "
                f"see {audit_path}"
            )
    if args.eval_n_limit:
        eval_n_limit = min(args.eval_n_limit, len(bench_tests))
        bench_tests = bench_tests.select(range(0, eval_n_limit)) if hasattr(bench_tests, 'select') else bench_tests[:eval_n_limit]
        logging.info(f'Limiting evaluation to first {eval_n_limit} instances.')

    manager = mp.Manager()
    queue = manager.Queue()
    output_file_lock, traj_file_lock = manager.Lock(), manager.Lock()

    # collect processed instances
    processed_instance = []
    if os.path.exists(args.output_file):
        traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
        locs = load_jsonl(args.output_file)        
        if args.rerun_empty_location:
            traj_datas = load_jsonl(traj_file)
            backup_loc_output = backup_file(args.output_file)
            backup_traj_output = backup_file(traj_file)
            clear_file(args.output_file)
            clear_file(traj_file)
            for loc in locs:
                if loc['found_files'] != [[]]:
                    append_to_jsonl(loc, args.output_file)
                    processed_instance.append(
                        loc.get('experiment_cell_id', loc['instance_id'])
                    )
                    
            for loc_traj in traj_datas:
                if loc_traj['found_files'] != [[]]:
                    append_to_jsonl(loc_traj, traj_file)
        else:
            processed_instance = [
                loc.get('experiment_cell_id', loc['instance_id']) for loc in locs
            ]
    
    num_bugs = 0
    for bug in bench_tests:
        instance_id = bug["instance_id"]
        cell_id = bug.get('_experiment_cell_id', instance_id)
        if cell_id in processed_instance:
        # if instance_id in processed_instance or instance_id in filtered_instances:
            print(f"instance {instance_id} has already been processed, skip.")
        else:
            queue.put(bug)
            num_bugs += 1

    log_queue = manager.Queue()
    queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    queue_listener.start()
    mp.spawn(
        run_localize,
        nprocs=min(num_bugs, args.num_processes) if args.num_processes > 0 else num_bugs,
        args=(args, queue, log_queue, output_file_lock, traj_file_lock),
        join=True
    )
    queue_listener.stop()
    
    if args.rerun_empty_location:
        try:
            delete_file(backup_loc_output)
            delete_file(backup_traj_output)
        except:
            return


def merge(args):
    args.merge_file = os.path.join(args.output_folder, 'merged_' + os.path.basename(args.output_file))
    
    if args.ranking_method == 'mrr':
        args.merge_file = args.merge_file.replace('.jsonl', f'_{args.ranking_method}.jsonl')
        
    clear_file(args.merge_file)
    with open(args.output_file, 'r') as file:
        for line in file:
            loc_data = json.loads(line)
            if loc_data['found_files'] == [[]]:
                loc_data['found_files'] = []
                loc_data['found_modules'] = []
                loc_data['found_entities'] = []
            else:
                loc_data['found_files'] = loc_data['found_files']
                loc_data['found_modules'] = loc_data['found_modules']
                loc_data['found_entities'] = loc_data['found_entities']
                ranked_files, ranked_modules, ranked_funcs = merge_sample_locations(loc_data['found_files'], 
                                                                    loc_data['found_modules'],
                                                                    loc_data['found_entities'],
                                                                    ranking_method=args.ranking_method,
                                                                    )
                loc_data['found_files'] = ranked_files
                loc_data['found_modules'] = ranked_modules
                loc_data['found_entities'] = ranked_funcs
            with open(args.merge_file, 'a') as f:
                f.write(json.dumps(loc_data) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--use_example", action="store_true")
    parser.add_argument("--ranking_method", type=str, default='mrr',
                        choices=['mrr', 'majority'])
    
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--used_list", type=str, default='selected_ids')
    parser.add_argument("--used_list_file", type=str, default=None,
                        help="Path to a JSON list of instance IDs to run.")
    
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--merge_file", type=str, default="merged_loc_outputs.jsonl")
    
    parser.add_argument(
        "--model", type=str,
        default="openai/gpt-4o-2024-05-13",
    )
    parser.add_argument(
        "--required_response_model", default=None,
        help="Fail the trajectory if the provider returns a different concrete model id.",
    )
    parser.add_argument("--use_function_calling", action="store_true",
                        help='Enable function calling features of LLMs. If disabled, codeact will be used to support function calling.')
    parser.add_argument("--use_repomem", action="store_true",
                        help="Enable RepoMem memory tools.")
    parser.add_argument(
        "--fixed_episodic_memory_manifest",
        default=None,
        help="Frozen task-memory cells to inject directly; memory tools stay disabled.",
    )
    parser.add_argument("--repomem_mode", type=str, default="episodic",
                        choices=["episodic", "semantic", "both"],
                        help="RepoMem memory variant to expose to the agent.")
    parser.add_argument(
        "--repomem_tool_profile", choices=["current", "paper-v1"], default="current",
        help="Memory-tool schema and observation rendering profile; current remains backward compatible.",
    )
    parser.add_argument(
        "--repomem_visibility_policy",
        choices=["strict", "legacy"],
        default="strict",
        help=(
            "Commit visibility policy. Strict requires a verified base-specific ancestor "
            "manifest; legacy preserves the unsafe history-index fallback for comparisons."
        ),
    )
    parser.add_argument(
        "--e1_supplemental_leakage",
        action="store_true",
        help=(
            "Run the frozen-prefix E1 control (supplemental propagation OFF) "
            "and lazy-fork treatment (ON); requires affected strict cases."
        ),
    )
    parser.add_argument(
        "--e2_search_topk",
        action="store_true",
        help=(
            "Run frozen-prefix E2 with environment-assigned SearchCommit "
            "Top5 control and lazy-fork Top20 treatment."
        ),
    )
    parser.add_argument(
        "--repomem_task_prompt",
        choices=["forced-early", "original"],
        default="forced-early",
        help=(
            "Keep the current main-task episodic instruction or expose only "
            "the original LocAgent task prompt while retaining memory tools."
        ),
    )
    parser.add_argument(
        "--e3_prompt_arm",
        choices=["control", "treatment"],
        default=None,
        help=(
            "Record one independently executed E3 prompt arm; control requires "
            "forced-early and treatment requires original."
        ),
    )
    parser.add_argument(
        "--repomem_semantic_layout", choices=["repo", "instance"], default="repo",
        help="Use one repository bank or a task-aligned bank at INDEX_ROOT/<instance_id>.",
    )
    parser.add_argument("--entity_grounding", choices=["off", "real", "placebo"], default="off")
    parser.add_argument(
        "--enable_asq",
        action="store_true",
        help="Run LocAgent search actions through the OrcaLoca priority queue.",
    )
    parser.add_argument(
        "--enable_disambiguation",
        action="store_true",
        help="Decompose detected ambiguity into priority-2 exact searches (requires --enable_asq).",
    )
    parser.add_argument("--grounding_ablation", choices=["full", "no_disambiguation", "no_file_class"], default="full")
    parser.add_argument("--simple_desc", action="store_true", 
                        help="Use simplified function descriptions due to certain LLM limitations. Set to False for better performance when using Claude.")
    
    parser.add_argument("--max_attempt_num", type=int, default=1, 
                        help='Only use in generating training trajectories.')
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_processes", type=int, default=-1)
    
    parser.add_argument("--log_level", type=str, default='INFO')
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--request_timeout", type=int, default=120)
    parser.add_argument("--max_iteration_num", type=int, default=20)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM sampling temperature. The historical effective default is 1.0.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Optional nucleus-sampling value; omitted from provider requests by default.",
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=None,
        help="Best-effort provider seed recorded with every response.",
    )
    parser.add_argument(
        "--context_token_budget",
        type=int,
        default=0,
        help="Maximum prompt tokens before a request; 0 disables the guard.",
    )
    parser.add_argument(
        "--tool_observation_token_cap",
        type=int,
        default=0,
        help="Maximum tokens retained from one tool observation; 0 disables it.",
    )
    parser.add_argument(
        "--fail_on_context_error",
        action="store_true",
        help="Abort instead of restarting after a provider context-limit error.",
    )
    parser.add_argument(
        "--bounded_graph_output",
        action="store_true",
        help="Cap graph traversal depth and output lines for context-safe runs.",
    )
    parser.add_argument(
        "--bounded_entity_output",
        action="store_true",
        help="Cap complete entity/file output lines for context-safe runs.",
    )
    parser.add_argument(
        "--bounded_search_output",
        action="store_true",
        help="Cap aggregate code-search output lines for context-safe runs.",
    )
    parser.add_argument("--rerun_empty_location", action="store_true")
    args = parser.parse_args()
    os.environ["REPOMEM_TOOL_PROFILE"] = args.repomem_tool_profile
    os.environ["REPOMEM_SEMANTIC_LAYOUT"] = args.repomem_semantic_layout
    os.environ["REPOMEM_VISIBILITY_POLICY"] = args.repomem_visibility_policy
    if not args.e1_supplemental_leakage:
        os.environ["REPOMEM_SUPPLEMENTAL_LEAKAGE"] = "on"
    if not args.e2_search_topk and not args.e3_prompt_arm:
        os.environ.pop("REPOMEM_SEARCH_COMMIT_FORCE_TOPK", None)
    if args.use_repomem and args.repomem_mode in ('episodic', 'both') and args.repomem_visibility_policy == 'legacy':
        logging.warning(
            "UNSAFE LEGACY EPISODIC VISIBILITY ENABLED: results may include non-ancestor history"
        )
    if args.enable_disambiguation and not args.enable_asq:
        parser.error("--enable_disambiguation requires --enable_asq")
    if args.e1_supplemental_leakage:
        if not args.use_repomem or args.repomem_mode != "episodic":
            parser.error("E1 requires --use_repomem --repomem_mode episodic")
        if args.repomem_visibility_policy != "strict":
            parser.error("E1 requires --repomem_visibility_policy strict")
        if args.repomem_tool_profile != "current":
            parser.error("E1 freezes --repomem_tool_profile current")
        if args.num_samples != 1:
            parser.error("E1 requires --num_samples 1 (one paired unit per case)")
        if args.enable_asq or args.enable_disambiguation:
            parser.error("E1 lazy replay does not support stateful ASQ scheduling")
        if args.context_token_budget or args.tool_observation_token_cap:
            parser.error("E1 freezes unguarded tool observations for exact replay")
        if int(os.environ.get("REPOMEM_SEARCH_COMMIT_TOPK_CAP", "5")) != 5:
            parser.error("E1 freezes REPOMEM_SEARCH_COMMIT_TOPK_CAP=5")
    if args.e2_search_topk:
        if args.e1_supplemental_leakage:
            parser.error("E1 and E2 cannot run together")
        if not args.use_repomem or args.repomem_mode != "episodic":
            parser.error("E2 requires --use_repomem --repomem_mode episodic")
        if args.repomem_visibility_policy != "strict":
            parser.error("E2 requires --repomem_visibility_policy strict")
        if args.repomem_tool_profile != "current":
            parser.error("E2 freezes --repomem_tool_profile current")
        if args.num_samples != 1:
            parser.error("E2 requires --num_samples 1 (one paired unit per case)")
        if args.enable_asq or args.enable_disambiguation:
            parser.error("E2 lazy replay does not support stateful ASQ scheduling")
        if args.context_token_budget or args.tool_observation_token_cap:
            parser.error("E2 freezes unguarded tool observations for exact replay")
        os.environ["REPOMEM_SEARCH_COMMIT_FORCE_TOPK"] = str(
            E2_CONTROL_TOP_K
        )
    if args.e3_prompt_arm:
        if args.e1_supplemental_leakage or args.e2_search_topk:
            parser.error("E3 cannot run together with E1 or E2")
        if not args.use_repomem or args.repomem_mode != "episodic":
            parser.error("E3 requires --use_repomem --repomem_mode episodic")
        if args.repomem_visibility_policy != "strict":
            parser.error("E3 requires --repomem_visibility_policy strict")
        if args.repomem_tool_profile != "current":
            parser.error("E3 reuses E2's --repomem_tool_profile current")
        expected_prompt = (
            "forced-early" if args.e3_prompt_arm == "control" else "original"
        )
        if args.repomem_task_prompt != expected_prompt:
            parser.error(
                f"E3 {args.e3_prompt_arm} requires "
                f"--repomem_task_prompt {expected_prompt}"
            )
        if args.num_samples != 1:
            parser.error("E3 requires --num_samples 1 per experiment cell")
        if args.enable_asq or args.enable_disambiguation:
            parser.error("E3 freezes ASQ and disambiguation OFF")
        if args.context_token_budget or args.tool_observation_token_cap:
            parser.error("E3 freezes unguarded tool observations")
        os.environ["REPOMEM_SEARCH_COMMIT_FORCE_TOPK"] = str(
            E2_TREATMENT_TOP_K
        )
    if args.enable_asq and not args.use_function_calling:
        parser.error("--enable_asq requires --use_function_calling")
    if args.fixed_episodic_memory_manifest and args.use_repomem:
        parser.error("fixed memory injection cannot be combined with memory tools")

    args.output_file = os.path.join(args.output_folder, args.output_file)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        level=logging.getLevelName(args.log_level),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.output_folder}/localize.log"),
            logging.StreamHandler()
        ]
    )
    
    if args.localize:
        localize(args)
    
    
    if args.merge:
        merge(args)


if __name__ == "__main__":

    start_time = time.time()
    main()
    end_time = time.time()
    logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))
