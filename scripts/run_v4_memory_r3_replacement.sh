#!/usr/bin/env bash
set -euo pipefail

cd /data/wjl/RepoMem
export PYTHONPATH="/data/wjl/RepoMem:${PYTHONPATH:-}"
source /data/wjl/RepoMem/openaigpt4o_env

export MPLCONFIGDIR=/tmp/repomem-matplotlib
export REPOMEM_INDEX_DIR=/data/wjl/RepoMem/repomem_index/verified_full500_episodic_with_issues_combined
export REPOMEM_PATCH_MAX_CHARS=12000
export REPOMEM_SEARCH_COMMIT_TOPK_CAP=20
export REPOMEM_SEARCH_COMMIT_SCHEMA_DEFAULT=20
export REPOMEM_EXAMINE_MAX_COMMITS=10
export REPOMEM_ISSUE_MAX_CHARS=200000

output_dir=results/context_safe_m3a/replication_v4/replacements/memory_r3_matplotlib_25479
if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite replacement output: ${output_dir}" >&2
    exit 1
fi

/data/conda_envs/locagent/bin/python auto_search_main.py \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --used_list_file reproduce_lists/v4_memory_r3_replacement.json \
    --model gpt-4o \
    --localize \
    --merge \
    --output_folder "${output_dir}" \
    --num_processes 1 \
    --num_samples 1 \
    --ranking_method mrr \
    --use_function_calling \
    --simple_desc \
    --timeout 240 \
    --request_timeout 60 \
    --max_iteration_num 8 \
    --max_attempt_num 1 \
    --temperature 1.0 \
    --context_token_budget 100000 \
    --tool_observation_token_cap 20000 \
    --fail_on_context_error \
    --bounded_graph_output \
    --bounded_entity_output \
    --bounded_search_output \
    --use_repomem \
    --repomem_mode episodic

/data/conda_envs/locagent/bin/python scripts/validate_context_safe_repeat.py \
    --run_dir "${output_dir}" \
    --expected_cases 1

/data/conda_envs/locagent/bin/python scripts/merge_v4_replacement_trajectory.py \
    --run_dir results/context_safe_m3a/replication_v4/runs/memory_r3 \
    --replacement_dir "${output_dir}" \
    --instance_id matplotlib__matplotlib-25479 \
    --reason execution_flow_timeout \
    --expected_cases 15

/data/conda_envs/locagent/bin/python scripts/repair_loc_outputs_from_raw.py \
    --input results/context_safe_m3a/replication_v4/runs/memory_r3/loc_trajs.jsonl \
    --output results/context_safe_m3a/replication_v4/runs/memory_r3/loc_trajs.repaired.jsonl \
    | tee results/context_safe_m3a/replication_v4/runs/memory_r3/parser_repair.log

/data/conda_envs/locagent/bin/python scripts/validate_context_safe_repeat.py \
    --run_dir results/context_safe_m3a/replication_v4/runs/memory_r3 \
    --expected_cases 15
/data/conda_envs/locagent/bin/python scripts/validate_replication_v4.py \
    --run_dir results/context_safe_m3a/replication_v4/runs/memory_r3 \
    --arm memory \
    --repeat 3 \
    --expected_cases 15
