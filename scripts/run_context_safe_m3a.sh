#!/usr/bin/env bash
set -euo pipefail

cd /data/wjl/RepoMem
export PYTHONPATH="${PYTHONPATH:-}"
source /data/wjl/RepoMem/openaigpt4o_env

export MPLCONFIGDIR=/tmp/repomem-matplotlib
export REPOMEM_INDEX_DIR=/data/wjl/RepoMem/repomem_index/verified_full500_episodic_with_issues_combined
export REPOMEM_PATCH_MAX_CHARS=12000
export REPOMEM_SEARCH_COMMIT_TOPK_CAP=20
export REPOMEM_SEARCH_COMMIT_SCHEMA_DEFAULT=20
export REPOMEM_EXAMINE_MAX_COMMITS=10
export REPOMEM_ISSUE_MAX_CHARS=200000

root=results/context_safe_m3a/runs_v2
selection=reproduce_lists/verified_m3a_matched_15.json

run_repeat() {
    local arm=$1
    local repeat=$2
    local output_dir="${root}/${arm}_r${repeat}"
    local memory_args=()
    if [[ "${arm}" == "memory" ]]; then
        memory_args=(--use_repomem --repomem_mode episodic)
    fi

    /data/conda_envs/locagent/bin/python auto_search_main.py \
        --dataset princeton-nlp/SWE-bench_Verified \
        --split test \
        --used_list_file "${selection}" \
        --model gpt-4o \
        --localize \
        --merge \
        --output_folder "${output_dir}" \
        --num_processes 4 \
        --num_samples 1 \
        --ranking_method mrr \
        --use_function_calling \
        --simple_desc \
        --timeout 240 \
        --request_timeout 60 \
        --max_iteration_num 8 \
        --max_attempt_num 1 \
        --context_token_budget 100000 \
        --tool_observation_token_cap 20000 \
        --fail_on_context_error \
        --bounded_graph_output \
        --bounded_entity_output \
        --bounded_search_output \
        "${memory_args[@]}"

    /data/conda_envs/locagent/bin/python scripts/repair_loc_outputs_from_raw.py \
        --input "${output_dir}/loc_trajs.jsonl" \
        --output "${output_dir}/loc_trajs.repaired.jsonl"
    /data/conda_envs/locagent/bin/python scripts/validate_context_safe_repeat.py \
        --run_dir "${output_dir}" \
        --expected_cases 15
}

for repeat in 1 2 3 4; do
    run_repeat baseline "${repeat}"
done
for repeat in 1 2 3 4; do
    run_repeat memory "${repeat}"
done
