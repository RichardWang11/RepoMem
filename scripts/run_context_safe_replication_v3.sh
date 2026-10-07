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

root=results/context_safe_m3a/replication_v3
runs_root="${root}/runs"
selection=reproduce_lists/verified_m3a_matched_15.json

mkdir -p "${runs_root}"

run_repeat() {
    local arm=$1
    local repeat=$2
    local output_dir="${runs_root}/${arm}_r${repeat}"
    local memory_args=()
    if [[ "${arm}" == "memory" ]]; then
        memory_args=(--use_repomem --repomem_mode episodic)
    fi
    if [[ -e "${output_dir}" ]]; then
        echo "Refusing to overwrite replication output: ${output_dir}" >&2
        return 1
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
        --temperature 1.0 \
        --context_token_budget 100000 \
        --tool_observation_token_cap 20000 \
        --fail_on_context_error \
        --bounded_graph_output \
        --bounded_entity_output \
        --bounded_search_output \
        "${memory_args[@]}"

    /data/conda_envs/locagent/bin/python scripts/repair_loc_outputs_from_raw.py \
        --input "${output_dir}/loc_trajs.jsonl" \
        --output "${output_dir}/loc_trajs.repaired.jsonl" \
        | tee "${output_dir}/parser_repair.log"
    /data/conda_envs/locagent/bin/python scripts/validate_context_safe_repeat.py \
        --run_dir "${output_dir}" \
        --expected_cases 15
    /data/conda_envs/locagent/bin/python scripts/validate_replication_v3_repeat.py \
        --run_dir "${output_dir}" \
        --arm "${arm}" \
        --expected_cases 15
}

# Position-balanced order: both arms have mean position 4.5 and variance 5.25.
run_repeat baseline 1
run_repeat memory 1
run_repeat memory 2
run_repeat baseline 2
run_repeat memory 3
run_repeat baseline 3
run_repeat baseline 4
run_repeat memory 4

/data/conda_envs/locagent/bin/python scripts/analyze_context_safe_m3a_audit.py \
    --run_root "${runs_root}" \
    --output_dir "${root}/analysis" \
    --expected_cases 15

/data/conda_envs/locagent/bin/python scripts/analyze_context_safe_replication_v3.py \
    --v3_baseline \
        "${runs_root}/baseline_r1/loc_trajs.repaired.jsonl" \
        "${runs_root}/baseline_r2/loc_trajs.repaired.jsonl" \
        "${runs_root}/baseline_r3/loc_trajs.repaired.jsonl" \
        "${runs_root}/baseline_r4/loc_trajs.repaired.jsonl" \
    --v3_memory \
        "${runs_root}/memory_r1/loc_trajs.repaired.jsonl" \
        "${runs_root}/memory_r2/loc_trajs.repaired.jsonl" \
        "${runs_root}/memory_r3/loc_trajs.repaired.jsonl" \
        "${runs_root}/memory_r4/loc_trajs.repaired.jsonl" \
    --v2_baseline \
        results/context_safe_m3a/runs_v2/baseline_r1/loc_trajs.repaired.jsonl \
        results/context_safe_m3a/runs_v2/baseline_r2/loc_trajs.repaired.jsonl \
        results/context_safe_m3a/runs_v2/baseline_r3/loc_trajs.repaired.jsonl \
        results/context_safe_m3a/runs_v2/baseline_r4/loc_trajs.repaired.jsonl \
    --v2_memory \
        results/context_safe_m3a/runs_v2/memory_r1/loc_trajs.repaired.jsonl \
        results/context_safe_m3a/runs_v2/memory_r2/loc_trajs.repaired.jsonl \
        results/context_safe_m3a/runs_v2/memory_r3/loc_trajs.repaired.jsonl \
        results/context_safe_m3a/runs_v2/memory_r4/loc_trajs.repaired.jsonl \
    --legacy_baseline \
        results/check9_label_stability_gpt4o_baseline_harm30/loc_trajs.repaired.jsonl \
        results/m3_memory_variance_runs/baseline_r1/loc_trajs.repaired.jsonl \
        results/m3_memory_variance_runs/baseline_r2/loc_trajs.repaired.jsonl \
        results/m3_memory_variance_runs/baseline_r3/loc_trajs.repaired.jsonl \
    --legacy_memory \
        results/check9_label_stability_gpt4o_episodic_harm30/loc_trajs.repaired.jsonl \
        results/m3_memory_variance_runs/memory_r1/loc_trajs.repaired.jsonl \
        results/m3_memory_variance_runs/memory_r2/loc_trajs.repaired.jsonl \
        results/m3_memory_variance_runs/memory_r3/loc_trajs.repaired.jsonl \
    --output_dir "${root}/analysis"
