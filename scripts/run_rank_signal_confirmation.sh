#!/usr/bin/env bash
set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

cd /data/wjl/RepoMem
export PYTHONPATH="/data/wjl/RepoMem:${PYTHONPATH:-}"
source /data/wjl/RepoMem/openaigpt4o_env
export GRAPH_INDEX_DIR=/data/wjl/RepoMem/index_data/SWE-bench_Verified/graph_index_v2.3
export BM25_INDEX_DIR=/data/wjl/RepoMem/index_data/SWE-bench_Verified/BM25_index
export REPOMEM_INDEX_DIR=/data/wjl/RepoMem/repomem_index/verified_full500_episodic_with_issues_combined
export REPOMEM_PATCH_MAX_CHARS=12000
export REPOMEM_ISSUE_MAX_CHARS=200000
export MPLCONFIGDIR=/tmp/repomem-rank-confirmation

python_bin=/data/conda_envs/locagent/bin/python
model=${EVAL_MODEL:-gpt-4o}
root=results/stale_memory_rank_confirmation
manifest="${root}/manifest.json"
dataset="${root}/selected_dataset.jsonl"
task_ids="${root}/task_ids.json"

if [[ ! -f "${manifest}" ]]; then
    echo "Missing frozen manifest. Run rank_signal_confirmation.py prepare first." >&2
    exit 1
fi

read -r status fixed_expected control_expected total_expected < <(
    "${python_bin}" -c \
        'import json,sys; x=json.load(open(sys.argv[1])); e=x["expected"]; print(x["status"], e["fixed_cells_per_repeat"], e["control_tasks_per_repeat"], e["successful_trajectories"])' \
        "${manifest}"
)
if [[ "${status}" != "FROZEN_READY" ]]; then
    echo "Density gate did not pass: ${status}" >&2
    exit 1
fi
if [[ ${total_expected} -gt 288 ]]; then
    echo "Frozen design exceeds the 288-trajectory cap." >&2
    exit 1
fi
if [[ -s "${root}/missing_index_task_ids.json" ]] && \
   [[ "$(tr -d '[:space:]' < "${root}/missing_index_task_ids.json")" != "[]" ]]; then
    echo "Missing graph/BM25 indexes listed in ${root}/missing_index_task_ids.json" >&2
    exit 1
fi

run_arm() {
    local arm=$1 repeat=$2 seed=$3 expected=$4
    local output="${root}/runs/${arm}_r${repeat}"
    local fixed_args=()
    if [[ "${arm}" == "fixed" ]]; then
        fixed_args=(--fixed_episodic_memory_manifest "${manifest}")
    fi
    local attempts=0 resume_args=()
    while true; do
        resume_args=()
        if [[ -e "${output}" ]]; then
            if [[ ! -f "${output}/loc_trajs.jsonl" || ! -f "${output}/loc_outputs.jsonl" || ! -f "${output}/args.json" ]]; then
                echo "Refusing malformed run directory: ${output}" >&2
                exit 1
            fi
            local count
            count=$(wc -l < "${output}/loc_trajs.jsonl")
            if [[ ${count} -eq ${expected} ]]; then
                "${python_bin}" scripts/rank_signal_confirmation.py validate-run \
                    --run-dir "${output}" --arm "${arm}" \
                    --repeat "${repeat}" --expected "${expected}"
                echo "Validated and skipping completed run: ${output}"
                return
            fi
            "${python_bin}" scripts/rank_signal_confirmation.py validate-run \
                --run-dir "${output}" --arm "${arm}" \
                --repeat "${repeat}" --expected "${expected}" --allow-retry
            resume_args=(--rerun_empty_location)
        fi
        if [[ ${dry_run} -eq 1 ]]; then
            echo "DRY RUN arm=${arm} repeat=${repeat} seed=${seed} expected=${expected} resume=${#resume_args[@]}"
            return
        fi
        if [[ ${attempts} -ge 3 ]]; then
            echo "Run remains incomplete after 3 attempts: ${output}" >&2
            exit 1
        fi
        attempts=$((attempts + 1))
        "${python_bin}" scripts/rank_signal_confirmation.py preflight \
            --model "${model}" --seed "${seed}"
        "${python_bin}" auto_search_main.py \
            --dataset "${dataset}" --split test \
            --used_list_file "${task_ids}" \
            --model "${model}" --localize --merge \
            --output_folder "${output}" --num_processes 4 \
            --num_samples 1 --ranking_method mrr \
            --use_function_calling --simple_desc \
            --timeout 900 --request_timeout 120 \
            --max_iteration_num 20 --max_attempt_num 1 \
            --temperature 0 --sampling_seed "${seed}" \
            --context_token_budget 100000 \
            --tool_observation_token_cap 20000 \
            --fail_on_context_error --bounded_graph_output \
            --bounded_entity_output --bounded_search_output \
            "${resume_args[@]}" "${fixed_args[@]}"
    done
}

# Counterbalanced order; fixed and control share a seed within each repeat.
run_arm fixed 1 101 "${fixed_expected}"
run_arm no_memory 1 101 "${control_expected}"
run_arm no_memory 2 202 "${control_expected}"
run_arm fixed 2 202 "${fixed_expected}"
run_arm fixed 3 303 "${fixed_expected}"
run_arm no_memory 3 303 "${control_expected}"

if [[ ${dry_run} -eq 0 ]]; then
    "${python_bin}" scripts/rank_signal_confirmation.py analyze
fi
