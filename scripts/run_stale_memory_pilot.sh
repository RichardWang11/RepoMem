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
export MPLCONFIGDIR=/tmp/repomem-stale-memory-pilot

python_bin=/data/conda_envs/locagent/bin/python
model=${EVAL_MODEL:-gpt-4o}
root=results/stale_memory_causal_pilot
manifest="${root}/pilot_manifest.json"
fixed_ids="${root}/fixed_task_ids.json"
control_ids="${root}/no_memory_task_ids.json"

if [[ ! -f "${manifest}" ]]; then
    "${python_bin}" scripts/stale_memory_pilot.py prepare
fi

run_arm() {
    local arm=$1
    local repeat=$2
    local seed=$3
    local output="${root}/runs/${arm}_r${repeat}"
    local expected=58
    if [[ "${arm}" == "fixed" ]]; then
        expected=65
    fi
    local selection="${control_ids}"
    local fixed_args=()
    if [[ "${arm}" == "fixed" ]]; then
        selection="${fixed_ids}"
        fixed_args=(--fixed_episodic_memory_manifest "${manifest}")
    fi
    local attempts=0
    local resume_args=()
    while true; do
        resume_args=()
        if [[ -e "${output}" ]]; then
            if [[ ! -f "${output}/loc_trajs.jsonl" || ! -f "${output}/loc_outputs.jsonl" || ! -f "${output}/args.json" ]]; then
                echo "Refusing malformed run directory: ${output}" >&2
                exit 1
            fi
            local trajectory_count
            trajectory_count=$(wc -l < "${output}/loc_trajs.jsonl")
            if [[ ${trajectory_count} -eq ${expected} ]]; then
                "${python_bin}" scripts/stale_memory_pilot.py validate-run \
                    --run-dir "${output}" --arm "${arm}" --repeat "${repeat}" \
                    --expected "${expected}"
                echo "Validated and skipping completed run: ${output}"
                return
            fi
            "${python_bin}" scripts/stale_memory_pilot.py validate-retry \
                --run-dir "${output}" --arm "${arm}" --repeat "${repeat}" \
                --expected "${expected}"
            resume_args=(--rerun_empty_location)
        fi
        if [[ ${dry_run} -eq 1 ]]; then
            echo "DRY RUN arm=${arm} repeat=${repeat} seed=${seed} output=${output} resume=${#resume_args[@]}"
            return
        fi
        if [[ ${attempts} -ge 3 ]]; then
            echo "Run remains incomplete after 3 attempts: ${output}" >&2
            exit 1
        fi
        attempts=$((attempts + 1))
        "${python_bin}" scripts/stale_memory_pilot.py preflight \
            --model "${model}" --seed "${seed}"
        "${python_bin}" auto_search_main.py \
        --dataset princeton-nlp/SWE-bench_Verified \
        --split test \
        --used_list_file "${selection}" \
        --model "${model}" \
        --localize --merge \
        --output_folder "${output}" \
        --num_processes 4 \
        --num_samples 1 \
        --ranking_method mrr \
        --use_function_calling \
        --simple_desc \
        --timeout 900 \
        --request_timeout 120 \
        --max_iteration_num 20 \
        --max_attempt_num 1 \
        --temperature 0 \
        --sampling_seed "${seed}" \
        --context_token_budget 100000 \
        --tool_observation_token_cap 20000 \
        --fail_on_context_error \
        --bounded_graph_output \
        --bounded_entity_output \
        --bounded_search_output \
        "${resume_args[@]}" \
        "${fixed_args[@]}"
    done
}

# Counterbalanced arm order across the three preregistered seeds.
run_arm fixed 1 42
run_arm no_memory 1 42
run_arm no_memory 2 43
run_arm fixed 2 43
run_arm fixed 3 44
run_arm no_memory 3 44

if [[ ${dry_run} -eq 0 ]]; then
    "${python_bin}" scripts/stale_memory_pilot.py analyze
fi
