#!/usr/bin/env bash
set -euo pipefail

cd /data/wjl/RepoMem
export PYTHONPATH="/data/wjl/RepoMem:${PYTHONPATH:-}"
source /data/wjl/RepoMem/openaigpt4o_env
export GRAPH_INDEX_DIR=/data/wjl/RepoMem/index_data/SWE-bench_Verified/graph_index_v2.3
export BM25_INDEX_DIR=/data/wjl/RepoMem/index_data/SWE-bench_Verified/BM25_index
export REPOMEM_INDEX_DIR=/data/wjl/RepoMem/repomem_index/verified_full500_episodic_with_issues_combined
export REPOMEM_PATCH_MAX_CHARS=12000
export REPOMEM_ISSUE_MAX_CHARS=200000
export MPLCONFIGDIR=/tmp/dual-bm25-nomic-agent-mpl

root=results/dual_bm25_nomic/agent
dataset="${root}/selected_dataset.jsonl"
python_bin=/data/conda_envs/locagent/bin/python
model=${EVAL_MODEL:-gpt-4o}
expected=$(wc -l < "${dataset}")

is_complete() {
    local path=$1
    [[ -s "${path}" ]] && [[ $("${python_bin}" -c \
        'import json,sys; print(len({json.loads(x)["instance_id"] for x in open(sys.argv[1]) if x.strip()}))' \
        "${path}") -eq ${expected} ]]
}

if [[ ! -s "${root}/bm25_manifest.json" || ! -s "${root}/nomic_manifest.json" ]]; then
    echo "No eligible Agent manifests. Run retrieval analysis first." >&2
    exit 2
fi

run_arm() {
    local arm=$1 repeat=$2 seed=$3
    local output="${root}/runs/${arm}_r${repeat}"
    if is_complete "${output}/merged_loc_outputs_mrr.jsonl"; then
        echo "Skipping completed ${arm} repeat ${repeat}"
        return
    fi
    "${python_bin}" auto_search_main.py \
        --dataset "${dataset}" --split test \
        --model "${model}" --localize --merge \
        --fixed_episodic_memory_manifest "${root}/${arm}_manifest.json" \
        --output_folder "${output}" --num_processes 4 \
        --num_samples 1 --ranking_method mrr \
        --use_function_calling --simple_desc \
        --timeout 900 --request_timeout 120 \
        --max_iteration_num 20 --max_attempt_num 1 \
        --temperature 0 --sampling_seed "${seed}" \
        --context_token_budget 100000 --tool_observation_token_cap 20000 \
        --fail_on_context_error --bounded_graph_output \
        --bounded_entity_output --bounded_search_output
    if ! is_complete "${output}/merged_loc_outputs_mrr.jsonl"; then
        echo "Incomplete ${arm} repeat ${repeat}: expected ${expected} merged outputs" >&2
        exit 1
    fi
}

run_arm bm25 1 101
run_arm nomic 1 101
run_arm nomic 2 202
run_arm bm25 2 202
"${python_bin}" scripts/dual_bm25_nomic_experiment.py analyze-agent
