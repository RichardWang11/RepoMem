#!/usr/bin/env bash
set -euo pipefail

cd /data/wjl/RepoMem
export PYTHONPATH="/data/wjl/RepoMem:${PYTHONPATH:-}"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

root=results/e2_search_topk
dataset=results/strict_visibility_pilot/preflight/verified500.jsonl
index_dir=repomem_index/verified_full500_episodic_with_issues_combined
mkdir -p "${root}/preflight" "${root}/run"

export GRAPH_INDEX_DIR=/data/wjl/RepoMem/index_data/SWE-bench_Verified/graph_index_v2.3
export BM25_INDEX_DIR=/data/wjl/RepoMem/index_data/SWE-bench_Verified/BM25_index
export REPOMEM_INDEX_DIR=/data/wjl/RepoMem/${index_dir}
export REPOMEM_VISIBILITY_POLICY=strict
export REPOMEM_BM25_PROFILE=bm25-code-v2
export REPOMEM_SEARCH_COMMIT_TOPK_CAP=5
export REPOMEM_SEARCH_COMMIT_SCHEMA_DEFAULT=5
export REPOMEM_LEAKAGE_FILTER=1
export REPOMEM_SUPPLEMENTAL_LEAKAGE=on
export REPOMEM_SEARCH_COMMIT_FORCE_TOPK=5
export MPLCONFIGDIR=/tmp/repomem-e2-mpl

/data/conda_envs/locagent/bin/python \
    scripts/prepare_e2_search_topk.py \
    --dataset "${dataset}" --output-dir "${root}/preflight"

if (( dry_run )); then
    echo "DRY RUN: E2 Top5/Top20 lazy-fork pilot on 12 frozen cases"
    exit 0
fi

source /data/wjl/RepoMem/openaigpt4o_env
# The credential file carries a historical 200-case index default.  Restore
# E2's frozen combined index after sourcing it so all 12 pilot repositories
# remain available to the strict preflight and both experiment arms.
export REPOMEM_INDEX_DIR=/data/wjl/RepoMem/${index_dir}
/data/conda_envs/locagent/bin/python scripts/preflight_exact_model.py \
    --model openai/gpt-4o \
    --required_response_model gpt-4o-2024-11-20 \
    --output "${root}/preflight/model.json"

/data/conda_envs/locagent/bin/python auto_search_main.py \
    --dataset "${dataset}" --split test \
    --used_list_file "${root}/preflight/selected_ids.json" \
    --model openai/gpt-4o \
    --required_response_model gpt-4o-2024-11-20 \
    --localize --merge --output_folder "${root}/run" \
    --num_processes 1 --num_samples 1 --ranking_method mrr \
    --use_function_calling --simple_desc \
    --use_repomem --repomem_mode episodic \
    --repomem_tool_profile current --repomem_visibility_policy strict \
    --e2_search_topk \
    --timeout 900 --request_timeout 120 \
    --max_iteration_num 20 --max_attempt_num 1 \
    --temperature 1.0 --sampling_seed 44

/data/conda_envs/locagent/bin/python scripts/analyze_e2_search_topk.py \
    --run-dir "${root}/run" --dataset "${dataset}" \
    --selected-ids "${root}/preflight/selected_ids.json"
