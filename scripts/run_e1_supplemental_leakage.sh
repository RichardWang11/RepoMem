#!/usr/bin/env bash
set -euo pipefail

cd /data/wjl/RepoMem
export PYTHONPATH="/data/wjl/RepoMem:${PYTHONPATH:-}"

dry_run=0
limit=20
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
if [[ "${1:-}" == "--limit" ]]; then
    limit=${2:?--limit requires an integer}
    shift 2
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run] [--limit N]" >&2
    exit 2
fi

root=results/e1_supplemental_leakage
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
export MPLCONFIGDIR=/tmp/repomem-e1-mpl

/data/conda_envs/locagent/bin/python \
    scripts/prepare_e1_supplemental_leakage.py \
    --dataset "${dataset}" \
    --index-dir "${index_dir}" \
    --output-dir "${root}/preflight" \
    --limit "${limit}" --seed 44

selected_count=$(/data/conda_envs/locagent/bin/python -c \
    'import json; print(len(json.load(open("results/e1_supplemental_leakage/preflight/selected_ids.json"))))')
if [[ "${selected_count}" == "0" ]]; then
    echo "E1 stopped before paid execution: no affected cases (D_i > 0)."
    exit 0
fi

if (( dry_run )); then
    echo "DRY RUN: frozen-prefix E1 on ${selected_count} affected cases"
    exit 0
fi

source /data/wjl/RepoMem/openaigpt4o_env
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
    --e1_supplemental_leakage \
    --timeout 900 --request_timeout 120 \
    --max_iteration_num 20 --max_attempt_num 1 \
    --temperature 1.0 --sampling_seed 44

/data/conda_envs/locagent/bin/python \
    scripts/analyze_e1_supplemental_leakage.py \
    --run-dir "${root}/run" --dataset "${dataset}"
