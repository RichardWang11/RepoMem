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

root=results/e3_prompt
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
export REPOMEM_SEARCH_COMMIT_FORCE_TOPK=20
export REPOMEM_LEAKAGE_FILTER=1
export REPOMEM_SUPPLEMENTAL_LEAKAGE=on
export MPLCONFIGDIR=/tmp/repomem-e3-mpl

/data/conda_envs/locagent/bin/python scripts/prepare_e3_prompt.py \
    --dataset "${dataset}" --output-dir "${root}/preflight"

if (( dry_run )); then
    /data/conda_envs/locagent/bin/python scripts/run_e3_prompt_cells.py \
        --dataset "${dataset}" \
        --manifest "${root}/preflight/cells.json" \
        --output-dir "${root}/run" --dry-run
    exit 0
fi

source /data/wjl/RepoMem/openaigpt4o_env
export REPOMEM_INDEX_DIR=/data/wjl/RepoMem/${index_dir}
export REPOMEM_VISIBILITY_POLICY=strict
export REPOMEM_BM25_PROFILE=bm25-code-v2
export REPOMEM_SEARCH_COMMIT_TOPK_CAP=5
export REPOMEM_SEARCH_COMMIT_SCHEMA_DEFAULT=5
export REPOMEM_SEARCH_COMMIT_FORCE_TOPK=20
export REPOMEM_LEAKAGE_FILTER=1
export REPOMEM_SUPPLEMENTAL_LEAKAGE=on

/data/conda_envs/locagent/bin/python scripts/preflight_exact_model.py \
    --model openai/gpt-4o \
    --required_response_model gpt-4o-2024-11-20 \
    --output "${root}/preflight/model.json"

/data/conda_envs/locagent/bin/python scripts/run_e3_prompt_cells.py \
    --dataset "${dataset}" \
    --manifest "${root}/preflight/cells.json" \
    --output-dir "${root}/run"

/data/conda_envs/locagent/bin/python scripts/analyze_e3_prompt.py \
    --run-dir "${root}/run" \
    --manifest "${root}/preflight/cells.json" \
    --dataset "${dataset}" \
    --e2-run-dir results/e2_search_topk/run
