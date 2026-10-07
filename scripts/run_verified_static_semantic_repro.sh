#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
python_bin="${PYTHON_BIN:-/data/conda_envs/locagent/bin/python}"
index_root="${REPOMEM_VERIFIED_SEMANTIC_ROOT:-repomem_index/verified_static_semantic_gpt4o_20241120_v1}"
result_root="${REPOMEM_VERIFIED_SEMANTIC_RESULTS:-results/verified_static_semantic_reproduction_v1}"
dataset="princeton-nlp/SWE-bench_Verified"

load_provider_env() {
  local env_file="${REPOMEM_ENV_FILE:-/data/wjl/RepoMem/openaigpt4o_env}"
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    source "$env_file"
  fi
  if [[ "${REPOMEM_DIRECT_API:-1}" == "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  fi
  : "${OPENAI_API_KEY:?OPENAI_API_KEY is required for paid model calls}"
}

prepare() {
  PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py prepare \
    --output-root "$index_root"
}

run_arm() {
  local cohort="$1"
  local arm="$2"
  local output="$3"
  local memory_args=()
  load_provider_env
  if [[ "$arm" == "semantic" ]]; then
    export REPOMEM_SEMANTIC_INDEX_DIR="$PWD/$index_root/banks"
    export REPOMEM_SEMANTIC_LAYOUT=repo
    export REPOMEM_SEMANTIC_MEMBERSHIP=exclude_tests
    export REPOMEM_SEMANTIC_DOCUMENT=summary_only
    export REPOMEM_BM25_PROFILE=bm25-code-v2
    memory_args=(
      --use_repomem --repomem_mode semantic
      --repomem_tool_profile paper-v1 --repomem_semantic_layout repo
    )
  else
    unset REPOMEM_SEMANTIC_INDEX_DIR REPOMEM_SEMANTIC_LAYOUT
    unset REPOMEM_SEMANTIC_MEMBERSHIP REPOMEM_SEMANTIC_DOCUMENT REPOMEM_BM25_PROFILE
  fi
  PYTHONPATH=. "$python_bin" auto_search_main.py \
    --dataset "$dataset" --split test --used_list_file "$cohort" \
    --output_folder "$output" --localize --merge \
    --model gpt-4o --required_response_model gpt-4o-2024-11-20 \
    --use_function_calling --num_samples 2 \
    --num_processes "${REPOMEM_NUM_PROCESSES:-8}" \
    --max_iteration_num 20 --temperature 1 --sampling_seed 44 \
    "${memory_args[@]}"
}

require_django_go() {
  "$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d["django_gate"]["passed"] else "Django gate is NO-GO")' \
    "$result_root/django/report.json"
}

case "$action" in
  prepare|preflight)
    prepare
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py status \
      --output-root "$index_root"
    ;;
  hydrate)
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py hydrate \
      --output-root "$index_root" --scope all
    ;;
  build-django)
    [[ -f "$index_root/experiment_manifest.json" ]] || prepare
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py indexes \
      --output-root "$index_root" --scope django --python "$python_bin"
    load_provider_env
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py banks \
      --output-root "$index_root" --scope django --pilot-limit 10 --python "$python_bin"
    PYTHONPATH=. "$python_bin" scripts/validate_semantic_memory_build.py \
      "$index_root/banks/django__django" --expected_count 10 \
      --max_median_cost_per_file 0.40 --max_single_cost 0.75
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py banks \
      --output-root "$index_root" --scope django --python "$python_bin"
    ;;
  validate-django)
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py validate \
      --output-root "$index_root" --scope django --python "$python_bin"
    ;;
  run-django-baseline)
    run_arm "$index_root/cohorts/django231.json" baseline "$result_root/django/baseline"
    ;;
  run-django-semantic)
    run_arm "$index_root/cohorts/django231.json" semantic "$result_root/django/semantic"
    ;;
  run-django)
    "$0" run-django-baseline
    "$0" run-django-semantic
    ;;
  analyze-django)
    PYTHONPATH=. "$python_bin" scripts/analyze_verified_static_semantic.py \
      --ids "$index_root/cohorts/django231.json" \
      --baseline "$result_root/django/baseline/merged_loc_outputs_mrr.jsonl" \
      --semantic "$result_root/django/semantic/merged_loc_outputs_mrr.jsonl" \
      --output "$result_root/django/report.json" --django-gate
    ;;
  build-all)
    [[ -f "$index_root/experiment_manifest.json" ]] || prepare
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py indexes \
      --output-root "$index_root" --scope remaining --python "$python_bin"
    load_provider_env
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py banks \
      --output-root "$index_root" --scope remaining --python "$python_bin"
    ;;
  validate-all)
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py validate \
      --output-root "$index_root" --scope all --python "$python_bin"
    ;;
  run-full-baseline)
    require_django_go
    run_arm "$index_root/cohorts/remaining269.json" baseline "$result_root/remaining/baseline"
    ;;
  run-full-semantic)
    require_django_go
    run_arm "$index_root/cohorts/remaining269.json" semantic "$result_root/remaining/semantic"
    ;;
  run-full)
    "$0" run-full-baseline
    "$0" run-full-semantic
    ;;
  analyze-full)
    PYTHONPATH=. "$python_bin" scripts/analyze_verified_static_semantic.py \
      --ids "$index_root/cohorts/full500.json" \
      --baseline "$result_root/django/baseline/merged_loc_outputs_mrr.jsonl" \
      --baseline "$result_root/remaining/baseline/merged_loc_outputs_mrr.jsonl" \
      --semantic "$result_root/django/semantic/merged_loc_outputs_mrr.jsonl" \
      --semantic "$result_root/remaining/semantic/merged_loc_outputs_mrr.jsonl" \
      --output "$result_root/full500/report.json"
    ;;
  status)
    PYTHONPATH=. "$python_bin" scripts/prepare_verified_static_semantic.py status \
      --output-root "$index_root"
    ;;
  *)
    echo "usage: $0 prepare|preflight|hydrate|build-django|validate-django|run-django[-baseline|-semantic]|analyze-django|build-all|validate-all|run-full[-baseline|-semantic]|analyze-full|status" >&2
    exit 2
    ;;
esac
