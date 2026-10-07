#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
python_bin="${PYTHON_BIN:-/data/conda_envs/locagent/bin/python}"
root="repomem_index/verified_django231_semantic_dynamic_gpt4o_v1"
dataset="princeton-nlp/SWE-bench_Verified"
ids="$root/django231_ids.json"

load_provider_env() {
  local env_file="${REPOMEM_ENV_FILE:-/data/wjl/RepoMem/openaigpt4o_env}"
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    source "$env_file"
  fi
  : "${OPENAI_API_KEY:?OPENAI_API_KEY is required for paid summary/localization calls}"
}

case "$action" in
  tools)
    PYTHONPATH=. "$python_bin" scripts/audit_repomem_paper_tools.py
    ;;
  plan|build-objects|materialize|validate)
    [[ "$action" != build-objects ]] || load_provider_env
    PYTHONPATH=. "$python_bin" scripts/build_django_dynamic_semantic_banks.py "$action"
    ;;
  baseline|semantic)
    load_provider_env
    output="results/repomem_semantic_django_reproduction/${action}"
    memory_args=()
    if [[ "$action" == semantic ]]; then
      export REPOMEM_SEMANTIC_INDEX_DIR="$PWD/$root/banks"
      export REPOMEM_SEMANTIC_LAYOUT=instance
      export REPOMEM_SEMANTIC_MEMBERSHIP=exclude_tests
      export REPOMEM_SEMANTIC_DOCUMENT=summary_only
      export REPOMEM_BM25_PROFILE=bm25-code-v2
      memory_args=(--use_repomem --repomem_mode semantic --repomem_tool_profile paper-v1 --repomem_semantic_layout instance)
    fi
    PYTHONPATH=. "$python_bin" auto_search_main.py \
      --dataset "$dataset" --split test --used_list_file "$ids" \
      --output_folder "$output" --localize --merge \
      --model gpt-4o --use_function_calling --num_samples 2 \
      --num_processes "${REPOMEM_NUM_PROCESSES:-8}" \
      --required_response_model gpt-4o-2024-11-20 \
      --max_iteration_num 20 --temperature 1 --sampling_seed 44 \
      "${memory_args[@]}"
    ;;
  analyze)
    PYTHONPATH=. "$python_bin" scripts/analyze_repomem_semantic_django.py
    ;;
  *)
    echo "usage: $0 tools|plan|build-objects|materialize|validate|baseline|semantic|analyze" >&2
    exit 2
    ;;
esac
