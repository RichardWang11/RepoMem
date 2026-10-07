#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
: "${DJANGO_REPO_DIR:?export DJANGO_REPO_DIR=/path/to/django first}"
python_bin="${PYTHON_BIN:-python}"

load_provider_env() {
  local env_file="${REPOMEM_ENV_FILE:-/data/wjl/RepoMem/openaigpt4o_env}"
  if [[ -f "$env_file" ]]; then
    export PYTHONPATH="${PYTHONPATH:-}"
    # shellcheck disable=SC1090
    source "$env_file"
  fi
  if [[ "${REPOMEM_DIRECT_API:-1}" == "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  fi
  : "${OPENAI_API_KEY:?OPENAI_API_KEY is missing; set it or REPOMEM_ENV_FILE}"
}

primary="repomem_index/verified_django231_semantic_membership_v2/django__django/static/membership.json"
sensitivity_root="repomem_index/verified_django231_semantic_membership_include_tests_v3"
sensitivity="$sensitivity_root/django__django/static/membership.json"
indexes="repomem_index/verified_django231_semantic_c0_indexes_v3"
pilot_output="repomem_index/verified_django231_semantic_agentic_pilot_v3"
full_output="repomem_index/verified_django231_semantic_agentic_full_v3"
instance="repomem_semantic_django_c0"

case "$action" in
  membership)
    PYTHONPATH=. "$python_bin" scripts/build_semantic_membership_manifests.py \
      --repo-dir "$DJANGO_REPO_DIR" --include-tests --mode static \
      --output-dir "$sensitivity_root"
    ;;
  indexes)
    PYTHONPATH=. "$python_bin" scripts/build_semantic_c0_indexes.py \
      --repo_dir "$DJANGO_REPO_DIR" --membership_manifest "$primary" \
      --additional_membership_manifest "$sensitivity" \
      --output_dir "$indexes" --instance_id "$instance" \
      --reuse_graph_index index_data/SWE-bench_Verified/graph_index_v2.3/django__django-7530.pkl \
      --reuse_code_bm25_index index_data/SWE-bench_Verified/BM25_index/django__django-7530
    ;;
  pilot|full)
    load_provider_env
    output="$full_output"
    [[ "$action" == pilot ]] && output="$pilot_output"
    limit=()
    [[ "$action" == pilot ]] && limit=(--pilot_limit 10)
    PYTHONPATH=. "$python_bin" scripts/build_repomem_semantic_index.py \
      --repo "django/django=$DJANGO_REPO_DIR" \
      --output_dir "$output" \
      --membership_manifest "$primary" \
      --membership_manifest "$sensitivity" \
      --builder_mode agent --model gpt-4o --temperature 0 --seed 44 \
      --max_iterations 4 \
      --required_response_model gpt-4o-2024-11-20 \
      --retrieval_profile bm25-code-v2 --num_workers 1 --retry_errors \
      --graph_index "$indexes/graph/$instance.pkl" \
      --code_bm25_index "$indexes/code_bm25/$instance" \
      "${limit[@]}"
    ;;
  validate-pilot)
    PYTHONPATH=. "$python_bin" scripts/validate_semantic_memory_build.py \
      "$pilot_output/django__django" --expected_count 10
    ;;
  validate-full)
    PYTHONPATH=. "$python_bin" scripts/validate_semantic_memory_build.py \
      "$full_output/django__django"
    ;;
  *)
    echo "usage: $0 membership|indexes|pilot|validate-pilot|full|validate-full" >&2
    exit 2
    ;;
esac
