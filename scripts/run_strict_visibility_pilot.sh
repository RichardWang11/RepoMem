#!/usr/bin/env bash
set -euo pipefail

cd /data/wjl/RepoMem
export PYTHONPATH="/data/wjl/RepoMem:${PYTHONPATH:-}"

dry_run=0
max_cases=0
cohort_scope=all
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --max-cases)
            max_cases=${2:?--max-cases requires an integer}
            shift 2
            ;;
        --cohort)
            cohort_scope=${2:?--cohort requires random, sensitivity, or all}
            shift 2
            ;;
        *)
            echo "Usage: $0 [--dry-run] [--max-cases N] [--cohort random|sensitivity|all]" >&2
            exit 2
            ;;
    esac
done
if ! [[ "${max_cases}" =~ ^[0-9]+$ ]]; then
    echo "--max-cases must be a non-negative integer" >&2
    exit 2
fi
if [[ "${cohort_scope}" != "all" && "${cohort_scope}" != "random" \
    && "${cohort_scope}" != "sensitivity" ]]; then
    echo "--cohort must be random, sensitivity, or all" >&2
    exit 2
fi

if (( ! dry_run )); then
    source /data/wjl/RepoMem/openaigpt4o_env
fi

export GRAPH_INDEX_DIR=/data/wjl/RepoMem/index_data/SWE-bench_Verified/graph_index_v2.3
export BM25_INDEX_DIR=/data/wjl/RepoMem/index_data/SWE-bench_Verified/BM25_index
export REPOMEM_INDEX_DIR=/data/wjl/RepoMem/repomem_index/verified_full500_episodic_with_issues_combined
export REPOMEM_BM25_PROFILE=bm25-code-v2
export REPOMEM_SEARCH_COMMIT_TOPK_CAP=5
export REPOMEM_SEARCH_COMMIT_SCHEMA_DEFAULT=5
export REPOMEM_LEAKAGE_FILTER=1
export MPLCONFIGDIR=/tmp/repomem-strict-visibility-mpl

root=results/strict_visibility_pilot
dataset="${root}/preflight/verified500.jsonl"
prereg="${root}/preregistration.json"
amendment="${root}/preregistration_amendment.json"
budget_cap=50
case_reserve=2
mkdir -p "${root}/runs" "${root}/case_lists" "${root}/preflight"

if [[ ! -f "${prereg}" ]]; then
    echo "Missing frozen preregistration: ${prereg}" >&2
    exit 1
fi
if [[ ! -f "${dataset}" ]]; then
    echo "Missing frozen dataset snapshot: ${dataset}" >&2
    exit 1
fi
if [[ ! -f "${amendment}" ]]; then
    echo "Missing model-substitution amendment: ${amendment}" >&2
    exit 1
fi

if (( ! dry_run )); then
    /data/conda_envs/locagent/bin/python scripts/preflight_exact_model.py \
        --model openai/gpt-4o \
        --required_response_model gpt-4o-2024-11-20 \
        --output "${root}/preflight/model.json"
fi

spent_cost() {
    /data/conda_envs/locagent/bin/python -c \
        'import glob,json,sys; total=0.0
for pattern in sys.argv[1:]:
    for p in glob.glob(pattern):
        for line in open(p,encoding="utf-8"):
            if not line.strip(): continue
            usage=json.loads(line).get("usage",{})
            cost=float(usage.get("cost($)",0) or 0)
            if cost == 0:
                cost=(int(usage.get("prompt_tokens",0) or 0)*0.000005 +
                      int(usage.get("completion_tokens",0) or 0)*0.000015)
            total += cost
print(f"{total:.5f}")' \
        "${root}/runs/*/loc_trajs.jsonl" \
        "${root}/rejected_runs/*/loc_trajs.jsonl"
}

case_ids() {
    /data/conda_envs/locagent/bin/python -c \
        'import json,sys; p=json.load(open(sys.argv[1])); print("\n".join(p[sys.argv[2]]))' \
        "${prereg}" "$1"
}

run_case() {
    local cohort=$1
    local position=$2
    local instance_id=$3
    local policy=$4
    local safe_id=${instance_id//\//__}
    local name
    name=$(printf '%s_%02d_%s_%s' "${cohort}" "${position}" "${safe_id}" "${policy}")
    local output_dir="${root}/runs/${name}"
    local list_file="${root}/case_lists/${safe_id}.json"
    /data/conda_envs/locagent/bin/python -c \
        'import json,sys; open(sys.argv[1],"w").write(json.dumps([sys.argv[2]],indent=2)+"\n")' \
        "${list_file}" "${instance_id}"
    if (( dry_run )); then
        echo "DRY RUN ${name}: ${instance_id} policy=${policy} samples=2"
        return
    fi
    if [[ -f "${output_dir}/loc_trajs.jsonl" ]]; then
        local completed_samples
        completed_samples=$(
            /data/conda_envs/locagent/bin/python -c \
                'import json,sys
rows=[json.loads(line) for line in open(sys.argv[1],encoding="utf-8") if line.strip()]
print(len(rows[0].get("loc_trajs",{}).get("trajs",[])) if len(rows)==1 else -1)' \
                "${output_dir}/loc_trajs.jsonl"
        )
        if [[ "${completed_samples}" == "2" ]]; then
            echo "Skipping completed case: ${name}"
            return
        fi
        echo "Refusing incomplete output directory: ${output_dir} " \
            "(successful samples=${completed_samples}, expected=2)" >&2
        exit 1
    fi
    if [[ -e "${output_dir}" ]]; then
        echo "Refusing incomplete output directory: ${output_dir}" >&2
        exit 1
    fi
    local spent
    spent=$(spent_cost)
    if ! /data/conda_envs/locagent/bin/python -c \
        'import sys; raise SystemExit(0 if float(sys.argv[1])+float(sys.argv[2]) <= float(sys.argv[3]) else 1)' \
        "${spent}" "${case_reserve}" "${budget_cap}"; then
        echo "Budget stop before ${name}: spent=${spent}, reserve=${case_reserve}, cap=${budget_cap}" >&2
        exit 3
    fi
    /data/conda_envs/locagent/bin/python auto_search_main.py \
        --dataset "${dataset}" \
        --split test \
        --used_list_file "${list_file}" \
        --model openai/gpt-4o \
        --required_response_model gpt-4o-2024-11-20 \
        --localize --merge \
        --output_folder "${output_dir}" \
        --num_processes 1 \
        --num_samples 2 \
        --ranking_method mrr \
        --use_function_calling --simple_desc \
        --use_repomem --repomem_mode episodic \
        --repomem_tool_profile current \
        --repomem_visibility_policy "${policy}" \
        --timeout 900 --request_timeout 120 \
        --max_iteration_num 20 --max_attempt_num 1 \
        --temperature 1.0 --sampling_seed 44
}

run_cohort() {
    local key=$1
    local label=$2
    local position=0
    while IFS= read -r instance_id; do
        [[ -z "${instance_id}" ]] && continue
        if (( max_cases > 0 && executed_cases >= max_cases )); then
            return
        fi
        position=$((position + 1))
        if (( position % 2 == 1 )); then
            run_case "${label}" "${position}" "${instance_id}" legacy
            run_case "${label}" "${position}" "${instance_id}" strict
        else
            run_case "${label}" "${position}" "${instance_id}" strict
            run_case "${label}" "${position}" "${instance_id}" legacy
        fi
        executed_cases=$((executed_cases + 1))
    done < <(case_ids "${key}")
}

executed_cases=0
if [[ "${cohort_scope}" == "all" || "${cohort_scope}" == "random" ]]; then
    run_cohort random_ids random
fi
if [[ "${cohort_scope}" == "all" || "${cohort_scope}" == "sensitivity" ]]; then
    run_cohort sensitivity_ids sensitivity
fi

if (( ! dry_run )); then
    analyze_args=(
        --root "${root}"
        --dataset "${dataset}"
    )
    if (( max_cases > 0 )); then
        analyze_args+=(--allow_partial)
    fi
    /data/conda_envs/locagent/bin/python \
        scripts/analyze_strict_visibility_pilot.py "${analyze_args[@]}"
fi
