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
export MPLCONFIGDIR=/tmp/repomem-matplotlib-paper2-pilot
export REPOMEM_PATCH_MAX_CHARS=12000
export REPOMEM_SEARCH_COMMIT_TOPK_CAP=20
export REPOMEM_SEARCH_COMMIT_SCHEMA_DEFAULT=20
export REPOMEM_EXAMINE_MAX_COMMITS=10
export REPOMEM_ISSUE_MAX_CHARS=200000

root=results/paper2_pilot
runs_root="${root}/runs"
prereg="${root}/preregistration.json"
selection=reproduce_lists/verified_pilot_random_20.json
cohort_a=reproduce_lists/verified_pilot_random_20_cohort_a.json
cohort_b=reproduce_lists/verified_pilot_random_20_cohort_b.json
reference_manifest=results/context_safe_m3a/replication_v4/experiment_manifest.json
budget_cap=100

if [[ ! -f "${prereg}" && ! -f "${selection}" ]]; then
    /data/conda_envs/locagent/bin/python scripts/sample_random_verified_pilot.py
elif [[ ! -f "${prereg}" || ! -f "${selection}" ]]; then
    echo "Refusing partial frozen sampling state; restore or remove both files." >&2
    exit 1
fi
mkdir -p "${runs_root}" "${root}/preflight"

write_status() {
    local status=$1
    local detail=${2:-}
    /data/conda_envs/locagent/bin/python -c \
        'import json,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":sys.argv[2],"detail":sys.argv[3],"timestamp_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")' \
        "${root}/run_status.json" "${status}" "${detail}"
}

cost_state() {
    /data/conda_envs/locagent/bin/python -c \
        'import glob,sys; from scripts.analyze_paper2_pilot import parse_cost; xs=[parse_cost(p)["cost_usd"] for p in glob.glob(sys.argv[1])]; total=sum(xs); reserve=max([10.0]+[1.25*x for x in xs]); print(f"{total:.2f} {reserve:.2f}")' \
        "${runs_root}/*/localize.log"
}

preflight() {
    local batch=$1
    local output_dir="${root}/preflight/batch_${batch}"
    mkdir -p "${output_dir}"
    /data/conda_envs/locagent/bin/python scripts/ping_gpt4o_fingerprint.py \
        --manifest "${reference_manifest}" \
        --output_dir "${output_dir}"
    /data/conda_envs/locagent/bin/python scripts/validate_replication_v4.py \
        --preflight "${output_dir}/fingerprint_ping_latest.json" \
        --max_age_seconds 300
}

run_half_batch() {
    local batch=$1
    local cohort=$2
    local arm=$3
    local repeat=$4
    local list_file=$cohort_a
    if [[ "${cohort}" == "b" ]]; then
        list_file=$cohort_b
    fi
    local name
    name=$(printf "batch_%02d_cohort_%s_%s_r%s" "${batch}" "${cohort}" "${arm}" "${repeat}")
    local output_dir="${runs_root}/${name}"
    local memory_args=()
    if [[ "${arm}" == "memory" ]]; then
        memory_args=(--use_repomem --repomem_mode episodic)
    fi

    if [[ -e "${output_dir}" ]]; then
        if [[ -f "${output_dir}/fingerprint_audit_summary.json" ]]; then
            echo "Validating and skipping completed pilot batch: ${name}"
            set +e
            /data/conda_envs/locagent/bin/python scripts/validate_paper2_pilot_batch.py \
                --run_dir "${output_dir}" --arm "${arm}" --repeat "${repeat}" \
                --expected_list "${list_file}"
            local rc=$?
            set -e
            if [[ ${rc} -eq 2 ]]; then
                write_status SNAPSHOT_DRIFT_ABORT "${name}"
                exit 2
            elif [[ ${rc} -ne 0 ]]; then
                write_status PROTOCOL_INVALID "${name}"
                exit 1
            fi
            return
        fi
        write_status PROTOCOL_INVALID "incomplete existing directory: ${name}"
        echo "Refusing incomplete pilot output: ${output_dir}" >&2
        exit 1
    fi

    read -r spent reserve < <(cost_state)
    if ! /data/conda_envs/locagent/bin/python -c \
        'import sys; raise SystemExit(0 if float(sys.argv[1])+float(sys.argv[2]) <= float(sys.argv[3]) else 1)' \
        "${spent}" "${reserve}" "${budget_cap}"; then
        write_status BUDGET_STOPPED "spent=${spent}, reserve=${reserve}, cap=${budget_cap}"
        exit 3
    fi

    if [[ ${dry_run} -eq 1 ]]; then
        echo "DRY RUN ${name}: list=${list_file} arm=${arm} repeat=${repeat}"
        return
    fi

    preflight "${batch}"
    /data/conda_envs/locagent/bin/python auto_search_main.py \
        --dataset princeton-nlp/SWE-bench_Verified \
        --split test \
        --used_list_file "${list_file}" \
        --model gpt-4o \
        --localize \
        --merge \
        --output_folder "${output_dir}" \
        --num_processes 4 \
        --num_samples 1 \
        --ranking_method mrr \
        --use_function_calling \
        --simple_desc \
        --timeout 240 \
        --request_timeout 60 \
        --max_iteration_num 8 \
        --max_attempt_num 1 \
        --temperature 1.0 \
        --context_token_budget 100000 \
        --tool_observation_token_cap 20000 \
        --fail_on_context_error \
        --bounded_graph_output \
        --bounded_entity_output \
        --bounded_search_output \
        "${memory_args[@]}"

    /data/conda_envs/locagent/bin/python scripts/repair_loc_outputs_from_raw.py \
        --input "${output_dir}/loc_trajs.jsonl" \
        --output "${output_dir}/loc_trajs.repaired.jsonl" \
        | tee "${output_dir}/parser_repair.log"

    set +e
    /data/conda_envs/locagent/bin/python scripts/validate_paper2_pilot_batch.py \
        --run_dir "${output_dir}" --arm "${arm}" --repeat "${repeat}" \
        --expected_list "${list_file}"
    local rc=$?
    set -e
    if [[ ${rc} -eq 2 ]]; then
        write_status SNAPSHOT_DRIFT_ABORT "${name}"
        exit 2
    elif [[ ${rc} -ne 0 ]]; then
        write_status PROTOCOL_INVALID "${name}"
        exit 1
    fi
}

# Position-balanced mirror schedules. Odd positions launch cohort A first;
# even positions launch cohort B first.
batches=(
    "a baseline 1" "b memory 1"
    "b baseline 1" "a memory 1"
    "a memory 2" "b baseline 2"
    "b memory 2" "a baseline 2"
    "a baseline 3" "b memory 3"
    "b baseline 3" "a memory 3"
    "a memory 4" "b baseline 4"
    "b memory 4" "a baseline 4"
)

batch_number=0
for entry in "${batches[@]}"; do
    batch_number=$((batch_number + 1))
    read -r cohort arm repeat <<< "${entry}"
    run_half_batch "${batch_number}" "${cohort}" "${arm}" "${repeat}"
done

if [[ ${dry_run} -eq 1 ]]; then
    echo "DRY RUN complete: validated 16 planned half-batches; no API calls made."
    exit 0
fi

/data/conda_envs/locagent/bin/python scripts/analyze_paper2_pilot.py
analysis_verdict=$(/data/conda_envs/locagent/bin/python -c \
    'import json; print(json.load(open("results/paper2_pilot/analysis/pilot_summary.json"))["verdict"])')
write_status "${analysis_verdict}" "all 160 trajectories validated and analyzed"
