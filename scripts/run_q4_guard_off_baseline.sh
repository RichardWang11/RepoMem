#!/usr/bin/env bash
set -euo pipefail

cd /data/wjl/RepoMem
export PYTHONPATH="/data/wjl/RepoMem:${PYTHONPATH:-}"
source /data/wjl/RepoMem/openaigpt4o_env

root=results/context_safe_m3a/q4_guard_off_baseline
runs_root="${root}/runs"
selection=reproduce_lists/verified_m3a_matched_15.json

mkdir -p "${runs_root}"

write_status() {
    /data/conda_envs/locagent/bin/python scripts/summarize_q4_run_status.py \
        --root "${root}" || true
}
trap write_status EXIT

run_repeat() {
    local repeat=$1
    local output_dir="${runs_root}/baseline_r${repeat}"
    if [[ -e "${output_dir}" ]]; then
        echo "Refusing to overwrite existing Q4 repeat: ${output_dir}" >&2
        return 1
    fi

    /data/conda_envs/locagent/bin/python auto_search_main.py \
        --dataset princeton-nlp/SWE-bench_Verified \
        --split test \
        --used_list_file "${selection}" \
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
        --max_attempt_num 1

    /data/conda_envs/locagent/bin/python scripts/validate_q4_guard_off_repeat.py \
        --run_dir "${output_dir}" \
        --expected_cases 15
    /data/conda_envs/locagent/bin/python scripts/repair_loc_outputs_from_raw.py \
        --input "${output_dir}/loc_trajs.jsonl" \
        --output "${output_dir}/loc_trajs.repaired.jsonl" \
        | tee "${output_dir}/parser_repair.log"
}

for repeat in 1 2 3 4; do
    run_repeat "${repeat}"
done

/data/conda_envs/locagent/bin/python scripts/analyze_q4_guard_off_baseline.py \
    --q4 \
        "${runs_root}/baseline_r1/loc_trajs.repaired.jsonl" \
        "${runs_root}/baseline_r2/loc_trajs.repaired.jsonl" \
        "${runs_root}/baseline_r3/loc_trajs.repaired.jsonl" \
        "${runs_root}/baseline_r4/loc_trajs.repaired.jsonl" \
    --guard_on \
        results/context_safe_m3a/runs_v2/baseline_r1/loc_trajs.repaired.jsonl \
        results/context_safe_m3a/runs_v2/baseline_r2/loc_trajs.repaired.jsonl \
        results/context_safe_m3a/runs_v2/baseline_r3/loc_trajs.repaired.jsonl \
        results/context_safe_m3a/runs_v2/baseline_r4/loc_trajs.repaired.jsonl \
    --original \
        results/check9_label_stability_gpt4o_baseline_harm30/loc_trajs.repaired.jsonl \
        results/m3_memory_variance_runs/baseline_r1/loc_trajs.repaired.jsonl \
        results/m3_memory_variance_runs/baseline_r2/loc_trajs.repaired.jsonl \
        results/m3_memory_variance_runs/baseline_r3/loc_trajs.repaired.jsonl \
    --output_dir "${root}/analysis"
