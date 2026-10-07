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
python_bin=/data/conda_envs/locagent/bin/python
dataset=results/stale_memory_rank_confirmation/selected_dataset.jsonl
task_ids=results/stale_memory_rank_confirmation/task_ids.json
graph_dir=/data/wjl/RepoMem/index_data/SWE-bench_Verified/graph_index_v2.3
bm25_dir=/data/wjl/RepoMem/index_data/SWE-bench_Verified/BM25_index
source_repo=/data/wjl/addressingmemory/results/orcaloca_experiment3/repos/django__django

if [[ ! -f "${dataset}" || ! -f "${task_ids}" ]]; then
    echo "Missing frozen confirmation dataset or task list." >&2
    exit 1
fi

graph_cmd=(
    "${python_bin}" dependency_graph/batch_build_graph.py
    --dataset "${dataset}" --split test --num_processes 4
    --download_repo --repo_path playground/rank_confirmation_graph
    --instance_id_path "${task_ids}" --output_dir "${graph_dir}"
)
bm25_cmd=(
    "${python_bin}" build_bm25_index.py
    --dataset "${dataset}" --split test --num_processes 4
    --download_repo --repo_path playground/rank_confirmation_bm25
    --instance_id_path "${task_ids}" --output_dir "${bm25_dir}"
)

if [[ ${dry_run} -eq 1 ]]; then
    printf 'DRY RUN:'; printf ' %q' "${graph_cmd[@]}"; printf '\n'
    printf 'DRY RUN:'; printf ' %q' "${bm25_cmd[@]}"; printf '\n'
    exit 0
fi

seed_workers() {
    local worker_root=$1
    mkdir -p "${worker_root}"
    for rank in 0 1 2 3; do
        local target="${worker_root}/${rank}/django_django"
        mkdir -p "${worker_root}/${rank}"
        if ! git -C "${target}" rev-parse --verify HEAD >/dev/null 2>&1; then
            if [[ -e "${target}" ]]; then
                mv "${target}" "/tmp/repomem-incomplete-$(basename "${worker_root}")-${rank}-$$"
            fi
            cp -a "${source_repo}" "${target}"
        fi
    done
}

seed_workers playground/rank_confirmation_graph
seed_workers playground/rank_confirmation_bm25
"${graph_cmd[@]}"
"${bm25_cmd[@]}"

"${python_bin}" - <<'PY'
import json
from pathlib import Path

root = Path("results/stale_memory_rank_confirmation")
ids = json.loads((root / "task_ids.json").read_text())
graph = Path("index_data/SWE-bench_Verified/graph_index_v2.3")
bm25 = Path("index_data/SWE-bench_Verified/BM25_index")
missing = [
    instance_id for instance_id in ids
    if not (graph / f"{instance_id}.pkl").exists()
    or not (bm25 / instance_id).exists()
]
(root / "missing_index_task_ids.json").write_text(
    json.dumps(missing, indent=2) + "\n"
)
if missing:
    raise SystemExit(f"Index build incomplete for {len(missing)} tasks")
print(f"Indexes ready for {len(ids)} confirmation tasks")
PY
