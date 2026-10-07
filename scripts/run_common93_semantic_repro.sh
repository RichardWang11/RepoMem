#!/usr/bin/env bash
set -euo pipefail

action="${1:-status}"
python_bin="${PYTHON_BIN:-/data/conda_envs/locagent/bin/python}"
source_root="${COMMON93_SOURCE_ROOT:-/data/wjl/RepoMem/repomem_index/common93_adaptive_file_memory_gpt4o_v1}"
output_root="${COMMON93_SCHEMA_V3_ROOT:-/data/wjl/RepoMem/repomem_index/common93_semantic_schema_v3_gpt4o}"
repo_root="${COMMON93_REPO_ROOT:-/data/yjm/orcaloca-common93-disambiguation-ablation/work/h1/repos}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/repomem-mpl}"

status() {
  "$python_bin" - "$source_root" "$output_root" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

source_root, output_root = map(Path, sys.argv[1:])
states = Counter()
rows = ok = 0
for instance_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
    repo_dirs = [path for path in instance_dir.iterdir() if path.is_dir()]
    if not repo_dirs:
        continue
    bank = repo_dirs[0]
    summaries = bank / "summaries.jsonl"
    manifest = bank / "manifest.json"
    source_rows = []
    if summaries.exists():
        source_rows = [json.loads(line) for line in summaries.read_text().splitlines() if line]
    state = json.loads(manifest.read_text()).get("status", "missing") if manifest.exists() else "missing"
    states[state] += 1
    rows += len(source_rows)
    ok += sum(row.get("status") == "ok" for row in source_rows)

materialized = []
if output_root.exists():
    for bank in sorted(path for path in output_root.iterdir() if path.is_dir()):
        manifest = bank / "build_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete":
            materialized.append(bank.name)

print(json.dumps({
    "source_banks": sum(states.values()),
    "source_statuses": dict(states),
    "source_rows": rows,
    "source_ok_rows": ok,
    "schema_v3_complete_banks": len(materialized),
    "schema_v3_instances": materialized,
}, indent=2))
PY
}

case "$action" in
  status)
    status
    ;;
  materialize-complete)
    while IFS= read -r manifest_path; do
      bank_dir="$(dirname "$manifest_path")"
      state="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$manifest_path")"
      [[ "$state" == "complete" ]] || continue
      repo_name="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"].replace("/", "__"))' "$manifest_path")"
      "$python_bin" scripts/materialize_common93_semantic_bank.py \
        "$bank_dir" \
        --repo-dir "$repo_root/$repo_name" \
        --output-root "$output_root"
    done < <(find "$source_root" -mindepth 3 -maxdepth 3 -name manifest.json -type f | sort)
    status
    ;;
  validate-materialized)
    while IFS= read -r build_manifest; do
      bank_dir="$(dirname "$build_manifest")"
      expected="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_count"])' "$build_manifest")"
      "$python_bin" scripts/validate_semantic_memory_build.py \
        "$bank_dir" --expected_count "$expected"
    done < <(find "$output_root" -mindepth 2 -maxdepth 2 -name build_manifest.json -type f | sort)
    ;;
  *)
    echo "usage: $0 status|materialize-complete|validate-materialized" >&2
    exit 2
    ;;
esac
