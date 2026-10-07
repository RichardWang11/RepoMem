#!/usr/bin/env bash
set -euo pipefail

cd /data/wjl/RepoMem
export PYTHONPATH="/data/wjl/RepoMem:${PYTHONPATH:-}"
export HF_HOME=/data/hf_cache
export MPLCONFIGDIR=/tmp/dual-bm25-nomic-mpl

python_loc=/data/conda_envs/locagent/bin/python
python_nomic=/data/conda_envs/hf-server/bin/python

"${python_loc}" -u scripts/dual_bm25_nomic_experiment.py retrieve
CUDA_VISIBLE_DEVICES=2 "${python_nomic}" -u scripts/dual_bm25_nomic_experiment.py rerank --device cuda:0
"${python_loc}" -u scripts/dual_bm25_nomic_experiment.py analyze

