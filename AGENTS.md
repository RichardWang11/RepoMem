# Repository Guidelines

## Project Structure & Module Organization

This repository implements LocAgent, a graph-guided code localization framework. Core indexing and repository abstractions live in `repo_index/`, with parser logic under `repo_index/codeblocks/parser/`. Graph construction utilities are in `dependency_graph/`. Agent runtime helpers, prompts, benchmark utilities, and output processing live in `util/`. Tool plugins are under `plugins/location_tools/`. Main entry points include `auto_search_main.py`, `build_bm25_index.py`, and `sft_train.py`. Evaluation assets are in `evaluation/`, shell scripts are in `scripts/`, images are in `assets/`, and sample outputs are under `results/`.

## Build, Test, and Development Commands

Create the documented environment before running project code:

```bash
conda create -n locagent python=3.12
conda activate locagent
pip install -r requirements.txt
```

Run LocAgent through the main localization entry point:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python auto_search_main.py --dataset 'czlll/SWE-bench_Lite' --split test --localize --merge
```

Generate graph indexes in batch with:

```bash
python dependency_graph/batch_build_graph.py --dataset 'czlll/Loc-Bench_V1' --split test --download_repo
```

Use `python build_bm25_index.py` for sparse retrieval indexes when needed. `scripts/run.sh` shows the expected environment variables, output path, and model flags for a full run.

## Coding Style & Naming Conventions

Write Python following PEP 8 with 4-space indentation. Use `snake_case` for modules, functions, variables, and CLI flags; use `PascalCase` for classes. Keep imports grouped by standard library, third-party packages, then local modules. Prefer explicit type hints and focused helper functions when modifying shared modules in `repo_index/`, `dependency_graph/`, or `util/`. Existing prompt templates use `.j2`; keep prompt filenames descriptive and colocated in `util/prompts/`.

## Testing Guidelines

There is no committed pytest configuration or formal test suite. Validate changes with the narrowest executable path affected by the edit, such as `python dependency_graph/batch_build_graph.py ...`, `python auto_search_main.py ...`, or `evaluation/run_evaluation.ipynb`. For evaluation logic, add small deterministic fixtures or JSONL samples near `evaluation/` or `results/` and document the command used to verify them.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive commit subjects such as `Update README.md`, `generate index in batch`, and `refactor dependency_graph`. Keep subjects concise and scoped. Pull requests should include a clear description, changed components, commands run, datasets or indexes used, and linked issues when applicable. Include screenshots only for documentation or visual asset changes.

## Security & Configuration Tips

Do not commit API keys, local dataset paths, generated indexes, or large benchmark outputs. Set provider credentials through environment variables; `scripts/run.sh` contains placeholder values only. Keep `GRAPH_INDEX_DIR`, `BM25_INDEX_DIR`, and result directories outside tracked source paths unless adding small intentional examples.
