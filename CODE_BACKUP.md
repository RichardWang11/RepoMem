# RepoMem code backup

Branch: `migration/code-backup-20261007`.
Base commit: `6b7bfebfb3cafedfe805bcafe6825bb4e7eae1b7`.
Source snapshot: `/data/wjl/RepoMem` on 2026-10-07.

This branch preserves the current source working tree, including local changes
and untracked source files, tests, documentation, and required small inputs.
Experiment outputs, generated indexes, downloaded benchmark repositories,
runtime caches, and local credential files are archived separately.

## Snapshot adjustments

- `scripts/run.sh`: read API key from caller environment instead of overwriting it.
- Ignore rules retain source tests and Markdown while excluding generated artifacts.
- The original source directory has not been rewritten.

## Validation

- Parsed 211 Python files using Python 3.12.
- Checked 23 shell scripts using `bash -n`.
- Checked credential patterns and link targets.
- No paid model calls or full experiments were run.

## Data not included

- `index_data/`
- `original_paper/`
- `playground/`
- `repomem_index/`
- `results/`
