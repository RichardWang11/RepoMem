#!/usr/bin/env python3
"""Run E3 cells sequentially in their pre-registered counterbalanced order."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def has_terminal_record(cell_dir: Path) -> bool:
    return any(
        (cell_dir / name).is_file()
        and (cell_dir / name).stat().st_size > 0
        for name in ("loc_trajs.jsonl", "loc_outputs.jsonl")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cells = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for sequence_index, cell in enumerate(cells):
        cell_dir = root / "cells" / cell["cell_id"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        selected_path = cell_dir / "selected_ids.json"
        selected_path.write_text(
            json.dumps([cell["instance_id"]], indent=2) + "\n",
            encoding="utf-8",
        )
        if has_terminal_record(cell_dir):
            print(f"SKIP {sequence_index + 1}/{len(cells)} {cell['cell_id']}")
            continue
        command = [
            "/data/conda_envs/locagent/bin/python", "auto_search_main.py",
            "--dataset", args.dataset, "--split", "test",
            "--used_list_file", str(selected_path),
            "--model", "openai/gpt-4o",
            "--required_response_model", "gpt-4o-2024-11-20",
            "--localize", "--merge", "--output_folder", str(cell_dir),
            "--num_processes", "1", "--num_samples", "1",
            "--ranking_method", "mrr", "--use_function_calling",
            "--simple_desc", "--use_repomem", "--repomem_mode", "episodic",
            "--repomem_tool_profile", "current",
            "--repomem_visibility_policy", "strict",
            "--repomem_task_prompt", cell["task_prompt"],
            "--e3_prompt_arm", cell["arm"],
            "--timeout", "900", "--request_timeout", "120",
            "--max_iteration_num", "20", "--max_attempt_num", "1",
            "--temperature", "1.0", "--sampling_seed", str(cell["seed"]),
        ]
        print(f"RUN {sequence_index + 1}/{len(cells)} {cell['cell_id']}")
        if args.dry_run:
            continue
        env = dict(os.environ)
        env["REPOMEM_SEARCH_COMMIT_FORCE_TOPK"] = "20"
        log_path = cell_dir / "runner.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        execution = {
            "cell": cell,
            "command": command,
            "returncode": completed.returncode,
        }
        (cell_dir / "execution.json").write_text(
            json.dumps(execution, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode != 0:
            print(f"FAILED {cell['cell_id']} rc={completed.returncode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
