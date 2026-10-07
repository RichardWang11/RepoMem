import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


FILES = ("loc_trajs.jsonl", "context_guard_audit.jsonl")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_rows(original, replacement, instance_id, expected_cases):
    original_by_id = {row["instance_id"]: row for row in original}
    replacement_rows = [
        row for row in replacement if row.get("instance_id") == instance_id
    ]
    if len(replacement_rows) != 1:
        raise ValueError(
            f"expected one replacement for {instance_id}, got "
            f"{len(replacement_rows)}"
        )
    if instance_id in original_by_id:
        raise ValueError(f"{instance_id} already exists in original rows")
    merged = original + replacement_rows
    if len(merged) != expected_cases:
        raise ValueError(
            f"expected {expected_cases} merged rows, got {len(merged)}"
        )
    if len({row["instance_id"] for row in merged}) != expected_cases:
        raise ValueError("merged rows contain duplicate instance IDs")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--replacement_dir", required=True)
    parser.add_argument("--instance_id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--expected_cases", type=int, default=15)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    replacement_dir = Path(args.replacement_dir)
    manifest = {
        "instance_id": args.instance_id,
        "reason": args.reason,
        "original_run_dir": str(run_dir),
        "replacement_run_dir": str(replacement_dir),
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    for filename in FILES:
        target = run_dir / filename
        source = replacement_dir / filename
        backup = run_dir / f"{filename}.pre_replacement"
        if backup.exists():
            raise ValueError(f"backup already exists: {backup}")
        original = load_jsonl(target)
        replacement = load_jsonl(source)
        merged = merge_rows(
            original, replacement, args.instance_id, args.expected_cases
        )
        shutil.copy2(target, backup)
        write_jsonl(target, merged)
        manifest["files"][filename] = {
            "original_rows": len(original),
            "replacement_rows": 1,
            "merged_rows": len(merged),
            "backup": str(backup),
        }
    (run_dir / "replacement_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
