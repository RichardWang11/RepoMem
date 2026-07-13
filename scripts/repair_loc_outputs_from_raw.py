import argparse
import json

from util.process_output import get_loc_results_from_raw_outputs


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repaired = 0
    total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for row in load_jsonl(args.input):
            total += 1
            raw_outputs = row.get("raw_output_loc", [])
            if raw_outputs:
                found_files, found_modules, found_entities = get_loc_results_from_raw_outputs(
                    row["instance_id"], raw_outputs
                )
                if found_files != row.get("found_files"):
                    repaired += 1
                row["found_files"] = found_files
                row["found_modules"] = found_modules
                row["found_entities"] = found_entities
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"total={total}, repaired={repaired}")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
