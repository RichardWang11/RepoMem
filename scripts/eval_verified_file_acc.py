import argparse
import json
import re
from collections import defaultdict
from datasets import load_dataset


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_path(path: str) -> str:
    path = path.strip()
    for prefix in ("a/", "b/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path.lstrip("/")


def files_from_patch(patch: str, py_only: bool = True):
    files = []
    seen = set()

    # Main source: diff --git a/foo.py b/foo.py
    for m in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch, flags=re.M):
        old_path, new_path = m.group(1), m.group(2)
        path = new_path if new_path != "/dev/null" else old_path
        path = normalize_path(path)

        if py_only and not path.endswith(".py"):
            continue

        if path not in seen:
            files.append(path)
            seen.add(path)

    # Fallback: +++ b/foo.py
    if not files:
        for m in re.finditer(r"^\+\+\+ b/(.*?)$", patch, flags=re.M):
            path = normalize_path(m.group(1))
            if path == "/dev/null":
                continue
            if py_only and not path.endswith(".py"):
                continue
            if path not in seen:
                files.append(path)
                seen.add(path)

    return files


def load_selected_ids(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data)


def get_pred_files(entry):
    pred = entry.get("found_files", [])

    # Raw LocAgent output before --merge is often list[list[str]].
    if pred and isinstance(pred[0], list):
        raise ValueError(
            f"{entry.get('instance_id')} has nested found_files. "
            "You are probably evaluating raw loc_outputs.jsonl. "
            "Please evaluate merged_loc_outputs_mrr.jsonl instead."
        )

    out = []
    seen = set()
    for p in pred:
        if not isinstance(p, str):
            continue
        p = normalize_path(p)
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def repo_group(instance):
    repo = instance.get("repo", "")
    if repo:
        name = repo.split("/")[-1]
    else:
        name = instance["instance_id"].split("__")[0]

    # Match RepoMem Table 2 naming style.
    mapping = {
        "scikit-learn": "scikit-learn",
        "pytest": "pytest",
        "django": "django",
        "sympy": "sympy",
        "matplotlib": "matplotlib",
        "astropy": "astropy",
        "sphinx": "sphinx",
    }
    return mapping.get(name, "others")


def evaluate(dataset_name, split, loc_file, selected_file=None, py_only=True, by_repo=False):
    selected = load_selected_ids(selected_file)
    ds = load_dataset(dataset_name, split=split)

    preds = {x["instance_id"]: get_pred_files(x) for x in load_jsonl(loc_file)}

    ks = [1, 3, 5]
    total = 0
    skipped_no_gt = 0
    correct = {k: 0 for k in ks}
    rows = []

    group_total = defaultdict(int)
    group_correct = defaultdict(lambda: {k: 0 for k in ks})

    diagnostics = {
        "missing_prediction": 0,
        "gt_len_hist": defaultdict(int),
        "pred_len_hist": defaultdict(int),
        "improve_1_to_3": 0,
        "improve_3_to_5": 0,
    }

    for inst in ds:
        iid = inst["instance_id"]
        if selected is not None and iid not in selected:
            continue

        gt = files_from_patch(inst["patch"], py_only=py_only)
        if not gt:
            skipped_no_gt += 1
            continue

        pred = preds.get(iid, [])
        if iid not in preds:
            diagnostics["missing_prediction"] += 1

        total += 1
        group = repo_group(inst)
        group_total[group] += 1

        diagnostics["gt_len_hist"][len(gt)] += 1
        diagnostics["pred_len_hist"][min(len(pred), 10)] += 1

        ok = {}
        gt_set = set(gt)
        for k in ks:
            ok[k] = gt_set.issubset(set(pred[:k]))
            if ok[k]:
                correct[k] += 1
                group_correct[group][k] += 1

        if (not ok[1]) and ok[3]:
            diagnostics["improve_1_to_3"] += 1
        if (not ok[3]) and ok[5]:
            diagnostics["improve_3_to_5"] += 1

        rows.append(
            {
                "instance_id": iid,
                "repo_group": group,
                "gt": gt,
                "pred_top5": pred[:5],
                "ok@1": ok[1],
                "ok@3": ok[3],
                "ok@5": ok[5],
            }
        )

    print("=" * 80)
    print(f"dataset: {dataset_name} / {split}")
    print(f"loc_file: {loc_file}")
    print(f"selected_file: {selected_file}")
    print(f"py_only_gt: {py_only}")
    print(f"evaluated instances: {total}")
    print(f"skipped_no_gt: {skipped_no_gt}")
    print(f"missing_prediction: {diagnostics['missing_prediction']}")
    print("-" * 80)

    for k in ks:
        acc = correct[k] / total if total else 0.0
        print(f"File Acc@{k}: {acc:.4f}  ({correct[k]}/{total})")

    print("-" * 80)
    print("gt_len_hist:", dict(sorted(diagnostics["gt_len_hist"].items())))
    print("pred_len_hist_capped_at_10:", dict(sorted(diagnostics["pred_len_hist"].items())))
    print("improve_1_to_3:", diagnostics["improve_1_to_3"])
    print("improve_3_to_5:", diagnostics["improve_3_to_5"])

    if by_repo:
        print("=" * 80)
        print("Per-repo File Acc@5")
        for g in sorted(group_total.keys()):
            n = group_total[g]
            acc5 = group_correct[g][5] / n if n else 0.0
            acc1 = group_correct[g][1] / n if n else 0.0
            acc3 = group_correct[g][3] / n if n else 0.0
            print(f"{g:15s} n={n:4d}  Acc@1={acc1:.4f}  Acc@3={acc3:.4f}  Acc@5={acc5:.4f}")

    # Save detailed cases for debugging.
    debug_path = loc_file + ".file_eval_debug.jsonl"
    with open(debug_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("=" * 80)
    print(f"debug cases saved to: {debug_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--loc_file", required=True)
    parser.add_argument("--selected_file", default=None)
    parser.add_argument("--all_files", action="store_true", help="Use all files in patch, not only .py files.")
    parser.add_argument("--by_repo", action="store_true")
    args = parser.parse_args()

    evaluate(
        dataset_name=args.dataset,
        split=args.split,
        loc_file=args.loc_file,
        selected_file=args.selected_file,
        py_only=not args.all_files,
        by_repo=args.by_repo,
    )