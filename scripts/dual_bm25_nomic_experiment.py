#!/usr/bin/env python3
"""Run the dual-BM25 -> Nomic episodic-memory retrieval experiment."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROFILES = ("bm25-code-v2", "bm25-experepair-v1")
QUERY_PREFIX = "Represent this query for searching relevant code: "


def normalize_path(path):
    path = str(path).strip()
    for prefix in ("a/", "b/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path.lstrip("/")


def files_from_patch(patch, py_only=False):
    files = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch or "", flags=re.M):
        old_path, new_path = match.groups()
        path = normalize_path(old_path if new_path == "/dev/null" else new_path)
        if (not py_only or path.endswith(".py")) and path not in files:
            files.append(path)
    if not files:
        for match in re.finditer(r"^\+\+\+ b/(.*?)$", patch or "", flags=re.M):
            path = normalize_path(match.group(1))
            if path != "/dev/null" and (not py_only or path.endswith(".py")) and path not in files:
                files.append(path)
    return files


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_tasks(dataset_path, ids_path):
    wanted = set(json.loads(Path(ids_path).read_text(encoding="utf-8")))
    tasks = [row for row in read_jsonl(dataset_path) if row["instance_id"] in wanted]
    if len(tasks) != len(wanted):
        found = {row["instance_id"] for row in tasks}
        raise ValueError(f"dataset is missing {len(wanted - found)} requested cases")
    tasks.sort(key=lambda row: row["instance_id"])
    return tasks


def candidate_record(score, memory, gold_files):
    changed = [normalize_path(path) for path in memory.get("changed_files") or []]
    overlap = sorted(set(changed) & set(gold_files))
    return {
        "sha": memory["sha"],
        "short_sha": memory.get("short_sha", memory["sha"][:9]),
        "bm25_score": score,
        "subject": memory.get("subject", ""),
        "message": memory.get("message", ""),
        "changed_files": changed,
        "issue_refs": memory.get("issue_refs") or [],
        "patch_path": memory.get("patch_path", ""),
        "overlap_files": overlap,
        "overlap_count": len(overlap),
        "gold_coverage": len(overlap) / len(gold_files) if gold_files else 0.0,
    }


def retrieve(args):
    os.environ["REPOMEM_INDEX_DIR"] = str(Path(args.index_dir).resolve())
    os.environ["REPOMEM_PATCH_MAX_CHARS"] = str(args.patch_chars)
    tasks = load_tasks(args.dataset, args.ids)
    from plugins.location_tools.repo_ops import repo_ops

    all_rows = {}
    for profile in PROFILES:
        os.environ["REPOMEM_BM25_PROFILE"] = profile
        rows = []
        for number, task in enumerate(tasks, 1):
            repo_ops.CURRENT_ISSUE_ID = task["instance_id"]
            repo_ops.CURRENT_INSTANCE = task
            memory_data = repo_ops._load_repomem_repo(task["repo"])
            if not memory_data:
                raise RuntimeError(f"missing episodic index for {task['repo']}")
            visible = list(repo_ops._iter_visible_commits(memory_data))
            ranked = repo_ops._rank_repomem_rows(task["problem_statement"], visible, 20)
            gold_files = files_from_patch(task["patch"], py_only=False)
            if not gold_files:
                raise ValueError(f"{task['instance_id']}: no gold files parsed")
            rows.append({
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "base_commit": task["base_commit"],
                "problem_statement": task["problem_statement"],
                "gold_files": gold_files,
                "visible_count": len(visible),
                "bm25_profile": profile,
                "candidates": [candidate_record(score, row, gold_files) for score, row in ranked],
            })
            if number % 50 == 0:
                print(f"{profile}: retrieved {number}/{len(tasks)}", flush=True)
        out = Path(args.output_dir) / profile / "bm25_top20.jsonl"
        write_jsonl(out, rows)
        all_rows[profile] = rows

    for instance_id in sorted({row["instance_id"] for row in all_rows[PROFILES[0]]}):
        for profile in PROFILES:
            row = next(x for x in all_rows[profile] if x["instance_id"] == instance_id)
            if len(row["candidates"]) < 5:
                raise ValueError(f"{profile}/{instance_id}: fewer than five BM25 matches")
    write_json(Path(args.output_dir) / "retrieval_manifest.json", {
        "dataset": str(Path(args.dataset).resolve()),
        "ids": str(Path(args.ids).resolve()),
        "index_dir": str(Path(args.index_dir).resolve()),
        "profiles": list(PROFILES),
        "query": "raw problem_statement",
        "top_k": 20,
        "patch_chars": args.patch_chars,
        "case_count": len(tasks),
    })


def load_raw_memory(index_dir, repo):
    repo_dir = Path(index_dir) / repo.replace("/", "__")
    issues = {}
    issues_path = repo_dir / "issues.jsonl"
    if issues_path.exists():
        issues = {
            str(row.get("issue_number")): row
            for row in read_jsonl(issues_path)
            if row.get("issue_number") is not None
        }
    return {"repo_dir": repo_dir, "issues": issues}


def read_index_patch(memory_data, candidate, patch_chars):
    relative = candidate.get("patch_path")
    if not relative:
        return ""
    path = memory_data["repo_dir"] / relative
    if not path.exists():
        return ""
    patch = path.read_text(encoding="utf-8", errors="replace").strip()
    return patch[:patch_chars] + ("\n...[patch truncated]..." if len(patch) > patch_chars else "")


def memory_document(memory_data, candidate, issue_chars, patch_chars):
    issue_parts = []
    for ref in candidate.get("issue_refs", [])[:3]:
        issue = memory_data["issues"].get(str(ref))
        if issue:
            issue_parts.append("\n".join(filter(None, [
                issue.get("issue_summary", ""), issue.get("issue_description", ""),
            ])))
    issues = "\n\n".join(issue_parts)[:issue_chars]
    patch = read_index_patch(memory_data, candidate, patch_chars)
    return "\n\n".join([
        f"Commit subject:\n{candidate.get('subject', '')}",
        f"Commit message:\n{candidate.get('message', '')}",
        "Changed files:\n" + "\n".join(candidate.get("changed_files") or []),
        f"Linked issue:\n{issues}",
        f"Patch:\n{patch}",
    ])


class EmbeddingCache:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(cache_key TEXT PRIMARY KEY, dims INTEGER NOT NULL, vector BLOB NOT NULL)"
        )

    def get(self, key):
        row = self.db.execute(
            "SELECT dims, vector FROM embeddings WHERE cache_key=?", (key,)
        ).fetchone()
        return None if row is None else np.frombuffer(row[1], dtype=np.float32, count=row[0]).copy()

    def put(self, key, vector):
        vector = np.asarray(vector, dtype=np.float32)
        self.db.execute(
            "INSERT OR REPLACE INTO embeddings(cache_key,dims,vector) VALUES(?,?,?)",
            (key, len(vector), vector.tobytes()),
        )
        self.db.commit()


class NomicEncoder:
    def __init__(self, model_id, revision, device, max_length, cache_path):
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.revision = HfApi().model_info(model_id, revision=revision).sha
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=self.revision, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=self.revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(device).eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or self.revision
        self.cache = EmbeddingCache(cache_path)

    def encode(self, text, kind):
        payload = QUERY_PREFIX + text if kind == "query" else text
        key = sha256_text("\0".join([
            self.model_id, str(self.resolved_revision), str(self.max_length), kind, payload,
        ]))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        tokens = self.tokenizer(
            [payload], padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        with self.torch.inference_mode():
            output = self.model(**tokens).last_hidden_state[:, -1, :].float()
            output = self.torch.nn.functional.normalize(output, p=2, dim=1)
        vector = output[0].cpu().numpy()
        self.cache.put(key, vector)
        return vector


def rerank(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    encoder = NomicEncoder(
        args.model, args.revision, args.device, args.max_length,
        str(Path(args.output_dir) / "nomic_embeddings.sqlite"),
    )
    for profile in PROFILES:
        os.environ["REPOMEM_BM25_PROFILE"] = profile
        rows = read_jsonl(Path(args.output_dir) / profile / "bm25_top20.jsonl")
        completed_path = Path(args.output_dir) / profile / "nomic_top20.jsonl"
        completed = {row["instance_id"]: row for row in read_jsonl(completed_path)} if completed_path.exists() else {}
        for number, row in enumerate(rows, 1):
            if row["instance_id"] in completed:
                continue
            memory_data = load_raw_memory(args.index_dir, row["repo"])
            query_vector = encoder.encode(row["problem_statement"], "query")
            reranked = []
            for bm25_rank, candidate in enumerate(row["candidates"], 1):
                document = memory_document(memory_data, candidate, args.issue_chars, args.patch_chars)
                vector = encoder.encode(document, "document")
                item = dict(candidate)
                item.update({
                    "bm25_rank": bm25_rank,
                    "nomic_score": float(np.dot(query_vector, vector)),
                    "document_sha256": sha256_text(document),
                    "document_chars": len(document),
                })
                reranked.append(item)
            reranked.sort(key=lambda item: (-item["nomic_score"], item["bm25_rank"]))
            completed[row["instance_id"]] = {
                **{key: value for key, value in row.items() if key != "candidates"},
                "model_id": args.model,
                "model_revision": encoder.resolved_revision,
                "max_length": args.max_length,
                "reranked_candidates": reranked,
            }
            write_jsonl(completed_path, [completed[key] for key in sorted(completed)])
            if number % 10 == 0:
                print(f"{profile}: reranked {number}/{len(rows)}", flush=True)
    write_json(Path(args.output_dir) / "nomic_manifest.json", {
        "model_id": args.model,
        "requested_revision": args.revision,
        "resolved_revision": encoder.resolved_revision,
        "max_length": args.max_length,
        "query_prefix": QUERY_PREFIX,
        "issue_chars": args.issue_chars,
        "patch_chars": args.patch_chars,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })


def ranking_metrics(candidates, gold_files, k):
    top = candidates[:k]
    gold = set(gold_files)
    covered = set().union(*(set(row.get("changed_files") or []) for row in top)) if top else set()
    relevant_ranks = [rank for rank, row in enumerate(top, 1) if set(row.get("changed_files") or []) & gold]
    hit = int(bool(relevant_ranks))
    coverage = len(covered & gold) / len(gold)
    mrr = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    gains = [int(bool(set(row.get("changed_files") or []) & gold)) for row in top]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_relevant = min(k, sum(
        int(bool(set(row.get("changed_files") or []) & gold)) for row in candidates
    ))
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return {"hit": hit, "coverage": coverage, "mrr": mrr, "ndcg": dcg / idcg if idcg else 0.0}


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return [0.0, 0.0]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_mcnemar_p(harmed, rescued):
    discordant = harmed + rescued
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(harmed, rescued) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def stratified_bootstrap(rows, left_key, right_key, iterations=10000, seed=44):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["repo"]].append(row)
    observed = sum(row[right_key] - row[left_key] for row in rows) / len(rows)
    samples = []
    for _ in range(iterations):
        selected = []
        for group in groups.values():
            selected.extend(rng.choice(group) for _ in group)
        samples.append(sum(row[right_key] - row[left_key] for row in selected) / len(selected))
    return {
        "difference": observed,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "iterations": iterations,
        "seed": seed,
    }


def profile_analysis(profile, rows):
    cases = []
    for row in rows:
        bm25 = row["candidates"]
        nomic = row["reranked_candidates"]
        metrics = {}
        for name, ranking in (("bm25", bm25), ("nomic", nomic)):
            for k in (5, 20):
                for metric, value in ranking_metrics(ranking, row["gold_files"], k).items():
                    metrics[f"{name}_{metric}@{k}"] = value
        cases.append({
            "instance_id": row["instance_id"], "repo": row["repo"],
            "base_commit": row["base_commit"], "gold_files": row["gold_files"],
            **metrics,
        })
    n = len(cases)
    ceiling = sum(row["bm25_hit@20"] for row in cases)
    headroom = sum(row["bm25_hit@20"] and not row["bm25_hit@5"] for row in cases)
    recoverable_ci = wilson(headroom, ceiling)
    a = sum(row["bm25_hit@5"] and row["nomic_hit@5"] for row in cases)
    b = sum(row["bm25_hit@5"] and not row["nomic_hit@5"] for row in cases)
    c = sum(not row["bm25_hit@5"] and row["nomic_hit@5"] for row in cases)
    d = n - a - b - c
    p_value = exact_mcnemar_p(b, c)
    summary = {
        "profile": profile,
        "case_count": n,
        "bm25_any_gold_hit@5": sum(row["bm25_hit@5"] for row in cases) / n,
        "bm25_any_gold_hit@20": ceiling / n,
        "nomic_any_gold_hit@5": sum(row["nomic_hit@5"] for row in cases) / n,
        "ranking_headroom_count": headroom,
        "recoverable_miss_rate": headroom / ceiling if ceiling else 0.0,
        "recoverable_miss_rate_wilson95": recoverable_ci,
        "stage1_gate": recoverable_ci[0] >= 0.15 and headroom >= 40,
        "mcnemar": {"a": a, "b_harmed": b, "c_rescued": c, "d": d, "p_raw": p_value},
        "net_gain_count": c - b,
        "net_gain_rate": (c - b) / n,
    }
    for metric in ("hit", "coverage", "mrr", "ndcg"):
        summary[f"paired_{metric}@5"] = stratified_bootstrap(
            cases, f"bm25_{metric}@5", f"nomic_{metric}@5"
        )
    for source in ("bm25", "nomic"):
        for metric in ("coverage", "mrr", "ndcg"):
            summary[f"{source}_{metric}@5"] = sum(row[f"{source}_{metric}@5"] for row in cases) / n
    repos = defaultdict(list)
    for row in cases:
        repos[row["repo"]].append(row)
    summary["repo_macro"] = {
        key: sum(sum(row[key] for row in group) / len(group) for group in repos.values()) / len(repos)
        for key in ("bm25_hit@5", "bm25_hit@20", "nomic_hit@5", "bm25_coverage@5", "nomic_coverage@5")
    }
    return summary, cases


def holm_adjust(summaries):
    ordered = sorted(summaries, key=lambda item: item["mcnemar"]["p_raw"])
    running = 0.0
    for index, summary in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - index) * summary["mcnemar"]["p_raw"])
        running = max(running, adjusted)
        summary["mcnemar"]["p_holm"] = running


def analyze(args):
    summaries = []
    per_profile_cases = {}
    reranked_by_profile = {}
    for profile in PROFILES:
        bm25_rows = {row["instance_id"]: row for row in read_jsonl(
            Path(args.output_dir) / profile / "bm25_top20.jsonl"
        )}
        reranked = read_jsonl(Path(args.output_dir) / profile / "nomic_top20.jsonl")
        combined = [{**bm25_rows[row["instance_id"]], **row} for row in reranked]
        summary, cases = profile_analysis(profile, combined)
        summaries.append(summary)
        per_profile_cases[profile] = cases
        reranked_by_profile[profile] = combined
        write_jsonl(Path(args.output_dir) / profile / "metrics_by_case.jsonl", cases)
    holm_adjust(summaries)
    for summary in summaries:
        hit_ci = summary["paired_hit@5"]
        summary["stage2_gate"] = bool(
            summary["stage1_gate"]
            and summary["nomic_any_gold_hit@5"] > summary["bm25_any_gold_hit@5"]
            and summary["mcnemar"]["p_holm"] < 0.05
            and hit_ci["ci_low"] > 0
            and summary["mcnemar"]["c_rescued"] > summary["mcnemar"]["b_harmed"]
        )
        summary["agent_eligible"] = summary["stage2_gate"] and summary["mcnemar"]["c_rescued"] >= 20

    eligible = [summary for summary in summaries if summary["agent_eligible"]]
    winner = None
    if eligible:
        winner = max(eligible, key=lambda item: (
            item["nomic_any_gold_hit@5"], item["net_gain_count"],
            item["nomic_coverage@5"], item["profile"] == "bm25-code-v2",
        ))["profile"]
    comparison = profile_overlap(reranked_by_profile)
    decision = {"winner": winner, "selection_is_post_hoc_exploratory": True}
    write_json(Path(args.output_dir) / "summary.json", {
        "profiles": summaries, "profile_comparison": comparison, "agent_decision": decision,
    })
    if winner:
        prepare_agent_artifacts(args, winner, reranked_by_profile[winner], per_profile_cases[winner])
    write_report(Path(args.output_dir) / "report.md", summaries, comparison, decision)


def profile_overlap(rows_by_profile):
    left = {row["instance_id"]: row for row in rows_by_profile[PROFILES[0]]}
    right = {row["instance_id"]: row for row in rows_by_profile[PROFILES[1]]}
    values = defaultdict(list)
    for instance_id in sorted(left):
        for label, key, k in (
            ("bm25_top20_jaccard", "candidates", 20),
            ("bm25_top5_jaccard", "candidates", 5),
            ("nomic_top5_jaccard", "reranked_candidates", 5),
        ):
            a = {row["sha"] for row in left[instance_id][key][:k]}
            b = {row["sha"] for row in right[instance_id][key][:k]}
            values[label].append(len(a & b) / len(a | b) if a | b else 1.0)
    return {key: sum(items) / len(items) for key, items in values.items()}


def prepare_agent_artifacts(args, winner, rows, cases):
    by_case = {row["instance_id"]: row for row in cases}
    rescued = [row for row in rows if not by_case[row["instance_id"]]["bm25_hit@5"] and by_case[row["instance_id"]]["nomic_hit@5"]]
    selected_ids = {row["instance_id"] for row in rescued}
    tasks = [row for row in load_tasks(args.dataset, args.ids) if row["instance_id"] in selected_ids]
    root = Path(args.output_dir) / "agent"
    write_jsonl(root / "selected_dataset.jsonl", tasks)
    write_json(root / "task_ids.json", sorted(selected_ids))
    for arm, key in (("bm25", "candidates"), ("nomic", "reranked_candidates")):
        cells = [{
            "cell_id": f"{arm}__{row['instance_id']}",
            "instance_id": row["instance_id"],
            "base_commit": row["base_commit"],
            "memory_shas": [item["sha"] for item in row[key][:5]],
            "cohort": "rerank_rescued",
            "bm25_profile": winner,
        } for row in rescued]
        write_json(root / f"{arm}_manifest.json", {"profile": winner, "arm": arm, "cells": cells})


def write_report(path, summaries, comparison, decision):
    lines = ["# Dual BM25 → Nomic Retrieval Experiment", ""]
    for summary in summaries:
        lines.extend([
            f"## {summary['profile']}", "",
            f"- BM25 Any-Gold-File Hit@5: {summary['bm25_any_gold_hit@5']:.4f}",
            f"- BM25 Any-Gold-File Hit@20: {summary['bm25_any_gold_hit@20']:.4f}",
            f"- Nomic Any-Gold-File Hit@5: {summary['nomic_any_gold_hit@5']:.4f}",
            f"- ranking headroom: {summary['ranking_headroom_count']}",
            f"- recoverable miss rate: {summary['recoverable_miss_rate']:.4f}",
            f"- rescued / harmed / net: {summary['mcnemar']['c_rescued']} / {summary['mcnemar']['b_harmed']} / {summary['net_gain_count']}",
            f"- McNemar raw / Holm p: {summary['mcnemar']['p_raw']:.6g} / {summary['mcnemar']['p_holm']:.6g}",
            f"- Stage 1 / Stage 2 / Agent eligible: {summary['stage1_gate']} / {summary['stage2_gate']} / {summary['agent_eligible']}", "",
        ])
    lines.extend(["## Profile comparison", ""])
    lines.extend(f"- {key}: {value:.4f}" for key, value in comparison.items())
    lines.extend(["", f"- exploratory Agent winner: {decision['winner'] or 'none'}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_agent(args):
    root = Path(args.output_dir) / "agent"
    tasks = {row["instance_id"]: row for row in read_jsonl(root / "selected_dataset.jsonl")}
    scores = defaultdict(dict)
    per_repeat = []
    duplicate_rows = {}
    for arm in ("bm25", "nomic"):
        for repeat in (1, 2):
            path = root / "runs" / f"{arm}_r{repeat}" / "merged_loc_outputs_mrr.jsonl"
            rows = read_jsonl(path)
            rows_by_id = {}
            for row in rows:
                rows_by_id.setdefault(row["instance_id"], row)
            missing = set(tasks) - set(rows_by_id)
            unexpected = set(rows_by_id) - set(tasks)
            if missing or unexpected:
                raise ValueError(
                    f"{path}: missing {sorted(missing)}, unexpected {sorted(unexpected)}"
                )
            duplicate_rows[f"{arm}_r{repeat}"] = len(rows) - len(rows_by_id)
            for row in rows_by_id.values():
                instance_id = row["instance_id"]
                gold = set(files_from_patch(tasks[instance_id]["patch"], py_only=False))
                predicted_files = row.get("found_files") or []
                if predicted_files and isinstance(predicted_files[0], list):
                    raise ValueError(f"{path}: expected merged flat found_files")
                predicted_files = [normalize_path(item) for item in predicted_files if isinstance(item, str)][:5]
                predicted = set(predicted_files)
                score = int(gold.issubset(predicted))
                scores[instance_id][f"{arm}_r{repeat}"] = score
                per_repeat.append({
                    "instance_id": instance_id, "repo": tasks[instance_id]["repo"],
                    "arm": arm, "repeat": repeat, "acc@5": score,
                    "gold_files": sorted(gold), "predicted_files@5": predicted_files,
                })
    cases = []
    for instance_id, values in sorted(scores.items()):
        bm25 = (values["bm25_r1"] + values["bm25_r2"]) / 2
        nomic = (values["nomic_r1"] + values["nomic_r2"]) / 2
        cases.append({
            "instance_id": instance_id, "repo": tasks[instance_id]["repo"],
            "bm25_mean_acc@5": bm25, "nomic_mean_acc@5": nomic,
            "paired_delta": nomic - bm25,
            "outcome": "win" if nomic > bm25 else "loss" if nomic < bm25 else "tie",
            **values,
        })
    bootstrap = stratified_bootstrap(cases, "bm25_mean_acc@5", "nomic_mean_acc@5")
    summary = {
        "analysis_type": "post-selection exploratory",
        "case_count": len(cases),
        "bm25_mean_acc@5": sum(row["bm25_mean_acc@5"] for row in cases) / len(cases),
        "nomic_mean_acc@5": sum(row["nomic_mean_acc@5"] for row in cases) / len(cases),
        "paired_bootstrap": bootstrap,
        "wins": sum(row["outcome"] == "win" for row in cases),
        "ties": sum(row["outcome"] == "tie" for row in cases),
        "losses": sum(row["outcome"] == "loss" for row in cases),
        "repeat_direction_agreement": sum(
            (row["nomic_r1"] - row["bm25_r1"]) * (row["nomic_r2"] - row["bm25_r2"]) > 0
            for row in cases
        ) / len(cases),
        "input_duplicate_rows": duplicate_rows,
    }
    write_jsonl(root / "agent_metrics_by_repeat.jsonl", per_repeat)
    write_jsonl(root / "agent_metrics_by_case.jsonl", cases)
    write_json(root / "agent_summary.json", summary)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("retrieve", "rerank", "analyze", "analyze-agent"))
    parser.add_argument("--dataset", default="/tmp/swebench_full_test.jsonl")
    parser.add_argument("--ids", default="reproduce_lists/verified_full_500.json")
    parser.add_argument(
        "--index-dir", default="repomem_index/verified_full500_episodic_with_issues_combined"
    )
    parser.add_argument("--output-dir", default="results/dual_bm25_nomic")
    parser.add_argument("--patch-chars", type=int, default=12000)
    parser.add_argument("--issue-chars", type=int, default=1200)
    parser.add_argument("--model", default="nomic-ai/nomic-embed-code")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=8192)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "retrieve":
        retrieve(args)
    elif args.command == "rerank":
        rerank(args)
    elif args.command == "analyze":
        analyze(args)
    else:
        analyze_agent(args)


if __name__ == "__main__":
    main()
