import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_POPULATION = "reproduce_lists/verified_full_500.json"
DEFAULT_EXCLUSIONS = "reproduce_lists/verified_m3a_matched_15.json"
DEFAULT_OUTPUT = "reproduce_lists/verified_pilot_random_20.json"
DEFAULT_PREREGISTRATION = "results/paper2_pilot/preregistration.json"
SEED = 20260715
SAMPLE_SIZE = 20
POPULATION_SIZE = 485

COHORT_SCHEDULES = {
    "a": [
        ("baseline", 1),
        ("memory", 1),
        ("memory", 2),
        ("baseline", 2),
        ("baseline", 3),
        ("memory", 3),
        ("memory", 4),
        ("baseline", 4),
    ],
    "b": [
        ("memory", 1),
        ("baseline", 1),
        ("baseline", 2),
        ("memory", 2),
        ("memory", 3),
        ("baseline", 3),
        ("baseline", 4),
        ("memory", 4),
    ],
}


def load_ids(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"{path}: expected a JSON list of strings")
    if len(data) != len(set(data)):
        raise ValueError(f"{path}: duplicate instance IDs")
    return data


def sha256_ids(ids):
    payload = "\n".join(ids) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def select_sample(population, exclusions, seed=SEED, sample_size=SAMPLE_SIZE):
    population_set = set(population)
    exclusion_set = set(exclusions)
    if not exclusion_set.issubset(population_set):
        raise ValueError("Paper 1 exclusions are not a subset of the population")
    pool = sorted(population_set - exclusion_set)
    return pool, random.Random(seed).sample(pool, sample_size)


def repository_key(instance_id):
    if "-" not in instance_id:
        raise ValueError(f"invalid SWE-bench instance ID: {instance_id}")
    return instance_id.rsplit("-", 1)[0]


def validate_indexes(ids, graph_dir, bm25_dir, memory_dir):
    missing = {"graph": [], "bm25": [], "episodic": []}
    for instance_id in ids:
        if not (Path(graph_dir) / f"{instance_id}.pkl").is_file():
            missing["graph"].append(instance_id)
        if not (Path(bm25_dir) / instance_id / "retriever.json").is_file():
            missing["bm25"].append(instance_id)
        commits = Path(memory_dir) / repository_key(instance_id) / "commits.jsonl"
        if not commits.is_file():
            missing["episodic"].append(instance_id)
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise ValueError(f"selected instances lack required indexes: {missing}")


def execution_batches():
    batches = []
    for position in range(8):
        cohort_order = ("a", "b") if position % 2 == 0 else ("b", "a")
        for cohort in cohort_order:
            arm, repeat = COHORT_SCHEDULES[cohort][position]
            batches.append({
                "batch": len(batches) + 1,
                "position": position + 1,
                "cohort": cohort,
                "arm": arm,
                "repeat": repeat,
            })
    return batches


def preregistration(population, exclusions, selected, cohort_a, cohort_b, args):
    repository_counts = {}
    for instance_id in selected:
        repo = repository_key(instance_id)
        repository_counts[repo] = repository_counts.get(repo, 0) + 1
    return {
        "experiment": "Paper 2 random-task stability pilot",
        "status": "FROZEN_PRE_API",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": {
            "population_file": args.population,
            "population_sha256": sha256_ids(sorted(population)),
            "population_size_before_exclusion": len(population),
            "exclusion_file": args.exclude,
            "exclusion_sha256": sha256_ids(sorted(exclusions)),
            "exclusion_count": len(exclusions),
            "candidate_pool_size": len(population) - len(exclusions),
            "candidate_pool_expected_size": POPULATION_SIZE,
            "method": "simple_random_without_replacement_from_sorted_pool",
            "rng": "python.random.Random",
            "seed": args.seed,
            "sample_size": args.sample_size,
            "selected_ids": selected,
            "selected_sha256": sha256_ids(selected),
            "cohort_a": cohort_a,
            "cohort_b": cohort_b,
            "repository_counts": repository_counts,
        },
        "design": {
            "arms": ["baseline", "memory"],
            "repetitions_per_case_arm": 4,
            "trajectory_count": args.sample_size * 2 * 4,
            "cohort_schedules": {
                key: [f"{arm[0].upper()}{repeat}" for arm, repeat in schedule]
                for key, schedule in COHORT_SCHEDULES.items()
            },
            "execution_batches": execution_batches(),
            "temporal_balance": (
                "best_effort under four-worker parallel execution; response-level "
                "fingerprint auditing is the primary drift control"
            ),
        },
        "runtime": {
            "requested_model": "gpt-4o",
            "expected_response_model": "gpt-4o-2024-11-20",
            "expected_system_fingerprint": "fp_40854b0bd2",
            "temperature": 1.0,
            "top_p": None,
            "timeout_seconds": 240,
            "request_timeout_seconds": 60,
            "max_iterations": 8,
            "max_attempts": 1,
            "num_processes": 4,
            "context_token_budget": 100000,
            "tool_observation_token_cap": 20000,
            "case_substitution": False,
            "trajectory_retry": False,
        },
        "budget": {
            "pilot_hard_cap_usd": 100.0,
            "initial_next_half_batch_reserve_usd": 10.0,
            "reserve_rule": "max(10, 1.25 * maximum_completed_half_batch_cost)",
        },
        "analysis": {
            "primary_outcome": "RepoMem-style file Acc@5 complete coverage",
            "secondary_outcomes": ["file_acc_at1", "file_acc_at3"],
            "stability": (
                "mean pairwise agreement over the six pairs from four binary "
                "outcomes per case and arm"
            ),
            "primary_estimand": "mean(case baseline stability - memory stability)",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 42,
        },
        "sample_size_target": {
            "locked_pre_analysis": True,
            "pilot_n": 20,
            "population_N": POPULATION_SIZE,
            "confidence_level": 0.95,
            "z": 1.96,
            "target_half_width": 0.03,
            "sd_definition": (
                "sample SD (ddof=1) of per-case baseline-minus-memory "
                "stability gaps"
            ),
            "step_1_uncorrected": "n_0 = (1.96 * s / 0.03) ** 2",
            "step_2_fpc": "n_fpc = n_0 / (1 + (n_0 - 1) / 485)",
            "step_3_integer_target": "n_target = ceil(max(n_fpc, 20))",
            "step_4_projection": (
                "h_target = h_20 * sqrt(20 / n_target), where h_20 is "
                "the pilot percentile-bootstrap CI half-width"
            ),
        },
        "credentials_recorded": False,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", default=DEFAULT_POPULATION)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--preregistration", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample_size", type=int, default=SAMPLE_SIZE)
    parser.add_argument(
        "--graph_dir",
        default="index_data/SWE-bench_Verified/graph_index_v2.3",
    )
    parser.add_argument(
        "--bm25_dir", default="index_data/SWE-bench_Verified/BM25_index"
    )
    parser.add_argument(
        "--memory_dir",
        default="repomem_index/verified_full500_episodic_with_issues_combined",
    )
    args = parser.parse_args()

    population = load_ids(args.population)
    exclusions = load_ids(args.exclude)
    exclusion_set = set(exclusions)
    pool, selected = select_sample(
        population, exclusions, args.seed, args.sample_size
    )
    if len(pool) != POPULATION_SIZE:
        raise ValueError(f"expected {POPULATION_SIZE} candidates, got {len(pool)}")
    if args.sample_size != SAMPLE_SIZE:
        raise ValueError(f"preregistered sample size is {SAMPLE_SIZE}")

    if set(selected) & exclusion_set:
        raise AssertionError("sample overlaps Paper 1 exclusions")
    validate_indexes(selected, args.graph_dir, args.bm25_dir, args.memory_dir)

    cohort_a = selected[: args.sample_size // 2]
    cohort_b = selected[args.sample_size // 2 :]
    output = Path(args.output)
    write_json(output, selected)
    write_json(output.with_name(f"{output.stem}_cohort_a.json"), cohort_a)
    write_json(output.with_name(f"{output.stem}_cohort_b.json"), cohort_b)
    write_json(
        args.preregistration,
        preregistration(
            population, exclusions, selected, cohort_a, cohort_b, args
        ),
    )
    print(f"saved {len(selected)} IDs to {output}")
    print(f"saved preregistration to {args.preregistration}")


if __name__ == "__main__":
    main()
