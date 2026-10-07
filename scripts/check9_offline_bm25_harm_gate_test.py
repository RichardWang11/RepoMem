import json
import os
import tempfile

import pandas as pd

from scripts.check9_offline_bm25_harm_gate import (
    OFFLINE_FEATURE_COLUMNS,
    bm25_search,
    build_offline_feature_dataframe,
    feature_columns,
    load_commits_by_repo,
    make_harm_rescue_labels,
    offline_commit_features,
)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_bm25_search_returns_issue_relevant_commits_with_scores():
    commits = [
        {"sha": "1", "subject": "Fix migration writer import ordering", "message": "Fix migration writer import ordering", "changed_files": ["django/db/migrations/writer.py"]},
        {"sha": "2", "subject": "Docs cleanup", "message": "Docs cleanup", "changed_files": ["docs/index.rst"]},
    ]

    top = bm25_search("migration import writer", commits, top_k=2)

    assert top[0]["sha"] == "1"
    assert top[0]["score"] > top[1]["score"]


def test_offline_commit_features_excludes_repo_pool_size():
    commits = [
        {"score": 3.0, "subject": "Fix model field", "message": "Fix model field", "changed_files": ["django/db/models/fields.py"]},
        {"score": 1.0, "subject": "Fix admin view", "message": "Fix admin view", "changed_files": ["django/contrib/admin/options.py"]},
    ]

    feats = offline_commit_features("`Field` fails in django/db/models/fields.py", commits)

    assert "repo_commit_pool_size" not in feats
    assert feats["offline_top20_unique_files_log"] > 0
    assert feats["offline_top20_unique_dirs_log"] > 0


def test_load_commits_by_repo_prefers_first_index_dir_for_duplicate_repo():
    with tempfile.TemporaryDirectory() as tmp:
        first = os.path.join(tmp, "first", "django__django")
        second = os.path.join(tmp, "second", "django__django")
        os.makedirs(first)
        os.makedirs(second)
        write_jsonl(os.path.join(first, "commits.jsonl"), [{"repo": "django/django", "sha": "first", "message": "first"}])
        write_jsonl(os.path.join(second, "commits.jsonl"), [{"repo": "django/django", "sha": "second", "message": "second"}])

        commits = load_commits_by_repo([os.path.join(tmp, "first"), os.path.join(tmp, "second")])

    assert commits["django/django"][0]["sha"] == "first"


def test_build_offline_feature_dataframe_adds_harm_labels_without_pool_feature():
    check5 = pd.DataFrame(
        [
            {
                "case_id": "django__django-1",
                "repo": "django",
                "repo_full": "django/django",
                "gold_files": ["django/db/migrations/writer.py"],
                "gold_files_count": 1,
                "baseline_predicted_files": ["django/db/migrations/writer.py"],
                "memory_predicted_files": ["django/db/models/fields.py"],
                "baseline_acc_at5_original": 1,
                "memory_acc_at5_original": 0,
            }
        ]
    )
    commits = {
        "django/django": [
            {
                "sha": "abc",
                "subject": "Fixed migration writer import",
                "message": "Fixed migration writer import",
                "changed_files": ["django/db/migrations/writer.py"],
            }
        ]
    }

    out = build_offline_feature_dataframe(check5, commits, max_commits=20, coverage_k=5)

    assert out.loc[0, "harm_at5"] == 1
    assert "repo_commit_pool_size" not in out.columns
    assert all(col in out.columns for col in OFFLINE_FEATURE_COLUMNS)
    assert "repo_commit_pool_size" not in feature_columns(out)


def test_make_harm_rescue_labels_for_all_k():
    df = pd.DataFrame(
        {
            "baseline_at1": [1, 0],
            "memory_at1": [0, 1],
            "baseline_at3": [1, 0],
            "memory_at3": [0, 1],
            "baseline_at5": [1, 0],
            "memory_at5": [0, 1],
        }
    )

    out = make_harm_rescue_labels(df)

    assert out["harm_at5"].tolist() == [1, 0]
    assert out["rescue_at3"].tolist() == [0, 1]


if __name__ == "__main__":
    tests = [
        test_bm25_search_returns_issue_relevant_commits_with_scores,
        test_offline_commit_features_excludes_repo_pool_size,
        test_load_commits_by_repo_prefers_first_index_dir_for_duplicate_repo,
        test_build_offline_feature_dataframe_adds_harm_labels_without_pool_feature,
        test_make_harm_rescue_labels_for_all_k,
    ]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
