import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plugins.location_tools.repo_ops import repo_ops
from util.runtime import episodic_visibility
from util.runtime.episodic_visibility import (
    VisibilityBuildError,
    audit_git_repository,
    build_window,
    write_visibility_windows,
)


def git(repo, *args, env=None):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def output(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def commit(repo, message, marker):
    path = Path(repo) / "history.txt"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(marker + "\n")
    git(repo, "add", "history.txt")
    git(repo, "commit", "-m", message)
    return output(repo, "rev-parse", "HEAD")


class EpisodicVisibilityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        git(self.temp.name, "init", str(self.repo))
        git(self.repo, "config", "user.email", "test@example.com")
        git(self.repo, "config", "user.name", "Test")
        self.root = commit(self.repo, "root", "root")
        self.main = commit(self.repo, "main", "main")
        git(self.repo, "checkout", "-b", "side", self.root)
        self.side = commit(self.repo, "side", "side")
        git(self.repo, "checkout", "master")
        self.base = commit(self.repo, "base", "base")

    def tearDown(self):
        self.temp.cleanup()

    def test_window_excludes_base_and_non_ancestor_branch(self):
        canonical, ancestors = build_window(self.repo, self.base[:12])
        self.assertEqual(canonical, self.base)
        self.assertEqual(ancestors, [self.main, self.root])
        self.assertNotIn(self.side, ancestors)
        self.assertNotIn(self.base, ancestors)

    def test_unknown_base_fails_closed(self):
        with self.assertRaises(VisibilityBuildError):
            build_window(self.repo, "f" * 40)

    def test_replace_ref_is_rejected(self):
        git(self.repo, "replace", self.main, self.side)
        try:
            with self.assertRaisesRegex(VisibilityBuildError, "replace refs"):
                audit_git_repository(self.repo, [self.base])
        finally:
            git(self.repo, "replace", "-d", self.main)

    def test_graft_is_rejected(self):
        graft = Path(output(self.repo, "rev-parse", "--git-path", "info/grafts"))
        if not graft.is_absolute():
            graft = self.repo / graft
        graft.parent.mkdir(parents=True, exist_ok=True)
        graft.write_text(f"{self.base}\n", encoding="utf-8")
        with self.assertRaisesRegex(VisibilityBuildError, "grafts"):
            audit_git_repository(self.repo, [self.base])

    def test_shallow_repository_is_rejected(self):
        shallow = Path(self.temp.name) / "shallow"
        subprocess.run(
            ["git", "clone", "--depth=1", f"file://{self.repo}", str(shallow)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self.assertRaisesRegex(VisibilityBuildError, "non-shallow"):
            audit_git_repository(shallow, [output(shallow, "rev-parse", "HEAD")])

    def test_partial_clone_configuration_is_rejected(self):
        git(self.repo, "config", "extensions.partialClone", "origin")
        with self.assertRaisesRegex(VisibilityBuildError, "partial/promisor"):
            audit_git_repository(self.repo, [self.base])

    def test_window_invokes_exact_7001_command_and_caps_at_7000(self):
        ancestors = [f"{value:040x}" for value in range(1, 7002)]
        result = mock.Mock(
            stdout="\n".join([self.base] + ancestors) + "\n",
            stderr="",
            returncode=0,
        )
        with mock.patch.object(
            episodic_visibility, "canonicalize_base", return_value=self.base
        ), mock.patch.object(
            episodic_visibility, "run_git", return_value=result
        ) as run_git:
            canonical, window = build_window(self.repo, self.base)
        self.assertEqual(canonical, self.base)
        self.assertEqual(window, ancestors[:7000])
        run_git.assert_called_once_with(
            self.repo,
            ["rev-list", "--max-count=7001", self.base],
            check=False,
        )

    def test_manifest_records_repository_audit_and_hash(self):
        index_dir = Path(self.temp.name) / "index" / "owner__repo"
        index_dir.mkdir(parents=True)
        commits = []
        for sha in (self.root, self.main, self.side, self.base):
            commits.append({"sha": sha})
        (index_dir / "commits.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in commits), encoding="utf-8"
        )
        manifest = write_visibility_windows(
            "owner/repo", self.repo, index_dir, [self.base]
        )
        self.assertFalse(manifest["repository_audit"]["is_shallow"])
        self.assertFalse(manifest["repository_audit"]["is_partial_clone"])
        self.assertEqual(manifest["repository_audit"]["replace_ref_count"], 0)
        self.assertFalse(manifest["repository_audit"]["grafts_present"])
        window_path = index_dir / manifest["bases"][self.base]["path"]
        window = json.loads(window_path.read_text(encoding="utf-8"))
        self.assertEqual(window["ancestor_shas"], [self.main, self.root])

    def test_runtime_strict_window_uses_manifest_order(self):
        index_root = Path(self.temp.name) / "index"
        index_dir = index_root / "owner__repo"
        index_dir.mkdir(parents=True)
        rows = []
        for index, sha in enumerate((self.root, self.main, self.side, self.base)):
            rows.append({
                "sha": sha,
                "history_index": index,
                "subject": sha,
                "message": "",
                "changed_files": [],
            })
        (index_dir / "commits.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (index_dir / "meta.json").write_text(
            json.dumps({
                "sha_to_history_index": {
                    row["sha"]: row["history_index"] for row in rows
                }
            }),
            encoding="utf-8",
        )
        write_visibility_windows("owner/repo", self.repo, index_dir, [self.base])
        instance = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": self.base,
            "problem_statement": "unrelated",
        }
        with mock.patch.dict(os.environ, {
            "REPOMEM_INDEX_DIR": str(index_root),
            "REPOMEM_VISIBILITY_POLICY": "strict",
            "REPOMEM_LEAKAGE_FILTER": "0",
        }):
            repo_ops.REPOMEM_CACHE.clear()
            audit, count = repo_ops.validate_episodic_visibility(instance)
            memory = repo_ops._load_repomem_repo("owner/repo")
            visible = repo_ops._visible_commits_for_instance(memory, instance)
        self.assertEqual([row["sha"] for row in visible], [self.main, self.root])
        self.assertEqual(count, 2)
        self.assertEqual(audit["policy"], "strict")

    def test_strict_supplement_adds_missing_ancestor_without_changing_legacy(self):
        index_root = Path(self.temp.name) / "index"
        index_dir = index_root / "owner__repo"
        index_dir.mkdir(parents=True)
        legacy_row = {
            "sha": self.root,
            "history_index": 0,
            "subject": "root",
            "message": "",
            "changed_files": ["history.txt"],
        }
        (index_dir / "commits.jsonl").write_text(
            json.dumps(legacy_row) + "\n", encoding="utf-8"
        )
        (index_dir / "meta.json").write_text(
            json.dumps({"sha_to_history_index": {self.root: 0}}),
            encoding="utf-8",
        )
        manifest = write_visibility_windows(
            "owner/repo", self.repo, index_dir, [self.base]
        )
        self.assertEqual(manifest["visibility_supplement_count"], 1)
        supplemental_rows = [
            json.loads(line)
            for line in (index_dir / "visibility_commits.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual([row["sha"] for row in supplemental_rows], [self.main])

        instance = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": self.base,
            "problem_statement": "unrelated",
        }
        common_env = {
            "REPOMEM_INDEX_DIR": str(index_root),
            "REPOMEM_LEAKAGE_FILTER": "0",
        }
        with mock.patch.dict(
            os.environ, {**common_env, "REPOMEM_VISIBILITY_POLICY": "strict"}
        ):
            repo_ops.REPOMEM_CACHE.clear()
            memory = repo_ops._load_repomem_repo("owner/repo")
            strict_visible = repo_ops._visible_commits_for_instance(memory, instance)
        with mock.patch.dict(
            os.environ, {**common_env, "REPOMEM_VISIBILITY_POLICY": "legacy"}
        ):
            repo_ops.REPOMEM_CACHE.clear()
            memory = repo_ops._load_repomem_repo("owner/repo")
            legacy_visible = repo_ops._visible_commits_for_instance(memory, instance)

        self.assertEqual(
            [row["sha"] for row in strict_visible], [self.main, self.root]
        )
        self.assertEqual([row["sha"] for row in legacy_visible], [self.root])

    def test_runtime_strict_missing_manifest_never_falls_back(self):
        index_root = Path(self.temp.name) / "index"
        index_dir = index_root / "owner__repo"
        index_dir.mkdir(parents=True)
        (index_dir / "commits.jsonl").write_text(
            json.dumps({
                "sha": self.root,
                "history_index": 0,
                "subject": "root",
                "message": "",
                "changed_files": [],
            }) + "\n",
            encoding="utf-8",
        )
        (index_dir / "meta.json").write_text(
            json.dumps({"sha_to_history_index": {self.root: 0}}), encoding="utf-8"
        )
        instance = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "f" * 40,
            "problem_statement": "unrelated",
        }
        with mock.patch.dict(os.environ, {
            "REPOMEM_INDEX_DIR": str(index_root),
            "REPOMEM_VISIBILITY_POLICY": "strict",
        }):
            repo_ops.REPOMEM_CACHE.clear()
            with self.assertRaisesRegex(
                repo_ops.EpisodicMemoryUnavailable, "manifest is missing"
            ):
                repo_ops.validate_episodic_visibility(instance)


if __name__ == "__main__":
    unittest.main()
