"""Build and validate fail-closed episodic-memory visibility windows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1
WINDOW_SIZE = 7000
ISSUE_REF_RE = re.compile(
    r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d+)|#(\d+)",
    flags=re.IGNORECASE,
)


class VisibilityBuildError(RuntimeError):
    """Raised when a Git repository cannot produce a trustworthy window."""


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    env["GIT_NO_LAZY_FETCH"] = "1"
    return env


def run_git(
    repo_dir: str | Path,
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env=_git_env(),
        timeout=600,
    )


def _optional_git_config(repo_dir: str | Path, key: str) -> str:
    result = run_git(repo_dir, ["config", "--get", key], check=False)
    if result.returncode not in (0, 1):
        raise VisibilityBuildError(result.stderr.strip() or f"failed to read Git config {key}")
    return result.stdout.strip()


def audit_git_repository(repo_dir: str | Path, base_commits: Iterable[str]) -> dict:
    """Reject incomplete or locally rewritten commit graphs before rev-list."""
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.is_dir():
        raise VisibilityBuildError(f"repository does not exist: {repo_dir}")

    shallow = run_git(repo_dir, ["rev-parse", "--is-shallow-repository"]).stdout.strip()
    if shallow != "false":
        raise VisibilityBuildError(f"strict visibility requires a non-shallow repository: {repo_dir}")

    partial_clone = _optional_git_config(repo_dir, "extensions.partialClone")
    promisor_result = run_git(
        repo_dir,
        ["config", "--get-regexp", r"^remote\..*\.promisor$"],
        check=False,
    )
    if promisor_result.returncode not in (0, 1):
        raise VisibilityBuildError(promisor_result.stderr.strip() or "failed to inspect promisor config")
    promisor_rows = [line for line in promisor_result.stdout.splitlines() if line.strip()]
    if partial_clone or any(line.rsplit(None, 1)[-1].lower() == "true" for line in promisor_rows):
        raise VisibilityBuildError(
            f"strict visibility rejects partial/promisor repositories: {repo_dir}"
        )

    replace_refs = [
        line.strip()
        for line in run_git(repo_dir, ["for-each-ref", "--format=%(refname)", "refs/replace/"]).stdout.splitlines()
        if line.strip()
    ]
    if replace_refs:
        raise VisibilityBuildError(f"replace refs are not allowed: {replace_refs}")

    graft_path = Path(run_git(repo_dir, ["rev-parse", "--git-path", "info/grafts"]).stdout.strip())
    if not graft_path.is_absolute():
        graft_path = repo_dir / graft_path
    grafts_present = graft_path.is_file() and bool(graft_path.read_text(errors="replace").strip())
    if grafts_present:
        raise VisibilityBuildError(f"Git grafts are not allowed: {graft_path}")

    canonical_bases = [canonicalize_base(repo_dir, base) for base in base_commits]
    strict_fsck = run_git(
        repo_dir,
        ["fsck", "--full", "--strict", "--no-dangling", *canonical_bases],
        check=False,
    )
    connectivity_fsck = run_git(
        repo_dir,
        [
            "fsck",
            "--full",
            "--connectivity-only",
            "--no-dangling",
            *canonical_bases,
        ],
        check=False,
    )
    if connectivity_fsck.returncode:
        raise VisibilityBuildError(
            f"Git object/connectivity check failed for {repo_dir}: "
            f"{connectivity_fsck.stderr.strip()}"
        )

    remotes = run_git(repo_dir, ["remote", "-v"], check=False).stdout.splitlines()
    reference_tips = sorted(
        line.strip()
        for line in run_git(
            repo_dir, ["for-each-ref", "--format=%(refname) %(objectname)"]
        ).stdout.splitlines()
        if line.strip()
    )
    return {
        "repo_dir": str(repo_dir),
        "git_version": run_git(repo_dir, ["--version"]).stdout.strip(),
        "repository_head": run_git(repo_dir, ["rev-parse", "HEAD"]).stdout.strip(),
        "object_format": run_git(repo_dir, ["rev-parse", "--show-object-format"]).stdout.strip(),
        "remotes": remotes,
        "ref_count": len(reference_tips),
        "reference_tips_sha256": hashlib.sha256(
            ("\n".join(reference_tips) + "\n").encode()
        ).hexdigest(),
        "is_shallow": False,
        "is_partial_clone": False,
        "replace_ref_count": 0,
        "grafts_present": False,
        "connectivity_check": "passed",
        "strict_object_check": (
            "passed" if strict_fsck.returncode == 0 else "legacy_object_warnings"
        ),
        "strict_object_warning_count": len(
            [line for line in strict_fsck.stderr.splitlines() if line.strip()]
        ),
        "strict_object_warnings_sha256": (
            hashlib.sha256(strict_fsck.stderr.encode()).hexdigest()
            if strict_fsck.stderr else None
        ),
        "git_no_replace_objects": True,
        "git_no_lazy_fetch": True,
        "canonical_bases": canonical_bases,
    }


def canonicalize_base(repo_dir: str | Path, base_commit: str) -> str:
    base_commit = str(base_commit or "").strip()
    if not base_commit:
        raise VisibilityBuildError("base_commit is empty")
    result = run_git(repo_dir, ["rev-parse", "--verify", f"{base_commit}^{{commit}}"], check=False)
    if result.returncode:
        raise VisibilityBuildError(
            f"cannot resolve base commit {base_commit!r}: {result.stderr.strip()}"
        )
    canonical = result.stdout.strip()
    if len(canonical) != 40:
        raise VisibilityBuildError(f"base commit did not resolve to a full SHA-1: {canonical!r}")
    return canonical


def build_window(repo_dir: str | Path, base_commit: str) -> tuple[str, list[str]]:
    canonical = canonicalize_base(repo_dir, base_commit)
    command = ["rev-list", f"--max-count={WINDOW_SIZE + 1}", canonical]
    result = run_git(repo_dir, command, check=False)
    if result.returncode:
        raise VisibilityBuildError(
            f"failed to enumerate ancestors for {canonical}: {result.stderr.strip()}"
        )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows or rows[0] != canonical:
        raise VisibilityBuildError(
            f"rev-list did not start with canonical base {canonical}: {rows[:1]}"
        )
    ancestors = [sha for sha in rows if sha != canonical][:WINDOW_SIZE]
    if len(ancestors) != len(set(ancestors)):
        raise VisibilityBuildError(f"rev-list returned duplicate ancestors for {canonical}")
    if canonical in ancestors:
        raise VisibilityBuildError(f"base commit leaked into its own visibility window: {canonical}")
    return canonical, ancestors


def build_supplemental_commit_row(
    repo_name: str,
    repo_dir: str | Path,
    index_repo_dir: Path,
    sha: str,
) -> dict:
    """Materialize a strict-only commit row absent from the legacy corpus."""
    record = run_git(
        repo_dir,
        ["show", "-s", "--format=%H%x00%ct%x00%s%x00%B", sha],
    ).stdout
    parts = record.rstrip("\n").split("\x00", 3)
    if len(parts) != 4 or parts[0] != sha:
        raise VisibilityBuildError(f"cannot parse supplemental commit metadata: {sha}")
    changed_files = [
        line.strip()
        for line in run_git(
            repo_dir,
            ["diff-tree", "--root", "--no-commit-id", "--name-only", "-r", sha],
        ).stdout.splitlines()
        if line.strip()
    ]
    subject, message = parts[2].strip(), parts[3].strip()
    issue_refs = []
    for match in ISSUE_REF_RE.finditer(subject + "\n" + message):
        issue_refs.extend(value for value in match.groups() if value)
    patches_dir = index_repo_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patches_dir / f"{sha}.diff"
    if not patch_path.exists():
        patch = run_git(
            repo_dir,
            ["show", "--format=", "--no-ext-diff", "--unified=20", sha],
        ).stdout
        patch_path.write_text(
            patch.strip() + "\n", encoding="utf-8", errors="replace"
        )
    return {
        "repo": repo_name,
        "sha": sha,
        "short_sha": sha[:9],
        "history_index": None,
        "commit_time": int(parts[1]),
        "subject": subject,
        "message": message,
        "changed_files": changed_files,
        "issue_refs": sorted(set(issue_refs), key=int),
        "patch_path": f"patches/{sha}.diff",
        "visibility_supplement": True,
    }


def write_visibility_windows(
    repo_name: str,
    repo_dir: str | Path,
    index_repo_dir: str | Path,
    base_commits: Iterable[str],
) -> dict:
    """Materialize exact Git windows beside an existing episodic index."""
    index_repo_dir = Path(index_repo_dir)
    commits_path = index_repo_dir / "commits.jsonl"
    if not commits_path.is_file():
        raise VisibilityBuildError(f"missing episodic commit corpus: {commits_path}")
    indexed_shas = {
        json.loads(line)["sha"]
        for line in commits_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    requested = list(dict.fromkeys(str(base).strip() for base in base_commits if str(base).strip()))
    audit = audit_git_repository(repo_dir, requested)
    visibility_dir = index_repo_dir / "visibility"
    visibility_dir.mkdir(parents=True, exist_ok=True)

    built_windows = []
    required_ancestor_shas = set()
    for requested_base in requested:
        canonical, ancestors = build_window(repo_dir, requested_base)
        built_windows.append((requested_base, canonical, ancestors))
        required_ancestor_shas.update(ancestors)

    supplemental_path = index_repo_dir / "visibility_commits.jsonl"
    missing_ancestors = sorted(required_ancestor_shas - indexed_shas)
    supplemental_rows = [
        build_supplemental_commit_row(repo_name, repo_dir, index_repo_dir, sha)
        for sha in missing_ancestors
    ]
    supplemental_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in supplemental_rows
        ),
        encoding="utf-8",
    )
    available_shas = indexed_shas | {row["sha"] for row in supplemental_rows}

    bases = {}
    for requested_base, canonical, ancestors in built_windows:
        unavailable = [sha for sha in ancestors if sha not in available_shas]
        if unavailable:
            raise VisibilityBuildError(
                f"strict corpus lacks {len(unavailable)} ancestors required by "
                f"{canonical}; first missing SHA: {unavailable[0]}"
            )
        row = {
            "schema_version": SCHEMA_VERSION,
            "repo": repo_name,
            "requested_base_commit": requested_base,
            "canonical_base_sha": canonical,
            "command": f"git rev-list --max-count={WINDOW_SIZE + 1} {canonical}",
            "base_excluded": True,
            "window_size_limit": WINDOW_SIZE,
            "ancestor_count": len(ancestors),
            "ancestor_shas": ancestors,
        }
        path = visibility_dir / f"{canonical}.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        bases[canonical] = {
            "requested_base_commit": requested_base,
            "path": str(path.relative_to(index_repo_dir)),
            "sha256": sha256_file(path),
            "ancestor_count": len(ancestors),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_definition": (
            "W(b)=[git rev-list --max-count=7001 <canonical-base-SHA>]\\{b}, "
            "truncated to at most 7000 commits"
        ),
        "commits_sha256": sha256_file(commits_path),
        "visibility_commits_sha256": sha256_file(supplemental_path),
        "visibility_supplement_count": len(supplemental_rows),
        "repository_audit": audit,
        "bases": bases,
    }
    manifest_path = index_repo_dir / "visibility_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
