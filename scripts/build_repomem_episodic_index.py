import argparse
import csv
import io
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ISSUE_REF_RE = re.compile(
    r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d+)|#(\d+)",
    flags=re.IGNORECASE,
)


def run_git(repo_dir, args, max_output=None):
    out = subprocess.run(
        ["git", "-C", repo_dir] + args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=120,
    ).stdout
    return out[:max_output] if max_output else out


def iter_commits(repo_dir, max_commits):
    fmt = "%x1e%H%x00%ct%x00%s%x00%B%x00--ENDMSG--"
    out = run_git(repo_dir, ["log", "--reverse", "--name-only", f"--format={fmt}"])
    records = [r for r in out.split("\x1e") if r.strip()]
    if max_commits:
        records = records[-max_commits:]
    for record in records:
        if "\x00--ENDMSG--" not in record:
            continue
        header, file_part = record.split("\x00--ENDMSG--", 1)
        parts = header.split("\x00", 3)
        if len(parts) < 4:
            continue
        sha, commit_time, subject, message = parts
        changed_files = [x.strip() for x in file_part.splitlines() if x.strip()]
        issue_refs = []
        for match in ISSUE_REF_RE.finditer(subject + "\n" + message):
            issue_refs.extend(ref for ref in match.groups() if ref)
        yield {
            "sha": sha.strip(),
            "commit_time": int(commit_time.strip()) if commit_time.strip().isdigit() else 0,
            "subject": subject.strip(),
            "message": message.strip(),
            "changed_files": changed_files,
            "issue_refs": sorted(set(issue_refs), key=int),
        }


def fetch_django_ticket(repo_name, issue_number, timeout=30):
    url = f"https://code.djangoproject.com/ticket/{issue_number}?format=tab"
    req = urllib.request.Request(url, headers={"User-Agent": "RepoMem-episodic-index-builder"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8-sig", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    except urllib.error.HTTPError as exc:
        return {
            "repo": repo_name,
            "issue_number": str(issue_number),
            "fetch_status": f"http_{exc.code}",
            "issue_summary": "",
            "issue_description": "",
            "url": f"https://code.djangoproject.com/ticket/{issue_number}",
        }
    except Exception as exc:
        return {
            "repo": repo_name,
            "issue_number": str(issue_number),
            "fetch_status": f"error:{type(exc).__name__}",
            "issue_summary": "",
            "issue_description": "",
            "url": f"https://code.djangoproject.com/ticket/{issue_number}",
        }

    if not rows:
        return {
            "repo": repo_name,
            "issue_number": str(issue_number),
            "fetch_status": "not_found",
            "issue_summary": "",
            "issue_description": "",
            "url": f"https://code.djangoproject.com/ticket/{issue_number}",
        }
    row = rows[0]
    return {
        "repo": repo_name,
        "issue_number": str(issue_number),
        "fetch_status": "ok",
        "issue_summary": row.get("summary") or "",
        "issue_description": row.get("description") or "",
        "state": row.get("status") or "",
        "component": row.get("component") or "",
        "resolution": row.get("resolution") or "",
        "url": f"https://code.djangoproject.com/ticket/{issue_number}",
    }


def fetch_github_issue(repo_name, issue_number, token=None, timeout=30):
    url = f"https://api.github.com/repos/{repo_name}/issues/{issue_number}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "RepoMem-episodic-index-builder",
        },
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        return {
            "repo": repo_name,
            "issue_number": str(issue_number),
            "fetch_status": f"http_{exc.code}",
            "issue_summary": "",
            "issue_description": "",
            "url": url,
        }
    except Exception as exc:
        return {
            "repo": repo_name,
            "issue_number": str(issue_number),
            "fetch_status": f"error:{type(exc).__name__}",
            "issue_summary": "",
            "issue_description": "",
            "url": url,
        }

    return {
        "repo": repo_name,
        "issue_number": str(issue_number),
        "fetch_status": "ok",
        "issue_summary": data.get("title") or "",
        "issue_description": data.get("body") or "",
        "state": data.get("state") or "",
        "created_at": data.get("created_at") or "",
        "updated_at": data.get("updated_at") or "",
        "closed_at": data.get("closed_at") or "",
        "url": data.get("html_url") or url,
    }


def fetch_linked_issue(repo_name, issue_number, token=None, timeout=30):
    if repo_name == "django/django":
        return fetch_django_ticket(repo_name, issue_number, timeout=timeout)
    return fetch_github_issue(repo_name, issue_number, token=token, timeout=timeout)


def load_existing_issues(issues_path):
    issues = {}
    if not issues_path.exists():
        return issues
    with issues_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            issues[str(row.get("issue_number"))] = row
    return issues


def write_issues(issues_path, issues):
    with issues_path.open("w", encoding="utf-8") as f:
        for issue_number in sorted(issues, key=lambda x: int(x) if x and x.isdigit() else 0):
            f.write(json.dumps(issues[issue_number], ensure_ascii=False) + "\n")


def write_patch(repo_dir, sha, patch_path):
    patch = run_git(repo_dir, ["show", "--format=", "--no-ext-diff", "--unified=20", sha])
    patch_path.write_text(patch.strip() + "\n", encoding="utf-8", errors="replace")


def build_repo_index(
    repo_name,
    repo_dir,
    output_dir,
    max_commits,
    crawl_issues=False,
    github_token=None,
    github_api_delay=0.0,
    issue_timeout=30,
    issue_workers=8,
    store_patches=True,
):
    repo_out = Path(output_dir) / repo_name.replace("/", "__")
    repo_out.mkdir(parents=True, exist_ok=True)
    patches_dir = repo_out / "patches"
    if store_patches:
        patches_dir.mkdir(parents=True, exist_ok=True)
    commits_path = repo_out / "commits.jsonl"
    issues_path = repo_out / "issues.jsonl"
    meta_path = repo_out / "meta.json"

    top_counter = Counter()
    rows = []
    issue_refs = set()
    for idx, row in enumerate(iter_commits(repo_dir, max_commits)):
        for path in row["changed_files"]:
            if path.endswith(".py"):
                top_counter[path] += 1
        issue_refs.update(row["issue_refs"])

        patch_rel_path = None
        if store_patches:
            patch_rel_path = f"patches/{row['sha']}.diff"
            patch_path = repo_out / patch_rel_path
            if not patch_path.exists():
                write_patch(repo_dir, row["sha"], patch_path)

        rows.append(
            {
                "repo": repo_name,
                "sha": row["sha"],
                "short_sha": row["sha"][:9],
                "history_index": idx,
                "commit_time": row["commit_time"],
                "subject": row["subject"],
                "message": row["message"],
                "changed_files": row["changed_files"],
                "issue_refs": row["issue_refs"],
                "patch_path": patch_rel_path,
            }
        )

    issues = load_existing_issues(issues_path)
    if crawl_issues:
        sorted_issue_refs = sorted(issue_refs, key=int)
        pending = [
            issue_number for issue_number in sorted_issue_refs
            if not (issue_number in issues and issues[issue_number].get("fetch_status") == "ok")
        ]
        done_before = len(sorted_issue_refs) - len(pending)
        if done_before:
            ok_count = sum(1 for row in issues.values() if row.get("fetch_status") == "ok")
            print(
                f"{repo_name}: resuming issue crawl with {done_before}/{len(sorted_issue_refs)} "
                f"already done ({ok_count} ok)",
                flush=True,
            )

        completed = done_before
        workers = max(1, int(issue_workers or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    fetch_linked_issue,
                    repo_name,
                    issue_number,
                    github_token,
                    issue_timeout,
                ): issue_number
                for issue_number in pending
            }
            for future in as_completed(futures):
                issue_number = futures[future]
                try:
                    issues[issue_number] = future.result()
                except Exception as exc:
                    issues[issue_number] = {
                        "repo": repo_name,
                        "issue_number": str(issue_number),
                        "fetch_status": f"error:{type(exc).__name__}",
                        "issue_summary": "",
                        "issue_description": "",
                        "url": "",
                    }
                completed += 1
                if completed == 1 or completed % 50 == 0 or completed == len(sorted_issue_refs):
                    write_issues(issues_path, issues)
                if completed == 1 or completed % 100 == 0 or completed == len(sorted_issue_refs):
                    ok_count = sum(1 for row in issues.values() if row.get("fetch_status") == "ok")
                    print(
                        f"{repo_name}: crawled {completed}/{len(sorted_issue_refs)} linked issues "
                        f"({ok_count} ok)",
                        flush=True,
                    )
                if github_api_delay:
                    time.sleep(github_api_delay)

    write_issues(issues_path, issues)

    with commits_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "repo": repo_name,
                "repo_dir": repo_dir,
                "num_commits": len(rows),
                "num_linked_issues": len(issue_refs),
                "num_crawled_issues": len(issues),
                "sha_to_history_index": {row["sha"]: row["history_index"] for row in rows},
                "top_python_files_by_commits": top_counter.most_common(500),
                "has_patches": store_patches,
                "has_issues": bool(issues),
            },
            f,
            indent=2,
        )

    print(f"{repo_name}: wrote {len(rows)} commits to {commits_path}")
    print(f"{repo_name}: wrote {len(issues)} issues to {issues_path}")


def parse_repo_map(items):
    repo_map = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected REPO=PATH, got {item}")
        repo, path = item.split("=", 1)
        if not os.path.isdir(path):
            raise FileNotFoundError(path)
        repo_map[repo] = path
    return repo_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        help="Repository mapping in the form owner/name=/path/to/clone.",
    )
    parser.add_argument("--max_commits", type=int, default=0)
    parser.add_argument(
        "--crawl_issues",
        action="store_true",
        help="Fetch linked GitHub issues referenced by commit messages and write issues.jsonl.",
    )
    parser.add_argument(
        "--github_token_env",
        default="GITHUB_TOKEN",
        help="Environment variable containing a GitHub token for issue crawling.",
    )
    parser.add_argument("--github_api_delay", type=float, default=0.0)
    parser.add_argument("--issue_timeout", type=int, default=10)
    parser.add_argument("--issue_workers", type=int, default=8)
    parser.add_argument(
        "--store_patches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize commit patches under patches/<sha>.diff.",
    )
    args = parser.parse_args()

    repo_map = parse_repo_map(args.repo)
    github_token = os.environ.get(args.github_token_env)
    for repo_name, repo_dir in repo_map.items():
        build_repo_index(
            repo_name=repo_name,
            repo_dir=repo_dir,
            output_dir=args.output_dir,
            max_commits=args.max_commits,
            crawl_issues=args.crawl_issues,
            github_token=github_token,
            github_api_delay=args.github_api_delay,
            issue_timeout=args.issue_timeout,
            issue_workers=args.issue_workers,
            store_patches=args.store_patches,
        )


if __name__ == "__main__":
    main()
