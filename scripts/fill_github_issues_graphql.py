import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


FIELDS = "title body state createdAt updatedAt closedAt url"


def load_issue_refs(commits_path):
    refs = set()
    with commits_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            refs.update(str(x) for x in row.get("issue_refs", []) if str(x).isdigit())
    return sorted(refs, key=int)


def load_existing(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row.get("issue_number"))] = row
    return rows


def write_issues(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for issue_number in sorted(rows, key=lambda x: int(x) if str(x).isdigit() else 0):
            f.write(json.dumps(rows[issue_number], ensure_ascii=False) + "\n")


def graphql_request(token, query, timeout):
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "RepoMem-episodic-index-builder-graphql",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def fetch_batch(token, repo_name, nums, timeout):
    owner, name = repo_name.split("/", 1)
    parts = []
    for idx, num in enumerate(nums):
        parts.append(f"i{idx}: issue(number: {num}) {{ {FIELDS} }}")
        parts.append(f"p{idx}: pullRequest(number: {num}) {{ {FIELDS} }}")
    query = (
        "query { repository(owner: %s, name: %s) { %s } "
        "rateLimit { remaining cost resetAt } }"
        % (json.dumps(owner), json.dumps(name), " ".join(parts))
    )
    data = graphql_request(token, query, timeout)
    repo_data = (data.get("data") or {}).get("repository") or {}
    rate = (data.get("data") or {}).get("rateLimit") or {}

    rows = {}
    for idx, num in enumerate(nums):
        item = repo_data.get(f"i{idx}") or repo_data.get(f"p{idx}")
        if item:
            rows[str(num)] = {
                "repo": repo_name,
                "issue_number": str(num),
                "fetch_status": "ok",
                "issue_summary": item.get("title") or "",
                "issue_description": item.get("body") or "",
                "state": item.get("state") or "",
                "created_at": item.get("createdAt") or "",
                "updated_at": item.get("updatedAt") or "",
                "closed_at": item.get("closedAt") or "",
                "url": item.get("url") or f"https://github.com/{repo_name}/issues/{num}",
            }
        else:
            rows[str(num)] = {
                "repo": repo_name,
                "issue_number": str(num),
                "fetch_status": "not_found",
                "issue_summary": "",
                "issue_description": "",
                "url": f"https://github.com/{repo_name}/issues/{num}",
            }
    return rows, rate, data.get("errors") or []


def update_meta(repo_dir, total_refs, issues):
    meta_path = repo_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["num_linked_issues"] = total_refs
    meta["num_crawled_issues"] = len(issues)
    meta["num_ok_issues"] = sum(1 for row in issues.values() if row.get("fetch_status") == "ok")
    meta["has_issues"] = bool(issues)
    meta["issue_fetch_mode"] = "github_graphql_issue_or_pull_request_batch"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def should_fetch(row):
    if not row:
        return True
    return row.get("fetch_status") not in {"ok", "not_found"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--github_token_env", default="GITHUB_TOKEN")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    token = os.environ.get(args.github_token_env)
    if not token:
        raise RuntimeError(f"{args.github_token_env} is not set")

    root = Path(args.index_dir)
    for repo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        commits_path = repo_dir / "commits.jsonl"
        issues_path = repo_dir / "issues.jsonl"
        if not commits_path.exists():
            continue

        with commits_path.open(encoding="utf-8") as f:
            repo_name = json.loads(next(f))["repo"]
        refs = load_issue_refs(commits_path)
        issues = load_existing(issues_path)
        pending = [num for num in refs if should_fetch(issues.get(num))]
        print(
            f"{repo_name}: refs={len(refs)} existing={len(issues)} pending={len(pending)}",
            flush=True,
        )

        done = 0
        for start in range(0, len(pending), args.batch_size):
            nums = pending[start : start + args.batch_size]
            last_exc = None
            for attempt in range(1, args.retries + 1):
                try:
                    got, rate, errors = fetch_batch(token, repo_name, nums, args.timeout)
                    issues.update(got)
                    done += len(nums)
                    if errors:
                        print(
                            f"{repo_name}: graphql errors in batch "
                            f"{start // args.batch_size + 1}: {len(errors)}",
                            flush=True,
                        )
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    print(
                        f"{repo_name}: retryable {type(exc).__name__} "
                        f"attempt {attempt}/{args.retries}",
                        flush=True,
                    )
                    time.sleep(min(30, 2**attempt))
            else:
                print(f"{repo_name}: failed batch after retries: {last_exc}", flush=True)
                continue

            if done == len(nums) or done % 250 == 0 or done >= len(pending):
                write_issues(issues_path, issues)
                ok = sum(1 for row in issues.values() if row.get("fetch_status") == "ok")
                print(
                    f"{repo_name}: fetched {done}/{len(pending)} pending, "
                    f"total={len(issues)}, ok={ok}, remaining={rate.get('remaining')}",
                    flush=True,
                )
            time.sleep(args.sleep)

        write_issues(issues_path, issues)
        update_meta(repo_dir, len(refs), issues)
        ok = sum(1 for row in issues.values() if row.get("fetch_status") == "ok")
        print(f"{repo_name}: complete total={len(issues)} ok={ok}", flush=True)


if __name__ == "__main__":
    main()
