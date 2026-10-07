#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 REPOSITORY_URL DESTINATION.git" >&2
    exit 2
fi

repository_url=$1
destination=$2
deepen_by=${GIT_DEEPEN_BY:-1000}
max_rounds=${GIT_DEEPEN_MAX_ROUNDS:-100}
fetch_timeout=${GIT_FETCH_TIMEOUT:-300}

# Incremental transfer avoids proxy truncation on large one-shot packfiles. The
# directory is not accepted as an input artifact until the final shallow=false
# check succeeds; visibility manifests are never built from an intermediate state.
if [[ ! -e "${destination}" ]]; then
    timeout "${fetch_timeout}" git clone --bare --no-tags --single-branch \
        --depth="${deepen_by}" \
        "${repository_url}" "${destination}"
elif [[ "$(git -C "${destination}" rev-parse --is-shallow-repository)" != "true" ]]; then
    echo "Existing destination is not a resumable shallow repository: ${destination}" >&2
    exit 1
else
    echo "Resuming shallow repository at $(git -C "${destination}" rev-list --count HEAD) commits"
fi

round=0
while [[ "$(git -C "${destination}" rev-parse --is-shallow-repository)" == "true" ]]; do
    round=$((round + 1))
    if (( round > max_rounds )); then
        echo "Repository remained shallow after ${max_rounds} deepen rounds" >&2
        exit 1
    fi
    attempt=0
    until timeout "${fetch_timeout}" git -C "${destination}" fetch \
        --no-tags --deepen="${deepen_by}" origin; do
        attempt=$((attempt + 1))
        if (( attempt >= 3 )); then
            echo "Deepen round ${round} failed after ${attempt} attempts" >&2
            exit 1
        fi
    done
    echo "Deepen round ${round}: $(git -C "${destination}" rev-list --count HEAD) commits"
done

if [[ -n "$(git -C "${destination}" config --get extensions.partialClone || true)" ]] \
    || git -C "${destination}" config --get-regexp '^remote\..*\.promisor$' >/dev/null 2>&1; then
    echo "Materialized repository unexpectedly has partial/promisor config" >&2
    exit 1
fi

git -C "${destination}" fsck --full --connectivity-only --no-dangling
echo "Complete non-shallow, non-partial graph ready: ${destination}"
