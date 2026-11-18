#!/usr/bin/env bash
# Container smoke test: build Dockerfile.api for this commit, run it with AI off, and check
# that /health answers 200 and names this commit. Local-only (127.0.0.1); removes what it made.
#
#   bash scripts/container_smoke.sh            (make container-smoke)
#
# When Docker is missing or its base images cannot be pulled, it prints "container-smoke:
# NOT RUN" with the reason and exits 0, so the gate records the gap instead of hiding it;
# any failure after the images are available fails.
set -euo pipefail
cd "$(dirname "$0")/.."
commit=$(git rev-parse HEAD)
tag="journalpulse-api:smoke-${commit:0:12}"
port=${JP_SMOKE_PORT:-18080}
mkdir -p outputs/container-smoke
result=outputs/container-smoke/RESULT.txt

not_run() {
  echo "container-smoke: NOT RUN ($1)" | tee "$result"
  exit 0
}

command -v docker >/dev/null || not_run "docker is not installed"
docker info >/dev/null 2>&1 || not_run "the Docker daemon is not reachable"
for image in node:22-slim python:3.12-slim ghcr.io/astral-sh/uv:0.11.28; do
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    if ! error=$(docker pull -q "$image" 2>&1); then
      not_run "base image $image could not be pulled: $(echo "$error" | tail -1 | grep -oE '[0-9]{3} [A-Za-z ]+|Forbidden|denied[^,]*' | head -1)"
    fi
  fi
done

started=$(date +%s)
docker build -q -f Dockerfile.api --build-arg JOURNALPULSE_GIT_COMMIT="$commit" -t "$tag" . >/dev/null
container=$(docker run -d --rm -p "127.0.0.1:${port}:8000" \
  -e JOURNALPULSE_LLM_ENABLED=false -e JOURNALPULSE_SEARCH_ENABLED=false "$tag")
trap 'docker rm -f "$container" >/dev/null 2>&1 || true; docker image rm "$tag" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 60); do
  if headers=$(curl -fsS -D - -o /dev/null "http://127.0.0.1:${port}/health" 2>/dev/null); then break; fi
  sleep 1
done
served=$(printf '%s' "${headers:-}" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-journalpulse-commit"{print $2}')
if [ "$served" != "$commit" ]; then
  echo "container-smoke: FAILED (/health named '${served:-nothing}', expected $commit)" | tee "$result"
  exit 1
fi
echo "container-smoke: PASS (image built and /health named $commit in $(( $(date +%s) - started )) s)" | tee "$result"
