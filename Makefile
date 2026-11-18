# JournalPulse task runner. Portfolio Standard gates: see upgrade/SPEC.md and upgrade/STATUS.md.
#
#   make setup      install the locked Python and Node dependencies
#   make test       backend tests (pytest + coverage gate) and frontend unit tests
#   make run        local API (AI and search off, SQLite, local identity) and web dev server
#   make gate-l1    everything a stranger needs: setup, test, CI lint checks, README and secret scan
#
# Postgres and browser suites run in CI and locally with: make test-postgres, make test-browser,
# make test-integration (they need JOURNALPULSE_PG_DSN pointing at a disposable local database).

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
UV ?= uv
NPM ?= npm
PG_VERIFIERS := verify_postgres_schema verify_journal_schema verify_integrity_upgrade \
	verify_activity_schema verify_sentinel_repairs verify_erasure_schema

.PHONY: setup test lint test-postgres test-browser test-integration test-all run \
	check-readme secret-scan gate-scripts gate-l1

setup:
	$(UV) sync --frozen --extra dev
	$(NPM) --prefix web ci

test:
	$(UV) run pytest --cov=journalpulse --cov-report=term
	$(NPM) --prefix web run test:unit

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run mypy src
	$(UV) run python scripts/export_openapi.py --check
	$(UV) run python scripts/validate_resources.py
	$(NPM) --prefix web run lint
	$(NPM) --prefix web run typecheck
	$(NPM) --prefix web run generate:api
	git diff --exit-code -- web/openapi.json web/lib/generated-api.ts

test-postgres:
	@test -n "$${JOURNALPULSE_PG_DSN:-}" || { echo "Set JOURNALPULSE_PG_DSN to a disposable local PostgreSQL"; exit 2; }
	@for script in $(PG_VERIFIERS); do echo "== $$script"; $(UV) run python scripts/$$script.py; done

test-browser:
	$(NPM) --prefix web run test:e2e

test-integration:
	$(NPM) --prefix web run test:integration

test-all: test lint test-postgres test-browser test-integration

run:
	@echo "API on http://127.0.0.1:8000 and web on http://127.0.0.1:3000 (AI and search off). Ctrl+C stops both."
	@trap 'kill 0' INT TERM EXIT; \
	JOURNALPULSE_LLM_ENABLED=false JOURNALPULSE_SEARCH_ENABLED=false \
	  $(UV) run uvicorn journalpulse.api:app --host 127.0.0.1 --port 8000 & \
	$(NPM) --prefix web run dev -- --hostname 127.0.0.1 & \
	wait

check-readme:
	$(UV) run python scripts/check_readme.py

secret-scan:
	$(UV) run python scripts/secret_scan.py

gate-scripts: check-readme secret-scan

gate-l1: setup test lint gate-scripts
	@echo "gate-l1: PASS"
