# Targeted Python dependency repairs — October 5, 2026

Updated only `uv.lock`: **GitPython 3.1.51 → 3.2.0** and **urllib3 2.7.0 → 2.8.0**, using `uv lock --upgrade-package gitpython --upgrade-package urllib3`. All other 78 package records (including the local project record), package membership, dependency roots, and Python constraints are unchanged. The before lock and advisory scan were preserved.

The original exact-version PyPI scan reported 38 GitPython advisory entries representing 21 unique CVEs; the highest minimum fixed version across those entries is 3.1.60. urllib3 had six entries representing CVE-2026-97687, CVE-2026-97688, and CVE-2026-97689, all fixed in 2.8.0. The compact `python-upgrade-advisory-summary.json` groups every ID/CVE by its advertised fixed version, avoiding duplicate counts. Official latest-release and exact-version PyPI metadata report no known advisories for the selected GitPython 3.2.0 and urllib3 2.8.0 versions.

Both packages are reached only through optional research dependencies: `journalpulse[research] → streamlit → GitPython`, and `streamlit → requests → urllib3`. The base/dev/test dependency closures do not include them. No direct product/script/test Python imports or call sites were found. The API Dockerfile installs the default dependencies and CI installs the dev extra. This source and installation-policy review does not attest to packages on an existing deployment.

Validation passed:

- `uv lock --check`: all 80 package records resolve.
- `uv sync --frozen --extra dev --dry-run`: checks 36 packages and would make no changes.
- `uv sync --frozen --all-extras --dry-run`: optional research selection resolves; it would add 43 packages, which were not installed in the project environment.
- An isolated environment with the two patched packages and GitPython's two small dependencies passed import/version checks, temporary repository initialization and benign config round-trip, author parsing, local gzip response decoding, and HTTPS pool construction. The smoke made no HTTP requests and no commits.

The shared `.venv` was unchanged. The actual optional Streamlit/research application was not run, so the compatibility evidence is dependency resolution plus focused package API smoke, not an end-to-end research workflow test. No deployment changed.

Evidence: `python-lock-before.toml`, `python-lock-before-manifest.json`, `python-lock-after-manifest.json`, `python-upgrade-pypi-metadata.json`, `python-upgrade-exact-version-check.json`, `python-upgrade-advisory-summary.json`, and `python-upgrade-validation.json`. The root audit is independently rescanning all 79 registry packages against the new lock into a separate after artifact.
