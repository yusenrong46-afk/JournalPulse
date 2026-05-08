import json
import subprocess
import sys
from pathlib import Path


def test_training_dry_run_reports_configurable_transformers():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--dry-run-candidates",
            "--transformers",
            "distilroberta-base,bert-base-uncased,distilroberta-base",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["transformer_candidates"] == ["distilroberta-base", "bert-base-uncased"]
    assert payload["will_train_transformers"] is True
    assert payload["calibration"]["linear_svc"]


def test_training_dry_run_can_skip_transformers():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--dry-run-candidates", "--skip-transformers"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["transformer_candidates"] == []
    assert payload["will_train_transformers"] is False
