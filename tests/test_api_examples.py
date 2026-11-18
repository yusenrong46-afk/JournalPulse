"""docs/API.md's curl examples run against the app exactly as written (contract test).

Each fenced bash block that starts with ``curl`` is parsed (method, path, headers, body)
and sent to the real FastAPI app in local mode; the ``# expect:`` status must match.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.config import Settings

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"```bash\n(curl .*?)```", re.DOTALL)


def examples() -> list[tuple[str, int]]:
    found = []
    for block in BLOCK.findall((ROOT / "docs" / "API.md").read_text()):
        expect = re.search(r"# expect: (\d{3})", block)
        assert expect, f"example without # expect: {block[:60]}"
        command = re.sub(r"\\\n\s*", " ", block.split("# expect:")[0]).strip()
        found.append((command, int(expect.group(1))))
    return found


def parse(command: str) -> tuple[str, str, dict[str, str], dict | None]:
    words = shlex.split(command)
    assert words[0] == "curl"
    method, url, headers, body = "GET", "", {}, None
    index = 1
    while index < len(words):
        word = words[index]
        if word == "-X":
            method, index = words[index + 1], index + 2
        elif word == "-H":
            name, value = words[index + 1].split(":", 1)
            headers[name.strip()] = value.strip()
            index += 2
        elif word == "-d":
            body, index = json.loads(words[index + 1]), index + 2
        elif word.startswith("-"):
            index += 1
        else:
            url, index = word, index + 1
    assert url.startswith("http://127.0.0.1:8000/"), url
    return method, url.removeprefix("http://127.0.0.1:8000"), headers, body


def test_docs_api_examples_return_their_documented_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("JOURNALPULSE_GIT_COMMIT", "0123abc")
    settings = Settings(
        environment="test", database_path=tmp_path / "api.db",
        resource_catalog_path=ROOT / "assets" / "resources" / "catalog.json",
        openrouter_api_key=None, openrouter_model="unused", openrouter_base_url="https://provider.example",
        openrouter_zdr=True, openrouter_timeout_seconds=1, raw_text_retention_default=False,
        supabase_url=None, supabase_anon_key=None, llm_feature_enabled=False,
    )
    documented = examples()
    assert len(documented) == 3
    with TestClient(create_app(settings=settings)) as client:
        responses = []
        for command, expected in documented:
            method, path, headers, body = parse(command)
            response = client.request(method, path, headers=headers, json=body)
            assert response.status_code == expected, (command, response.text)
            responses.append(response)
        health, status, saved = responses
        assert health.json() == {"status": "ok"} and health.headers["X-JournalPulse-Commit"] == "0123abc"
        assert status.json()["build"] == {"commit": "0123abc", "source": "JOURNALPULSE_GIT_COMMIT"}
        assert status.json()["analysis_mode"] == "local_only"
        assert saved.json()["text"].startswith("Fictional example")
        # The documented idempotency claims: same retry is safe, a changed retry is refused.
        method, path, headers, body = parse(documented[2][0])
        assert client.request(method, path, headers=headers, json=body).status_code in {200, 201}
        changed = {**body, "text": "Changed"}
        assert client.request(method, path, headers=headers, json=changed).status_code == 409
        owner_list = client.get("/v1/journal/entries", headers=headers).json()["items"]
        assert [item["id"] for item in owner_list] == [saved.json()["id"]]
        other = {"X-JournalPulse-User": "00000000-0000-4000-8000-0000000000b2"}
        assert client.get("/v1/journal/entries", headers=other).json()["items"] == []
