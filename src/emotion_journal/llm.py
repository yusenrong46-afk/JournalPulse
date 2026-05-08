import json
import os
from typing import Iterable, List, Optional

import httpx

from .config import LLM_API_KEY_ENV, LLM_BASE_URL_ENV, LLM_MODEL_ENV, LLM_TIMEOUT_SECONDS


class OpenAICompatibleCoachAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> Optional["OpenAICompatibleCoachAdapter"]:
        api_key = os.getenv(LLM_API_KEY_ENV)
        model = os.getenv(LLM_MODEL_ENV)
        if not api_key or not model:
            return None
        base_url = os.getenv(LLM_BASE_URL_ENV, "https://api.openai.com/v1")
        return cls(api_key=api_key, base_url=base_url, model=model)

    def rewrite(self, *, draft_message: str, suggested_replies: Iterable[str], context: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the assistant draft in a calm, concise, non-clinical tone. "
                        "Do not add diagnosis, therapy claims, or new safety advice. "
                        "Keep the same intent and keep it under 70 words."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "draft_message": draft_message,
                            "suggested_replies": list(suggested_replies),
                            "context": context,
                        }
                    ),
                },
            ],
            "temperature": 0.5,
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()


def llm_adapter_available() -> bool:
    return OpenAICompatibleCoachAdapter.from_env() is not None


def maybe_rewrite_coach_message(
    draft_message: str,
    suggested_replies: List[str],
    *,
    context: dict,
    use_llm: bool,
) -> tuple:
    if not use_llm:
        return draft_message, False

    adapter = OpenAICompatibleCoachAdapter.from_env()
    if adapter is None:
        return draft_message, False

    try:
        rewritten = adapter.rewrite(
            draft_message=draft_message,
            suggested_replies=suggested_replies,
            context=context,
        )
    except Exception:
        return draft_message, False
    return rewritten, True
