"""One benign two-turn Luna check. Run only when a live smoke test is explicitly authorized."""

from __future__ import annotations

from journalpulse.config import load_settings
from journalpulse.intelligence import ConversationProviderError, OpenRouterConversationClient

INPUT_USD_PER_MILLION = 0.20
OUTPUT_USD_PER_MILLION = 1.20
TURNS = (
    "I finished a hard meeting and I still feel a little restless. I want to notice that.",
    "A short walk might help. What is one small thing I could try?",
)


def main() -> None:
    settings = load_settings()
    try:
        client = OpenRouterConversationClient(settings)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    history: list[dict[str, str]] = []
    prompt_tokens = 0
    completion_tokens = 0
    offered = False
    for text in TURNS:
        history.append({"role": "user", "content": text})
        try:
            result = client.complete(history)
        except ConversationProviderError as exc:
            raise SystemExit(str(exc)) from exc
        if result.model_run.used_fallback or not result.model_run.schema_valid:
            raise SystemExit("Luna did not return a valid conversation turn.")
        prompt_tokens += result.model_run.prompt_tokens or 0
        completion_tokens += result.model_run.completion_tokens or 0
        offered = offered or result.offer_action
        history.append({"role": "assistant", "content": result.reply})
        print(f"model: {result.model_run.model}")
        print(f"provider: {result.model_run.provider}")
        print(f"latency_ms: {result.model_run.latency_ms}")
        print(f"prompt_tokens: {result.model_run.prompt_tokens}")
        print(f"completion_tokens: {result.model_run.completion_tokens}")
        print(f"card_offered: {result.offer_action}")

    cost = (prompt_tokens * INPUT_USD_PER_MILLION + completion_tokens * OUTPUT_USD_PER_MILLION) / 1_000_000
    print(f"estimated_cost_usd: {cost:.6f}")
    print("card_offered_in_conversation: " + ("yes" if offered else "no"))
    print("OpenRouter conversation verification passed")


if __name__ == "__main__":
    main()
