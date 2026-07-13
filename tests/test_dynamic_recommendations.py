from pathlib import Path
from types import SimpleNamespace

from emotion_journal.config import RESOURCE_DOMAIN_SAFELIST
from emotion_journal.experience import build_prediction_experience
from emotion_journal.llm import (
    OpenAICompatibleCoachAdapter,
    url_domain_allowed,
    validate_resource_recommendation_payload,
)
from emotion_journal.resources import (
    build_recommendation_set,
    load_resource_catalog,
    make_generated_resource_card,
    recommend_resources,
)


def _catalog_ids(n=2):
    return [resource["id"] for resource in load_resource_catalog() if resource["resource_type"] != "support"][:n]


def test_content_aware_ranking_changes_with_text(tmp_path: Path):
    db = tmp_path / "jp.db"
    # A query packed with grounding/breathing language should not error and should
    # still return a full, catalog-only set when no LLM is configured.
    recs = recommend_resources(
        "fear",
        query="my chest is tight and i need a calm breathing grounding exercise",
        db_path=db,
    )
    assert recs, "expected catalog recommendations"
    assert all(resource.get("source", "catalog") == "catalog" for resource in recs)


def test_url_domain_allowed_blocks_lookalikes():
    assert url_domain_allowed("https://www.nimh.nih.gov/health", RESOURCE_DOMAIN_SAFELIST)
    assert url_domain_allowed("https://youtu.be/abc123", RESOURCE_DOMAIN_SAFELIST)
    # Lookalike host that merely ends in a safelisted-looking string must be rejected.
    assert not url_domain_allowed("https://totally-not-nih.gov.evil.com/x", RESOURCE_DOMAIN_SAFELIST)
    assert not url_domain_allowed("https://sketchy.example.com/page", RESOURCE_DOMAIN_SAFELIST)
    assert not url_domain_allowed("notaurl", RESOURCE_DOMAIN_SAFELIST)


def test_validate_recommendation_filters_unsafe_and_unknown():
    catalog_ids = _catalog_ids(2)
    payload = {
        "ranked_catalog_ids": catalog_ids + ["does_not_exist"],
        "generated": [
            {
                "title": "Box Breathing Guide",
                "url": "https://www.headspace.com/meditation/box-breathing",
                "resource_type": "website",
                "coping_style": "read",
                "provider": "Headspace",
                "summary": "A short box-breathing walkthrough.",
                "why": "Matches the need to settle a racing mind.",
                "goal_tags": ["ground"],
            },
            {
                "title": "Sketchy link",
                "url": "https://malware.example.com/x",
                "resource_type": "website",
                "coping_style": "read",
                "provider": "Unknown",
                "summary": "Should be dropped.",
                "why": "Off the safelist.",
            },
        ],
    }
    result = validate_resource_recommendation_payload(
        payload,
        allowed_catalog_ids=catalog_ids,
        allowed_domains=RESOURCE_DOMAIN_SAFELIST,
        max_generated=3,
    )
    assert result["ranked_catalog_ids"] == catalog_ids  # unknown id dropped
    assert len(result["generated"]) == 1  # off-safelist suggestion dropped
    assert result["generated"][0]["url"].startswith("https://www.headspace.com")


def test_make_generated_card_is_badged_and_youtube_aware():
    card = make_generated_resource_card(
        {
            "title": "Calming Nature Sounds",
            "url": "https://youtu.be/abcdef",
            "resource_type": "video",
            "coping_style": "watch",
            "provider": "YouTube",
            "summary": "Ambient nature footage.",
            "why": "Helps slow down after a tense day.",
            "goal_tags": ["ground"],
        },
        emotion="fear",
    )
    assert card["source"] == "ai_suggested"
    assert card["embed_kind"] == "youtube"
    assert card["id"].startswith("ai_")
    assert card["rationale"]
    assert card["is_crisis_safe"] is False


def test_llm_recommender_merges_generated(monkeypatch, tmp_path: Path):
    catalog_ids = _catalog_ids(2)

    class FakeAdapter:
        model = "test-model"

        def recommend(self, *, text, analysis_context, catalog_options, allowed_domains, max_generated):
            return {
                "ranked_catalog_ids": catalog_ids,
                "generated": [
                    {
                        "title": "5-Minute Grounding Audio",
                        "url": "https://insighttimer.com/grounding",
                        "resource_type": "website",
                        "coping_style": "read",
                        "provider": "Insight Timer",
                        "summary": "A guided grounding audio.",
                        "why": "Built for exactly this kind of overwhelm.",
                        "goal_tags": ["ground"],
                    }
                ],
            }

    monkeypatch.setenv("JOURNALPULSE_LLM_MODE", "structured")
    monkeypatch.setattr(
        OpenAICompatibleCoachAdapter, "from_env", classmethod(lambda cls: FakeAdapter())
    )

    resources, meta = build_recommendation_set(
        "I feel overwhelmed and my thoughts keep racing about work",
        emotion="fear",
        analysis_context={"emotion": "fear", "emotion_tags": ["overwhelm"]},
        db_path=tmp_path / "jp.db",
        use_llm=True,
    )

    assert meta["used_llm_recommender"] is True
    assert meta["generated_count"] == 1
    assert meta["recommender_model"] == "test-model"
    ai_cards = [r for r in resources if r.get("source") == "ai_suggested"]
    assert len(ai_cards) == 1
    assert ai_cards[0]["title"] == "5-Minute Grounding Audio"
    # AI suggestions are surfaced first, catalog grounding still present.
    assert resources[0]["source"] == "ai_suggested"
    assert any(r.get("source") == "catalog" for r in resources)


def test_llm_recommender_falls_back_on_error(monkeypatch, tmp_path: Path):
    class BoomAdapter:
        model = "test-model"

        def recommend(self, **kwargs):
            raise RuntimeError("network down")

    monkeypatch.setenv("JOURNALPULSE_LLM_MODE", "structured")
    monkeypatch.setattr(
        OpenAICompatibleCoachAdapter, "from_env", classmethod(lambda cls: BoomAdapter())
    )

    resources, meta = build_recommendation_set(
        "I feel anxious",
        emotion="fear",
        db_path=tmp_path / "jp.db",
        use_llm=True,
    )
    assert meta["used_llm_recommender"] is False
    assert meta["generated_count"] == 0
    assert meta["recommender_fallback_reason"].startswith("resource_llm_invalid")
    assert all(r.get("source") == "catalog" for r in resources)


def _fake_prediction(emotion="fear", is_crisis=False):
    return SimpleNamespace(
        emotion=emotion,
        is_crisis=is_crisis,
        confidence=0.6,
        confidence_band="medium",
        scores={"sadness": 0.1, "joy": 0.1, "love": 0.05, "anger": 0.1, "fear": 0.6, "surprise": 0.05},
        secondary_emotions=[],
        emotion_tags=["overwhelm"],
        is_mixed=False,
        uncertainty_reason=None,
        classifier_source="artifact",
        classifier_mode="calibrated",
        recommendation="Take one small grounding step.",
    )


def test_experience_exposes_recommendation_meta(tmp_path: Path):
    class FakePredictor:
        def predict(self, text, *, location=None, activity=None):
            return _fake_prediction()

    experience = build_prediction_experience(
        FakePredictor(),
        "I feel anxious about my workload and cannot settle",
        db_path=tmp_path / "jp.db",
        use_llm=False,
    )
    assert "recommendation_meta" in experience
    assert experience["recommendation_meta"]["used_llm_recommender"] is False
    assert experience["resources"], "expected catalog recommendations in the experience"
    assert all(r.get("source", "catalog") == "catalog" for r in experience["resources"])


def test_crisis_skips_llm_recommender(monkeypatch, tmp_path: Path):
    called = {"recommend": False}

    class TrackingAdapter:
        model = "test-model"

        def recommend(self, **kwargs):
            called["recommend"] = True
            return {"ranked_catalog_ids": [], "generated": []}

    monkeypatch.setenv("JOURNALPULSE_LLM_MODE", "structured")
    monkeypatch.setattr(
        OpenAICompatibleCoachAdapter, "from_env", classmethod(lambda cls: TrackingAdapter())
    )

    resources, meta = build_recommendation_set(
        "I want to die",
        emotion="sadness",
        is_crisis=True,
        db_path=tmp_path / "jp.db",
        use_llm=True,
    )
    assert called["recommend"] is False, "crisis entries must never call the LLM recommender"
    assert meta["used_llm_recommender"] is False
    assert all(r.get("source", "catalog") == "catalog" for r in resources)
    assert all(r.get("is_crisis_safe") or r.get("resource_type") == "support" for r in resources)
