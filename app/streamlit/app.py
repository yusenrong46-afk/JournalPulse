import html
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import httpx
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_journal.coach import llm_mode_available, respond_with_coach
from emotion_journal.config import (
    API_BASE_URL_ENV,
    COPING_STYLES,
    DEFAULT_DB_PATH,
    DEFAULT_RESOURCE_TYPES,
    LABELS,
    LLM_MODE_ENV,
    MODELS_DIR,
    RESOURCE_GOAL_TAGS,
    SOURCE_TIERS,
    api_base_url,
)
from emotion_journal.db import (
    initialize_database,
    insert_entry,
    list_entries,
    record_resource_interaction,
)
from emotion_journal.demo import demo_entries
from emotion_journal.experience import build_prediction_experience
from emotion_journal.llm import configured_llm_mode
from emotion_journal.model import get_default_predictor
from emotion_journal.resources import (
    filter_resources,
    get_resource_lookup,
    load_resource_catalog,
    recommend_resources,
    resource_catalog_summary,
    resource_titles,
    resources_by_style,
)

st.set_page_config(page_title="JournalPulse", page_icon="JP", layout="wide")


EMOTION_DETAILS = {
    "sadness": {
        "color": "#3b82f6",
        "surface": "#eff6ff",
        "focus": "Name what feels heavy and what still feels changeable.",
    },
    "joy": {
        "color": "#16a34a",
        "surface": "#ecfdf5",
        "focus": "Capture what created the lift so it is easier to repeat.",
    },
    "love": {
        "color": "#db2777",
        "surface": "#fdf2f8",
        "focus": "Notice the connection and translate it into one small action.",
    },
    "anger": {
        "color": "#dc2626",
        "surface": "#fef2f2",
        "focus": "Separate the crossed boundary from the next deliberate move.",
    },
    "fear": {
        "color": "#7c3aed",
        "surface": "#f5f3ff",
        "focus": "Shrink the worry into one next safe step.",
    },
    "surprise": {
        "color": "#0891b2",
        "surface": "#ecfeff",
        "focus": "Capture the new information before the moment blurs.",
    },
}

STYLE_LABELS = {
    "watch": "Watch",
    "read": "Read",
    "play": "Play",
    "move": "Move",
}
STYLE_FROM_LABEL = {label: style for style, label in STYLE_LABELS.items()}

SOURCE_TIER_LABELS = {
    "official": "Official",
    "nonprofit": "Nonprofit",
    "educational": "Educational",
    "activity": "Activity",
    "crisis_support": "Crisis support",
}

GOAL_TAG_LABELS = {
    "ground": "Ground",
    "planning": "Plan",
    "reframing": "Reframe",
    "connection": "Connect",
    "movement": "Move",
    "reading": "Read",
    "watching": "Watch",
    "play": "Play",
}

FEEDBACK_LABELS = {
    "helpful": "Helpful",
    "not_helpful": "Not helpful",
    "unsure": "Unsure",
    None: "Unrated",
}

CHAT_STARTERS = (
    (
        "Meeting friction",
        "My manager dismissed my idea in the meeting and I am still frustrated. I keep replaying what I should have said, but I do not want to spiral tonight.",
    ),
    (
        "Anxious loop",
        "I am waiting for a result and my thoughts keep jumping to worst-case scenarios. I want help separating facts from fear.",
    ),
    (
        "Small win",
        "I finally finished something I had been avoiding, and I feel lighter than I expected. I want to remember what helped me start.",
    ),
)

EMOTION_ORDER = [LABELS[index] for index in sorted(LABELS)]
PREDICTION_FIELDS = (
    "emotion",
    "confidence",
    "recommendation",
    "disclaimer",
    "is_crisis",
    "scores",
    "support_message",
    "model_name",
    "confidence_band",
    "reflection_summary",
    "interpretation",
    "follow_up_prompts",
    "explanation_phrases",
    "secondary_emotions",
    "emotion_tags",
    "top_margin",
    "is_mixed",
    "uncertainty_reason",
    "calibration_notes",
    "classifier_mode",
    "classifier_source",
    "classifier_fallback_reason",
)
def backend_readiness_status(base_url: str) -> dict:
    if not base_url:
        return {"status": "local", "detail": "Streamlit is using in-process demo services."}
    try:
        response = httpx.get(f"{base_url}/ready", timeout=4.0)
        response.raise_for_status()
    except Exception as exc:
        return {"status": "unreachable", "detail": f"{exc.__class__.__name__}"}
    return response.json()


def backend_enabled() -> bool:
    return bool(api_base_url())


def api_get(path: str, *, params: dict = None):
    response = httpx.get(f"{api_base_url()}{path}", params=params, timeout=20.0)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = httpx.post(f"{api_base_url()}{path}", json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def api_patch(path: str, payload: dict):
    response = httpx.patch(f"{api_base_url()}{path}", json=payload, timeout=20.0)
    response.raise_for_status()
    return response.json()


def prediction_from_payload(payload: dict):
    return SimpleNamespace(**{field: payload.get(field) for field in PREDICTION_FIELDS})


def experience_from_prediction_payload(payload: dict) -> dict:
    return {
        "prediction": prediction_from_payload(payload),
        "resources": payload.get("resources", []),
        "recommendation_meta": {
            "used_llm_recommender": payload.get("used_llm_recommender", False),
            "generated_count": payload.get("generated_resource_count", 0),
            "recommender_model": payload.get("recommender_model"),
            "recommender_fallback_reason": payload.get("recommender_fallback_reason"),
        },
        "coach": {
            "assistant_message": payload.get("coach_opening"),
            "coach_state": payload.get("coach_state", {}),
            "suggested_replies": payload.get("suggested_replies", []),
            "tips": payload.get("tips", []),
            "practical_steps": payload.get("practical_steps", []),
            "reflection_question": payload.get("reflection_question"),
            "communication_draft": payload.get("communication_draft"),
            "confidence_note": payload.get("confidence_note"),
            "used_llm": payload.get("used_llm", False),
            "coach_mode": payload.get("coach_mode", "deterministic"),
            "agent_mode": payload.get("agent_mode", "deterministic"),
            "agent_model": payload.get("agent_model"),
            "agent_fallback_reason": payload.get("agent_fallback_reason"),
            "fallback_reason": payload.get("fallback_reason"),
            "resource_rationales": payload.get("resource_rationales", {}),
            "resource_intent": payload.get("resource_intent"),
        },
    }


def build_experience_for_ui(*, predictor, text: str, location: str, activity: str, use_llm: bool):
    if backend_enabled():
        payload = api_post(
            "/predict",
            {
                "text": text,
                "location": location,
                "activity": activity,
                "use_llm": use_llm,
            },
        )
        return experience_from_prediction_payload(payload)
    return build_prediction_experience(
        predictor,
        text,
        location=location,
        activity=activity,
        db_path=DEFAULT_DB_PATH,
        use_llm=use_llm,
    )


def list_entries_for_ui() -> list:
    if backend_enabled():
        return api_get("/entries").get("entries", [])
    return list_entries(db_path=DEFAULT_DB_PATH)
def resources_for_ui(*, emotion=None, resource_type=None, coping_style=None):
    if backend_enabled():
        params = {
            key: value
            for key, value in {
                "emotion": emotion,
                "resource_type": resource_type,
                "coping_style": coping_style,
            }.items()
            if value
        }
        return api_get("/resources", params=params).get("resources", [])
    return filter_resources(emotion=emotion, coping_style=coping_style, resource_type=resource_type)


def resource_summary_for_ui() -> dict:
    if backend_enabled():
        return api_get("/resources/summary")
    return resource_catalog_summary()


def save_entry_for_ui(pending: dict, feedback: str) -> dict:
    prediction = pending["prediction"]
    coach_summary = safe_coach_summary_from_pending(pending)
    if backend_enabled():
        return api_post(
            "/entries",
            {
                "text": pending["text"],
                "location": pending["location"],
                "activity": pending["activity"],
                "feedback": feedback,
                "coach_summary": coach_summary,
            },
        )
    return insert_entry(
        text=pending["text"],
        emotion=prediction.emotion,
        confidence=prediction.confidence,
        recommendation=prediction.recommendation,
        location=pending["location"],
        activity=pending["activity"],
        feedback=feedback,
        reflection_summary=prediction.reflection_summary,
        interpretation=prediction.interpretation,
        confidence_band=prediction.confidence_band,
        model_name=prediction.model_name,
        classifier_mode=getattr(prediction, "classifier_mode", "calibrated"),
        classifier_source=getattr(prediction, "classifier_source", "artifact"),
        classifier_fallback_reason=getattr(prediction, "classifier_fallback_reason", None),
        support_message=prediction.support_message,
        follow_up_prompts=prediction.follow_up_prompts,
        explanation_phrases=prediction.explanation_phrases,
        coach_state_summary=(
            f"step={pending['coach_state'].get('step')}|"
            f"emotion={pending['coach_state'].get('framing_emotion')}|"
            f"style={pending['coach_state'].get('selected_coping_style') or 'none'}"
        ),
        coach_summary=coach_summary,
        suggested_resource_ids=[resource["id"] for resource in pending["resources"]],
        db_path=DEFAULT_DB_PATH,
    )


def escape(value) -> str:
    return html.escape("" if value is None else str(value))


def emotion_detail(emotion: str) -> dict:
    return EMOTION_DETAILS.get(
        emotion,
        {"color": "#334155", "surface": "#f8fafc", "focus": "Stay specific about what happened next."},
    )


def render_global_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --jp-ink: #172033;
                --jp-muted: #667085;
                --jp-border: #d7dee8;
                --jp-panel: #ffffff;
                --jp-soft: #f6f8fb;
                --jp-accent: #0f766e;
                --jp-accent-strong: #115e59;
                --jp-warm: #a16207;
            }

            .block-container {
                padding-top: 1.15rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }

            div[data-testid="stSidebar"] {
                background: #fbfcfe;
                border-right: 1px solid var(--jp-border);
            }

            .jp-sidebar-brand {
                border-bottom: 1px solid var(--jp-border);
                padding-bottom: 0.85rem;
                margin-bottom: 0.9rem;
            }

            .jp-sidebar-brand h2 {
                margin: 0;
                font-size: 1.35rem;
            }

            .jp-sidebar-brand p {
                color: var(--jp-muted);
                font-size: 0.9rem;
                line-height: 1.45;
                margin: 0.35rem 0 0 0;
            }

            h1, h2, h3 {
                color: var(--jp-ink);
                letter-spacing: 0;
            }

            .jp-hero {
                background: #ffffff;
                border: 1px solid var(--jp-border);
                border-radius: 8px;
                padding: 1.15rem 1.25rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(21, 32, 52, 0.04);
            }

            .jp-hero h1 {
                margin: 0 0 0.35rem 0;
                font-size: 2rem;
                line-height: 1.1;
            }

            .jp-hero p {
                margin: 0;
                color: #46566c;
                max-width: 780px;
                line-height: 1.55;
            }

            .jp-mode-row {
                display: flex;
                gap: 0.45rem;
                flex-wrap: wrap;
                margin-top: 0.85rem;
            }

            .jp-panel {
                background: var(--jp-panel);
                border: 1px solid var(--jp-border);
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 26px rgba(21, 32, 52, 0.04);
            }

            .jp-panel-tight {
                background: var(--jp-panel);
                border: 1px solid var(--jp-border);
                border-radius: 8px;
                padding: 0.85rem 0.95rem;
                margin-bottom: 0.75rem;
            }

            .jp-workspace-toolbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.85rem;
                margin: 0.25rem 0 0.9rem 0;
                flex-wrap: wrap;
            }

            .jp-workspace-toolbar p {
                color: var(--jp-muted);
                margin: 0;
            }

            .jp-result {
                border-radius: 8px;
                border: 1px solid var(--jp-border);
                padding: 1.05rem 1.1rem;
                margin: 0.5rem 0 1rem 0;
                background: #ffffff;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            }

            .jp-result-top {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: flex-start;
                flex-wrap: wrap;
            }

            .jp-eyebrow {
                color: var(--jp-muted);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .jp-emotion {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 1.85rem;
                font-weight: 800;
                color: var(--jp-ink);
            }

            .jp-focus-note {
                border-left: 3px solid var(--jp-accent);
                background: #f0fdfa;
                border-radius: 6px;
                color: #134e4a;
                margin-top: 0.9rem;
                padding: 0.7rem 0.8rem;
                line-height: 1.45;
            }

            .jp-pill {
                display: inline-flex;
                align-items: center;
                width: fit-content;
                border-radius: 999px;
                border: 1px solid rgba(15, 23, 42, 0.12);
                padding: 0.2rem 0.55rem;
                font-size: 0.78rem;
                font-weight: 700;
                color: #334155;
                background: #ffffff;
                margin: 0.15rem 0.25rem 0.15rem 0;
                white-space: nowrap;
            }

            .jp-stat-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                gap: 0.75rem;
                margin-top: 1rem;
            }

            .jp-stat {
                background: var(--jp-soft);
                border: 1px solid var(--jp-border);
                border-radius: 8px;
                padding: 0.8rem;
            }

            .jp-stat span {
                display: block;
                color: var(--jp-muted);
                font-size: 0.76rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .jp-stat strong {
                display: block;
                color: var(--jp-ink);
                font-size: 1rem;
                margin-top: 0.2rem;
                overflow-wrap: anywhere;
            }

            .jp-step-list {
                display: grid;
                gap: 0.65rem;
                margin: 0;
                padding: 0;
            }

            .jp-step {
                display: grid;
                grid-template-columns: 1.8rem 1fr;
                gap: 0.65rem;
                align-items: start;
            }

            .jp-step-number {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1.55rem;
                height: 1.55rem;
                border-radius: 999px;
                background: #e6fffb;
                color: #115e59;
                font-size: 0.82rem;
                font-weight: 800;
            }

            .jp-card-title {
                margin: 0 0 0.35rem 0;
                font-size: 1.08rem;
                font-weight: 800;
                color: var(--jp-ink);
            }

            .jp-muted {
                color: var(--jp-muted);
            }

            .jp-copy {
                color: #334155;
                line-height: 1.55;
            }

            .jp-score-row {
                display: grid;
                grid-template-columns: 92px 1fr 52px;
                align-items: center;
                gap: 0.7rem;
                margin: 0.48rem 0;
                color: #334155;
                font-size: 0.88rem;
            }

            .jp-score-track {
                height: 0.55rem;
                border-radius: 999px;
                background: #e5edf6;
                overflow: hidden;
            }

            .jp-score-fill {
                height: 100%;
                border-radius: 999px;
            }

            .jp-entry-preview {
                background: #f8fafc;
                border: 1px solid var(--jp-border);
                border-radius: 8px;
                padding: 0.9rem;
                color: #334155;
                line-height: 1.55;
                white-space: pre-wrap;
            }

            .jp-resource-card {
                padding: 0.1rem 0 0 0;
            }

            .jp-resource-card--ai .jp-card-title {
                color: var(--jp-accent-strong);
            }

            .jp-ai-badge {
                display: inline-block;
                font-size: 0.68rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                text-transform: uppercase;
                color: #ffffff;
                background: linear-gradient(135deg, var(--jp-accent), var(--jp-accent-strong));
                padding: 0.12rem 0.5rem;
                border-radius: 999px;
                vertical-align: middle;
                margin-left: 0.3rem;
            }

            .jp-section-label {
                font-size: 0.82rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: var(--jp-muted);
                margin: 1.1rem 0 0.4rem 0;
            }

            .jp-resource-meta {
                color: var(--jp-muted);
                font-size: 0.85rem;
                margin-bottom: 0.55rem;
            }

            .jp-resource-rationale {
                border-left: 3px solid var(--jp-accent);
                background: #f0fdfa;
                border-radius: 6px;
                padding: 0.65rem 0.75rem;
                color: #134e4a;
                font-size: 0.9rem;
                line-height: 1.45;
                margin: 0.75rem 0;
            }

            .jp-assistant-card {
                border: 1px solid var(--jp-border);
                border-left: 4px solid var(--jp-accent);
                border-radius: 8px;
                padding: 0.95rem 1rem;
                background: #ffffff;
                margin-bottom: 0.9rem;
            }

            .jp-assistant-card p {
                margin: 0;
                color: #26354a;
                line-height: 1.55;
            }

            .jp-chat-intro {
                border: 1px solid var(--jp-border);
                border-left: 4px solid var(--jp-accent);
                border-radius: 8px;
                padding: 0.95rem 1rem;
                background: #ffffff;
                margin-bottom: 0.9rem;
            }

            .jp-chat-intro p {
                margin: 0;
                color: #26354a;
                line-height: 1.55;
            }

            .jp-context-label {
                color: var(--jp-muted);
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin: 0.35rem 0 0.5rem 0;
            }

            .jp-tip-list {
                display: grid;
                gap: 0.55rem;
                margin-bottom: 0.9rem;
            }

            .jp-tip {
                background: #f8fafc;
                border: 1px solid var(--jp-border);
                border-radius: 8px;
                padding: 0.72rem 0.82rem;
                color: #334155;
                line-height: 1.45;
            }

            .jp-coach-meta {
                color: var(--jp-muted);
                font-size: 0.85rem;
                margin-top: -0.2rem;
                margin-bottom: 0.75rem;
            }

            .jp-demo-note {
                background: #fff7ed;
                border: 1px solid #fed7aa;
                border-radius: 8px;
                color: #7c2d12;
                padding: 0.8rem 0.9rem;
                margin-bottom: 1rem;
            }

            .jp-empty {
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                border-radius: 8px;
                padding: 1rem;
                color: #475569;
            }

            div[data-testid="stTabs"] button {
                font-weight: 700;
            }

            div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
                border-radius: 8px;
                min-height: 2.5rem;
                font-weight: 700;
            }

            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--jp-border);
                border-radius: 8px;
                padding: 0.8rem;
                box-shadow: 0 8px 22px rgba(21, 32, 52, 0.04);
            }

            div[data-testid="stMetric"] label {
                color: var(--jp-muted);
                font-weight: 700;
            }

            @media (max-width: 760px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .jp-hero h1 {
                    font-size: 1.75rem;
                }

                .jp-stat-grid {
                    grid-template-columns: 1fr;
                }

                .jp-score-row {
                    grid-template-columns: 78px 1fr 46px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_predictor():
    return get_default_predictor()


def load_model_card() -> dict:
    production_path = MODELS_DIR / "production.json"
    if not production_path.exists():
        return {}
    return json.loads(production_path.read_text())


def load_model_quality_report() -> dict:
    report_path = PROJECT_ROOT / "artifacts" / "reports" / "model_quality_eval.json"
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text())


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="jp-sidebar-brand">
            <h2>JournalPulse</h2>
            <p>Journaling support with emotion signals, coaching, and resources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pages = ["Chat", "Resources", "History", "Model"]
    page = st.sidebar.radio(
        "Workspace",
        pages,
    )
    st.sidebar.divider()
    backend_url = api_base_url()
    backend_status = backend_readiness_status(backend_url)
    st.sidebar.markdown("**Product boundary**")
    st.sidebar.caption("Reflection support, not therapy or diagnosis. Crisis language switches to support-first guidance.")
    use_llm = False
    llm_mode = configured_llm_mode()
    if llm_mode_available():
        label = "Use AI reflection agent" if llm_mode == "structured" else "AI-polished coach wording"
        use_llm = st.sidebar.toggle(label, value=False)
    with st.sidebar.expander("Developer status", expanded=False):
        if backend_url:
            status = backend_status.get("status", "unknown")
            st.caption(f"API backend: `{status}`")
            st.caption(f"`{API_BASE_URL_ENV}` = {backend_url}")
        else:
            st.caption("API backend: local in-process mode.")
        if llm_mode_available():
            st.caption(f"LLM mode: `{llm_mode}` via `{LLM_MODE_ENV}`.")
        else:
            st.caption("Coach: deterministic fallback active.")
    st.session_state["use_llm"] = use_llm
    return page


def render_page_hero(title: str, body: str) -> None:
    st.markdown(
        f"""
        <section class="jp-hero">
            <h1>{escape(title)}</h1>
            <p>{escape(body)}</p>
            <div class="jp-mode-row">
                <span class="jp-pill">Transformer NLP</span>
                <span class="jp-pill">Explainable signals</span>
                <span class="jp-pill">Safety routed</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_phrase_chips(phrases) -> None:
    if not phrases:
        st.caption("No phrase-level explanation is available for this entry.")
        return
    chips = "".join(f'<span class="jp-pill">{escape(phrase)}</span>' for phrase in phrases)
    st.markdown(chips, unsafe_allow_html=True)


def render_label_chips(labels) -> None:
    labels = labels or []
    if not labels:
        st.caption("No extra tags available yet.")
        return
    chips = "".join(
        f'<span class="jp-pill">{escape(str(label).replace("_", " ").title())}</span>'
        for label in labels
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_prompt_cards(prompts) -> None:
    if not prompts:
        return
    columns = st.columns(len(prompts))
    for index, (column, prompt) in enumerate(zip(columns, prompts), start=1):
        with column:
            st.markdown(
                f"""
                <div class="jp-panel">
                    <div class="jp-eyebrow">Prompt {index}</div>
                    <div class="jp-copy">{escape(prompt)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_link_button(label: str, url: str, key: str):
    if hasattr(st, "link_button"):
        st.link_button(label, url, use_container_width=True)
    else:
        st.markdown(f"[{label}]({url})")


def format_percent(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0%}"


def render_demo_seed_notice(page_name: str) -> None:
    st.markdown(
        f"""
        <div class="jp-demo-note">
            Showing seeded recruiter-demo {escape(page_name)} because this local SQLite store is empty.
            Save a reflection to replace the demo view with your own session data.
        </div>
        """,
        unsafe_allow_html=True,
    )


def classifier_source_label(prediction) -> str:
    source = getattr(prediction, "classifier_source", "artifact") or "artifact"
    mode = getattr(prediction, "classifier_mode", "calibrated") or "calibrated"
    fallback_reason = getattr(prediction, "classifier_fallback_reason", None)
    if source == "llm":
        return "Structured LLM"
    if mode in {"llm", "hybrid"} and fallback_reason:
        return "Artifact fallback"
    return "Calibrated artifact"


def render_prediction_summary(prediction) -> None:
    detail = emotion_detail(prediction.emotion)
    band = prediction.confidence_band or "unknown"
    is_mixed = bool(getattr(prediction, "is_mixed", False))
    support_label = "Safety mode" if prediction.is_crisis else ("Mixed signal" if is_mixed else "Reflection mode")
    secondary_emotions = getattr(prediction, "secondary_emotions", []) or []
    secondary_text = ", ".join(emotion.title() for emotion in secondary_emotions)
    focus_text = detail["focus"]
    if secondary_text:
        focus_text = f"{focus_text} Also check whether {secondary_text} is present."
    margin = getattr(prediction, "top_margin", None)
    margin_label = f"{margin:.0%} margin" if margin is not None else None
    st.markdown(
        f"""
        <section class="jp-result" style="border-top: 5px solid {detail['color']};">
            <div class="jp-result-top">
                <div>
                    <div class="jp-eyebrow">Current signal</div>
                    <div class="jp-emotion">{escape(prediction.emotion.title())}</div>
                </div>
                <div>
                    <span class="jp-pill" style="background:{detail['surface']}; border-color:{detail['color']}33;">
                        {escape(support_label)}
                    </span>
                    <span class="jp-pill">{escape(band.title())} confidence</span>
                    {f'<span class="jp-pill">{escape(margin_label)}</span>' if margin_label else ''}
                </div>
            </div>
            <div class="jp-stat-grid">
                <div class="jp-stat"><span>Confidence</span><strong>{format_percent(prediction.confidence)}</strong></div>
                <div class="jp-stat"><span>Model</span><strong>{escape(prediction.model_name or "Unknown")}</strong></div>
                <div class="jp-stat"><span>Classifier</span><strong>{escape(classifier_source_label(prediction))}</strong></div>
            </div>
            <div class="jp-focus-note"><strong>Suggested focus:</strong> {escape(focus_text)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_score_bars(scores: dict, dominant_emotion: str) -> None:
    if not scores:
        return
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    rows = []
    for emotion, score in ordered:
        detail = emotion_detail(emotion)
        color = detail["color"] if emotion == dominant_emotion else "#94a3b8"
        rows.append(
            f"""
            <div class="jp-score-row">
                <div>{escape(emotion.title())}</div>
                <div class="jp-score-track">
                    <div class="jp-score-fill" style="width:{max(0, min(score, 1)) * 100:.1f}%; background:{color};"></div>
                </div>
                <div>{score:.0%}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def record_resource_action(resource: dict, action: str, *, emotion: str, entry_id=None) -> None:
    payload = {
        "resource_id": resource["id"],
        "action": action,
        "emotion": emotion,
        "entry_id": entry_id if isinstance(entry_id, int) else None,
    }
    if backend_enabled():
        api_post("/resource-interactions", payload)
        return
    record_resource_interaction(db_path=DEFAULT_DB_PATH, **payload)


def render_resource_card(
    resource: dict,
    *,
    emotion: str,
    entry_id=None,
    show_actions: bool = True,
    show_video_preview: bool = True,
):
    duration = resource.get("duration_minutes")
    duration_label = f"{duration} min" if duration else "Flexible"
    source_tier = resource.get("source_tier") or "educational"
    source_label = SOURCE_TIER_LABELS.get(source_tier, source_tier.replace("_", " ").title())
    goal_chips = "".join(
        f'<span class="jp-pill">{escape(GOAL_TAG_LABELS.get(tag, tag.title()))}</span>'
        for tag in resource.get("goal_tags", [])[:4]
    )
    rationale = resource.get("rationale")
    is_ai = resource.get("source") == "ai_suggested"
    card_class = "jp-resource-card jp-resource-card--ai" if is_ai else "jp-resource-card"
    ai_badge = '<span class="jp-ai-badge">✨ AI-suggested</span>' if is_ai else ""
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="jp-card-title">{escape(resource['title'])} {ai_badge}</div>
                <div class="jp-resource-meta">
                    {escape(resource['provider'])} | {escape(resource['resource_type'].title())} | {escape(duration_label)}
                </div>
                <div>
                    <span class="jp-pill">{escape(source_label)}</span>
                    <span class="jp-pill">{escape(STYLE_LABELS.get(resource['coping_style'], resource['coping_style'].title()))}</span>
                    <span class="jp-pill">{escape(resource['embed_kind'].title())}</span>
                    {goal_chips}
                </div>
                <p class="jp-copy">{escape(resource["summary"])}</p>
                {f'<div class="jp-resource-rationale"><strong>Why this resource:</strong> {escape(rationale)}</div>' if rationale else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if resource.get("embed_kind") == "youtube" and show_video_preview:
            with st.expander("Preview video", expanded=False):
                st.video(resource["url"])

        render_link_button("Open", resource["url"], key=f"open-link-{resource['id']}")

        if not show_actions:
            return

        left, middle, right = st.columns(3)
        if left.button("Opened", key=f"opened-{entry_id}-{resource['id']}", use_container_width=True):
            record_resource_action(resource, "opened", emotion=emotion, entry_id=entry_id)
            st.success("Marked as opened.")
        if middle.button("Helpful", key=f"helpful-{entry_id}-{resource['id']}", use_container_width=True):
            record_resource_action(resource, "helpful", emotion=emotion, entry_id=entry_id)
            st.success("Marked as helpful.")
        if right.button("Dismiss", key=f"dismissed-{entry_id}-{resource['id']}", use_container_width=True):
            record_resource_action(resource, "dismissed", emotion=emotion, entry_id=entry_id)
            st.info("Dismissed for future ranking.")


def render_resource_card_grid(
    resources: list,
    *,
    emotion: str = None,
    entry_id=None,
    limit: int = 4,
    show_actions: bool = True,
    show_video_preview: bool = True,
) -> None:
    visible_resources = resources[:limit]
    if not visible_resources:
        st.markdown('<div class="jp-empty">No curated cards match this preference yet.</div>', unsafe_allow_html=True)
        return

    columns = st.columns(2, gap="large") if len(visible_resources) > 1 else [st.container()]
    for index, resource in enumerate(visible_resources):
        card_emotion = emotion or resource.get("emotion_tags", ["joy"])[0]
        with columns[index % len(columns)]:
            render_resource_card(
                resource,
                emotion=card_emotion,
                entry_id=entry_id,
                show_actions=show_actions,
                show_video_preview=show_video_preview,
            )

    hidden_count = len(resources) - len(visible_resources)
    if hidden_count > 0:
        st.caption(f"Showing the strongest {len(visible_resources)} matches here. Open Resources for {hidden_count} more options.")


def selected_resource_style(label: str):
    return None if label == "Blend" else STYLE_FROM_LABEL[label]


def current_resource_set(pending: dict, selected_label: str) -> list:
    prediction = pending["prediction"]
    style = selected_resource_style(selected_label)
    goal = pending.get("resource_intent") or pending.get("coach_state", {}).get("last_intent")
    if backend_enabled():
        return api_get(
            "/resources/recommendations",
            params={key: value for key, value in {
                "emotion": prediction.emotion,
                "coping_style": style,
                "goal": goal,
                "is_crisis": prediction.is_crisis,
            }.items() if value is not None},
        ).get("resources", [])
    return recommend_resources(
        prediction.emotion,
        coping_style=style,
        goal=goal,
        db_path=DEFAULT_DB_PATH,
        is_crisis=prediction.is_crisis,
    )


def rerank_pending_resources(pending):
    style = selected_resource_style(pending.get("resource_style_choice", "Blend"))
    goal = pending.get("resource_intent") or pending.get("coach_state", {}).get("last_intent")
    if backend_enabled():
        pending["resources"] = api_get(
            "/resources/recommendations",
            params={key: value for key, value in {
                "emotion": pending["coach_state"].get("framing_emotion", pending["prediction"].emotion),
                "coping_style": style,
                "goal": goal,
                "is_crisis": pending["prediction"].is_crisis,
            }.items() if value is not None},
        ).get("resources", [])
        return
    pending["resources"] = recommend_resources(
        pending["coach_state"].get("framing_emotion", pending["prediction"].emotion),
        coping_style=style,
        goal=goal,
        db_path=DEFAULT_DB_PATH,
        is_crisis=pending["prediction"].is_crisis,
    )


def safe_coach_summary_from_pending(pending: dict) -> dict:
    coach_state = pending.get("coach_state", {})
    return {
        "turn_count": int(coach_state.get("turns", 0) or 0),
        "final_step": coach_state.get("step"),
        "framing_emotion": coach_state.get("framing_emotion"),
        "selected_coping_style": coach_state.get("selected_coping_style"),
        "resource_ids": [resource["id"] for resource in pending.get("resources", [])],
        "used_llm": bool(pending.get("coach_used_llm")),
        "safety_mode": bool(pending["prediction"].is_crisis),
    }


def apply_coach_turn(pending, user_message: str):
    if backend_enabled():
        response = api_post(
            "/coach/respond",
            {
                "text": pending["text"],
                "emotion": pending["prediction"].emotion,
                "confidence_band": pending["prediction"].confidence_band,
                "coach_state": pending["coach_state"],
                "user_message": user_message,
                "is_crisis": pending["prediction"].is_crisis,
                "use_llm": pending["use_llm"],
            },
        )
    else:
        response = respond_with_coach(
            entry_text=pending["text"],
            emotion=pending["prediction"].emotion,
            confidence_band=pending["prediction"].confidence_band,
            coach_state=pending["coach_state"],
            user_message=user_message,
            is_crisis=pending["prediction"].is_crisis,
            use_llm=pending["use_llm"],
            db_path=DEFAULT_DB_PATH,
        )
    pending["coach_transcript"].append({"role": "user", "content": user_message})
    pending["coach_transcript"].append({"role": "assistant", "content": response["assistant_message"]})
    pending["coach_state"] = response["coach_state"]
    pending["suggested_replies"] = response["suggested_replies"]
    pending["coach_tips"] = response.get("tips", [])
    pending["practical_steps"] = response.get("practical_steps", []) or response.get("tips", [])
    pending["reflection_question"] = response.get("reflection_question")
    pending["communication_draft"] = response.get("communication_draft")
    pending["confidence_note"] = response.get("confidence_note")
    pending["coach_used_llm"] = bool(pending.get("coach_used_llm") or response.get("used_llm"))
    pending["coach_mode"] = response.get("coach_mode", "deterministic")
    pending["coach_fallback_reason"] = response.get("fallback_reason")
    pending["agent_mode"] = response.get("agent_mode", pending.get("agent_mode", "deterministic"))
    pending["agent_model"] = response.get("agent_model") or pending.get("agent_model")
    pending["agent_fallback_reason"] = response.get("agent_fallback_reason")
    pending["resource_intent"] = response.get("resource_intent")
    pending["coach_resource_rationales"] = response.get("resource_rationales", {})
    selected_style = pending["coach_state"].get("selected_coping_style")
    if response["resource_ids"] and selected_style in STYLE_LABELS:
        pending["resource_style_choice"] = STYLE_LABELS[selected_style]
    if response["resource_ids"]:
        lookup = get_resource_lookup()
        resources = []
        for resource_id in response["resource_ids"]:
            if resource_id not in lookup:
                continue
            resource = dict(lookup[resource_id])
            rationale = pending["coach_resource_rationales"].get(resource_id)
            if rationale:
                resource["rationale"] = rationale
            resources.append(resource)
        pending["resources"] = resources


def store_pending_entry(*, text: str, location: str, activity: str, use_llm: bool) -> None:
    with st.spinner("Reading the entry and preparing the reflection..."):
        experience = build_experience_for_ui(
            predictor=load_predictor() if not backend_enabled() else None,
            text=text,
            location=location.strip() or None,
            activity=activity.strip() or None,
            use_llm=use_llm,
        )
    recommendation_meta = experience.get("recommendation_meta", {})
    all_resources = experience["resources"]
    st.session_state["pending_entry"] = {
        "text": text,
        "location": location.strip() or None,
        "activity": activity.strip() or None,
        "prediction": experience["prediction"],
        "resources": all_resources,
        # Stable copy of AI-suggested cards so catalog re-ranking never drops them.
        "generated_resources": [r for r in all_resources if r.get("source") == "ai_suggested"],
        "used_llm_recommender": recommendation_meta.get("used_llm_recommender", False),
        "recommender_model": recommendation_meta.get("recommender_model"),
        "coach_state": experience["coach"]["coach_state"],
        "suggested_replies": experience["coach"]["suggested_replies"],
        "coach_transcript": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": experience["coach"]["assistant_message"]}
        ],
        "coach_tips": experience["coach"].get("tips", []),
        "practical_steps": experience["coach"].get("practical_steps", []) or experience["coach"].get("tips", []),
        "reflection_question": experience["coach"].get("reflection_question"),
        "communication_draft": experience["coach"].get("communication_draft"),
        "confidence_note": experience["coach"].get("confidence_note"),
        "coach_used_llm": experience["coach"].get("used_llm", False),
        "coach_mode": experience["coach"].get("coach_mode", "deterministic"),
        "coach_fallback_reason": experience["coach"].get("fallback_reason"),
        "agent_mode": experience["coach"].get("agent_mode", "deterministic"),
        "agent_model": experience["coach"].get("agent_model"),
        "agent_fallback_reason": experience["coach"].get("agent_fallback_reason"),
        "coach_resource_rationales": experience["coach"].get("resource_rationales", {}),
        "resource_intent": experience["coach"].get("resource_intent"),
        "resource_style_choice": "Blend",
        "use_llm": use_llm,
    }


def render_entry_composer(
    use_llm: bool,
    *,
    form_key: str = "entry-form",
    context_in_expander: bool = True,
) -> None:
    prefill_key = f"{form_key}-prefill"
    text_key = f"{form_key}-message"
    location_key = f"{form_key}-location"
    activity_key = f"{form_key}-activity"
    st.session_state.setdefault(text_key, "")
    st.session_state.setdefault(location_key, "")
    st.session_state.setdefault(activity_key, "")

    st.markdown(
        """
        <div class="jp-chat-intro">
            <div class="jp-eyebrow">JournalPulse</div>
            <p>Tell me what happened, what felt unresolved, or what you want help thinking through. I’ll respond with a grounded next step and keep the model details available off to the side.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    starter_columns = st.columns(len(CHAT_STARTERS))
    for column, (label, starter_text) in zip(starter_columns, CHAT_STARTERS):
        if column.button(label, key=f"{form_key}-starter-{label}", use_container_width=True):
            st.session_state[text_key] = starter_text
            st.session_state[prefill_key] = starter_text
            st.rerun()

    if prefill_key in st.session_state:
        st.session_state.pop(prefill_key, None)

    st.text_area(
        "Message",
        height=210,
        placeholder="Example: I left the meeting frustrated because I felt dismissed, and I still do not know whether to confront it or let it go.",
        key=text_key,
    )
    if context_in_expander:
        context_container = st.expander("Optional context", expanded=False)
    else:
        st.markdown("**Optional context**")
        context_container = st.container()
    with context_container:
        left, right = st.columns(2)
        left.text_input("Location", placeholder="Optional", key=location_key)
        right.text_input("Activity", placeholder="Optional", key=activity_key)
    submitted = st.button("Send to JournalPulse", key=f"{form_key}-submit", use_container_width=True)

    st.caption("JournalPulse is reflection support, not therapy or diagnosis. Crisis language switches to support-first guidance.")

    if submitted:
        submitted_text = st.session_state.get(text_key, "").strip()
        if not submitted_text:
            st.error("Write a few sentences first so I have something real to respond to.")
        else:
            store_pending_entry(
                text=submitted_text,
                location=st.session_state.get(location_key, ""),
                activity=st.session_state.get(activity_key, ""),
                use_llm=use_llm,
            )
            st.rerun()


def render_reflection_details(pending: dict) -> None:
    prediction = pending["prediction"]
    overview_column, signals_column = st.columns([1.15, 0.85], gap="large")
    with overview_column:
        st.markdown(
            f"""
            <div class="jp-panel-tight">
                <div class="jp-eyebrow">Guidance</div>
                <div class="jp-copy">{escape(prediction.recommendation)}</div>
            </div>
            <div class="jp-panel-tight">
                <div class="jp-eyebrow">Summary</div>
                <div class="jp-copy">{escape(prediction.reflection_summary)}</div>
            </div>
            <div class="jp-panel-tight">
                <div class="jp-eyebrow">Interpretation</div>
                <div class="jp-copy">{escape(prediction.interpretation)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with signals_column:
        with st.container(border=True):
            st.markdown("**Score distribution**")
            render_score_bars(prediction.scores, prediction.emotion)
            if getattr(prediction, "is_mixed", False):
                st.caption(
                    f"Mixed-signal note: {getattr(prediction, 'uncertainty_reason', 'top scores are close')}."
                )
            st.markdown("**Emotion tags**")
            render_label_chips(getattr(prediction, "emotion_tags", []))
            st.markdown("**Explanation phrases**")
            render_phrase_chips(prediction.explanation_phrases)
            fallback_reason = getattr(prediction, "classifier_fallback_reason", None)
            if fallback_reason:
                st.caption(f"Classifier fallback: {fallback_reason}")

    if prediction.follow_up_prompts:
        st.markdown("**Reflection prompts**")
        render_prompt_cards(prediction.follow_up_prompts)


def render_resource_recommendations(
    pending: dict,
    *,
    show_actions: bool = True,
    show_video_preview: bool = True,
) -> None:
    prediction = pending["prediction"]
    if prediction.is_crisis:
        st.caption("Normal entertainment and distraction links are suppressed in safety mode.")
        render_resource_card_grid(
            pending["resources"],
            emotion=prediction.emotion,
            limit=3,
            show_actions=show_actions,
            show_video_preview=show_video_preview,
        )
        return

    generated = pending.get("generated_resources", [])
    if generated:
        model_note = pending.get("recommender_model")
        st.markdown(
            f'<div class="jp-section-label">✨ Personalized for this entry'
            f'{f" · {escape(model_note)}" if model_note else ""}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "AI-suggested links matched to what you wrote, from a vetted set of reputable "
            "domains. Open in a new tab to confirm they fit."
        )
        render_resource_card_grid(
            generated,
            emotion=prediction.emotion,
            limit=4,
            show_actions=show_actions,
            show_video_preview=show_video_preview,
        )
        st.markdown('<div class="jp-section-label">From the curated library</div>', unsafe_allow_html=True)

    resource_choice = st.radio(
        "What would help right now?",
        ["Blend"] + [STYLE_LABELS[style] for style in COPING_STYLES],
        horizontal=True,
        key="resource-style-choice",
        index=(["Blend"] + [STYLE_LABELS[style] for style in COPING_STYLES]).index(
            pending.get("resource_style_choice", "Blend")
        ),
    )
    pending["resource_style_choice"] = resource_choice
    pending["resources"] = current_resource_set(pending, resource_choice)

    if resource_choice == "Blend":
        st.caption("Showing a compact mix. Choose a style above for a narrower set.")
        render_resource_card_grid(
            pending["resources"],
            emotion=prediction.emotion,
            limit=4,
            show_actions=show_actions,
            show_video_preview=show_video_preview,
        )
    else:
        render_resource_card_grid(
            pending["resources"],
            emotion=prediction.emotion,
            limit=4,
            show_actions=show_actions,
            show_video_preview=show_video_preview,
        )


def render_coach_panel(pending: dict) -> None:
    mode_label = pending.get("coach_mode", "deterministic").replace("_", " ").title()
    fallback = pending.get("coach_fallback_reason")
    agent_model = pending.get("agent_model")
    agent_fallback = pending.get("agent_fallback_reason")
    st.markdown(
        f"""
        <div class="jp-coach-meta">
            Coach mode: <strong>{escape(mode_label)}</strong>
            {f' | Model: {escape(agent_model)}' if agent_model else ''}
            {f' | Fallback: {escape(fallback or agent_fallback)}' if (fallback or agent_fallback) else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in pending["coach_transcript"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    practical_steps = pending.get("practical_steps") or pending.get("coach_tips") or []
    if practical_steps:
        st.markdown("**Practical steps**")
        tip_markup = "".join(f'<div class="jp-tip">{escape(tip)}</div>' for tip in practical_steps)
        st.markdown(f'<div class="jp-tip-list">{tip_markup}</div>', unsafe_allow_html=True)

    if pending.get("communication_draft"):
        st.markdown("**One thing you could say**")
        st.markdown(f'<div class="jp-tip">{escape(pending["communication_draft"])}</div>', unsafe_allow_html=True)

    if pending.get("reflection_question"):
        st.markdown("**Reflection question**")
        st.markdown(f'<div class="jp-tip">{escape(pending["reflection_question"])}</div>', unsafe_allow_html=True)

    if pending.get("confidence_note"):
        st.caption(pending["confidence_note"])

    if pending.get("resource_intent") and pending.get("resource_intent") != "none":
        st.caption(f"Resource intent: {pending['resource_intent']}")

    if pending["suggested_replies"]:
        reply_columns = st.columns(len(pending["suggested_replies"]))
        for column, reply in zip(reply_columns, pending["suggested_replies"]):
            if column.button(reply, key=f"reply-{reply}", use_container_width=True):
                apply_coach_turn(pending, reply)
                rerank_pending_resources(pending)
                st.rerun()

    with st.form("coach-form"):
        coach_input = st.text_input("Reply", placeholder="Example: show me something to watch")
        coach_submit = st.form_submit_button("Send", use_container_width=True)
    if coach_submit and coach_input.strip():
        apply_coach_turn(pending, coach_input.strip())
        rerank_pending_resources(pending)
        st.rerun()


def render_save_panel(pending: dict) -> None:
    prediction = pending["prediction"]
    st.markdown(
        f"""
        <div class="jp-panel">
            <div class="jp-eyebrow">Review before saving</div>
            <div class="jp-copy">
                Emotion: <strong>{escape(prediction.emotion.title())}</strong><br>
                Confidence: <strong>{format_percent(prediction.confidence)}</strong><br>
                Context: <strong>{escape(pending["activity"] or "Not set")}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    feedback = st.selectbox(
        "Reflection usefulness",
        options=["helpful", "not_helpful", "unsure"],
        format_func=lambda value: FEEDBACK_LABELS[value],
    )
    if st.button("Save reflection", use_container_width=True):
        saved = save_entry_for_ui(pending, feedback)
        st.success(f"Saved entry #{saved['id']} to the journal history.")
        st.session_state.pop("pending_entry", None)

    st.caption(prediction.disclaimer)


def render_chat_page(use_llm: bool) -> None:
    render_page_hero(
        "JournalPulse Chat",
        "Start with the conversation. Emotion signals, resources, and analytics stay available when they are useful.",
    )

    pending = st.session_state.get("pending_entry")
    if not pending:
        render_entry_composer(use_llm)
        return

    prediction = pending["prediction"]
    action_left, action_right = st.columns([0.78, 0.22])
    with action_left:
        st.markdown(
            """
            <div class="jp-workspace-toolbar">
                <p>Keep chatting, ask for resources, or save the reflection when the conversation feels useful.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with action_right:
        if st.button("New chat", use_container_width=True):
            st.session_state.pop("pending_entry", None)
            st.rerun()

    if prediction.support_message:
        st.warning(prediction.support_message)

    chat_column, context_column = st.columns([1.25, 0.82], gap="large")
    with chat_column:
        render_coach_panel(pending)

    with context_column:
        st.markdown('<div class="jp-context-label">Conversation context</div>', unsafe_allow_html=True)
        render_prediction_summary(prediction)
        render_save_panel(pending)

        with st.expander("Matched resources", expanded=True):
            render_resource_recommendations(pending, show_actions=False, show_video_preview=False)

        with st.expander("Why this response", expanded=False):
            render_reflection_details(pending)


def summarize_entries_for_table(entries: list) -> pd.DataFrame:
    rows = []
    for entry in entries:
        rows.append(
            {
                "id": entry["id"],
                "created_at": entry["created_at"],
                "emotion": entry["emotion"],
                "confidence": f"{float(entry['confidence']):.0%}",
                "band": entry.get("confidence_band") or "unknown",
                "feedback": FEEDBACK_LABELS.get(entry.get("feedback"), entry.get("feedback") or "Unrated"),
                "summary": entry.get("reflection_summary") or entry.get("recommendation"),
                "resources": ", ".join(resource_titles(entry.get("suggested_resource_ids", []))),
            }
        )
    return pd.DataFrame(rows)


def render_saved_entry(entry: dict) -> None:
    detail = emotion_detail(entry["emotion"])
    resource_names = resource_titles(entry.get("suggested_resource_ids", []))
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="jp-result-top">
                <div>
                    <div class="jp-eyebrow">Saved reflection #{entry['id']}</div>
                    <div class="jp-emotion" style="font-size:1.45rem;">{escape(entry["emotion"].title())}</div>
                </div>
                <div>
                    <span class="jp-pill" style="background:{detail['surface']}; border-color:{detail['color']}33;">
                        {escape((entry.get("confidence_band") or "unknown").title())}
                    </span>
                    <span class="jp-pill">{escape(FEEDBACK_LABELS.get(entry.get("feedback"), "Unrated"))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.markdown("**Reflection summary**")
            st.write(entry.get("reflection_summary") or entry.get("recommendation"))
            st.markdown("**Interpretation**")
            st.write(entry.get("interpretation") or "No interpretation saved.")
            st.markdown("**Explanation phrases**")
            render_phrase_chips(entry.get("explanation_phrases", []))
        with right:
            st.markdown("**Original entry**")
            st.markdown(
                f'<div class="jp-entry-preview">{escape(entry["text"])}</div>',
                unsafe_allow_html=True,
            )

        if entry.get("follow_up_prompts"):
            st.markdown("**Follow-up prompts**")
            for prompt in entry["follow_up_prompts"]:
                st.markdown(f"- {prompt}")

        if resource_names:
            st.markdown("**Suggested resources**")
            st.write(", ".join(resource_names))

        context_bits = [
            f"Location: {entry['location']}" if entry.get("location") else None,
            f"Activity: {entry['activity']}" if entry.get("activity") else None,
            entry.get("coach_state_summary"),
        ]
        st.caption(" | ".join(bit for bit in context_bits if bit))

        coach_summary = entry.get("coach_summary")
        if coach_summary:
            with st.expander("Coach summary", expanded=False):
                st.json(coach_summary)


def render_history_page() -> None:
    render_page_hero(
        "History",
        "Review saved reflections, filter by emotional state, and inspect the model signals attached to each entry.",
    )
    entries = list_entries_for_ui()
    if not entries:
        render_demo_seed_notice("history")
        entries = demo_entries()

    emotion_options = ["All"] + sorted({entry["emotion"] for entry in entries})
    band_options = ["All"] + sorted({entry.get("confidence_band") or "unknown" for entry in entries})
    feedback_options = ["All", "helpful", "not_helpful", "unsure", "unrated"]

    filter_columns = st.columns(3)
    selected_emotion = filter_columns[0].selectbox("Emotion", emotion_options)
    selected_band = filter_columns[1].selectbox("Confidence band", band_options)
    selected_feedback = filter_columns[2].selectbox(
        "Feedback",
        feedback_options,
        format_func=lambda value: "All" if value == "All" else FEEDBACK_LABELS.get(None if value == "unrated" else value, value.title()),
    )

    filtered = []
    for entry in entries:
        feedback_value = entry.get("feedback") or "unrated"
        if selected_emotion != "All" and entry["emotion"] != selected_emotion:
            continue
        if selected_band != "All" and (entry.get("confidence_band") or "unknown") != selected_band:
            continue
        if selected_feedback != "All" and feedback_value != selected_feedback:
            continue
        filtered.append(entry)

    metric_row = st.columns(3)
    metric_row[0].metric("Visible entries", len(filtered))
    metric_row[1].metric("Saved entries", len(entries))
    metric_row[2].metric(
        "Average confidence",
        "n/a" if not filtered else f"{sum(float(entry['confidence']) for entry in filtered) / len(filtered):.0%}",
    )

    if not filtered:
        st.markdown('<div class="jp-empty">No saved entries match those filters.</div>', unsafe_allow_html=True)
        return

    table = summarize_entries_for_table(filtered)
    st.dataframe(table, use_container_width=True, hide_index=True)

    options = {
        f"#{entry['id']} | {entry['created_at']} | {entry['emotion'].title()}": entry
        for entry in filtered
    }
    selected_key = st.selectbox("Inspect saved reflection", list(options.keys()))
    render_saved_entry(options[selected_key])


def render_resource_library_page() -> None:
    render_page_hero(
        "Resources",
        "Browse the curated catalog when you want more options than the chat recommends.",
    )
    summary = resource_summary_for_ui()
    metric_row = st.columns(4)
    metric_row[0].metric("Resources", summary["total_resources"])
    metric_row[1].metric("Crisis-safe", summary["crisis_safe_count"])
    metric_row[2].metric("Coverage gaps", len(summary["coverage_gaps"]))
    metric_row[3].metric("Validation issues", len(summary["validation_errors"]))

    if summary["validation_errors"]:
        st.warning("Catalog validation found issues.")
        for error in summary["validation_errors"]:
            st.caption(error)

    filter_columns = st.columns(3)
    emotion = filter_columns[0].selectbox("Emotion", ["All"] + sorted(EMOTION_DETAILS))
    style = filter_columns[1].selectbox("Coping style", ["All"] + [STYLE_LABELS[item] for item in COPING_STYLES])
    resource_type = filter_columns[2].selectbox("Resource type", ["All"] + [item.title() for item in DEFAULT_RESOURCE_TYPES])

    selected_emotion = None if emotion == "All" else emotion
    selected_style = None if style == "All" else STYLE_FROM_LABEL[style]
    selected_type = None if resource_type == "All" else resource_type.lower()
    resources = resources_for_ui(
        emotion=selected_emotion,
        coping_style=selected_style,
        resource_type=selected_type,
    )

    if not resources:
        st.markdown('<div class="jp-empty">No resources match those filters.</div>', unsafe_allow_html=True)
        return

    st.caption("Showing a compact catalog view. Use filters to narrow the list.")
    render_resource_card_grid(
        resources,
        emotion=selected_emotion,
        entry_id="library",
        limit=8 if selected_style is None and selected_emotion is None and selected_type is None else 12,
        show_actions=False,
    )
def render_confusion_matrix(metrics: dict) -> None:
    matrix = metrics.get("confusion_matrix")
    if not matrix:
        return
    rows = []
    for true_index, values in enumerate(matrix):
        for predicted_index, count in enumerate(values):
            rows.append(
                {
                    "true_label": LABELS.get(true_index, str(true_index)),
                    "predicted_label": LABELS.get(predicted_index, str(predicted_index)),
                    "count": int(count),
                }
            )
    frame = pd.DataFrame(rows)
    chart = (
        alt.Chart(frame)
        .mark_rect()
        .encode(
            x=alt.X("predicted_label:N", sort=EMOTION_ORDER, title="Predicted"),
            y=alt.Y("true_label:N", sort=EMOTION_ORDER, title="Actual"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="teals"), title="Count"),
            tooltip=["true_label", "predicted_label", "count"],
        )
        .properties(height=340)
    )
    labels = (
        alt.Chart(frame)
        .mark_text(fontSize=12)
        .encode(
            x=alt.X("predicted_label:N", sort=EMOTION_ORDER),
            y=alt.Y("true_label:N", sort=EMOTION_ORDER),
            text="count:Q",
            color=alt.condition(alt.datum.count > 250, alt.value("white"), alt.value("#152034")),
        )
    )
    st.subheader("Confusion Matrix")
    st.altair_chart(chart + labels, use_container_width=True)
def render_about_page() -> None:
    render_page_hero(
        "Model",
        "JournalPulse pairs a transformer classifier with a transparent explainer, curated support resources, and a constrained coaching flow.",
    )
    model_card = load_model_card()
    if not model_card:
        st.warning("No trained production model found yet. Run the training script first.")
        return

    metrics = model_card.get("metrics", {})
    summary_columns = st.columns(4)
    summary_columns[0].metric("Selected model", model_card.get("model_name", "n/a"))
    summary_columns[1].metric("Accuracy", format_percent(metrics.get("accuracy")))
    summary_columns[2].metric("Macro F1", format_percent(metrics.get("macro_f1")))
    summary_columns[3].metric("Max tokens", model_card.get("max_length", "n/a"))

    quality_report = load_model_quality_report()
    if quality_report:
        st.subheader("Product-Shaped Eval")
        quality_columns = st.columns(4)
        quality_columns[0].metric("Journal cases", quality_report.get("total_cases", "n/a"))
        quality_columns[1].metric(
            "Accepted accuracy",
            format_percent(quality_report.get("accepted_accuracy")),
        )
        quality_columns[2].metric(
            "Non-crisis accepted",
            format_percent(quality_report.get("non_crisis_accepted_accuracy")),
        )
        quality_columns[3].metric(
            "Top-3 recall",
            format_percent(quality_report.get("top3_primary_recall")),
        )
        with st.expander("Model quality misses", expanded=False):
            misses = quality_report.get("misses", [])
            if misses:
                st.dataframe(
                    pd.DataFrame(misses)[
                        ["id", "expected_primary", "accepted_emotions", "predicted", "confidence", "is_mixed", "top3"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("No misses in the latest product-shaped eval report.")

    render_confusion_matrix(metrics)

    st.subheader("How the Pieces Work Together")
    columns = st.columns(2, gap="large")
    with columns[0]:
        st.markdown(
            """
            <div class="jp-panel">
                <div class="jp-card-title">Production classifier</div>
                <div class="jp-copy">A fine-tuned DistilRoBERTa model scores six emotions and returns confidence bands for the app.</div>
            </div>
            <div class="jp-panel">
                <div class="jp-card-title">Explanation layer</div>
                <div class="jp-copy">A classical TF-IDF linear model stays in the product to expose phrase-level signals.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            """
            <div class="jp-panel">
                <div class="jp-card-title">Resource engine</div>
                <div class="jp-copy">Curated videos, reading, games, movement links, and crisis resources are ranked by emotion and feedback.</div>
            </div>
            <div class="jp-panel">
                <div class="jp-card-title">Coach flow</div>
                <div class="jp-copy">A finite-state coach keeps the interaction bounded, with optional AI wording when configured.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    report_path = PROJECT_ROOT / "artifacts" / "reports" / "evaluation.md"
    if report_path.exists():
        with st.expander("Evaluation summary", expanded=True):
            st.markdown(report_path.read_text())

    with st.expander("Production artifact metadata"):
        st.json(model_card)


if not backend_enabled():
    initialize_database(DEFAULT_DB_PATH)
render_global_styles()
page = render_sidebar()

if page == "Chat":
    render_chat_page(st.session_state.get("use_llm", False))
elif page == "Resources":
    render_resource_library_page()
elif page == "History":
    render_history_page()
elif page == "Model":
    render_about_page()
