import html
import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_journal.coach import llm_mode_available, respond_with_coach
from emotion_journal.config import (
    ADMIN_MODE_ENV,
    COPING_STYLES,
    DEFAULT_DB_PATH,
    DEFAULT_RESOURCE_TYPES,
    LABELS,
    MODELS_DIR,
)
from emotion_journal.db import (
    get_analytics,
    initialize_database,
    insert_entry,
    list_entries,
    record_resource_interaction,
)
from emotion_journal.experience import build_prediction_experience
from emotion_journal.model import get_default_predictor
from emotion_journal.resources import (
    build_resource_draft,
    filter_resources,
    get_resource_lookup,
    recommend_resources,
    resource_admin_snapshot,
    resource_catalog_summary,
    resource_titles,
    resources_by_style,
    validate_resource_catalog,
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

FEEDBACK_LABELS = {
    "helpful": "Helpful",
    "not_helpful": "Not helpful",
    "unsure": "Unsure",
    None: "Unrated",
}

EMOTION_ORDER = [LABELS[index] for index in sorted(LABELS)]
EMOTION_COLORS = [EMOTION_DETAILS[emotion]["color"] for emotion in EMOTION_ORDER]
RESOURCE_ACTION_ORDER = ["opened", "helpful", "dismissed"]


def admin_mode_enabled() -> bool:
    return os.getenv(ADMIN_MODE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


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
                --jp-ink: #152034;
                --jp-muted: #64748b;
                --jp-border: #d9e2ec;
                --jp-panel: #ffffff;
                --jp-soft: #f5f7fb;
                --jp-accent: #0f766e;
            }

            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 3rem;
                max-width: 1220px;
            }

            div[data-testid="stSidebar"] {
                background: #f8fafc;
                border-right: 1px solid var(--jp-border);
            }

            h1, h2, h3 {
                color: var(--jp-ink);
                letter-spacing: 0;
            }

            .jp-hero {
                background: linear-gradient(135deg, #f8fafc 0%, #eef7f4 100%);
                border: 1px solid var(--jp-border);
                border-radius: 8px;
                padding: 1.35rem 1.45rem;
                margin-bottom: 1.1rem;
            }

            .jp-hero h1 {
                margin: 0 0 0.35rem 0;
                font-size: 2.15rem;
                line-height: 1.1;
            }

            .jp-hero p {
                margin: 0;
                color: #46566c;
                max-width: 780px;
                line-height: 1.55;
            }

            .jp-panel {
                background: var(--jp-panel);
                border: 1px solid var(--jp-border);
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 26px rgba(21, 32, 52, 0.04);
            }

            .jp-result {
                border-radius: 8px;
                border: 1px solid var(--jp-border);
                padding: 1.1rem;
                margin: 1rem 0;
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
                grid-template-columns: repeat(3, minmax(150px, 1fr));
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
                font-size: 1.25rem;
                margin-top: 0.2rem;
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

            .jp-resource-meta {
                color: var(--jp-muted);
                font-size: 0.85rem;
                margin-bottom: 0.55rem;
            }

            .jp-empty {
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                border-radius: 8px;
                padding: 1rem;
                color: #475569;
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


def render_sidebar() -> str:
    st.sidebar.title("JournalPulse")
    st.sidebar.caption("Reflective journaling support with emotion classification, curated resources, and saved trends.")
    pages = ["New Entry", "Resource Library", "History", "Insights", "About the Model"]
    if admin_mode_enabled():
        pages.append("Resource Admin")
    page = st.sidebar.radio(
        "Navigate",
        pages,
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Support boundary**")
    st.sidebar.caption("This is not therapy or diagnosis. If you feel unsafe in the U.S., call or text 988.")
    use_llm = False
    if llm_mode_available():
        use_llm = st.sidebar.toggle("AI-polished coach wording", value=False)
    else:
        st.sidebar.caption("Deterministic coach wording is active.")
    st.session_state["use_llm"] = use_llm
    return page


def render_page_hero(title: str, body: str) -> None:
    st.markdown(
        f"""
        <section class="jp-hero">
            <h1>{escape(title)}</h1>
            <p>{escape(body)}</p>
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


def render_prediction_summary(prediction) -> None:
    detail = emotion_detail(prediction.emotion)
    band = prediction.confidence_band or "unknown"
    support_label = "Safety mode" if prediction.is_crisis else "Reflection mode"
    st.markdown(
        f"""
        <section class="jp-result" style="border-top: 5px solid {detail['color']};">
            <div class="jp-result-top">
                <div>
                    <div class="jp-eyebrow">Detected state</div>
                    <div class="jp-emotion">{escape(prediction.emotion.title())}</div>
                </div>
                <div>
                    <span class="jp-pill" style="background:{detail['surface']}; border-color:{detail['color']}33;">
                        {escape(support_label)}
                    </span>
                    <span class="jp-pill">{escape(band.title())} confidence</span>
                </div>
            </div>
            <div class="jp-stat-grid">
                <div class="jp-stat"><span>Confidence</span><strong>{format_percent(prediction.confidence)}</strong></div>
                <div class="jp-stat"><span>Model</span><strong>{escape(prediction.model_name or "Unknown")}</strong></div>
                <div class="jp-stat"><span>Focus</span><strong>{escape(detail["focus"])}</strong></div>
            </div>
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


def render_resource_card(resource: dict, *, emotion: str, entry_id=None):
    duration = resource.get("duration_minutes")
    duration_label = f"{duration} min" if duration else "Flexible"
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="jp-card-title">{escape(resource['title'])}</div>
            <div class="jp-resource-meta">
                {escape(resource['provider'])} | {escape(resource['resource_type'].title())} | {escape(duration_label)}
            </div>
            <div>
                <span class="jp-pill">{escape(STYLE_LABELS.get(resource['coping_style'], resource['coping_style'].title()))}</span>
                <span class="jp-pill">{escape(resource['embed_kind'].title())}</span>
            </div>
            <p class="jp-copy">{escape(resource["summary"])}</p>
            """,
            unsafe_allow_html=True,
        )

        if resource.get("embed_kind") == "youtube":
            st.video(resource["url"])

        render_link_button("Open resource", resource["url"], key=f"open-link-{resource['id']}")

        left, middle, right = st.columns(3)
        if left.button("Opened", key=f"opened-{entry_id}-{resource['id']}"):
            record_resource_interaction(
                resource_id=resource["id"],
                action="opened",
                emotion=emotion,
                entry_id=entry_id,
                db_path=DEFAULT_DB_PATH,
            )
            st.success("Marked as opened.")
        if middle.button("Helpful", key=f"helpful-{entry_id}-{resource['id']}"):
            record_resource_interaction(
                resource_id=resource["id"],
                action="helpful",
                emotion=emotion,
                entry_id=entry_id,
                db_path=DEFAULT_DB_PATH,
            )
            st.success("Marked as helpful.")
        if right.button("Dismiss", key=f"dismissed-{entry_id}-{resource['id']}"):
            record_resource_interaction(
                resource_id=resource["id"],
                action="dismissed",
                emotion=emotion,
                entry_id=entry_id,
                db_path=DEFAULT_DB_PATH,
            )
            st.info("Dismissed for future ranking.")


def selected_resource_style(label: str):
    return None if label == "Blend" else STYLE_FROM_LABEL[label]


def current_resource_set(pending: dict, selected_label: str) -> list:
    prediction = pending["prediction"]
    style = selected_resource_style(selected_label)
    return recommend_resources(
        prediction.emotion,
        coping_style=style,
        db_path=DEFAULT_DB_PATH,
        is_crisis=prediction.is_crisis,
    )


def rerank_pending_resources(pending):
    style = selected_resource_style(pending.get("resource_style_choice", "Blend"))
    pending["resources"] = recommend_resources(
        pending["coach_state"].get("framing_emotion", pending["prediction"].emotion),
        coping_style=style,
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
    pending["coach_used_llm"] = bool(pending.get("coach_used_llm") or response.get("used_llm"))
    if response["resource_ids"]:
        lookup = get_resource_lookup()
        pending["resources"] = [
            lookup[resource_id]
            for resource_id in response["resource_ids"]
            if resource_id in lookup
        ]


def render_new_entry_page(use_llm: bool) -> None:
    render_page_hero(
        "JournalPulse",
        "Write one honest entry, then review the emotional signal, reflection prompts, support cards, and coach turn before saving it.",
    )

    form_column, guide_column = st.columns([1.4, 0.9], gap="large")
    with form_column:
        with st.form("entry-form"):
            text = st.text_area(
                "Journal entry",
                height=240,
                placeholder="Example: I left the meeting frustrated because I felt dismissed, and I still do not know whether to confront it or let it go.",
            )
            left, right = st.columns(2)
            location = left.text_input("Location", placeholder="Optional")
            activity = right.text_input("Activity", placeholder="Optional")
            submitted = st.form_submit_button("Reflect on this entry", use_container_width=True)

    with guide_column:
        st.markdown(
            """
            <div class="jp-panel">
                <div class="jp-eyebrow">Reflection frame</div>
                <div class="jp-copy">
                    Specific moments usually produce clearer predictions than abstract summaries. Include what happened,
                    what changed in your body or behavior, and what still feels unresolved.
                </div>
            </div>
            <div class="jp-panel">
                <div class="jp-eyebrow">Safety boundary</div>
                <div class="jp-copy">
                    Acute crisis language switches the product into a support-first response with ordinary resource cards suppressed.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if submitted:
        if not text.strip():
            st.error("Write a few sentences first so the reflection has something real to work with.")
        else:
            with st.spinner("Reading the entry and preparing the reflection..."):
                experience = build_prediction_experience(
                    load_predictor(),
                    text,
                    location=location.strip() or None,
                    activity=activity.strip() or None,
                    db_path=DEFAULT_DB_PATH,
                    use_llm=use_llm,
                )
            st.session_state["pending_entry"] = {
                "text": text,
                "location": location.strip() or None,
                "activity": activity.strip() or None,
                "prediction": experience["prediction"],
                "resources": experience["resources"],
                "coach_state": experience["coach"]["coach_state"],
                "suggested_replies": experience["coach"]["suggested_replies"],
                "coach_transcript": [
                    {"role": "assistant", "content": experience["coach"]["assistant_message"]}
                ],
                "coach_used_llm": experience["coach"].get("used_llm", False),
                "resource_style_choice": "Blend",
                "use_llm": use_llm,
            }

    pending = st.session_state.get("pending_entry")
    if not pending:
        return

    prediction = pending["prediction"]
    render_prediction_summary(prediction)

    if prediction.support_message:
        st.warning(prediction.support_message)

    overview_column, signals_column = st.columns([1.2, 0.8], gap="large")
    with overview_column:
        st.subheader("Reflection")
        st.markdown(
            f"""
            <div class="jp-panel">
                <div class="jp-eyebrow">Headline guidance</div>
                <div class="jp-copy">{escape(prediction.recommendation)}</div>
            </div>
            <div class="jp-panel">
                <div class="jp-eyebrow">Summary</div>
                <div class="jp-copy">{escape(prediction.reflection_summary)}</div>
            </div>
            <div class="jp-panel">
                <div class="jp-eyebrow">Interpretation</div>
                <div class="jp-copy">{escape(prediction.interpretation)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with signals_column:
        st.subheader("Model Signals")
        with st.container(border=True):
            st.caption("Score distribution")
            render_score_bars(prediction.scores, prediction.emotion)
            st.caption("Phrase-level explanation")
            render_phrase_chips(prediction.explanation_phrases)

    if prediction.follow_up_prompts:
        st.subheader("Try These Next")
        render_prompt_cards(prediction.follow_up_prompts)

    st.subheader("Curated Resources")
    if prediction.is_crisis:
        st.caption("Normal entertainment and distraction links are suppressed in safety mode.")
        for resource in pending["resources"]:
            render_resource_card(resource, emotion=prediction.emotion)
    else:
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
            grouped = resources_by_style(pending["resources"])
            tabs = st.tabs([STYLE_LABELS[style] for style in COPING_STYLES])
            for tab, style in zip(tabs, COPING_STYLES):
                with tab:
                    style_resources = grouped.get(style, [])
                    if not style_resources:
                        st.markdown('<div class="jp-empty">No curated cards yet for this style.</div>', unsafe_allow_html=True)
                    for resource in style_resources:
                        render_resource_card(resource, emotion=prediction.emotion)
        else:
            if not pending["resources"]:
                st.markdown('<div class="jp-empty">No curated cards match this preference yet.</div>', unsafe_allow_html=True)
            for resource in pending["resources"]:
                render_resource_card(resource, emotion=prediction.emotion)

    st.subheader("Guided Coach")
    coach_column, save_column = st.columns([1.25, 0.75], gap="large")
    with coach_column:
        for message in pending["coach_transcript"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if pending["suggested_replies"]:
            reply_columns = st.columns(len(pending["suggested_replies"]))
            for column, reply in zip(reply_columns, pending["suggested_replies"]):
                if column.button(reply, key=f"reply-{reply}"):
                    apply_coach_turn(pending, reply)
                    rerank_pending_resources(pending)
                    st.rerun()

        with st.form("coach-form"):
            coach_input = st.text_input("Coach message", placeholder="Example: show me something to watch")
            coach_submit = st.form_submit_button("Send", use_container_width=True)
        if coach_submit and coach_input.strip():
            apply_coach_turn(pending, coach_input.strip())
            rerank_pending_resources(pending)
            st.rerun()

    with save_column:
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
            saved = insert_entry(
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
                support_message=prediction.support_message,
                follow_up_prompts=prediction.follow_up_prompts,
                explanation_phrases=prediction.explanation_phrases,
                coach_state_summary=(
                    f"step={pending['coach_state'].get('step')}|"
                    f"emotion={pending['coach_state'].get('framing_emotion')}|"
                    f"style={pending['coach_state'].get('selected_coping_style') or 'none'}"
                ),
                coach_summary=safe_coach_summary_from_pending(pending),
                suggested_resource_ids=[resource["id"] for resource in pending["resources"]],
                db_path=DEFAULT_DB_PATH,
            )
            st.success(f"Saved entry #{saved['id']} to the journal history.")
            st.session_state.pop("pending_entry", None)

        st.caption(prediction.disclaimer)


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
    entries = list_entries(db_path=DEFAULT_DB_PATH)
    if not entries:
        st.markdown('<div class="jp-empty">No entries yet. Save a reflection from the New Entry page.</div>', unsafe_allow_html=True)
        return

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
        "Resource Library",
        "Browse the curated catalog by emotion, coping style, and resource type.",
    )
    summary = resource_catalog_summary()
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
    resources = filter_resources(
        emotion=selected_emotion,
        coping_style=selected_style,
        resource_type=selected_type,
    )

    if not resources:
        st.markdown('<div class="jp-empty">No resources match those filters.</div>', unsafe_allow_html=True)
        return

    grouped = resources_by_style(resources)
    tabs = st.tabs([STYLE_LABELS[style] for style in COPING_STYLES])
    for tab, tab_style in zip(tabs, COPING_STYLES):
        with tab:
            style_resources = grouped.get(tab_style, [])
            if not style_resources:
                st.markdown('<div class="jp-empty">No resources in this style for the current filters.</div>', unsafe_allow_html=True)
            for resource in style_resources:
                interaction_emotion = selected_emotion or resource.get("emotion_tags", ["joy"])[0]
                render_resource_card(resource, emotion=interaction_emotion, entry_id="library")


def render_resource_admin_page() -> None:
    render_page_hero(
        "Resource Admin",
        "Validate catalog coverage, inspect resource rows, and draft additions without changing the checked-in catalog.",
    )
    snapshot = resource_admin_snapshot()
    summary = snapshot["summary"]

    metric_row = st.columns(4)
    metric_row[0].metric("Resources", summary["total_resources"])
    metric_row[1].metric("Coverage gaps", len(snapshot["coverage_gaps"]))
    metric_row[2].metric("Validation issues", len(snapshot["validation_errors"]))
    metric_row[3].metric("Crisis-safe", summary["crisis_safe_count"])

    if snapshot["validation_errors"]:
        st.warning("Catalog validation found issues.")
        st.dataframe(pd.DataFrame({"issue": snapshot["validation_errors"]}), use_container_width=True, hide_index=True)
    else:
        st.success("Catalog validation is clean.")

    if snapshot["coverage_gaps"]:
        st.subheader("Coverage Gaps")
        st.dataframe(pd.DataFrame(snapshot["coverage_gaps"]), use_container_width=True, hide_index=True)

    with st.expander("Catalog rows", expanded=False):
        st.dataframe(pd.DataFrame(snapshot["resources"]), use_container_width=True, hide_index=True)

    st.subheader("Draft Resource")
    with st.form("resource-draft-form"):
        title = st.text_input("Title")
        url = st.text_input("URL", placeholder="https://")
        provider = st.text_input("Provider")
        summary_text = st.text_area("Summary", height=100)
        first_row = st.columns(3)
        resource_type = first_row[0].selectbox("Resource type", list(DEFAULT_RESOURCE_TYPES))
        coping_style = first_row[1].selectbox("Coping style", list(COPING_STYLES), format_func=lambda value: STYLE_LABELS[value])
        embed_kind = first_row[2].selectbox("Embed kind", ["link", "youtube", "external"])
        second_row = st.columns(3)
        duration = second_row[0].number_input("Duration minutes", min_value=0, value=0, step=1)
        is_browser_safe = second_row[1].checkbox("Browser safe", value=True)
        is_crisis_safe = second_row[2].checkbox("Crisis safe", value=False)
        emotion_tags = st.multiselect("Emotion tags", EMOTION_ORDER)
        tone_tags_raw = st.text_input("Tone tags", placeholder="comma-separated, optional")
        submitted = st.form_submit_button("Preview draft", use_container_width=True)

    if submitted:
        draft = build_resource_draft(
            title=title,
            url=url,
            provider=provider,
            summary=summary_text,
            resource_type=resource_type,
            coping_style=coping_style,
            embed_kind=embed_kind,
            duration_minutes=int(duration) if duration else None,
            emotion_tags=emotion_tags,
            tone_tags=[tag.strip() for tag in tone_tags_raw.split(",")],
            is_browser_safe=is_browser_safe,
            is_crisis_safe=is_crisis_safe,
        )
        proposed_catalog = snapshot["resources"] + [draft]
        proposed_errors = validate_resource_catalog(proposed_catalog)
        left, right = st.columns([0.9, 1.1], gap="large")
        with left:
            st.markdown("**Draft JSON**")
            st.json(draft)
        with right:
            st.markdown("**Validation after adding draft**")
            if proposed_errors:
                st.warning("The proposed catalog still has validation issues.")
                st.dataframe(pd.DataFrame({"issue": proposed_errors}), use_container_width=True, hide_index=True)
            else:
                st.success("The proposed catalog validates cleanly.")
            st.download_button(
                "Download proposed catalog JSON",
                data=json.dumps(proposed_catalog, indent=2) + "\n",
                file_name="catalog.proposed.json",
                mime="application/json",
                use_container_width=True,
            )


def render_chart_or_empty(title: str, data: dict, index_name: str, value_name: str = "count") -> None:
    st.subheader(title)
    if not data:
        st.markdown('<div class="jp-empty">No data for this chart yet.</div>', unsafe_allow_html=True)
        return
    frame = pd.DataFrame([{index_name: key, value_name: value} for key, value in data.items()])
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{index_name}:N", sort="-y", title=None),
            y=alt.Y(f"{value_name}:Q", title=value_name.replace("_", " ").title()),
            color=alt.Color(f"{index_name}:N", legend=None),
            tooltip=[index_name, value_name],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_emotion_trend_chart(trend_buckets: list) -> None:
    if not trend_buckets:
        return
    st.subheader("Trend Over Time")
    trend_frame = pd.DataFrame(trend_buckets)
    chart = (
        alt.Chart(trend_frame)
        .mark_area(opacity=0.78, interpolate="monotone")
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("count:Q", stack="zero", title="Entries"),
            color=alt.Color(
                "emotion:N",
                scale=alt.Scale(domain=EMOTION_ORDER, range=EMOTION_COLORS),
                title="Emotion",
            ),
            tooltip=["date", "emotion", "count"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_confidence_by_emotion(entries: list) -> None:
    if not entries:
        return
    frame = pd.DataFrame(
        [
            {"emotion": entry["emotion"], "confidence": float(entry["confidence"])}
            for entry in entries
        ]
    )
    summary = frame.groupby("emotion", as_index=False)["confidence"].mean()
    chart = (
        alt.Chart(summary)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("emotion:N", sort=EMOTION_ORDER, title=None),
            y=alt.Y("confidence:Q", title="Average confidence", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "emotion:N",
                scale=alt.Scale(domain=EMOTION_ORDER, range=EMOTION_COLORS),
                legend=None,
            ),
            tooltip=["emotion", alt.Tooltip("confidence:Q", format=".0%")],
        )
        .properties(height=260)
    )
    st.subheader("Confidence by Emotion")
    st.altair_chart(chart, use_container_width=True)


def render_resource_action_funnel(action_counts: dict) -> None:
    st.subheader("Resource Action Funnel")
    if not action_counts:
        st.markdown('<div class="jp-empty">No resource interactions yet.</div>', unsafe_allow_html=True)
        return
    frame = pd.DataFrame(
        [
            {"action": action, "count": int(action_counts.get(action, 0))}
            for action in RESOURCE_ACTION_ORDER
        ]
    )
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("action:N", sort=RESOURCE_ACTION_ORDER, title=None),
            y=alt.Y("count:Q", title="Interactions"),
            color=alt.Color("action:N", legend=None),
            tooltip=["action", "count"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


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


def render_insights_page() -> None:
    render_page_hero(
        "Insights",
        "Track emotion mix, confidence patterns, explanation phrases, and which resource styles are actually helping.",
    )
    analytics = get_analytics(db_path=DEFAULT_DB_PATH)
    entries = list_entries(db_path=DEFAULT_DB_PATH)
    total_entries = analytics["total_entries"]

    top_emotion = "n/a"
    if analytics["counts_by_emotion"]:
        top_emotion = max(analytics["counts_by_emotion"].items(), key=lambda item: item[1])[0].title()

    top_row = st.columns(4)
    top_row[0].metric("Total entries", total_entries)
    top_row[1].metric("Feedback usefulness", format_percent(analytics["feedback_usefulness_rate"]))
    top_row[2].metric("Tracked emotions", len(analytics["counts_by_emotion"]))
    top_row[3].metric("Most common", top_emotion)

    if total_entries == 0:
        st.markdown('<div class="jp-empty">Insights appear after you save a reflection.</div>', unsafe_allow_html=True)
        return

    left, right = st.columns(2, gap="large")
    with left:
        render_chart_or_empty("Emotion Distribution", analytics["counts_by_emotion"], "emotion")
    with right:
        render_chart_or_empty("Confidence Bands", analytics["confidence_band_counts"], "confidence_band")

    render_emotion_trend_chart(analytics["trend_buckets"])

    left, right = st.columns(2, gap="large")
    with left:
        render_resource_action_funnel(analytics["resource_action_counts"])
    with right:
        render_confidence_by_emotion(entries)

    render_chart_or_empty("Preferred Coping Styles", analytics["preferred_coping_styles"], "coping_style", "score")

    if analytics["top_helpful_resources"]:
        st.subheader("Helpful Resources")
        st.dataframe(pd.DataFrame(analytics["top_helpful_resources"]), use_container_width=True, hide_index=True)

    if analytics["top_explanation_phrases_by_emotion"]:
        st.subheader("Common Explanation Phrases")
        for emotion, phrase_rows in analytics["top_explanation_phrases_by_emotion"].items():
            with st.expander(emotion.title(), expanded=False):
                render_phrase_chips([f"{row['phrase']} ({row['count']})" for row in phrase_rows])


def render_about_page() -> None:
    render_page_hero(
        "About the Model",
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


initialize_database(DEFAULT_DB_PATH)
render_global_styles()
page = render_sidebar()

if page == "New Entry":
    render_new_entry_page(st.session_state.get("use_llm", False))
elif page == "Resource Library":
    render_resource_library_page()
elif page == "History":
    render_history_page()
elif page == "Insights":
    render_insights_page()
elif page == "About the Model":
    render_about_page()
elif page == "Resource Admin":
    render_resource_admin_page()
