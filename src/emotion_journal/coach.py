from typing import Dict, List, Optional

from .config import COACH_SUGGESTED_REPLY_LIMIT, CRISIS_COACH_OPENING
from .llm import (
    configured_llm_mode,
    llm_adapter_available,
    maybe_generate_structured_coach_response,
    maybe_rewrite_coach_message,
)
from .preprocessing import normalize_text
from .resources import recommend_resources


def coach_available() -> bool:
    return True


def llm_mode_available() -> bool:
    return llm_adapter_available()


def _trim_replies(replies: List[str]) -> List[str]:
    seen = []
    for reply in replies:
        if reply and reply not in seen:
            seen.append(reply)
    return seen[:COACH_SUGGESTED_REPLY_LIMIT]


BASE_TIPS = {
    "ground": [
        "Put both feet on the floor and name five things you can see before deciding what to do next.",
        "Slow the exhale for three breaths; aim for a longer out-breath than in-breath.",
        "Pick one concrete fact from the entry and separate it from the story your mind is adding.",
    ],
    "plan": [
        "Write the next action as something doable in ten minutes or less.",
        "Choose one person, place, or tool that would make the next step easier.",
        "Decide what can wait until tomorrow so tonight has a clear boundary.",
    ],
    "reframe": [
        "Name the harshest thought, then rewrite it as something accurate but less punishing.",
        "Ask what you would say to a friend who wrote the same entry.",
        "Look for one alternative explanation that does not require blaming yourself.",
    ],
    "connect": [
        "Send one low-pressure message to someone safe: no big explanation required.",
        "If talking feels like too much, ask for company around a simple activity.",
        "Name what kind of support you want before reaching out: listening, advice, distraction, or help with a task.",
    ],
}

EMOTION_TIPS = {
    "sadness": [
        "Lower the bar: food, water, light, or a short walk counts as progress.",
        "Name the loss or disappointment in one sentence so it stops taking up the whole room.",
        "Do one small care action before analyzing the feeling further.",
    ],
    "joy": [
        "Capture what made this feel good while the details are still available.",
        "Turn the lift into one small action that future-you will appreciate.",
        "Share the good moment with someone who can help it feel more real.",
    ],
    "love": [
        "Translate warmth into a concrete gesture: a message, thanks, time, or repair.",
        "Notice what made closeness feel possible and protect that condition.",
        "Let the feeling guide one honest sentence you have been avoiding.",
    ],
    "anger": [
        "Delay the response until you can name the boundary, not just the heat.",
        "Move the energy physically before choosing words.",
        "Write the clean version of the ask: what changed behavior would actually help?",
    ],
    "fear": [
        "Split the fear into facts, predictions, and next protective action.",
        "Choose the smallest step that makes the situation safer or clearer.",
        "Borrow calm from the environment: light, water, breath, and one grounded object.",
    ],
    "surprise": [
        "Give the new information a little time before deciding what it means.",
        "Ask whether this is pleasant surprise, threat surprise, or disorientation.",
        "Write down what changed and what has not changed.",
    ],
}


def _intent_from_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "none"
    if any(token in normalized for token in ("done", "enough", "stop", "thats enough")):
        return "done"
    if any(token in normalized for token in ("not this", "different", "another option")):
        return "not_this"
    if any(token in normalized for token in ("plan", "next step", "action", "decide", "decision")):
        return "plan"
    if any(token in normalized for token in ("ground", "grounding", "calm down", "breathe", "breath", "settle")):
        return "ground"
    if any(token in normalized for token in ("reframe", "perspective", "think about", "spiral", "overthinking")):
        return "reframe"
    if any(token in normalized for token in ("connect", "talk to someone", "friend", "family", "lonely", "alone")):
        return "connect"
    if any(token in normalized for token in ("tip", "tips", "advice", "what should i do", "help me")):
        return "tips"
    if any(token in normalized for token in ("watch", "video", "youtube")):
        return "watch"
    if any(token in normalized for token in ("read", "article", "website")):
        return "read"
    if any(token in normalized for token in ("play", "game")):
        return "play"
    if any(token in normalized for token in ("move", "walk", "exercise", "stretch")):
        return "move"
    if any(token in normalized for token in ("show something", "show me something", "resource", "helpful")):
        return "resource"
    if any(token in normalized for token in ("sad", "sadness")):
        return "sadness"
    if any(token in normalized for token in ("angry", "anger", "frustrated", "mad")):
        return "anger"
    if any(token in normalized for token in ("fear", "scared", "anxious", "afraid", "nervous")):
        return "fear"
    if any(token in normalized for token in ("unpack", "reflect", "talk it through", "process")):
        return "unpack"
    return "reflect"


def _suggested_replies(step: str) -> List[str]:
    if step == "clarify":
        return _trim_replies(["Give me tips", "Ground me", "Show resources", "Help me plan"])
    if step == "resource_follow_up":
        return _trim_replies(["Watch", "Read", "Move", "Give me tips"])
    if step == "tips":
        return _trim_replies(["Ground me", "Help me plan", "Show resources", "Done"])
    if step == "close":
        return _trim_replies(["Done"])
    return _trim_replies(["Give me tips", "Ground me", "Show resources", "Help me plan"])


def _tips_for_intent(intent: str, emotion: str) -> List[str]:
    if intent in BASE_TIPS:
        return BASE_TIPS[intent]
    if intent in {"tips", "reflect", "unpack", "not_this"}:
        return EMOTION_TIPS.get(emotion, BASE_TIPS["ground"])
    return []


def _style_for_intent(intent: str) -> Optional[str]:
    if intent in {"watch", "read", "play", "move"}:
        return intent
    if intent == "ground":
        return "move"
    if intent in {"plan", "reframe"}:
        return "read"
    if intent == "connect":
        return "read"
    return None


def _resource_intent_for(intent: str) -> Optional[str]:
    if intent in {"ground", "plan", "reframe", "connect", "watch", "read", "play", "move", "resource", "tips"}:
        return intent
    if intent == "not_this":
        return "resource"
    return None


def _resource_rationales(resources: List[dict], resource_ids: Optional[List[str]] = None) -> Dict[str, str]:
    allowed_ids = set(resource_ids or [resource["id"] for resource in resources])
    return {
        resource["id"]: resource["rationale"]
        for resource in resources
        if resource["id"] in allowed_ids and resource.get("rationale")
    }


def _apply_optional_llm(
    payload: dict,
    *,
    allowed_resources: List[dict],
    context: dict,
    use_llm: bool,
    is_crisis: bool,
) -> dict:
    if is_crisis:
        if use_llm:
            payload["fallback_reason"] = "crisis_mode_llm_bypassed"
            payload["agent_fallback_reason"] = "crisis_mode_llm_bypassed"
            payload["agent_mode"] = "fallback"
        return payload

    mode = configured_llm_mode()
    if not use_llm or mode == "off":
        return payload

    if mode == "structured":
        structured_payload, used_llm, fallback_reason = maybe_generate_structured_coach_response(
            payload,
            allowed_resources=allowed_resources,
            context=context,
            use_llm=True,
        )
        if structured_payload is None:
            payload["fallback_reason"] = fallback_reason
            payload["agent_fallback_reason"] = fallback_reason
            payload["agent_mode"] = "fallback"
            return payload

        allowed_ids = {resource["id"] for resource in allowed_resources}
        resource_ids = [
            resource_id
            for resource_id in structured_payload.get("resource_ids", [])
            if resource_id in allowed_ids
        ] or payload.get("resource_ids", [])

        payload.update(
            {
                "assistant_message": structured_payload["assistant_message"],
                "tips": structured_payload.get("tips") or payload.get("tips", []),
                "practical_steps": (
                    structured_payload.get("practical_steps")
                    or structured_payload.get("tips")
                    or payload.get("practical_steps", [])
                    or payload.get("tips", [])
                ),
                "suggested_replies": structured_payload.get("suggested_replies") or payload.get("suggested_replies", []),
                "resource_intent": (
                    structured_payload.get("resource_intent")
                    if structured_payload.get("resource_intent") != "none"
                    else payload.get("resource_intent")
                ),
                "resource_ids": resource_ids,
                "used_llm": used_llm,
                "coach_mode": "structured",
                "agent_mode": "structured",
                "agent_model": structured_payload.get("agent_model"),
                "agent_fallback_reason": None,
                "reflection_question": structured_payload.get("reflection_question"),
                "communication_draft": structured_payload.get("communication_draft"),
                "confidence_note": structured_payload.get("confidence_note"),
                "fallback_reason": structured_payload.get("refusal_reason"),
            }
        )
        payload["resource_rationales"] = _resource_rationales(allowed_resources, payload["resource_ids"])
        return payload

    message, used_llm = maybe_rewrite_coach_message(
        payload["assistant_message"],
        payload["suggested_replies"],
        context=context,
        use_llm=True,
    )
    payload["assistant_message"] = message
    payload["used_llm"] = used_llm
    payload["coach_mode"] = "rewrite" if used_llm else "deterministic"
    payload["agent_mode"] = "deterministic"
    if not used_llm:
        payload["fallback_reason"] = "rewrite_llm_unavailable"
        payload["agent_fallback_reason"] = "rewrite_llm_unavailable"
    return payload


def _message_with_tips(
    *,
    framing_emotion: str,
    intent: str,
    tips: List[str],
    has_resources: bool,
) -> str:
    if intent == "ground":
        opener = f"Let's make this {framing_emotion}-leaning moment smaller and more physical."
    elif intent == "plan":
        opener = f"Let's turn the {framing_emotion}-leaning signal into a next step instead of a loop."
    elif intent == "reframe":
        opener = f"Let's loosen the interpretation around this {framing_emotion}-leaning moment."
    elif intent == "connect":
        opener = f"This may go better with a little human support around the {framing_emotion}-leaning part."
    else:
        opener = f"Here are a few practical things to try for this {framing_emotion}-leaning moment."

    tip_lines = " ".join(f"{index}. {tip}" for index, tip in enumerate(tips, start=1))
    resource_note = " I also pulled matching resources below." if has_resources else ""
    return f"{opener} {tip_lines}{resource_note}"


def _draft_message(
    *,
    step: str,
    framing_emotion: str,
    confidence_band: Optional[str],
    selected_coping_style: Optional[str],
    is_crisis: bool,
    user_intent: str = "opening",
) -> str:
    if is_crisis:
        return CRISIS_COACH_OPENING

    if step == "clarify":
        return (
            f"The signal looks a little mixed, even though `{framing_emotion}` is leading. "
            "Does this feel closer to sadness, anger, fear, or would you rather skip straight to something helpful?"
        )

    if step == "resource_follow_up":
        if selected_coping_style:
            return (
                f"I’ll lean into {selected_coping_style}-style support for this {framing_emotion}-leaning moment. "
                "I pulled matching resources below, and you can ask for tips if you want a smaller next step first."
            )
        return (
            f"For a {framing_emotion}-leaning moment, I can point you toward something to watch, read, play, or do. "
            "Pick the kind of help that feels easiest right now."
        )

    if step == "close":
        return "That is enough for now. If you want another pass later, we can reopen from the feeling or from the resource side."

    if user_intent == "unpack":
        return (
            f"Stay with the {framing_emotion}-leaning part for one more beat. "
            "What feels most unresolved: what happened, what it meant, or what you want to do next?"
        )

    if confidence_band == "low":
        return (
            f"This looks emotionally mixed, but `{framing_emotion}` is the strongest signal. "
            "We can either unpack it a bit or jump straight to something grounding."
        )

    return (
        f"I’m reading this as mostly {framing_emotion} right now. "
        "I can help in three useful ways: give practical tips, ground the feeling, or pull resources that fit the moment."
    )


def build_initial_coach_turn(
    *,
    entry_text: str,
    emotion: str,
    confidence_band: Optional[str],
    is_crisis: bool,
    use_llm: bool = False,
    allowed_resources: Optional[List[dict]] = None,
    nlp_context: Optional[dict] = None,
) -> dict:
    allowed_resources = allowed_resources or []
    step = "close" if is_crisis else ("clarify" if confidence_band == "low" else "opening")
    suggested_replies = _suggested_replies(step)
    draft = _draft_message(
        step=step,
        framing_emotion=emotion,
        confidence_band=confidence_band,
        selected_coping_style=None,
        is_crisis=is_crisis,
    )
    payload = {
        "assistant_message": draft,
        "coach_state": {
            "step": step,
            "framing_emotion": emotion,
            "selected_coping_style": None,
            "turns": 0,
            "is_crisis": is_crisis,
        },
        "suggested_replies": suggested_replies,
        "resource_ids": [],
        "tips": [],
        "practical_steps": [],
        "reflection_question": None,
        "communication_draft": None,
        "confidence_note": None,
        "used_llm": False,
        "coach_mode": "deterministic",
        "agent_mode": "deterministic",
        "agent_model": None,
        "agent_fallback_reason": None,
        "resource_intent": None,
        "resource_rationales": {},
        "fallback_reason": None,
    }
    return _apply_optional_llm(
        payload,
        allowed_resources=allowed_resources,
        context={
            "entry_text": entry_text,
            "emotion": emotion,
            "confidence_band": confidence_band,
            "step": step,
            "nlp": nlp_context or {},
        },
        use_llm=use_llm,
        is_crisis=is_crisis,
    )


def respond_with_coach(
    *,
    entry_text: str,
    emotion: str,
    confidence_band: Optional[str],
    coach_state: Optional[Dict[str, object]],
    user_message: str,
    is_crisis: bool,
    use_llm: bool = False,
    db_path=None,
) -> dict:
    current_state = dict(coach_state or {})
    framing_emotion = str(current_state.get("framing_emotion") or emotion)
    intent = _intent_from_text(user_message)
    resource_intent = _resource_intent_for(intent)

    tips: List[str] = []
    resources: List[dict] = []

    if is_crisis:
        step = "close"
        selected_coping_style = None
        resources = recommend_resources(emotion, goal="resource", is_crisis=True, db_path=db_path)
        resource_ids = [resource["id"] for resource in resources]
        resource_intent = "resource"
        tips = [
            "Move toward immediate human support instead of handling this alone.",
            "If there is immediate danger, call emergency services now.",
            "If you can, stay near another person or contact someone trusted while you get help.",
        ]
    else:
        selected_coping_style = current_state.get("selected_coping_style")
        step = "opening"
        resource_ids: List[str] = []

        selected_style = _style_for_intent(intent)
        if selected_style:
            selected_coping_style = selected_style
            step = "resource_follow_up"
            resources = recommend_resources(
                framing_emotion,
                coping_style=selected_coping_style,
                goal=intent,
                db_path=db_path,
            )
            resource_ids = [resource["id"] for resource in resources]
            tips = _tips_for_intent(intent, framing_emotion)
        elif intent == "resource":
            step = "resource_follow_up"
            resources = recommend_resources(framing_emotion, goal=intent, db_path=db_path)
            resource_ids = [resource["id"] for resource in resources]
        elif intent in {"sadness", "anger", "fear"} and confidence_band == "low":
            framing_emotion = intent
            step = "opening"
        elif intent == "done":
            step = "close"
        elif intent in {"not_this", "reflect", "unpack", "tips"}:
            step = "tips"
            tips = _tips_for_intent(intent, framing_emotion)

        if use_llm and step != "close" and not resources:
            resources = recommend_resources(
                framing_emotion,
                goal=resource_intent or intent,
                db_path=db_path,
            )

    suggested_replies = _suggested_replies(step)
    if tips and not is_crisis:
        draft = _message_with_tips(
            framing_emotion=framing_emotion,
            intent=intent,
            tips=tips,
            has_resources=bool(resource_ids),
        )
    else:
        draft = _draft_message(
            step=step,
            framing_emotion=framing_emotion,
            confidence_band=confidence_band,
            selected_coping_style=selected_coping_style,
            is_crisis=is_crisis,
            user_intent=intent,
        )
    payload = {
        "assistant_message": draft,
        "coach_state": {
            "step": step,
            "framing_emotion": framing_emotion,
            "selected_coping_style": selected_coping_style,
            "turns": int(current_state.get("turns", 0)) + 1,
            "is_crisis": is_crisis,
            "last_intent": intent,
        },
        "suggested_replies": suggested_replies,
        "resource_ids": resource_ids,
        "tips": tips,
        "practical_steps": tips,
        "reflection_question": None,
        "communication_draft": None,
        "confidence_note": None,
        "used_llm": False,
        "coach_mode": "deterministic",
        "agent_mode": "deterministic",
        "agent_model": None,
        "agent_fallback_reason": None,
        "resource_intent": resource_intent,
        "resource_rationales": _resource_rationales(resources, resource_ids),
        "fallback_reason": None,
    }
    return _apply_optional_llm(
        payload,
        allowed_resources=resources,
        context={
            "entry_text": entry_text,
            "emotion": emotion,
            "framing_emotion": framing_emotion,
            "confidence_band": confidence_band,
            "step": step,
            "selected_coping_style": selected_coping_style,
            "user_message": user_message,
            "tips": tips,
            "resource_intent": resource_intent,
        },
        use_llm=use_llm,
        is_crisis=is_crisis,
    )
