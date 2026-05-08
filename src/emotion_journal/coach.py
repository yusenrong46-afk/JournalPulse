from typing import Dict, List, Optional

from .config import COACH_SUGGESTED_REPLY_LIMIT, CRISIS_COACH_OPENING
from .llm import llm_adapter_available, maybe_rewrite_coach_message
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


def _intent_from_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "none"
    if any(token in normalized for token in ("done", "enough", "stop", "thats enough")):
        return "done"
    if any(token in normalized for token in ("not this", "different", "another option")):
        return "not_this"
    if any(token in normalized for token in ("watch", "video", "youtube")):
        return "watch"
    if any(token in normalized for token in ("read", "article", "website")):
        return "read"
    if any(token in normalized for token in ("play", "game")):
        return "play"
    if any(token in normalized for token in ("move", "walk", "exercise", "breath", "breathe", "calm")):
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
        return _trim_replies(["Sadness fits", "More angry", "More anxious", "Show resources"])
    if step == "resource_follow_up":
        return _trim_replies(["Watch", "Read", "Play", "Move"])
    if step == "close":
        return _trim_replies(["Done"])
    return _trim_replies(["Unpack it", "Calm down", "Show something helpful", "Done"])


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
                f"I’ll lean into `{selected_coping_style}`-style support for this {framing_emotion}-leaning moment. "
                "If that’s off, switch styles and I’ll pivot."
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
        f"I’m reading this as mostly `{framing_emotion}` right now. "
        "Do you want to unpack it, settle your system, or see something that might help immediately?"
    )


def build_initial_coach_turn(
    *,
    entry_text: str,
    emotion: str,
    confidence_band: Optional[str],
    is_crisis: bool,
    use_llm: bool = False,
) -> dict:
    step = "close" if is_crisis else ("clarify" if confidence_band == "low" else "opening")
    suggested_replies = _suggested_replies(step)
    draft = _draft_message(
        step=step,
        framing_emotion=emotion,
        confidence_band=confidence_band,
        selected_coping_style=None,
        is_crisis=is_crisis,
    )
    message, used_llm = maybe_rewrite_coach_message(
        draft,
        suggested_replies,
        context={
            "entry_text": entry_text,
            "emotion": emotion,
            "confidence_band": confidence_band,
            "step": step,
        },
        use_llm=use_llm and not is_crisis,
    )
    return {
        "assistant_message": message,
        "coach_state": {
            "step": step,
            "framing_emotion": emotion,
            "selected_coping_style": None,
            "turns": 0,
            "is_crisis": is_crisis,
        },
        "suggested_replies": suggested_replies,
        "resource_ids": [],
        "used_llm": used_llm,
    }


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

    if is_crisis:
        step = "close"
        selected_coping_style = None
        resource_ids = [resource["id"] for resource in recommend_resources(emotion, is_crisis=True, db_path=db_path)]
    else:
        selected_coping_style = current_state.get("selected_coping_style")
        step = "opening"
        resource_ids: List[str] = []

        if intent in {"watch", "read", "play", "move"}:
            selected_coping_style = intent
            step = "resource_follow_up"
            resources = recommend_resources(
                framing_emotion,
                coping_style=selected_coping_style,
                db_path=db_path,
            )
            resource_ids = [resource["id"] for resource in resources]
        elif intent == "resource":
            step = "resource_follow_up"
            resources = recommend_resources(framing_emotion, db_path=db_path)
            resource_ids = [resource["id"] for resource in resources]
        elif intent in {"sadness", "anger", "fear"} and confidence_band == "low":
            framing_emotion = intent
            step = "opening"
        elif intent == "done":
            step = "close"
        elif intent in {"not_this", "reflect", "unpack"}:
            step = "opening"

    suggested_replies = _suggested_replies(step)
    draft = _draft_message(
        step=step,
        framing_emotion=framing_emotion,
        confidence_band=confidence_band,
        selected_coping_style=selected_coping_style,
        is_crisis=is_crisis,
        user_intent=intent,
    )
    message, used_llm = maybe_rewrite_coach_message(
        draft,
        suggested_replies,
        context={
            "entry_text": entry_text,
            "emotion": emotion,
            "framing_emotion": framing_emotion,
            "confidence_band": confidence_band,
            "step": step,
            "selected_coping_style": selected_coping_style,
            "user_message": user_message,
        },
        use_llm=use_llm and not is_crisis,
    )

    return {
        "assistant_message": message,
        "coach_state": {
            "step": step,
            "framing_emotion": framing_emotion,
            "selected_coping_style": selected_coping_style,
            "turns": int(current_state.get("turns", 0)) + 1,
            "is_crisis": is_crisis,
        },
        "suggested_replies": suggested_replies,
        "resource_ids": resource_ids,
        "used_llm": used_llm,
    }
