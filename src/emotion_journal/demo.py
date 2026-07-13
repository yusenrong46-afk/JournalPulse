from .analytics import build_analytics


DEMO_ENTRIES = [
    {
        "id": "demo-1",
        "created_at": "2026-05-05 09:18:00",
        "text": "I walked before work and felt lighter than I expected. The week still looks busy, but I want to protect that little bit of momentum.",
        "emotion": "joy",
        "confidence": 0.88,
        "recommendation": "Anchor the bright spot before the day blurs together.",
        "location": "Vancouver",
        "activity": "walking",
        "feedback": "helpful",
        "reflection_summary": "The entry reads like relief paired with genuine lift.",
        "interpretation": "The model is picking up positive language around energy, ease, and a specific restorative moment.",
        "confidence_band": "high",
        "model_name": "distilroberta-base",
        "support_message": None,
        "follow_up_prompts": [
            "What created the lift most clearly?",
            "How do you want to use it today?",
            "What would make this easier to repeat?",
        ],
        "explanation_phrases": ["felt lighter", "protect momentum"],
        "coach_state_summary": "step=resource_follow_up|emotion=joy|style=play",
        "coach_summary": {
            "turn_count": 2,
            "final_step": "resource_follow_up",
            "framing_emotion": "joy",
            "selected_coping_style": "play",
            "resource_ids": ["game_autodraw", "site_gratitude_thanks"],
            "used_llm": False,
            "safety_mode": False,
        },
        "suggested_resource_ids": ["game_autodraw", "site_gratitude_thanks"],
    },
    {
        "id": "demo-2",
        "created_at": "2026-05-06 18:42:00",
        "text": "The meeting made me angry because I felt talked over. I do not want to explode, but I also do not want to pretend it was fine.",
        "emotion": "anger",
        "confidence": 0.76,
        "recommendation": "Separate the crossed boundary from the next deliberate move.",
        "location": "Office",
        "activity": "meeting",
        "feedback": "helpful",
        "reflection_summary": "There is direct friction, but also a wish to respond cleanly.",
        "interpretation": "The model is reacting to language about being dismissed, restraint, and a boundary that still needs attention.",
        "confidence_band": "medium",
        "model_name": "distilroberta-base",
        "support_message": None,
        "follow_up_prompts": [
            "What boundary felt crossed?",
            "What would a clean ask sound like?",
            "What can wait until you are calmer?",
        ],
        "explanation_phrases": ["made me angry", "talked over"],
        "coach_state_summary": "step=tips|emotion=anger|style=read",
        "coach_summary": {
            "turn_count": 1,
            "final_step": "tips",
            "framing_emotion": "anger",
            "selected_coping_style": "read",
            "resource_ids": ["site_mind_manage_anger", "site_nhs_breathing"],
            "used_llm": False,
            "safety_mode": False,
        },
        "suggested_resource_ids": ["site_mind_manage_anger", "site_nhs_breathing"],
    },
    {
        "id": "demo-3",
        "created_at": "2026-05-07 22:10:00",
        "text": "I keep imagining everything going wrong tomorrow. I know some of it is guessing, but my body still feels like it is bracing.",
        "emotion": "fear",
        "confidence": 0.81,
        "recommendation": "Shrink the worry into one next safe step you can control.",
        "location": "Home",
        "activity": "planning",
        "feedback": "unsure",
        "reflection_summary": "The entry sounds future-focused and physically tense.",
        "interpretation": "The model is picking up worry, prediction language, and a body-based description of threat readiness.",
        "confidence_band": "high",
        "model_name": "distilroberta-base",
        "support_message": None,
        "follow_up_prompts": [
            "Which part is fact and which part is prediction?",
            "What is the smallest protective action?",
            "Who or what could make tomorrow easier?",
        ],
        "explanation_phrases": ["going wrong", "body still feels"],
        "coach_state_summary": "step=resource_follow_up|emotion=fear|style=move",
        "coach_summary": {
            "turn_count": 3,
            "final_step": "resource_follow_up",
            "framing_emotion": "fear",
            "selected_coping_style": "move",
            "resource_ids": ["site_nhs_breathing", "site_nhs_mindfulness"],
            "used_llm": False,
            "safety_mode": False,
        },
        "suggested_resource_ids": ["site_nhs_breathing", "site_nhs_mindfulness"],
    },
]

DEMO_RESOURCE_INTERACTIONS = [
    {"resource_id": "game_autodraw", "action": "opened", "emotion": "joy", "entry_id": "demo-1"},
    {"resource_id": "game_autodraw", "action": "helpful", "emotion": "joy", "entry_id": "demo-1"},
    {"resource_id": "site_mind_manage_anger", "action": "opened", "emotion": "anger", "entry_id": "demo-2"},
    {"resource_id": "site_mind_manage_anger", "action": "helpful", "emotion": "anger", "entry_id": "demo-2"},
    {"resource_id": "site_nhs_breathing", "action": "opened", "emotion": "fear", "entry_id": "demo-3"},
]


def demo_entries() -> list:
    return [dict(entry) for entry in DEMO_ENTRIES]


def demo_resource_interactions() -> list:
    return [dict(interaction) for interaction in DEMO_RESOURCE_INTERACTIONS]


def demo_analytics(resource_catalog: list) -> dict:
    return build_analytics(
        demo_entries(),
        resource_interactions=demo_resource_interactions(),
        resource_catalog=resource_catalog,
    )
