from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DEFAULT_DB_PATH = ARTIFACTS_DIR / "journal.db"
RESOURCE_CATALOG_PATH = ASSETS_DIR / "resources" / "catalog.json"

RANDOM_SEED = 42
MAX_TEXT_LENGTH = 5000
MAX_TRANSFORMER_LENGTH = 128
BASELINE_MODEL_NAME = "tfidf-logreg"
LINEAR_SVC_MODEL_NAME = "tfidf-linearsvc"
TRANSFORMER_MODEL_NAME = "distilroberta-base"
SECOND_TRANSFORMER_MODEL_NAME = "bert-base-uncased"
DEFAULT_TRANSFORMER_CANDIDATES = (TRANSFORMER_MODEL_NAME,)
EXPLANATION_PHRASE_LIMIT = 5
RESOURCE_LIMIT_PER_STYLE = 2
COACH_SUGGESTED_REPLY_LIMIT = 4
COPING_STYLES = ("watch", "read", "play", "move")
RESOURCE_ACTIONS = ("opened", "helpful", "dismissed")
DEFAULT_RESOURCE_TYPES = ("video", "website", "game", "support")

LLM_API_KEY_ENV = "JOURNALPULSE_LLM_API_KEY"
LLM_BASE_URL_ENV = "JOURNALPULSE_LLM_BASE_URL"
LLM_MODEL_ENV = "JOURNALPULSE_LLM_MODEL"
LLM_TIMEOUT_SECONDS = 10.0
ADMIN_MODE_ENV = "JOURNALPULSE_ADMIN_MODE"

LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
EMOTION_TO_ID = {label: idx for idx, label in LABELS.items()}

DEFAULT_DISCLAIMER = (
    "This tool offers reflective journaling support and emotion classification. "
    "It is not therapy, diagnosis, or medical advice."
)
CRISIS_DISCLAIMER = (
    "This entry may describe acute distress. The assistant is switching to a "
    "supportive safety response instead of a normal journaling recommendation."
)
CRISIS_SUPPORT_MESSAGE = (
    "If you might act on thoughts of self-harm, call or text 988 right now in "
    "the United States, or call 911 if there is immediate danger. If you can, "
    "reach out to a trusted person nearby while you seek help."
)
CRISIS_RECOMMENDATION = "Pause and reach for immediate human support."
CRISIS_REFLECTION_SUMMARY = (
    "Your entry reads like acute distress, so this app is switching into safety mode."
)
CRISIS_INTERPRETATION = (
    "This is not a moment for ordinary journaling coaching; urgent human support matters more than a model label."
)
CRISIS_COACH_OPENING = (
    "I’m switching out of ordinary reflection mode. Right now the priority is getting you to a human support option, not asking you to do more emotional work alone."
)

CRISIS_KEYWORDS = {
    "suicide",
    "kill myself",
    "end my life",
    "self harm",
    "hurt myself",
    "overdose",
    "want to die",
    "can't go on",
    "hopeless",
    "not safe",
}

EMOTION_RECOMMENDATIONS = {
    "sadness": [
        "Take ten quiet minutes to name what feels heavy and what feels changeable.",
        "Try a low-pressure reset like a short walk, water, or a check-in with someone you trust.",
        "Write one small thing that helped you cope before and repeat that today.",
    ],
    "joy": [
        "Capture what made today feel good so you can recreate it intentionally.",
        "Share the highlight with someone close and let the moment stay social.",
        "Turn the good energy into momentum on one small goal you care about.",
    ],
    "love": [
        "Write a note about who or what made you feel connected today.",
        "Translate the feeling into action with a message, gratitude, or quality time.",
        "Notice what conditions helped you feel close and supported.",
    ],
    "anger": [
        "Pause before reacting and write the boundary or value that feels crossed.",
        "Discharge some tension first, then decide whether this needs action or distance.",
        "Name the specific trigger so the next step feels deliberate instead of explosive.",
    ],
    "fear": [
        "Shrink the situation into the next safe, concrete step you can control.",
        "Use a grounding exercise to separate current facts from future worries.",
        "Write what you know, what you fear, and what support would help right now.",
    ],
    "surprise": [
        "Capture what happened and whether it felt energizing, disruptive, or both.",
        "Use the unexpected moment to notice what you value or want to protect.",
        "Turn the surprise into a learning note while the details are still fresh.",
    ],
}
