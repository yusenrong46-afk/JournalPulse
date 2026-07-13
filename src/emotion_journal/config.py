from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DEFAULT_DB_PATH = ARTIFACTS_DIR / "journal.db"
RESOURCE_CATALOG_PATH = ASSETS_DIR / "resources" / "catalog.json"

APP_ENV_ENV = "JOURNALPULSE_ENV"
API_BASE_URL_ENV = "JOURNALPULSE_API_BASE_URL"
DB_PATH_ENV = "JOURNALPULSE_DB_PATH"
HF_MODEL_ID_ENV = "JOURNALPULSE_HF_MODEL_ID"
DEPLOYMENT_MODE_ENV = "JOURNALPULSE_DEPLOYMENT_MODE"

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
RESOURCE_GOAL_TAGS = (
    "ground",
    "planning",
    "reframing",
    "connection",
    "movement",
    "reading",
    "watching",
    "play",
)
SOURCE_TIERS = ("official", "nonprofit", "educational", "activity", "crisis_support")

LLM_API_KEY_ENV = "JOURNALPULSE_LLM_API_KEY"
LLM_BASE_URL_ENV = "JOURNALPULSE_LLM_BASE_URL"
LLM_MODEL_ENV = "JOURNALPULSE_LLM_MODEL"
LLM_MODE_ENV = "JOURNALPULSE_LLM_MODE"
LLM_APP_URL_ENV = "JOURNALPULSE_LLM_APP_URL"
LLM_APP_TITLE_ENV = "JOURNALPULSE_LLM_APP_TITLE"
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODE = "off"
LLM_TIMEOUT_SECONDS = 10.0
CLASSIFIER_MODE_ENV = "JOURNALPULSE_CLASSIFIER_MODE"
DEFAULT_CLASSIFIER_MODE = "calibrated"

# AI-generated resource suggestions may only link to these reputable, browser-safe
# domains. This is the guardrail that keeps the LLM from inventing risky links.
# A suggestion's host must equal one of these or be a subdomain of one.
RESOURCE_DOMAIN_SAFELIST = (
    # Official / government health
    "nimh.nih.gov",
    "nih.gov",
    "cdc.gov",
    "samhsa.gov",
    "who.int",
    "nhs.uk",
    "betterhealth.vic.gov.au",
    # Nonprofit / advocacy
    "mhanational.org",
    "nami.org",
    "apa.org",
    "mind.org.uk",
    "helpguide.org",
    "988lifeline.org",
    "actionforhappiness.org",
    "self-compassion.org",
    "greatergood.berkeley.edu",
    # Education / reputable wellness
    "headspace.com",
    "calm.com",
    "insighttimer.com",
    "mindful.org",
    "ted.com",
    "khanacademy.org",
    "coursera.org",
    "edx.org",
    "verywellmind.com",
    "psychologytoday.com",
    "positivepsychology.com",
    "sleepfoundation.org",
    "nutrition.org",
    # Browser-safe media / play
    "youtube.com",
    "youtu.be",
    "open.spotify.com",
    "freerice.com",
    "quickdraw.withgoogle.com",
)


def app_environment() -> str:
    return os.getenv(APP_ENV_ENV, "local").strip().lower() or "local"


def api_base_url() -> str:
    return os.getenv(API_BASE_URL_ENV, "").strip().rstrip("/")


def deployment_mode() -> str:
    return os.getenv(DEPLOYMENT_MODE_ENV, "demo").strip().lower() or "demo"


def database_path() -> Path:
    configured = os.getenv(DB_PATH_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_DB_PATH


def hf_model_id() -> str:
    return os.getenv(HF_MODEL_ID_ENV, "").strip()

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
