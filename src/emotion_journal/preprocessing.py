import re
import string

from .config import CRISIS_KEYWORDS

HIGH_RISK_CRISIS_PHRASES = {
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "take my life",
    "want to die",
    "wanna die",
    "wish i was dead",
    "wish i were dead",
    "self harm",
    "hurt myself",
    "overdose",
    "not safe",
    "cant go on",
    "can't go on",
    "cannot go on",
    "nothing to live for",
    "no reason to live",
}

BROAD_DISTRESS_PHRASES = {
    "hopeless",
}

NEGATION_CUES = {
    "not",
    "never",
    "no",
    "dont",
    "do not",
    "isnt",
    "is not",
    "wasnt",
    "was not",
    "without",
}

HIGH_RISK_CONTEXT_TERMS = {
    "die",
    "dying",
    "death",
    "unsafe",
    "hurt",
    "harm",
    "life",
    "living",
    "go",
}


def normalize_text(text: str) -> str:
    """Standardize raw journal text for both training and inference."""
    lowered = text.lower().strip()
    no_urls = re.sub(r"https?://\S+|www\.\S+", " ", lowered)
    no_punct = no_urls.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\s+", " ", no_punct).strip()
    return normalized


def _phrase_pattern(phrase: str) -> re.Pattern:
    normalized_phrase = normalize_text(phrase)
    escaped = re.escape(normalized_phrase).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def _has_negation_guard(normalized: str, start_index: int, *, window: int = 5) -> bool:
    prefix = normalized[:start_index].split()[-window:]
    prefix_text = " ".join(prefix)
    return any(cue in prefix or cue in prefix_text for cue in NEGATION_CUES)


def _matches_unnegated_phrase(normalized: str, phrase: str) -> bool:
    for match in _phrase_pattern(phrase).finditer(normalized):
        if not _has_negation_guard(normalized, match.start()):
            return True
    return False


def _matches_contextual_distress(normalized: str, phrase: str) -> bool:
    for match in _phrase_pattern(phrase).finditer(normalized):
        if _has_negation_guard(normalized, match.start()):
            continue
        surrounding = normalized[max(0, match.start() - 80) : match.end() + 80]
        tokens = set(surrounding.split())
        if tokens & HIGH_RISK_CONTEXT_TERMS:
            return True
    return False


def contains_crisis_language(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False

    high_risk_phrases = HIGH_RISK_CRISIS_PHRASES | (set(CRISIS_KEYWORDS) - BROAD_DISTRESS_PHRASES)
    if any(_matches_unnegated_phrase(normalized, phrase) for phrase in high_risk_phrases):
        return True

    return any(
        _matches_contextual_distress(normalized, phrase)
        for phrase in BROAD_DISTRESS_PHRASES
    )
