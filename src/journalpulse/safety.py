from __future__ import annotations

import re

from .domain import SafetyMode, SafetyResult

HIGH_RISK_PATTERNS = (
    r"\b(?:want|plan|going) to (?:die|kill myself|end my life)\b",
    r"\bi(?: will|'ll) (?:kill myself|end my life)\b",
    r"\b(?:hurt|harm) myself\b",
    r"\b(?:not|do not|don't) feel safe(?: (?:right now|tonight|today|alone))?\b",
    r"\bi (?:have|made) a suicide plan\b",
    r"\bi might act on (?:it|these thoughts)\b",
    r"\boverdose (?:myself|tonight|today)\b",
)

NEGATED_PATTERNS = (
    r"\b(?:not|never) suicidal\b",
    r"\b(?:do not|don't|never) want to (?:die|hurt myself|harm myself|kill myself|end my life)\b",
)

SUPPORT_BY_LOCALE = {
    "CA": (
        "If you may act on thoughts of suicide or self-harm, call or text 9-8-8 in Canada now. "
        "Call 9-1-1 if there is immediate danger, and reach a trusted person nearby if you can.",
        ["support_988_canada", "support_befrienders"],
    ),
    "US": (
        "If you may act on thoughts of suicide or self-harm, call or text 988 in the United States now. "
        "Call 911 if there is immediate danger, and reach a trusted person nearby if you can.",
        ["support_988", "support_befrienders"],
    ),
}


SUPPORT_FALLBACK_MESSAGE = (
    "If you may be in immediate danger, contact your local emergency service now and reach a trusted "
    "person nearby. Befrienders Worldwide can help locate crisis support in your country."
)

_CLAUSE_BREAK = re.compile(r"[.!?]+|;|\s+\bbut\b\s+|,\s*")


def _clauses(normalized: str) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_BREAK.split(normalized) if part.strip()]
    return parts or [normalized]


def _clause_has_unnegated_risk(clause: str) -> bool:
    negated_spans = [
        match.span() for pattern in NEGATED_PATTERNS for match in re.finditer(pattern, clause)
    ]
    # A denial applies only to the phrase it contains. "I don't want to die and
    # I might act on these thoughts" still contains a separate affirmative risk.
    return any(
        not any(start <= match.start() and match.end() <= end for start, end in negated_spans)
        for pattern in HIGH_RISK_PATTERNS
        for match in re.finditer(pattern, clause)
    )


def assess_safety(text: str, locale: str = "CA") -> SafetyResult:
    normalized = " ".join(text.lower().replace("’", "'").split())
    if not any(_clause_has_unnegated_risk(clause) for clause in _clauses(normalized)):
        return SafetyResult(mode=SafetyMode.NORMAL, locale=locale.upper(), exploration_allowed=True)

    locale_key = locale.upper()
    message, resources = SUPPORT_BY_LOCALE.get(
        locale_key,
        (SUPPORT_FALLBACK_MESSAGE, ["support_befrienders"]),
    )
    return SafetyResult(
        mode=SafetyMode.SUPPORT,
        reasons=["high_risk_language"],
        locale=locale_key,
        exploration_allowed=False,
        support_message=message,
        resource_ids=resources,
    )
