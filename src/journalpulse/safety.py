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
    # Indirect first-person phrasings (2026-10-05 audit). Bare idioms such as "killing
    # myself at work" stay out because support mode closes the composer for the chat.
    r"\b(?:everyone|everybody|people|they|the world|my (?:family|friends|kids|children|partner|parents)) "
    r"(?:would|will|'d) be better off without me\b",
    r"\bbetter off if i (?:was|were) (?:dead|gone|not here|not around)\b",
    r"\b(?:end|ending) it all\b",
    r"\bi (?:won't|will not|'m not going to|am not going to) be (?:around|here) much longer\b",
    r"\bsuicidal\b",
    r"\b(?:thinking|thoughts?) (?:about|of) (?:killing myself|suicide|ending my life)\b",
    r"\bi wish i (?:was|were) dead\b",
    r"\bi (?:don't|do not) want to (?:be alive|live|exist) anymore\b",
)

# A risk phrase preceded in its clause by one of these reads as history ("I used to want
# to die"). It still routes to support if the clause also has a present-time marker, so
# "I used to want to die and now I want to die again" is never treated as past.
HISTORICAL_PREFIX = re.compile(
    r"\b(?:used\s+to|years\s+ago|months\s+ago|long\s+ago|in\s+the\s+past|back\s+then|"
    r"when\s+i\s+was\s+(?:(?:a|an)\s+)?(?:kid|child|teen|teenager|younger|student|\d+))\b"
)
PRESENT_MARKER = re.compile(
    r"\b(?:now|again|still|today|tonight|currently|lately|these days|this (?:week|month))\b"
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

# Clause ownership depends on the subject, not an immediate verb whitelist:
# "and I really want ..." and "and sometimes I think I might ..." are independent
# statements. Shared past predicates retain the earlier historical scope.
_CLAUSE_BREAK = re.compile(r"[.!?]+|;|\s+\bbut\b\s+|,\s*")
_COORDINATION = re.compile(
    # Colons, dashes and newlines may introduce a new subject or continue a past
    # predicate. Decide below instead of discarding inherited history globally.
    # Keep hyphenated words and the historical marker "back then" intact.
    r"[:/\u2013\u2014]+|--+|(?<!\w)-|-(?!\w)|\n+|"
    r"\s+(?:and|or|yet|so|because|although|while|however|(?<!\bback )then)\s+"
)
_INDEPENDENT_SUBJECT = re.compile(
    r"\b(?:i\b|everyone\b|everybody\b|people\b|they\b|the\s+world\b|"
    r"my\s+(?:family|friends|kids|children|partner|parents)\b)"
)
_PAST_PREDICATE = re.compile(
    # "Would" is also a present conditional, so it cannot establish past scope.
    r"^i\s+(?:(?:[a-z]+ly|also|even)\s+)*(?:was|were|felt|wanted|wished|thought|used to)\b"
)
_PAST_HABIT = re.compile(r"^i\s+(?:(?:[a-z]+ly|also|even)\s+)*would\b")
_PRESENT_CONTINUATION = re.compile(r"\bi (?:still do|still am|still feel that way)\b")
_REPORTED_PREDICATE = re.compile(
    # A belief's embedded subject shares the belief's tense: "I was convinced
    # my family ..." is past; "I am convinced my family ..." remains current.
    r"\b(?:think|thought|thinking|feel|felt|feeling|believe|believed|remember|remembered|said|convinced|sure)"
    r"(?:\s+(?:that|like|about how))?$"
)


def _subject_parts(part: str) -> list[str]:
    """Separate a later current subject without treating a reported thought as new."""
    # An explicit historical statement owns its embedded subjects until a real
    # clause delimiter. This avoids treating objects/quoted thoughts as current
    # statements in "I used to tell myself I want to die".
    if HISTORICAL_PREFIX.search(part):
        return [part]
    parts = []
    start = 0
    for subject in list(_INDEPENDENT_SUBJECT.finditer(part))[1:]:
        prefix = part[:subject.start()].rstrip()
        if (
            _REPORTED_PREDICATE.search(prefix)
            or _PAST_PREDICATE.match(part[subject.start():])
            or _PRESENT_CONTINUATION.match(part[subject.start():])
        ):
            continue
        parts.append(part[start:subject.start()].strip())
        start = subject.start()
    parts.append(part[start:])
    return parts


def _clauses(normalized: str) -> list[str]:
    clauses = []
    for sentence in _CLAUSE_BREAK.split(normalized):
        # Line wrapping within "my family", "the world" or a historical marker
        # does not introduce a clause boundary. Retain other newlines for scope.
        for phrase in (HISTORICAL_PREFIX, _INDEPENDENT_SUBJECT):
            sentence = phrase.sub(lambda match: " ".join(match.group().split()), sentence)
        combined = ""
        parts = [
            part
            for raw in _COORDINATION.split(sentence)
            for part in _subject_parts(" ".join(raw.split()))
        ]
        for part in parts:
            if not part:
                continue
            subject = _INDEPENDENT_SUBJECT.search(part)
            subject_clause = part[subject.start():] if subject else ""
            shared_past = (
                not PRESENT_MARKER.search(part)
                and (
                    _PAST_PREDICATE.match(subject_clause)
                    # A standalone history heading establishes past scope for
                    # "When I was a teenager: I would hurt myself". "Would"
                    # after a complete statement remains a possible current wish.
                    or (HISTORICAL_PREFIX.fullmatch(combined) and _PAST_HABIT.match(subject_clause))
                )
            )
            # "I still do" explicitly carries the earlier predicate into now.
            carries_predicate = shared_past or _PRESENT_CONTINUATION.search(part)
            if combined and subject and not carries_predicate:
                clauses.append(combined)
                combined = part
            else:
                # A wrapped phrase such as "I do not\nwant to die" must retain
                # its denial rather than gaining an artificial conjunction.
                combined = f"{combined} {part}" if combined else part
        if combined:
            clauses.append(combined)
    return clauses or [normalized]


def _clause_has_unnegated_risk(clause: str) -> bool:
    negated_spans = [
        match.span() for pattern in NEGATED_PATTERNS for match in re.finditer(pattern, clause)
    ]
    # A denial applies only to the phrase it contains. "I don't want to die and
    # I might act on these thoughts" still contains a separate affirmative risk.
    present = PRESENT_MARKER.search(clause) is not None

    def is_risk(match: re.Match[str]) -> bool:
        if any(start <= match.start() and match.end() <= end for start, end in negated_spans):
            return False
        historical = HISTORICAL_PREFIX.search(clause, 0, match.start()) is not None
        return present or not historical

    return any(
        is_risk(match) for pattern in HIGH_RISK_PATTERNS for match in re.finditer(pattern, clause)
    )


def assess_safety(text: str, locale: str = "CA") -> SafetyResult:
    normalized = "\n".join(
        " ".join(line.split()) for line in text.lower().replace("’", "'").splitlines()
    )
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
