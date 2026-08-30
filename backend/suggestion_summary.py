"""Plain-language summary of the ranked suggestions.

The response returns up to four separate sentences and leaves the reader to work
out what they add up to. This builds one short paragraph from them.

Deterministic by construction. Every word shown either comes from the catalog's
``approved_text`` or from the fixed scaffolding in this module — nothing is
generated, reworded or intensified. That matters because the catalog carries
``approved`` and ``forbidden`` columns: the strings went through review, and text
that did not is not approved text.

The summary is an addition, not a replacement. The individual suggestions stay
visible beneath it, so a paragraph that drops an item to stay short is not hiding
anything.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

# Advice about taking the photograph, as opposed to what the measurements found.
# The two are different kinds of statement and reading them in one sentence
# implies the capture problem caused the measurement, which is not claimed.
CAPTURE_CATEGORIES = frozenset({"capture", "general"})

OBSERVATION_LEAD = "Compared with typical values for your comparison group, "
CAPTURE_LEAD = "For a clearer reading next time, "

# Soft cap. Prose past roughly this length stops being a summary; the full list is
# directly below it either way.
MAX_WORDS = 60

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    """Split approved text into sentences, preserving each one verbatim."""
    return [s.strip() for s in _SENTENCE_END.split(str(text).strip()) if s.strip()]


def _observation(text: str) -> str:
    """The first sentence, stripped of its full stop, for use as a clause."""
    parts = _sentences(text)
    return parts[0].rstrip(".").strip() if parts else ""


def _advice(text: str) -> str:
    """Everything after the first sentence — the 'what to do' half, where there is one."""
    parts = _sentences(text)
    return " ".join(parts[1:]).strip() if len(parts) > 1 else ""


def _decapitalise(clause: str) -> str:
    """Lower a leading capital so a sentence can be used mid-sentence.

    Only when the word is ordinary capitalisation, never when it is an acronym or a
    hyphenated proper form, so 'Left-right alignment' survives intact.
    """
    if not clause:
        return clause
    first = clause.split(" ", 1)[0]
    if first.isupper() or (len(first) > 1 and first[1].isupper()):
        return clause
    return clause[0].lower() + clause[1:]


def _join(clauses: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c' — a comma-separated list is not a sentence."""
    clauses = [c for c in clauses if c]
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f" and {clauses[-1]}"


def _word_count(text: str) -> int:
    return len(text.split())


def summarize_suggestions(suggestions: Iterable[Mapping[str, object]]) -> str:
    """One short paragraph built from ranked suggestions, or "" when there are none.

    Expects the dicts predict_suggestions returns: text, category, trigger_class.
    """
    items = [s for s in (suggestions or []) if str(s.get("text", "")).strip()]

    # Nothing to say. The caller renders nothing rather than an empty container.
    if not items:
        return ""

    # A single suggestion is already a sentence. Wrapping one sentence in scaffolding
    # adds words without adding meaning.
    if len(items) == 1:
        return str(items[0]["text"]).strip()

    # Nothing was flagged. The "compared with typical values" framing implies a
    # finding, so use the approved positive text instead of asserting one.
    if all(str(s.get("trigger_class", "")).lower() == "ok" for s in items):
        return str(items[0]["text"]).strip()

    capture, observations = [], []
    for s in items:
        category = str(s.get("category", "")).strip().lower()
        text = str(s["text"])
        if category in CAPTURE_CATEGORIES:
            # Capture rows lead with the problem and follow with the instruction;
            # the instruction is the useful half here.
            capture.append(_decapitalise(_advice(text) or _observation(text)))
        else:
            observations.append(_decapitalise(_observation(text)))

    # Rank order is preserved. Trimming to the word cap drops from the end — the
    # lowest-ranked items — rather than reordering or reselecting anything.
    while True:
        parts = []
        if observations:
            parts.append(OBSERVATION_LEAD + _join(observations) + ".")
        if capture:
            parts.append(CAPTURE_LEAD + _join(capture).rstrip(".") + ".")
        summary = " ".join(parts)
        if _word_count(summary) <= MAX_WORDS:
            return summary
        if len(observations) > 1:
            observations.pop()
        elif capture:
            capture.pop()
        else:
            return summary
