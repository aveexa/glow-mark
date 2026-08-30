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

Layer 2 (bottom of this module) is an optional LLM pass (currently Groq) that
polishes that paragraph's wording. It is guarded and fail-soft: ``summarize_suggestions``
is the product and always works; the LLM path is never required and falls back to the
template on any failure. That separation is deliberate — this function stays
deterministic and dependency-light, while the LLM code is quarantined at the end
of the module.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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


def _strip_capture_lead(clause: str) -> str:
    """Drop a redundant 'for a clearer reading next time' prefix so the clause can
    take the module-level CAPTURE_LEAD without doubling it up."""
    clause = clause.strip()
    if clause.lower().startswith("for a clearer reading next time"):
        remainder = clause[len("For a clearer reading next time"):].lstrip(", ").strip()
        return remainder or clause
    return clause


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
            capture.append(_decapitalise(_strip_capture_lead(_advice(text) or _observation(text))))
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


# --- Layer 2: optional LLM polish (Groq) ---------------------------------------------
# Everything above is deterministic and never touches the network. This section is
# an optional pass that rewrites that template's wording via Groq's OpenAI-compatible
# chat-completions API. It is guarded and fail-soft by construction: if the flag is
# off, no GROQ_API_KEY is set, the request times out, the HTTP call errors, or the
# guard trips, the caller gets the template back. It is never required, never reports
# an error, and never invents content — only approved catalog text is sent to the model.

# Words the model may use purely to join sentences, plus the template's own
# scaffolding ("compared with typical values for your comparison group") and the
# capture lead-in. These carry no claim, so a rewrite may use them freely.
_CONNECTIVES: frozenset = frozenset(
    """
    a an the and or but not so if then with for to of in on at from by as
    is are was were be been being have has had my your you its it this that
    these those we they i than more less their our commonly compared typical
    values group comparison clearer reading next time please relax keep eyes open
    across between than when where while also overall while though however then
    very too
    """.split()
)

# Hard block. Words that would invent a compliment, an intensifying judgement, or
# a claim about a person's attractiveness never appear in the reviewed catalog, so
# a rewrite that introduces them is always rejected. This is the whole gate: a
# rewrite may reword and reorder freely as long as none of these words appear.
_FORBIDDEN: frozenset = frozenset(
    """
    beautiful beauty gorgeous stunning attractive flawless perfect impeccable
    ideal radiant elegant handsome lovely pretty cute adorable glowing flawless
    stunning extraordinary remarkably amazingly incredibly astonishing
    outstanding phenomenal spectacular wonderful fantastic incredible amazing
    awesome excellent superb magnificent exceptional strikingly dramatically
    perfectly
    """.split()
)

# The LLM layer is provider-neutral in spirit but currently wired to Groq's
# OpenAI-compatible chat-completions API. Set the model here or via
# ``summary.model`` in gate_config.json; authentication is the GROQ_API_KEY env var.
GROQ_MODEL = "qwen/qwen3.6-27b"
_LLM_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
_LLM_TIMEOUT_SECS = 10.0
_LLM_MAX_OUTPUT_TOKENS = 200
_LLM_TEMPERATURE = 0.2

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_CONFIG_PATH = _REPO_ROOT / "data" / "interim" / "gate_config.json"

# Load GROQ_API_KEY (and anything else) from the repo-root .env so the summary
# works without the operator exporting the env var by hand. override=False means a
# real exported variable still wins over .env.
load_dotenv(_REPO_ROOT / ".env", override=False)

# Cache hit rate matters here because the calls are billed. Most runs inspect only a
# handful of the ~50 catalog suggestions, so the ranked combinations repeat often
# and the same wording should not be bought twice. Key on the sorted suggestion ids.
_LLM_CACHE: dict[tuple[str, ...], str] = {}

_WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")


def _words(text: str) -> set:
    return set(_WORD_RE.findall(str(text).lower()))


def _read_config() -> dict:
    """Read the whole gate_config.json fresh. Never cached: the Settings UI toggles
    summary.use_llm at runtime and must take effect on the very next analyze without
    a server restart."""
    try:
        with open(_GATE_CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:  # noqa: BLE001 — a broken config must not break analyze
        return {}


def _summary_config() -> dict:
    return dict(_read_config().get("summary", {}) or {})


def ai_summary_enabled() -> bool:
    """Public, user-facing flag: is the AI summary turned on in Settings?

    This is exactly the persisted ``summary.use_llm`` flag the Settings page writes.
    It does NOT require a GROQ_API_KEY — wiring that only affects whether the polish
    is the model's or the deterministic template's fallback, not whether a summary
    is returned at all. Read fresh so the toggle applies to the next analyze.
    """
    env = os.environ.get("SUMMARY_USE_LLM")
    if env is not None and env.strip():
        return env.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_read_config().get("summary", {}).get("use_llm", False))


def set_ai_summary_enabled(enabled: bool) -> bool:
    """Persist summary.use_llm to gate_config.json (atomic replace on same volume).

    Returns the value that was written. The empty-`GROQ_API_KEY` note in the file's
    ``_why`` keeps the disconnect visible to anyone who reads the config.
    """
    cfg = _read_config()
    summary = dict(cfg.get("summary", {}) or {})
    summary["use_llm"] = bool(enabled)
    cfg["summary"] = summary
    tmp = _GATE_CONFIG_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    tmp.replace(_GATE_CONFIG_PATH)
    return bool(enabled)


def _llm_enabled() -> bool:
    """Internal gate for whether to actually call Groq (needs a key in addition to
    the user-facing flag). Without a key the summary still renders, but as the
    deterministic template rather than a model-polished paragraph."""
    if not os.environ.get("GROQ_API_KEY"):
        return False
    return ai_summary_enabled()


def _llm_model() -> str:
    return str(_summary_config().get("model", GROQ_MODEL) or GROQ_MODEL)


def _llm_messages(template: str, items: Sequence[Mapping[str, object]]) -> Sequence[Mapping[str, str]]:
    sentences = "\n".join(f"- {str(s.get('text', '')).strip()}" for s in items)
    # Single user message (no system role). Groq's reasoning models are documented
    # to follow instructions best when they all live in the user message. The
    # "keep every point" constraint stops the polish from silently dropping a
    # particular recommendation while staying within the word cap.
    user = (
        "Write one short, friendly paragraph that combines the points below into "
        "natural, conversational advice for the reader. Treat each bullet as a finding "
        "plus what to do about it, and weave them together so it flows like a person "
        "talking, not like a list.\n"
        "Rules:\n"
        "- Write directly to the reader ('your face', 'try', 'add', 'soften') in plain "
        "English.\n"
        "- State each finding naturally (e.g. 'reads a little higher/lower than typical') "
        "and turn its instruction into a 'try ...', 'soften ...', or 'add ...' "
        "recommendation.\n"
        "- Cover every point from the source sentences; do not drop or contradict any "
        "of them.\n"
        "- Do not invent measurements, values, comparisons, or advice that are not "
        "present in the source.\n"
        "- Do not comment on a person's attractiveness or appearance (no 'beautiful', "
        "'perfect', and so on).\n"
        "- Keep it to a single paragraph under 60 words.\n"
        "- Output ONLY the paragraph, with no intro, headings, reasoning, or markdown.\n\n"
        "Source approved sentences:\n"
        f"{sentences}"
    )
    return [{"role": "user", "content": user}]


def _llm_call(messages: Sequence[Mapping[str, str]]) -> str:
    payload = {
        "model": _llm_model(),
        "messages": messages,
        "temperature": _LLM_TEMPERATURE,
        "max_tokens": _LLM_MAX_OUTPUT_TOKENS,
        # reasoning_effort "none" disables Qwen's thinking preamble; without it the
        # model emits a "thinking" block that trips the word guard. (Only the Qwen
        # family supports "none"; gpt-oss takes low/medium/high instead.)
        "reasoning_effort": "none",
    }
    resp = requests.post(
        _LLM_BASE_URL,
        headers={"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"},
        json=payload,
        timeout=_LLM_TIMEOUT_SECS,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM returned no choices")
    content = (choices[0].get("message", {}) or {}).get("content") or ""
    return content.strip()


def _guard_passes(proposed: str, items: Sequence[Mapping[str, object]]) -> bool:
    """Reject rewrites that introduce forbidden wording.

    The only hard rules are the ``_FORBIDDEN`` words: invented compliments,
    intensifying judgements, or claims about a person's attractiveness. Grammar,
    phrasing, and reordering are left entirely to the model — a rewrite is served
    as long as none of those words appear. ``items`` is accepted for call-site
    compatibility only; it is not used for enforcement.
    """
    proposed_words = _words(proposed)
    if proposed_words & _FORBIDDEN:
        return False
    return True


def summarize_suggestions_with_llm(
    suggestions: Iterable[Mapping[str, object]],
) -> str:
    """Deterministic summary, optionally polished by Gemini when enabled and keyed.

    The deterministic ``summarize_suggestions`` is always the fallback; the LLM pass
    only substitutes a reworded paragraph when it is enabled and its output survives
    the word guard. Keyed on the sorted suggestion ids.
    """
    items = [s for s in (suggestions or []) if str(s.get("text", "")).strip()]
    template = summarize_suggestions(items)

    # The LLM only adds value where the template joined two or more sentences. A
    # single suggestion is returned unchanged, and the all-ok case is one approved
    # sentence — rewriting either would just add words without adding meaning.
    if len(items) < 2:
        return template
    if all(str(s.get("trigger_class", "")).lower() == "ok" for s in items):
        return template
    if not _llm_enabled():
        return template

    cache_key = tuple(sorted(str(s.get("id", "")) for s in items))
    cached = _LLM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        proposed = _llm_call(_llm_messages(template, items))
    except Exception:  # noqa: BLE001 — the template must always win on any failure
        logger.warning("LLM summary unavailable; using template", exc_info=True)
        return template

    # Accept only non-empty output that survives the word guard; empty or guard-
    # tripping output falls back to the template (never cached, never served).
    if proposed and _guard_passes(proposed, items):
        _LLM_CACHE[cache_key] = proposed
        return proposed

    logger.warning("LLM summary rejected (empty or guard-tripped); using template: %r", proposed[:200])
    return template

