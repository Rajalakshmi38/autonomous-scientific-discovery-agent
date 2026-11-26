"""Small helpers for starting a research run from the dashboard. No Streamlit and no language model here.

The run itself is src.main.run_research(), the same function `python -m src.main run` uses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

MIN_TOPIC_CHARS = 5
MAX_TOPIC_CHARS = 300


@dataclass
class TopicCheck:
    topic: str                       # the cleaned topic (whitespace collapsed); empty when invalid
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def validate_topic(raw: object) -> TopicCheck:
    """Reject empty, tiny, huge or non-text topics before any model is called."""
    text = " ".join(str(raw or "").split())
    if not text:
        return TopicCheck("", "Enter a research topic first.")
    if any((ord(ch) < 32 and ch not in "\t\n\r") or ord(ch) == 127 for ch in str(raw)):
        return TopicCheck("", "The topic contains control characters. Please type plain text.")
    if len(text) < MIN_TOPIC_CHARS:
        return TopicCheck("", f"The topic is too short. Use at least {MIN_TOPIC_CHARS} characters.")
    if len(text) > MAX_TOPIC_CHARS:
        return TopicCheck("", f"The topic is too long ({len(text)} characters). Keep it under {MAX_TOPIC_CHARS}.")
    if sum(ch.isalpha() for ch in text) < 3:
        return TopicCheck("", "The topic should describe a research subject in words, for example "
                              "\"Diabetes prediction using machine learning\".")
    return TopicCheck(text)


def format_event(event) -> str:
    """One markdown line per real pipeline event (a workflow.ProgressEvent). Nothing is estimated."""
    where = f"Step {event.index} of {event.total}: {event.title}"
    took = f" in {event.seconds:g} s" if event.seconds is not None else ""
    extra = f" - {event.message}" if event.message else ""
    if event.phase == "running":
        return f":blue[:material/hourglass_top:] {where} - running..."
    if event.phase == "done":
        return f":green[:material/check_circle:] {where} - finished{took}.{extra}"
    if event.phase == "warning":
        return f":orange[:material/warning:] {where} - finished{took} with a warning.{extra}"
    if event.phase == "skipped":
        return f":gray[:material/skip_next:] {where} - not run.{extra}"
    return f":red[:material/error:] {where} - failed{took}.{extra}"


def waiting_line(index: int, total: int, title: str) -> str:
    return f":gray[:material/radio_button_unchecked:] Step {index} of {total}: {title} - waiting"


def friendly_error(exc: BaseException, step_title: Optional[str] = None) -> str:
    """A plain-language reason for a failed run, naming the step that was running when it failed."""
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    where = f"The step \"{step_title}\" failed. " if step_title else "The research run failed. "
    if isinstance(exc, ConnectionError) or "connection refused" in lowered or "failed to establish a new connection" in lowered:
        return where + ("The local language model (Ollama) could not be reached. Start Ollama, check that the model is "
                        "pulled (python -m src.main check-llm) and try again. Details: " + text)
    if "timed out" in lowered or "timeout" in lowered:
        return where + "The model took too long to answer. Raise LLM_TIMEOUT_SECONDS in .env or use a smaller model. Details: " + text
    if isinstance(exc, OSError):
        return where + "A file could not be read or written (disk space or permissions?). Details: " + text
    return where + "Details: " + text + " (more in storage/logs/app.log)"


_FILLER = {"the", "and", "with", "without", "for", "from", "into", "onto", "that", "this", "using", "versus", "between", "among",
           "than", "then", "of", "on", "in", "by", "to", "an", "is", "are", "its", "their", "effect", "effects", "accuracy",
           "dataset", "datasets", "study", "analysis", "approach", "method", "methods", "based", "new"}


def fallback_queries(topic: str) -> List[str]:
    """Search texts to try when fetching papers: the whole topic first, then its main words (arXiv often finds nothing for a long
    sentence). At most three, in order, and never the same text twice."""
    words: List[str] = []
    for w in re.findall(r"[A-Za-z][A-Za-z\-]+", topic):
        if len(w) > 2 and w.lower() not in _FILLER and w.lower() not in [x.lower() for x in words]:     # each word once
            words.append(w)
    queries = [" ".join(str(topic).split())]
    if len(words) <= 3:                                           # already short: shorter text would only search worse
        return queries
    for count in (4, 2):
        if len(words) >= count:
            candidate = " ".join(words[:count])
            if candidate.lower() not in [q.lower() for q in queries]:
                queries.append(candidate)
    return queries
