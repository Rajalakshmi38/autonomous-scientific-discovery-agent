"""RULE-BASED MOCK "LLM" -- TEST MODE ONLY.

This is NOT a language model and performs NO reasoning. It returns fixed
templates that echo the paper identifiers/titles found in the prompt so that
the pipeline plumbing (prompts, parsing, citation checks, storage) can be
tested without any model installed. Every output is labelled accordingly.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

TEST_MODE_LABEL = "[TEST MODE - rule-based mock, NOT genuine LLM reasoning]"
_PAPER_LINE = re.compile(r"^\[((?:arxiv|pubmed):[^\]\s]+)\]\s+(.+)$", re.MULTILINE)

_STUB_SCRIPT = '''\
# TEST MODE placeholder script. It uses SYNTHETIC random data and proves nothing
# about the research topic. It only exists to exercise the execution pipeline.
import json, random

random.seed(0)
baseline = [random.gauss(0.0, 1.0) for _ in range(30)]
treatment = [random.gauss(0.0, 1.0) for _ in range(30)]   # same distribution: no real effect
print(json.dumps({
    "data_origin": "synthetic",
    "comparisons": [{"name": "treatment vs baseline (synthetic)", "labels": ["baseline", "treatment"],
                     "a": baseline, "b": treatment}],
}))
'''


class RuleBasedStubChatModel(BaseChatModel):
    """Chooses a template by the ``[TASK:...]`` tag in the system prompt."""

    @property
    def _llm_type(self) -> str:
        return "rule-based-stub-test-mode"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        text = self._respond(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @staticmethod
    def _respond(prompt: str) -> str:
        papers = _PAPER_LINE.findall(prompt)
        topic_match = re.search(r"^Topic: (.+)$", prompt, re.MULTILINE)
        topic = topic_match.group(1).strip() if topic_match else "(unknown topic)"
        cites = ", ".join(f"[{uid}]" for uid, _ in papers[:3]) or "(no papers supplied)"

        if "[TASK:LITERATURE_REVIEW]" in prompt:
            lines = [f"- [{uid}] {title}" for uid, title in papers]
            return (
                f"{TEST_MODE_LABEL}\nPapers retrieved for topic '{topic}' (titles only, not summarised):\n"
                + "\n".join(lines)
            )
        if "[TASK:HYPOTHESIS]" in prompt:
            return (
                f"HYPOTHESIS: {TEST_MODE_LABEL} Placeholder hypothesis for '{topic}': a method discussed in "
                f"{cites} differs measurably from a baseline on a synthetic benchmark.\n"
                f"RATIONALE: Placeholder; evidence papers listed: {cites}.\n"
                "TEST PLAN: Generate synthetic data, compare two simple estimators, report the metric difference."
            )
        if "[TASK:EXPERIMENT_SPEC]" in prompt:
            return json.dumps({
                "note": TEST_MODE_LABEL,
                "dataset": "iris",
                "baseline": {"model": "logistic_regression", "params": {"C": 1}, "scale": False},
                "treatment": {"model": "logistic_regression", "params": {"C": 10}, "scale": False},
                "metric": "accuracy", "n_resamples": 20,
            })
        if "[TASK:EXPERIMENT_CODE]" in prompt or "[TASK:EXPERIMENT_REPAIR]" in prompt:
            return f"```python\n{_STUB_SCRIPT}```"
        return f"{TEST_MODE_LABEL} Unrecognised task."
