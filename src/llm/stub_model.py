"""RULE-BASED MOCK "LLM" -- TEST MODE ONLY.

This is NOT a language model and performs NO reasoning. It returns fixed
templates that echo the paper identifiers/titles found in the prompt so that
the pipeline plumbing (prompts, parsing, citation checks, storage) can be
tested without any model installed. Every output is labelled accordingly.
"""
from __future__ import annotations

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
import json, random, statistics

random.seed(0)
group_a = [random.gauss(0.0, 1.0) for _ in range(100)]
group_b = [random.gauss(0.1, 1.0) for _ in range(100)]
print(json.dumps({
    "data_type": "synthetic",
    "mean_a": statistics.mean(group_a),
    "mean_b": statistics.mean(group_b),
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
        if "[TASK:EXPERIMENT_CODE]" in prompt:
            return f"```python\n{_STUB_SCRIPT}```"
        return f"{TEST_MODE_LABEL} Unrecognised task."
