"""Code Executor Agent.

WEEK 1 SCOPE: this agent only PREPARES experiment code. It does NOT run it.
Execution inside the Docker sandbox is implemented in Week 2; until then
``execution_status`` is always PENDING. Generated code is untrusted.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.base import AgentError
from src.agents.hypothesis_generator import Hypothesis

SYSTEM = (
    "[TASK:EXPERIMENT_CODE]\n"
    "You write small, self-contained Python 3 experiment scripts. Rules: standard library only (plus numpy if "
    "essential); no network access; no reading environment variables; no file writes except under /tmp; "
    "set random seeds; print the final results as a single JSON object to stdout; label any synthetic data as "
    "synthetic. Reply with exactly one ```python code block and nothing else."
)
HUMAN = "Hypothesis: {hypothesis}\n\nTest plan: {test_plan}"

PENDING_STATUS = "PENDING - sandbox execution is not available until Week 2 (Docker)"
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_RISKY_MODULES = {"os", "subprocess", "socket", "shutil", "ctypes", "requests", "urllib", "http", "sys"}


@dataclass
class ExperimentPlan:
    code: str
    syntax_ok: bool
    syntax_error: Optional[str] = None
    advisory_warnings: List[str] = field(default_factory=list)   # NOT a security control
    execution_status: str = PENDING_STATUS


def extract_code(text: str) -> str:
    """Return the first fenced code block, or the raw text if no fence is present."""
    match = _FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


def analyse_code(code: str) -> tuple[bool, Optional[str], List[str]]:
    """Syntax check + advisory scan for risky imports. The sandbox, not this scan, provides safety."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno})", []
    warnings = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        warnings += [f"imports '{n}' (review before running)" for n in names if n in _RISKY_MODULES]
    return True, None, sorted(set(warnings))


class CodeExecutorAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
        self._chain = prompt | llm | StrOutputParser()

    def prepare_experiment(self, hypothesis: Hypothesis) -> ExperimentPlan:
        try:
            raw = self._chain.invoke(
                {"hypothesis": hypothesis.statement, "test_plan": hypothesis.test_plan or "(none given)"}
            )
        except Exception as exc:
            raise AgentError(f"Code Executor model call failed: {exc}") from exc
        code = extract_code(raw)
        ok, error, warnings = analyse_code(code)
        return ExperimentPlan(code=code, syntax_ok=ok, syntax_error=error, advisory_warnings=warnings)
