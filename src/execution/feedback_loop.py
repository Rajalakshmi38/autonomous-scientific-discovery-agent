"""Bounded error-feedback loop: run -> read the error -> ask the model for a fix -> run again.

THIS IS NOT REINFORCEMENT LEARNING. Nothing is trained and nothing is remembered between runs; the
model simply sees the error text once and writes new code. The loop is capped by ``max_repairs`` so a
failing script costs at most ``1 + max_repairs`` sandbox runs. See docs/rl_proposal.md for what a
genuine RL component would require.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from src.agents.base import AgentError
from src.execution.sandbox import ExecutionResult, Status, missing_module

logger = logging.getLogger(__name__)


class _Runner(Protocol):
    def run_code(self, code: str, timeout_seconds: float | None = None) -> ExecutionResult: ...


class _Repairer(Protocol):
    def repair_experiment(self, hypothesis: Any, code: str, error_text: str) -> Any: ...


@dataclass
class Attempt:
    number: int
    code: str
    result: ExecutionResult
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"number": self.number, "code": self.code, "note": self.note, "result": self.result.to_dict()}


@dataclass
class ExecutionOutcome:
    attempts: List[Attempt] = field(default_factory=list)
    final_status: str = Status.NOT_RUN
    message: str = ""

    @property
    def final_result(self) -> ExecutionResult | None:
        return self.attempts[-1].result if self.attempts else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_status": self.final_status,
            "message": self.message,
            "attempt_count": len(self.attempts),
            "attempts": [a.to_dict() for a in self.attempts],
        }


def summarize_error(result: ExecutionResult, max_chars: int = 1500) -> str:
    """Compact, model-readable description of a failed run (the END of stderr holds the real error)."""
    lines = [f"Status: {result.status} (exit code {result.exit_code})", f"Meaning: {result.message}"]
    module = missing_module(result.stderr)
    if module:
        lines.append(f"The module '{module}' is NOT installed. Only the standard library and numpy exist.")
    if result.status == Status.TIMEOUT:
        lines.append("Make the script much faster: fewer iterations, smaller data, no infinite loops.")
    if result.status == Status.OUT_OF_MEMORY:
        lines.append("Use much smaller data structures.")
    if result.stderr.strip():
        lines.append("stderr (last part):\n" + result.stderr.strip()[-max_chars:])
    if result.stdout.strip():
        lines.append("stdout (last part):\n" + result.stdout.strip()[-500:])
    return "\n".join(lines)


def run_with_repair(
    sandbox: _Runner,
    repairer: _Repairer,
    hypothesis: Any,
    initial_code: str,
    max_repairs: int = 2,
) -> ExecutionOutcome:
    """Run ``initial_code``; on a repairable failure, ask for a fix and retry, at most ``max_repairs`` times."""
    outcome = ExecutionOutcome()
    code = initial_code
    max_repairs = max(0, max_repairs)

    for number in range(1, max_repairs + 2):
        result = sandbox.run_code(code)
        attempt = Attempt(number, code, result)
        outcome.attempts.append(attempt)
        logger.info("Attempt %d/%d finished with %s", number, max_repairs + 1, result.status)

        if result.ok:
            break
        if not result.repairable:
            attempt.note = "Not retried: a code fix cannot solve a sandbox problem."
            break
        if number > max_repairs:
            attempt.note = f"Stopped: repair limit ({max_repairs}) reached."
            break
        try:
            repaired = repairer.repair_experiment(hypothesis, code, summarize_error(result))
        except AgentError as exc:
            attempt.note = f"Stopped: the model could not repair the code ({exc})."
            break
        if repaired.code.strip() == code.strip():
            attempt.note = "Stopped: the model returned the same code."
            break
        attempt.note = "Sent the error to the model and retried with its fix."
        code = repaired.code

    final = outcome.attempts[-1].result
    outcome.final_status = final.status
    outcome.message = final.message
    return outcome
