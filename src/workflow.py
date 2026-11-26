"""Research workflow: retrieve (RAG) -> review -> hypothesis -> experiment code -> sandbox run (+ repair).

Built as a LangChain runnable pipeline (``RunnableLambda`` steps joined with ``|``). Each step receives
and returns a state dict. If no evidence is found the pipeline stops early WITHOUT calling any model.
Generated code is only ever executed inside the Docker sandbox, never on the host.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableLambda

from src.agents.code_executor import CodeExecutorAgent, ExperimentPlan
from src.agents.hypothesis_generator import Hypothesis, HypothesisGeneratorAgent
from src.agents.literature_reviewer import LiteratureReview, LiteratureReviewerAgent
from src.critic.critic import CriticReport, StatisticalCritic
from src.execution.feedback_loop import ExecutionOutcome, run_with_repair
from src.execution.sandbox import Status
from src.models import RetrievedPaper, publication_status
from src.retrieval.rag_pipeline import GroundingReport, RagPipeline, Source
from src.retrieval.retriever import PaperRetriever

logger = logging.getLogger(__name__)

# The six pipeline steps in order: (key, title). The dashboard shows the same titles.
PIPELINE_STEPS = [("retrieve", "Retrieve papers"), ("review", "Literature review"), ("hypothesis", "Hypothesis"),
                  ("experiment_design", "Experiment design"), ("execute", "Sandbox execution"), ("critic", "Statistical critic")]


@dataclass
class ProgressEvent:
    """One real moment in a run, sent to the optional ``on_progress`` callback. Nothing here is estimated.

    phase: running (the step just started) | done | warning (finished with a caveat) | skipped | failed (raised an error).
    """
    step: str
    phase: str
    seconds: Optional[float] = None
    message: str = ""

    @property
    def index(self) -> int:
        return [k for k, _ in PIPELINE_STEPS].index(self.step) + 1

    @property
    def total(self) -> int:
        return len(PIPELINE_STEPS)

    @property
    def title(self) -> str:
        return dict(PIPELINE_STEPS)[self.step]


def _notify(callback: Optional[Callable[[ProgressEvent], None]], event: ProgressEvent) -> None:
    """A broken progress listener must never break the research run. (Only ``Exception`` is caught, so a UI that wants
    to cancel the run by raising a BaseException still can.)"""
    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:
        logger.warning("Progress callback failed (ignored): %s", exc)


def _step_outcome(name: str, state: dict) -> tuple[str, str]:
    """What actually happened in a step that did not raise, read from the real pipeline state."""
    if name != "retrieve" and state.get("stop"):
        return "skipped", "Skipped because no relevant papers were retrieved."
    if name == "retrieve":
        n = len(state.get("evidence") or [])
        return ("done", f"{n} paper(s) retrieved.") if n else ("warning", "No relevant papers found in the local index.")
    if name in ("review", "hypothesis"):
        item = state.get(name)
        unknown, cited = list(getattr(item, "unknown_citations", []) or []), list(getattr(item, "cited_ids", []) or [])
        if unknown:
            return "warning", f"Cites {len(unknown)} paper id(s) that were not retrieved."
        if not cited:
            return "warning", "Cites no retrieved paper by id."
        return "done", f"Cites {len(cited)} retrieved paper(s)."
    if name == "experiment_design":
        exp = state.get("experiment")
        if exp is None:
            return "warning", "No experiment was prepared."
        if exp.note or not exp.syntax_ok:
            return "warning", exp.note or "The generated code has a syntax error."
        return "done", exp.design_summary or "Experiment code prepared."
    if name == "execute":
        outcome = state.get("execution")
        if outcome is None:
            return "skipped", "Execution was disabled for this run."
        if outcome.final_status == Status.SANDBOX_UNAVAILABLE:
            return "skipped", outcome.message or "The sandbox is not available."
        if outcome.final_status == Status.SUCCESS:
            return "done", f"Sandbox run succeeded ({len(outcome.attempts)} attempt(s))."
        return "warning", f"Ended with {outcome.final_status} after {len(outcome.attempts)} attempt(s)."
    if name == "critic":
        report = state.get("critic")
        return ("done", f"Verdict: {report.verdict}.") if report is not None else ("skipped", "No experiment output to check.")
    return "done", ""


@dataclass
class WorkflowResult:
    topic: str
    llm_mode: str
    status: str                                   # COMPLETED | NO_EVIDENCE
    evidence: List[RetrievedPaper] = field(default_factory=list)
    review: Optional[LiteratureReview] = None
    hypothesis: Optional[Hypothesis] = None
    experiment: Optional[ExperimentPlan] = None
    review_grounding: Optional[GroundingReport] = None
    hypothesis_grounding: Optional[GroundingReport] = None
    execution: Optional[ExecutionOutcome] = None
    critic: Optional[CriticReport] = None
    step_seconds: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    @property
    def execution_status(self) -> str:
        return self.execution.final_status if self.execution else Status.NOT_RUN

    @property
    def sources(self) -> List[Source]:
        """Papers the hypothesis actually cites, with ids and urls."""
        return self.hypothesis_grounding.cited_sources if self.hypothesis_grounding else []

    def to_dict(self) -> Dict[str, Any]:
        def obj(x):
            return dataclasses.asdict(x) if x is not None else None
        return {
            "topic": self.topic,
            "created_at": self.created_at,
            "llm_mode": self.llm_mode,
            "status": self.status,
            "execution_status": self.execution_status,
            "evidence": [
                {"id": e.paper.uid, "title": e.paper.title, "url": e.paper.url, "score": round(e.score, 4),
                 "status": publication_status(e.paper)}
                for e in self.evidence
            ],
            "sources_cited_by_hypothesis": obj_list(self.sources),
            "review": obj(self.review),
            "review_grounding": obj(self.review_grounding),
            "hypothesis": obj(self.hypothesis),
            "hypothesis_grounding": obj(self.hypothesis_grounding),
            "experiment": obj(self.experiment),
            "execution": self.execution.to_dict() if self.execution else None,
            "critic": self.critic.to_dict() if self.critic else None,
            "step_seconds": self.step_seconds,
            "warnings": self.warnings,
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        path, n = directory / f"{stem}.json", 1
        while path.exists():                      # two runs in the same second must not overwrite each other
            n += 1
            path = directory / f"{stem}_{n}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def obj_list(items) -> List[Dict[str, Any]]:
    return [dataclasses.asdict(i) for i in items]


class ResearchWorkflow:
    def __init__(
        self,
        retriever: PaperRetriever,
        reviewer: LiteratureReviewerAgent,
        hypothesis_agent: HypothesisGeneratorAgent,
        executor: CodeExecutorAgent,
        llm_mode: str,
        top_k: int = 5,
        sandbox: Any = None,            # a DockerSandbox; None disables execution
        max_repairs: int = 2,
        char_budget: int = 6000,
        critic_alpha: float = 0.05,
        memory: Any = None,                                   # a MemoryStore; None = no memory
        memory_budget: int = 1200,
        before_execute: Optional[Callable[[], None]] = None,  # e.g. unload the language model to free RAM
        figures_dir: Any = None,                              # folder of extracted figure captions
        published_only: bool = False,                         # skip preprints when retrieving
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,   # called as each step starts and ends
    ) -> None:
        self.rag = RagPipeline(retriever, char_budget=char_budget, figures_dir=figures_dir, published_only=published_only)
        self.reviewer, self.hypothesis_agent, self.executor = reviewer, hypothesis_agent, executor
        self.llm_mode, self.top_k = llm_mode, top_k
        self.sandbox, self.max_repairs = sandbox, max_repairs
        self.critic = StatisticalCritic(critic_alpha)
        self.memory, self.memory_budget, self.before_execute = memory, memory_budget, before_execute
        self.on_progress = on_progress
        self._pipeline = (
            RunnableLambda(self._timed("retrieve", self._step_retrieve, self._emit))
            | RunnableLambda(self._timed("review", self._step_review, self._emit))
            | RunnableLambda(self._timed("hypothesis", self._step_hypothesis, self._emit))
            | RunnableLambda(self._timed("experiment_design", self._step_experiment, self._emit))
            | RunnableLambda(self._timed("execute", self._step_execute, self._emit))
            | RunnableLambda(self._timed("critic", self._step_critic, self._emit))
        )

    def _emit(self, event: ProgressEvent) -> None:
        _notify(self.on_progress, event)

    @staticmethod
    def _timed(name: str, step: Callable[[dict], dict], notify: Optional[Callable[[ProgressEvent], None]] = None) -> Callable[[dict], dict]:
        """Record how long each step took (shown on the dashboard) and, if asked, report the real start and end."""
        def wrapper(state: dict) -> dict:
            started = time.monotonic()
            _notify(notify, ProgressEvent(name, "running"))
            failure: Optional[Exception] = None
            try:
                state = step(state)
                return state
            except Exception as exc:
                failure = exc
                raise
            finally:
                seconds = round(time.monotonic() - started, 2)
                state.setdefault("timings", {})[name] = seconds
                if failure is not None:
                    _notify(notify, ProgressEvent(name, "failed", seconds, f"{type(failure).__name__}: {failure}"))
                else:
                    phase, message = _step_outcome(name, state)
                    _notify(notify, ProgressEvent(name, phase, seconds, message))
        return wrapper

    # ---- steps ------------------------------------------------------------
    def _step_retrieve(self, state: dict) -> dict:
        state["context"] = self.rag.prepare(state["topic"], k=self.top_k)
        state["evidence"] = state["context"].evidence
        if not state["evidence"]:
            state["stop"] = True
            state["warnings"].append("No relevant papers found in the local index. Run 'ingest' first.")
        elif state["context"].dropped:
            state["warnings"].append(f"{state['context'].dropped} lower-ranked paper(s) were left out to fit the prompt budget.")
        return state

    def _step_review(self, state: dict) -> dict:
        if state.get("stop"):
            return state
        review = state["review"] = self.reviewer.run(state["topic"], state["evidence"])
        state["review_grounding"] = self.rag.check(review.summary, state["context"])
        if review.unknown_citations:
            state["warnings"].append(f"Reviewer cited unretrieved ids: {review.unknown_citations}")
        if not review.cited_ids:
            state["warnings"].append("Reviewer cited no retrieved paper by id, so its claims are not traceable.")
        for sentence in state["review_grounding"].weakly_supported[:2]:
            state["warnings"].append(f"Review sentence with weak lexical support from its cited paper: {sentence!r}")
        return state

    def _step_hypothesis(self, state: dict) -> dict:
        if state.get("stop"):
            return state
        state["memory_text"] = ""
        if self.memory is not None and self.memory_budget > 0:
            state["memory_text"] = self.memory.render(state["topic"], self.memory_budget, include_mock="TEST MODE" in self.llm_mode)
        hyp = state["hypothesis"] = self.hypothesis_agent.run(
            state["topic"], state["review"], state["evidence"], memory_text=state["memory_text"])
        state["hypothesis_grounding"] = self.rag.check(hyp.raw_text, state["context"])
        if hyp.unknown_citations:
            state["warnings"].append(f"Hypothesis cited unretrieved ids: {hyp.unknown_citations}")
        if not hyp.cited_ids:
            state["warnings"].append("Hypothesis cited no retrieved paper by id, so it is not traceable to evidence.")
        for sentence in state["hypothesis_grounding"].weakly_supported[:2]:
            state["warnings"].append(f"Hypothesis sentence with weak lexical support from its cited paper: {sentence!r}")
        return state

    def _step_experiment(self, state: dict) -> dict:
        if state.get("stop"):
            return state
        exp = state["experiment"] = self.executor.prepare_experiment(state["hypothesis"], memory_text=state.get("memory_text", ""))
        if exp.note:
            state["warnings"].append(f"Experiment design: {exp.note}")
        if exp.truncated:
            state["warnings"].append("Generated code looks cut off (token limit reached). Raise LLM_MAX_TOKENS in .env.")
        if not exp.syntax_ok:
            state["warnings"].append(f"Generated code has a syntax error: {exp.syntax_error}")
        state["warnings"].extend(f"Code check: {w}" for w in exp.advisory_warnings)
        return state

    def _step_execute(self, state: dict) -> dict:
        if state.get("stop"):
            return state
        if self.sandbox is None:
            state["warnings"].append("Experiment was not executed (execution disabled for this run).")
            return state
        if self.before_execute is not None:
            try:
                self.before_execute()
            except Exception as exc:                          # freeing memory is a courtesy, never a reason to stop
                logger.warning("Could not run the before-execute hook: %s", exc)
        availability = self.sandbox.check_available()
        if not availability.ok:
            state["execution"] = ExecutionOutcome([], Status.SANDBOX_UNAVAILABLE, availability.message)
            state["warnings"].append(f"Experiment was NOT executed: {availability.message}")
            return state
        outcome = state["execution"] = run_with_repair(
            self.sandbox, self.executor, state["hypothesis"], state["experiment"].code, self.max_repairs)
        if outcome.final_status != Status.SUCCESS:
            state["warnings"].append(
                f"Experiment ended with {outcome.final_status} after {len(outcome.attempts)} attempt(s): {outcome.message}")
        return state

    def _step_critic(self, state: dict) -> dict:
        """Check the statistics of the experiment output. Runs only if something was executed."""
        outcome = state.get("execution")
        if state.get("stop") or outcome is None or outcome.final_result is None:
            return state
        last = outcome.final_result
        report = state["critic"] = self.critic.evaluate(last.stdout, last.status, last.output_truncated)
        state["warnings"].extend(
            f"Critic: {f.message}" for f in report.findings if f.severity in ("error", "warning") and f.code != "NO_RESULT")
        return state

    # ---- public -------------------------------------------------------------
    def run(self, topic: str) -> WorkflowResult:
        if not topic or not topic.strip():
            raise ValueError("Topic must be a non-empty string.")
        state = self._pipeline.invoke({"topic": topic.strip(), "warnings": [], "timings": {}})
        result = WorkflowResult(
            topic=state["topic"],
            llm_mode=self.llm_mode,
            status="NO_EVIDENCE" if state.get("stop") else "COMPLETED",
            evidence=state.get("evidence", []),
            review=state.get("review"),
            hypothesis=state.get("hypothesis"),
            experiment=state.get("experiment"),
            review_grounding=state.get("review_grounding"),
            hypothesis_grounding=state.get("hypothesis_grounding"),
            execution=state.get("execution"),
            critic=state.get("critic"),
            step_seconds=state.get("timings", {}),
            warnings=state["warnings"],
        )
        if self.memory is not None and result.status == "COMPLETED":
            try:
                self.memory.add(
                    topic=result.topic, hypothesis=(result.hypothesis.statement if result.hypothesis else "")[:300],
                    design=(result.experiment.design_summary or "free-form code") if result.experiment else "none",
                    execution_status=result.execution_status,
                    critic_verdict=result.critic.verdict if result.critic else "none", llm_mode=self.llm_mode)
            except OSError as exc:
                logger.warning("Could not save to memory: %s", exc)
        return result
