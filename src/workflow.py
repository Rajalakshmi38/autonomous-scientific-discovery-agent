"""Week 1 workflow: retrieve -> review -> hypothesise -> prepare experiment code.

Built as a LangChain runnable pipeline (``RunnableLambda`` steps joined with ``|``).
Each step receives and returns a state dict. If no evidence is found the
pipeline stops early WITHOUT calling any model.

Execution is deliberately absent: it arrives in Week 2 with the Docker sandbox.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableLambda

from src.agents.code_executor import PENDING_STATUS, CodeExecutorAgent, ExperimentPlan
from src.agents.hypothesis_generator import Hypothesis, HypothesisGeneratorAgent
from src.agents.literature_reviewer import LiteratureReview, LiteratureReviewerAgent
from src.models import RetrievedPaper
from src.retrieval.retriever import PaperRetriever

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    topic: str
    llm_mode: str
    status: str                                   # COMPLETED_WEEK1 | NO_EVIDENCE
    evidence: List[RetrievedPaper] = field(default_factory=list)
    review: Optional[LiteratureReview] = None
    hypothesis: Optional[Hypothesis] = None
    experiment: Optional[ExperimentPlan] = None
    warnings: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        def obj(x):
            return x.__dict__ if x is not None else None
        return {
            "topic": self.topic,
            "created_at": self.created_at,
            "llm_mode": self.llm_mode,
            "status": self.status,
            "execution_status": PENDING_STATUS,
            "evidence": [
                {"id": e.paper.uid, "title": e.paper.title, "url": e.paper.url, "score": round(e.score, 4)}
                for e in self.evidence
            ],
            "review": obj(self.review),
            "hypothesis": obj(self.hypothesis),
            "experiment": obj(self.experiment),
            "warnings": self.warnings,
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


class ResearchWorkflow:
    def __init__(
        self,
        retriever: PaperRetriever,
        reviewer: LiteratureReviewerAgent,
        hypothesis_agent: HypothesisGeneratorAgent,
        executor: CodeExecutorAgent,
        llm_mode: str,
        top_k: int = 5,
    ) -> None:
        self.retriever, self.reviewer = retriever, reviewer
        self.hypothesis_agent, self.executor = hypothesis_agent, executor
        self.llm_mode, self.top_k = llm_mode, top_k
        self._pipeline = (
            RunnableLambda(self._step_retrieve)
            | RunnableLambda(self._step_review)
            | RunnableLambda(self._step_hypothesis)
            | RunnableLambda(self._step_experiment)
        )

    # ---- steps ------------------------------------------------------------
    def _step_retrieve(self, state: dict) -> dict:
        state["evidence"] = self.retriever.retrieve(state["topic"], k=self.top_k)
        if not state["evidence"]:
            state["stop"] = True
            state["warnings"].append("No relevant papers found in the local index. Run 'ingest' first.")
        return state

    def _step_review(self, state: dict) -> dict:
        if state.get("stop"):
            return state
        state["review"] = self.reviewer.run(state["topic"], state["evidence"])
        if state["review"].unknown_citations:
            state["warnings"].append(f"Reviewer cited unretrieved ids: {state['review'].unknown_citations}")
        return state

    def _step_hypothesis(self, state: dict) -> dict:
        if state.get("stop"):
            return state
        state["hypothesis"] = self.hypothesis_agent.run(state["topic"], state["review"], state["evidence"])
        if state["hypothesis"].unknown_citations:
            state["warnings"].append(f"Hypothesis cited unretrieved ids: {state['hypothesis'].unknown_citations}")
        return state

    def _step_experiment(self, state: dict) -> dict:
        if state.get("stop"):
            return state
        state["experiment"] = self.executor.prepare_experiment(state["hypothesis"])
        if not state["experiment"].syntax_ok:
            state["warnings"].append(f"Generated code has a syntax error: {state['experiment'].syntax_error}")
        return state

    # ---- public -------------------------------------------------------------
    def run(self, topic: str) -> WorkflowResult:
        if not topic or not topic.strip():
            raise ValueError("Topic must be a non-empty string.")
        state = self._pipeline.invoke({"topic": topic.strip(), "warnings": []})
        return WorkflowResult(
            topic=state["topic"],
            llm_mode=self.llm_mode,
            status="NO_EVIDENCE" if state.get("stop") else "COMPLETED_WEEK1",
            evidence=state.get("evidence", []),
            review=state.get("review"),
            hypothesis=state.get("hypothesis"),
            experiment=state.get("experiment"),
            warnings=state["warnings"],
        )
