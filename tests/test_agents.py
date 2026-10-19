"""Agent + workflow tests. Everything here runs in TEST MODE (rule-based stub) or with
hand-written fake models. Passing these tests proves the PLUMBING works, not that any
model reasons well. Papers are FICTIONAL TEST DATA."""
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from config.settings import load_settings
from src.agents.base import extract_citations, split_citations
from src.agents.code_executor import PENDING_STATUS, CodeExecutorAgent, analyse_code, extract_code
from src.agents.hypothesis_generator import Hypothesis, HypothesisGeneratorAgent, parse_hypothesis_text
from src.agents.literature_reviewer import LiteratureReview, LiteratureReviewerAgent
from src.ingestion.paper_store import PaperStore
from src.llm.factory import describe_llm, get_llm
from src.llm.stub_model import RuleBasedStubChatModel
from src.main import main
from src.models import Paper, RetrievedPaper
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.retriever import PaperRetriever
from src.retrieval.vector_store import NumpyVectorStore
from src.workflow import ResearchWorkflow

import pytest


def rp(pid, title, abstract="Abstract text.", score=0.5):
    return RetrievedPaper(Paper("arxiv", pid, title, abstract, url=f"https://arxiv.org/abs/{pid}"), score)


EVIDENCE = [rp("1", "Graph networks"), rp("2", "Bayesian search")]
REVIEW = LiteratureReview(summary="x")


# ------------------------------------------------------------- citations
def test_citation_extraction_and_validation():
    text = "Claim [arxiv:1] and [arxiv:2], again [arxiv:1], fake [arxiv:999]."
    assert extract_citations(text) == ["arxiv:1", "arxiv:2", "arxiv:999"]
    valid, unknown = split_citations(text, ["arxiv:1", "arxiv:2"])
    assert valid == ["arxiv:1", "arxiv:2"] and unknown == ["arxiv:999"]


# ------------------------------------------------------------- agents (stub)
def test_reviewer_in_test_mode_is_labelled_and_cites_only_retrieved():
    review = LiteratureReviewerAgent(RuleBasedStubChatModel()).run("graphs", EVIDENCE)
    assert "TEST MODE" in review.summary
    assert review.cited_ids == ["arxiv:1", "arxiv:2"] and review.unknown_citations == []


def test_hypothesis_agent_parses_sections():
    h = HypothesisGeneratorAgent(RuleBasedStubChatModel()).run("graphs", REVIEW, EVIDENCE)
    assert "TEST MODE" in h.statement
    assert h.test_plan and h.cited_ids


def test_code_agent_returns_valid_unexecuted_code():
    hyp = Hypothesis("s", "r", "plan", "raw")
    plan = CodeExecutorAgent(RuleBasedStubChatModel()).prepare_experiment(hyp)
    assert plan.syntax_ok and "synthetic" in plan.code.lower()
    assert plan.execution_status == PENDING_STATUS


# ------------------------------------------------------------- hallucination guard
def test_unknown_citation_is_flagged():
    fake = FakeListChatModel(responses=["Great results [arxiv:424242]."])
    review = LiteratureReviewerAgent(fake).run("t", EVIDENCE)
    assert review.unknown_citations == ["arxiv:424242"] and review.cited_ids == []


def test_model_failure_becomes_agent_error():
    from src.agents.base import AgentError
    from langchain_core.language_models.chat_models import BaseChatModel

    class Broken(RuleBasedStubChatModel):
        def _generate(self, *a, **k):
            raise ConnectionError("ollama not running")

    with pytest.raises(AgentError, match="ollama not running"):
        LiteratureReviewerAgent(Broken()).run("t", EVIDENCE)


def test_agents_refuse_empty_evidence():
    with pytest.raises(ValueError):
        LiteratureReviewerAgent(RuleBasedStubChatModel()).run("t", [])


# ------------------------------------------------------------- parsing / code analysis
def test_parse_hypothesis_fallback_when_format_ignored():
    assert parse_hypothesis_text("just some prose")[0] == "just some prose"


def test_extract_code_and_syntax_check():
    assert extract_code("text\n```python\nprint(1)\n```\nmore") == "print(1)"
    ok, err, _ = analyse_code("def broken(:\n  pass")
    assert not ok and err
    _, _, warnings = analyse_code("import subprocess\nimport json")
    assert any("subprocess" in w for w in warnings) and not any("json" in w for w in warnings)


# ------------------------------------------------------------- workflow
@pytest.fixture
def workflow(tmp_path):
    corpus = [
        Paper("arxiv", "1", "Graph neural networks", "Message passing on graph neural networks.", url="u1"),
        Paper("arxiv", "2", "Bayesian optimisation", "Gaussian process hyperparameter search.", url="u2"),
    ]
    store = PaperStore(tmp_path / "p.db")
    store.upsert_many(corpus)
    emb = HashingEmbedder()
    retriever = PaperRetriever(store, emb, NumpyVectorStore())
    retriever.index_papers(corpus)
    llm = RuleBasedStubChatModel()
    yield ResearchWorkflow(
        retriever, LiteratureReviewerAgent(llm), HypothesisGeneratorAgent(llm),
        CodeExecutorAgent(llm), llm_mode="TEST MODE", top_k=3,
    )
    store.close()


def test_workflow_end_to_end_test_mode(workflow, tmp_path):
    result = workflow.run("graph neural networks")
    assert result.status == "COMPLETED_WEEK1"
    assert result.evidence[0].paper.uid == "arxiv:1"
    assert result.experiment.syntax_ok
    data = result.to_dict()
    assert data["execution_status"] == PENDING_STATUS and data["llm_mode"] == "TEST MODE"
    assert result.save(tmp_path / "runs").exists()


def test_workflow_stops_cleanly_without_evidence(workflow):
    result = workflow.run("quantum chromodynamics lattice")
    assert result.status == "NO_EVIDENCE"
    assert result.review is None and result.hypothesis is None and result.warnings


def test_workflow_rejects_blank_topic(workflow):
    with pytest.raises(ValueError):
        workflow.run("  ")


# ------------------------------------------------------------- factory / CLI
def test_factory_default_is_test_mode():
    s = load_settings(env={}, load_env_file=False)
    assert isinstance(get_llm(s), RuleBasedStubChatModel)
    assert "TEST MODE" in describe_llm(s)


def test_cli_status_and_empty_search(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    assert main(["status"]) == 0
    assert "Papers stored     : 0" in capsys.readouterr().out
    assert main(["run", "--topic", "anything"]) == 0
    out = capsys.readouterr().out
    assert "TEST MODE" in out and "NO_EVIDENCE" in out

# ------------------------------------------------------------- truncated / untraceable model output
def test_unclosed_code_fence_is_detected_and_cleaned():
    from src.agents.code_executor import is_truncated
    raw = "```python\nimport torch\ndef f():\n    # cut off"
    assert is_truncated(raw) and not is_truncated("```python\nprint(1)\n```")
    assert extract_code(raw).startswith("import torch")      # fence marker removed


def test_third_party_imports_are_flagged_even_when_code_does_not_parse():
    ok, _, warnings = analyse_code("import torch\nimport numpy\nimport json\ndef f():\n    # cut off")
    assert not ok
    assert any("torch" in w for w in warnings)
    assert not any("numpy" in w or "json" in w for w in warnings)


def test_workflow_warns_about_untraceable_and_truncated_output(workflow):
    fake = FakeListChatModel(responses=[
        "A summary with no citations.",
        "HYPOTHESIS: h\nRATIONALE: r\nTEST PLAN: p",
        "```python\nimport torch\nx = (",
    ])
    workflow.reviewer = LiteratureReviewerAgent(fake)
    workflow.hypothesis_agent = HypothesisGeneratorAgent(fake)
    workflow.executor = CodeExecutorAgent(fake)
    text = " | ".join(workflow.run("graph neural networks").warnings)
    assert "Reviewer cited no retrieved paper" in text
    assert "Hypothesis cited no retrieved paper" in text
    assert "cut off" in text and "torch" in text
