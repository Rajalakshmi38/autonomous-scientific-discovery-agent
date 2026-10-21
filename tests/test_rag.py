"""RAG layer and citation-grounding tests. Papers are FICTIONAL TEST DATA."""
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.hypothesis_generator import HypothesisGeneratorAgent
from src.agents.literature_reviewer import LiteratureReview, LiteratureReviewerAgent
from src.ingestion.paper_store import PaperStore
from src.models import Paper, RetrievedPaper
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.rag_pipeline import RagContext, RagPipeline, Source
from src.retrieval.retriever import PaperRetriever
from src.retrieval.vector_store import NumpyVectorStore

import pytest

LONG = "graph neural network message passing " * 40


def rp(pid, title="Graph networks", abstract="Message passing graph neural networks for molecules."):
    return RetrievedPaper(Paper("arxiv", pid, title, abstract, url=f"https://arxiv.org/abs/{pid}", published="2024-05-01"), 0.5)


def context(*papers):
    return RagContext("t", list(papers), [Source(p.paper.uid, p.paper.title, p.paper.url) for p in papers])


@pytest.fixture
def pipeline(tmp_path):
    corpus = [Paper("arxiv", str(i), f"Graph neural networks study {i}", LONG, url=f"u{i}") for i in range(5)]
    store = PaperStore(tmp_path / "p.db")
    store.upsert_many(corpus)
    retriever = PaperRetriever(store, HashingEmbedder(), NumpyVectorStore())
    retriever.index_papers(corpus)
    yield retriever
    store.close()


def test_prepare_keeps_papers_within_budget_and_reports_drops(pipeline):
    ctx = RagPipeline(pipeline, char_budget=2000).prepare("graph neural networks", k=5)
    assert 1 <= len(ctx.evidence) < 5 and ctx.dropped == 5 - len(ctx.evidence)
    assert all(s.url for s in ctx.sources)


def test_prepare_always_keeps_the_top_paper_even_if_over_budget(pipeline):
    ctx = RagPipeline(pipeline, char_budget=10).prepare("graph neural networks", k=5)
    assert len(ctx.evidence) == 1


def test_prepare_with_no_matches_returns_empty_context(pipeline):
    ctx = RagPipeline(pipeline).prepare("quantum chromodynamics lattice", k=3)
    assert ctx.evidence == [] and ctx.sources == []


def test_grounding_report_lists_ids_and_urls():
    ctx = context(rp("1"), rp("2", "Bayesian optimisation", "Gaussian process hyperparameter search."))
    text = "Graph neural networks use message passing on molecules [arxiv:1]. Also see [arxiv:999]."
    report = RagPipeline(None).check(text, ctx)
    assert report.cited_ids == ["arxiv:1"] and report.unknown_ids == ["arxiv:999"]
    assert report.cited_sources[0].url == "https://arxiv.org/abs/1"
    assert not report.grounded                       # an unknown id means the answer is not fully grounded


def test_grounded_requires_at_least_one_valid_citation():
    ctx = context(rp("1"))
    pipe = RagPipeline(None)
    assert pipe.check("Message passing graph neural networks help molecules [arxiv:1].", ctx).grounded
    assert not pipe.check("No citations at all in this answer.", ctx).grounded


def test_weak_lexical_support_is_flagged_but_supported_sentences_are_not():
    ctx = context(rp("1"))
    pipe = RagPipeline(None)
    good = pipe.check("Message passing graph neural networks predict molecular properties [arxiv:1].", ctx)
    bad = pipe.check("Submarine cables transport underwater telecommunications traffic worldwide [arxiv:1].", ctx)
    assert good.weakly_supported == []
    assert len(bad.weakly_supported) == 1 and bad.grounded     # cited, valid, but suspicious


# ------------------------------------------------------------- citation retry in the agents
def test_reviewer_retries_once_when_nothing_is_cited():
    fake = FakeListChatModel(responses=["A summary with no citation.", "Better summary [arxiv:1]."])
    review = LiteratureReviewerAgent(fake, citation_retries=1).run("t", [rp("1")])
    assert review.cited_ids == ["arxiv:1"] and review.citation_retries_used == 1


def test_retry_is_bounded_and_can_be_disabled():
    fake = FakeListChatModel(responses=["still nothing cited"])
    assert LiteratureReviewerAgent(fake, citation_retries=2).run("t", [rp("1")]).citation_retries_used == 2
    assert LiteratureReviewerAgent(fake, citation_retries=0).run("t", [rp("1")]).citation_retries_used == 0


def test_hypothesis_agent_retries_for_citations():
    fake = FakeListChatModel(responses=[
        "HYPOTHESIS: h\nRATIONALE: r\nTEST PLAN: p",
        "HYPOTHESIS: h\nRATIONALE: supported by [arxiv:1]\nTEST PLAN: p",
    ])
    hyp = HypothesisGeneratorAgent(fake, citation_retries=1).run("t", LiteratureReview("s"), [rp("1")])
    assert hyp.cited_ids == ["arxiv:1"] and hyp.citation_retries_used == 1


def test_no_retry_when_the_first_answer_already_cites():
    fake = FakeListChatModel(responses=["Good [arxiv:1].", "should not be used"])
    assert LiteratureReviewerAgent(fake, citation_retries=3).run("t", [rp("1")]).citation_retries_used == 0
