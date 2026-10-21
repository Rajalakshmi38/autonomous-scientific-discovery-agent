"""Command-line entry point.

Commands (run from the project root):
    python -m src.main status
    python -m src.main ingest  --query "graph neural networks" --max-results 10
    python -m src.main search  --query "graph neural networks" -k 5
    python -m src.main run     --topic "graph neural networks" [--no-execute]
    python -m src.main sandbox-build | sandbox-status | sandbox-test
    python -m src.main reindex
    python -m src.main check-system     # can my PC run a local LLM?
    python -m src.main check-llm        # is Ollama running and the model pulled?
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from typing import Optional, Sequence

from config.settings import PROJECT_ROOT, ConfigError, Settings, load_settings
from src.agents.code_executor import CodeExecutorAgent
from src.agents.hypothesis_generator import HypothesisGeneratorAgent
from src.agents.literature_reviewer import LiteratureReviewerAgent
from src.execution.sandbox import DockerSandbox, SandboxConfig, Status
from src.execution.self_test import run_self_test
from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.base import IngestionError
from src.ingestion.paper_store import PaperStore
from src.llm.factory import available_providers, describe_llm, get_llm, get_provider_spec
from src.llm.ollama_check import check_ollama
from src.retrieval.embeddings import EmbeddingError, get_embedder
from src.retrieval.retriever import PaperRetriever
from src.retrieval.vector_store import NumpyVectorStore, VectorStoreError
from src.utils.logging_config import setup_logging
from src.utils.system_info import get_system_info, recommend
from src.workflow import ResearchWorkflow

logger = logging.getLogger("main")


def build_retriever(settings: Settings) -> tuple[PaperStore, PaperRetriever]:
    store = PaperStore(settings.papers_db_path)
    embedder = get_embedder(settings)
    vectors = NumpyVectorStore(settings.vector_dir, embedder.name)
    return store, PaperRetriever(store, embedder, vectors, settings.retrieval_min_score)


def cmd_status(settings: Settings, _args) -> int:
    store, retriever = build_retriever(settings)
    print(f"LLM provider      : {describe_llm(settings)}")
    print(f"Embedding backend : {retriever.embedder.name}")
    print(f"Papers stored     : {store.count()}")
    print(f"Vectors indexed   : {retriever.vector_store.count()}")
    if get_provider_spec(settings.llm_provider).needs_local_server:
        print(f"Ollama server     : {check_ollama(settings.ollama_base_url, settings.ollama_model).message}")
    print(f"Docker sandbox    : {make_sandbox(settings).check_available().message}")
    store.close()
    return 0


def make_sandbox(settings: Settings) -> DockerSandbox:
    return DockerSandbox(SandboxConfig.from_settings(settings))


def cmd_sandbox_build(settings: Settings, _args) -> int:
    print(f"Building image '{settings.sandbox_image}' (first time takes a few minutes; needs internet)...")
    code = make_sandbox(settings).build_image(PROJECT_ROOT / "sandbox")
    print("Image built. Next: python -m src.main sandbox-test" if code == 0 else f"Build failed (exit code {code}).")
    return 0 if code == 0 else 1


def cmd_sandbox_status(settings: Settings, _args) -> int:
    status = make_sandbox(settings).check_available()
    print(status.message)
    return 0 if status.ok else 1


def cmd_sandbox_test(settings: Settings, _args) -> int:
    sandbox = make_sandbox(settings)
    status = sandbox.check_available()
    if not status.ok:
        print(f"Cannot test the sandbox: {status.message}")
        return 1
    print("Running restriction probes inside the sandbox (about 1 minute)...\n")
    results = run_self_test(sandbox)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name:22s} {r.what}\n       {r.detail}")
    failed = [r.name for r in results if not r.passed]
    if failed:
        print(f"\n{len(failed)} probe(s) FAILED: {', '.join(failed)}. Do NOT treat the sandbox as secure.")
        return 1
    print("\nAll probes passed. The restrictions above work on this machine.")
    print("This is evidence, not a guarantee: keep running only code you accept the risk of running in a container.")
    return 0


def cmd_check_system(settings: Settings, _args) -> int:
    info = get_system_info()
    rec = recommend(info)
    print(f"OS   : {info.os_name}")
    print(f"CPU  : {info.cpu_count} logical cores")
    print(f"RAM  : {info.ram_gb if info.ram_gb is not None else 'unknown'} GB")
    print(f"GPU  : {info.gpu_name or 'no NVIDIA GPU detected'}" + (f" ({info.vram_gb} GB VRAM)" if info.vram_gb else ""))
    print(f"\nGuidance ({rec.tier}): {rec.summary}")
    if rec.suggested_models:
        print("Example models (confirm names at https://ollama.com/library): " + ", ".join(rec.suggested_models))
    print("\nThis is a rule of thumb. The real test is: ollama run <model> \"hello\"")
    return 0


def cmd_check_llm(settings: Settings, _args) -> int:
    spec = get_provider_spec(settings.llm_provider)
    print(f"Configured provider: {describe_llm(settings)}")
    if spec.is_mock:
        print("Mock/test mode needs nothing installed. It does NOT use a language model.")
        return 0
    if spec.needs_local_server:
        status = check_ollama(settings.ollama_base_url, settings.ollama_model)
        print(status.message)
        if status.reachable and status.models:
            print("Downloaded models: " + ", ".join(status.models))
        return 0 if status.ok else 1
    return 0


def cmd_ingest(settings: Settings, args) -> int:
    client = ArxivClient(
        timeout=settings.arxiv_timeout_seconds,
        retries=settings.arxiv_retries,
        delay_seconds=settings.arxiv_delay_seconds,
        contact=settings.arxiv_contact,
        raw_dir=settings.raw_dir,
    )
    papers = client.search(args.query, max_results=args.max_results, category=args.category)
    if not papers:
        print("arXiv returned no papers for this query. Try different or fewer keywords.")
        return 0
    store, retriever = build_retriever(settings)
    try:
        store.upsert_many(papers)
        retriever.index_papers(papers)
    finally:
        store.close()
    print(f"Stored and indexed {len(papers)} paper(s):")
    for p in papers:
        print(f"  [{p.uid}] {p.title[:90]} ({p.year})")
    return 0


def cmd_search(settings: Settings, args) -> int:
    store, retriever = build_retriever(settings)
    try:
        results = retriever.retrieve(args.query, k=args.k)
    finally:
        store.close()
    if not results:
        print("No relevant papers in the local index. Run 'ingest' first, or rephrase the query.")
        return 0
    for r in results:
        print(f"{r.score:.3f}  [{r.paper.uid}] {r.paper.title[:80]}\n       {r.paper.url}")
    return 0


def cmd_reindex(settings: Settings, _args) -> int:
    """Rebuild the index from stored papers; works even if the old index used a different embedder."""
    store = PaperStore(settings.papers_db_path)
    try:
        embedder = get_embedder(settings)
        vectors = NumpyVectorStore(settings.vector_dir, embedder.name, load_existing=False)
        n = PaperRetriever(store, embedder, vectors, settings.retrieval_min_score).reindex_all()
    finally:
        store.close()
    print(f"Re-indexed {n} paper(s).")
    return 0


def cmd_run(settings: Settings, args) -> int:
    if args.provider:
        get_provider_spec(args.provider)   # validates the name
        settings = dataclasses.replace(settings, llm_provider=args.provider)
    spec = get_provider_spec(settings.llm_provider)
    if spec.needs_local_server:
        status = check_ollama(settings.ollama_base_url, settings.ollama_model)
        if not status.ok:
            print(f"Cannot start: {status.message}", file=sys.stderr)
            print("Tip: run with --provider test to try the pipeline in clearly labelled mock mode.", file=sys.stderr)
            return 1
    if settings.is_test_mode:
        print("=" * 72)
        print("TEST MODE: the LLM is a rule-based mock. Output below is NOT genuine reasoning.")
        print("For real (free, local) model output: install Ollama, then see docs/setup_guide.md.")
        print("=" * 72)
    store, retriever = build_retriever(settings)
    try:
        llm = get_llm(settings)
        sandbox = None if args.no_execute else make_sandbox(settings)
        workflow = ResearchWorkflow(
            retriever=retriever,
            reviewer=LiteratureReviewerAgent(llm, settings.citation_retries),
            hypothesis_agent=HypothesisGeneratorAgent(llm, settings.citation_retries),
            executor=CodeExecutorAgent(llm),
            llm_mode=describe_llm(settings),
            top_k=args.k,
            sandbox=sandbox,
            max_repairs=settings.max_repair_retries,
            char_budget=settings.context_char_budget,
        )
        result = workflow.run(args.topic)
    finally:
        store.close()

    path = result.save(settings.runs_dir)
    print(f"\nStatus: {result.status}   (LLM: {result.llm_mode})")
    print(f"Execution: {result.execution_status}")
    for w in result.warnings:
        print(f"WARNING: {w}")
    if result.status == "COMPLETED":
        print("\n--- Evidence retrieved ---")
        for e in result.evidence:
            print(f"[{e.paper.uid}] {e.paper.title[:80]}  score={e.score:.3f}\n    {e.paper.url}")
        print(f"\n--- Literature review ---\n{result.review.summary}")
        print(f"\n--- Hypothesis ---\n{result.hypothesis.statement}")
        print("\n--- Sources cited by the hypothesis ---")
        if result.sources:
            for src in result.sources:
                print(f"[{src.id}] {src.title[:80]}\n    {src.url}")
        else:
            print("(none: the hypothesis is not traceable to the retrieved papers)")
        print(f"\n--- Experiment code (first version) ---\n{result.experiment.code}")
        if result.execution and result.execution.attempts:
            print("\n--- Sandbox execution ---")
            for a in result.execution.attempts:
                r = a.result
                print(f"Attempt {a.number}: {r.status} (exit {r.exit_code}, {r.duration_seconds}s) {r.message}")
                if a.note:
                    print(f"  note: {a.note}")
            last = result.execution.final_result
            if last.stdout.strip():
                print(f"\nOutput of the last run:\n{last.stdout.strip()[:2000]}")
            if last.stderr.strip() and not last.ok:
                print(f"\nError output of the last run:\n{last.stderr.strip()[-1500:]}")
            if last.ok:
                print("\nNote: these numbers come from model-written code. They are real outputs of that code, "
                      "not verified research findings.")
    print(f"\nSaved run record: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.main", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show configuration and index size").set_defaults(func=cmd_status)

    p = sub.add_parser("ingest", help="fetch papers from arXiv, store and index them")
    p.add_argument("--query", required=True)
    p.add_argument("--max-results", type=int, default=10)
    p.add_argument("--category", default=None, help="e.g. cs.LG")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="search the local index")
    p.add_argument("--query", required=True)
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("run", help="run the Week 1 multi-agent workflow")
    p.add_argument("--topic", required=True)
    p.add_argument("-k", type=int, default=5, help="papers to retrieve (use 3 with small local models)")
    p.add_argument("--provider", choices=available_providers(), default=None,
                   help="override LLM_PROVIDER for this run")
    p.add_argument("--no-execute", action="store_true", help="prepare the code but do not run it in the sandbox")
    p.set_defaults(func=cmd_run)

    sub.add_parser("check-system", help="report RAM/GPU and what size of local LLM is realistic").set_defaults(func=cmd_check_system)
    sub.add_parser("check-llm", help="check that the configured LLM provider is usable").set_defaults(func=cmd_check_llm)
    sub.add_parser("sandbox-build", help="build the Docker sandbox image (once)").set_defaults(func=cmd_sandbox_build)
    sub.add_parser("sandbox-status", help="is Docker running and the image built?").set_defaults(func=cmd_sandbox_status)
    sub.add_parser("sandbox-test", help="verify the sandbox restrictions on real Docker").set_defaults(func=cmd_sandbox_test)
    sub.add_parser("reindex", help="rebuild the vector index from stored papers").set_defaults(func=cmd_reindex)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):          # avoid Windows cp1252 crashes on non-ASCII titles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings()
        setup_logging(settings.log_level, settings.logs_dir)
        return args.func(settings, args)
    except (ConfigError, ValueError) as exc:
        print(f"Configuration/input error: {exc}", file=sys.stderr)
        return 2
    except (IngestionError, VectorStoreError, EmbeddingError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # last resort: readable message, details in the log file
        logger.exception("Unexpected failure")
        print(f"Unexpected error: {exc} (see storage/logs/app.log)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
