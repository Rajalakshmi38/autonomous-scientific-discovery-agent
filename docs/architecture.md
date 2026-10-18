# Architecture (Week 1 built; later weeks planned)

```
 arXiv API --> ArxivClient --> PaperStore (SQLite) --+
                                                     +--> PaperRetriever --> LiteratureReviewer
 (PubMed: placeholder)       Embedder + VectorStore -+                          |
                                                                      HypothesisGenerator
                                                                                |
                                                                       CodeExecutor (prepares code only)
                                                                                |
                                                          [Week 2: Docker sandbox + repair loop]
                                                          [Week 3: Critic, VLM] [Week 4: draft, dashboard]
```

## Design decisions
- **LangChain (`langchain-core` only)**: agents are `prompt | llm | parser` chains; the workflow is a `RunnableLambda` pipeline. Only the core package is required, which keeps dependencies small. Provider packages (`langchain-ollama`) are optional.
- **Plain `requests` + stdlib XML for arXiv**: fewer dependencies than a client wrapper, and fully testable with fixture XML.
- **SQLite for metadata, NumPy for vectors**: zero setup. `VectorStore` is an abstract interface so a Milvus/Pinecone/Chroma adapter can replace `NumpyVectorStore` without touching agents.
- **Embeddings (all free)**: default `hashing` is lexical only (not semantic). `ollama` (nomic-embed-text) gives semantic embeddings via the same local server with no PyTorch; `sentence-transformers` is the PyTorch option. Switch by `EMBEDDING_BACKEND` and run `reindex`.
- **Provider registry** (`src/llm/factory.py`): each LLM provider is one `ProviderSpec`. Adding another provider later means one build function plus one name in `VALID_LLM_PROVIDERS`; agents depend only on LangChain's `BaseChatModel`. Only free options (`ollama`, mock `test`) ship by default and no API key is read anywhere (a test guards this).
- **Local-model safeguards**: explicit `num_ctx` (avoids silent prompt truncation), `num_predict` cap, request timeout, and an Ollama preflight check before a run.
- **Test mode**: `RuleBasedStubChatModel` (a mock) returns labelled templates. It exists to test plumbing, never to demonstrate reasoning. Every saved run records which LLM mode produced it.
- **Citation guard**: agents list cited ids and flag any id that was not in the retrieved set (`unknown_citations`). This detects some fabricated citations; it does not verify that a claim is actually supported.
- **Changes from the suggested structure**: added `src/models.py`, `src/llm/`, `src/agents/base.py`, `src/ingestion/{base,paper_store}.py`, `src/retrieval/retriever.py`, `src/workflow.py`; `rag_pipeline.py`, `sandbox.py`, `feedback_loop.py` and `sandbox/Dockerfile` are deferred to Week 2. `data/` and `storage/` are created at runtime and git-ignored.

## Trust boundaries
Paper text and model output are untrusted. Prompts wrap paper text in delimiters and instruct the model to ignore instructions inside it (a mitigation, not a guarantee). Generated code is never run in Week 1.
