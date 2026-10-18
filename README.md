# Autonomous Multimodal Scientific Discovery Agent

MSc Data Science internship project. **Current status: Week 1 (foundation) complete and tested; Week 2+ pending.**

## What works today (Week 1)
| Capability | Status |
|---|---|
| arXiv retrieval, parsing, SQLite storage | Implemented; parsing/retry logic unit-tested on fixture data. **Live arXiv call: run it yourself (see testing guide).** |
| Local vector index + retrieval (lexical hashing embedder) | Implemented, tested |
| Literature Reviewer / Hypothesis Generator / Code Executor agents (LangChain) | Implemented; tested **in TEST MODE only** (rule-based mock, not an LLM) and against a fake Ollama server |
| Workflow: retrieve -> review -> hypothesis -> experiment code | Implemented, tested in TEST MODE |
| Free local LLM via Ollama | Code path + preflight written and tested against a **fake** Ollama server. **Not yet run against real Ollama on your PC.** |
| Free semantic embeddings via Ollama | Same status as above (optional; default embeddings are lexical) |
| Hardware check (`check-system`) | Implemented; Windows RAM detection untested on Windows |
| PubMed ingestion | Placeholder only |
| RAG grounding, Docker sandbox, execution, error feedback | **Pending (Week 2)** |
| Critic agent, multimodal, dashboard, drafts | **Pending (Weeks 3-4)** |
| Reinforcement learning | **Not implemented.** Planned separately; retries are not RL. |

Nothing in this project executes generated code yet.

## Quick start (Windows PowerShell)
See `docs/setup_guide.md`. Short version:
**Cost: everything is free.** No API key, account or billing is needed. See `docs/setup_guide.md` for full Windows + VS Code steps, Ollama install and hardware guidance.
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # use requirements.txt only if you will never use Ollama
Copy-Item .env.example .env
python -m pytest -q
python -m src.main check-system      # can my PC run a local LLM?
python -m src.main status
python -m src.main ingest --query "graph neural networks" --max-results 10
python -m src.main run --topic "graph neural networks"
```
