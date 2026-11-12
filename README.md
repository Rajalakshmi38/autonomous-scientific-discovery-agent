# Autonomous Multimodal Scientific Discovery Agent

An AI research assistant that finds recent papers, summarises them, proposes a testable hypothesis, runs a small experiment in an isolated Docker container, and checks the result with statistics.

It uses free tools only: a local language model (Ollama), local search and Docker. No API key or paid account is needed. The experiments are small checks on toy datasets, not research findings.

## What it does

- Fetches papers from arXiv and PubMed into a local database. Saved searches can be refreshed with one command.
- Searches the papers by topic and cites them by identifier. Each paper is marked as published (PubMed, or a journal reference or DOI) or as a preprint, and retrieval can be limited to published papers.
- Three agents (LangChain): Literature Reviewer, Hypothesis Generator and Code Executor.
- Runs the experiment in a restricted Docker sandbox (no network, limited memory and time).
- A statistical critic reports the difference, a 95% confidence interval and a p-value.
- A small reinforcement learning component (a bandit) can learn how many papers to retrieve from the results of earlier runs.
- Extras: figure captions from PDFs, a short memory of earlier runs, a draft (or failed run report) in Markdown and LaTeX, a monitoring dashboard, and an evaluation on recent papers.

## Requirements

- Windows 10 or 11 with PowerShell, and Python 3.11 or 3.12
- Ollama (https://ollama.com/download) for the language model
- Docker Desktop for the sandbox
- About 8 GB of RAM is enough for a small model

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

In `.env`, set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=llama3.2:3b`. Start Docker Desktop, then check everything:

```powershell
python -m src.main check-llm
python -m src.main sandbox-build
python -m src.main sandbox-test
```

## Usage

```powershell
python -m src.main ingest --query "random forest" --max-results 30 --category cs.LG
python -m src.main run --topic "random forest number of trees" -k 3
python -m src.main draft
python -m src.main dashboard
```

Other commands (`python -m src.main --help` lists all of them):

- `ingest --source pubmed --query "..."` fetches papers from PubMed.
- `watch add ...` and `update` save searches and add only new papers. Schedule `update` with Windows Task Scheduler for regular refreshes.
- `search`, `status`, `summary` and `memory` show what is stored and what earlier runs did.
- `figures --paper arxiv:ID` extracts figure captions and page images from a paper.
- `evaluate` runs the agent on recent papers that are not in the index and writes a rating sheet.
- `run --topic "..." --adaptive` lets the bandit choose how many papers to retrieve and learn from the result. `bandit` shows what it has learned.
- `search --published-only` skips preprints.
- `run --topic "..." --provider test` tries everything with a mock model, clearly labelled TEST MODE.

Each run is saved in `storage/runs/`. The dashboard needs `pip install -r requirements-dashboard.txt`, and figure extraction needs `pip install -r requirements-figures.txt`.

## Settings

Settings are read from `.env` (see `.env.example` for all of them). The main ones:

- `LLM_PROVIDER` and `OLLAMA_MODEL`: which language model to use.
- `EMBEDDING_BACKEND`: `ollama` for semantic search, or `hashing` for simple word matching.
- `SANDBOX_TIMEOUT_SECONDS`: time limit for one experiment (default 120).
- `RETRIEVAL_PUBLISHED_ONLY`: `true` skips preprints.
- `VECTOR_BACKEND`: `local` (default) or `milvus`. Milvus needs a running Milvus server (`MILVUS_URI`) and `pip install -r requirements-milvus.txt`.

## Project structure

```
config/settings.py        settings
src/main.py               command-line interface
src/workflow.py           runs the steps in order
src/agents/               reviewer, hypothesis generator, code executor
src/experiments/          checked experiment form and script template
src/critic/               statistical critic
src/memory/               note about earlier runs
src/ingestion/            arXiv, PubMed, saved searches, figure extraction
src/retrieval/            embeddings, vector index (local or Milvus), search, citation checks
src/rl/                   strategy bandit
src/execution/            Docker sandbox, repair loop, self-test
src/draft/                draft and failed run report
src/dashboard/            monitoring dashboard (Streamlit)
src/evaluation/           holdout evaluation
tests/                    automatic tests
sandbox/Dockerfile        the sandbox image
```

## Limitations

- Output quality depends on the language model. A small local model can be inaccurate, so summaries and hypotheses need human review.
- Experiments use a few small datasets and simple methods. A hypothesis about anything else can only be tested through a stand-in comparison, and the program warns when the design does not seem to fit.
- Results describe the toy dataset only. So far they have mostly been inconclusive.
- The repair loop is a plain retry loop, not reinforcement learning. The bandit is real but small, and it has not been shown to improve results (see below).
- "Published" is a proxy for peer review: a PubMed record, or an arXiv paper with a journal reference or DOI. Most papers in the index are preprints.
- Milvus was tested with Milvus Lite on Linux. On Windows it needs a Milvus server, and that setup was not tested here.
- Figure descriptions from a vision model are optional. A small model was tried and was not reliable, so only the figure captions are used as evidence.
- The evaluation uses a handful of papers, and its ratings were made by an AI assistant, so it is a rough check and not a benchmark.
- The dashboard and `update` work on saved data. Nothing runs in the background.

## Reinforcement learning

The bandit in `src/rl/` is a simple form of reinforcement learning with one state and a few actions. The actions are how many papers to retrieve (2, 3 or 5). After each `run --adaptive` the reward comes from automatic checks on the saved run: the sandbox succeeded, the hypothesis cited a retrieved paper, and the critic gave a verdict without a design warning. The model never grades itself.

The reward is a proxy for a clean run, not for good science. Tests show that the algorithm learns on simulated rewards. Whether it helps real runs is not shown: each run takes minutes, so it needs many runs compared with a fixed and a random strategy. Until then, claim only that the mechanism exists.

## Troubleshooting

- "Cannot reach Ollama": start Ollama, then run `check-llm`.
- "Engine is not running": start Docker Desktop and wait until it is ready.
- "Image is not built yet": run `python -m src.main sandbox-build`.
