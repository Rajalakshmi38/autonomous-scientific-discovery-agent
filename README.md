# Autonomous Multimodal Scientific Discovery Agent

An AI research assistant that finds recent papers, summarises them, proposes a testable hypothesis, runs a small experiment in an isolated Docker container, and checks the result with statistics.

It uses free tools only: a local language model (Ollama), local search and Docker. No API key or paid account is needed.

## What it does

- Fetches papers from arXiv and PubMed into a local database and searches them by topic.
- Three agents (LangChain): Literature Reviewer, Hypothesis Generator and Code Executor.
- Runs the experiment in a restricted Docker sandbox (no network, limited memory and time).
- A statistical critic reports the difference, a 95% confidence interval and a p-value.
- A small bandit (reinforcement learning) can learn how many papers to retrieve.
- Extras: a draft or failed run report (Markdown and LaTeX), a dashboard, and an evaluation on recent papers.

## Requirements

- Windows 10 or 11 with PowerShell, and Python 3.11 or 3.12
- Ollama (https://ollama.com/download) and Docker Desktop
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
python -m src.main run "random forest number of trees" -k 3
python -m src.main draft
python -m src.main dashboard
```

Other useful commands (`python -m src.main --help` lists all of them):

- `run "..." --adaptive` lets the bandit choose how many papers to retrieve. `bandit` shows what it has learned.
- `evaluate` runs the agent on recent papers that are not in the index and writes a rating sheet.
- `ingest --source pubmed --query "..."` fetches papers from PubMed.
- `run "..." --provider test` tries everything with a mock model, clearly labelled TEST MODE.

Each run is saved in `storage/runs/`.

## Dashboard

Start it from the project folder with `python -m src.main dashboard`. The menu on the left has:

- **Start New Research:** type a topic and start a run. It uses the same code as the `run` command. You can set the number of papers, let the bandit choose, and fetch new papers first.
- **Runs:** a table of saved runs. Tick a row, then Open or Delete it. Clear All Runs asks for confirmation. Deleting a run removes only its own record and drafts.
- **Logic tree, Literature Citations, Execution Performance & Costs, Draft:** details of the run chosen at the top of the page.
- **Evaluation:** results of the evaluation, with a form to rate the papers and to run a new evaluation.
- **Learning (bandit):** what the bandit has learned, with a button to reset it.

- Don't click or reload the page while a research run is in progress, because that stops it and nothing is saved.

Colours are set in `.streamlit/config.toml`. The dashboard needs `pip install -r requirements-dashboard.txt`.

## Settings

Settings are read from `.env` (see `.env.example` for all of them). The main ones:

- `LLM_PROVIDER` and `OLLAMA_MODEL`: which language model to use.
- `EMBEDDING_BACKEND`: `ollama` for semantic search, or `hashing` for simple word matching.
- `SANDBOX_TIMEOUT_SECONDS`: time limit for one experiment (default 120).
- `VECTOR_BACKEND`: `local` (default) or `milvus` (needs a Milvus server; only tested on Linux).

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
src/retrieval/            embeddings, vector index, search, citation checks
src/rl/                   strategy bandit
src/execution/            Docker sandbox, repair loop, self-test
src/draft/                draft and failed run report
src/dashboard/            Streamlit dashboard
src/evaluation/           holdout evaluation
tests/                    automatic tests
sandbox/Dockerfile        the sandbox image
```

## Limitations

- Output quality depends on the local language model, so summaries and hypotheses need human review.
- Experiments use small toy datasets and simple methods, so results are small checks, not research findings.
- The bandit and the evaluation are small (a few runs and a handful of papers, and the evaluation ratings were made by an AI assistant), so they show that the mechanism works, not that it improves results.


## Troubleshooting

- "Cannot reach Ollama": start Ollama, then run `check-llm`.
- "Engine is not running": start Docker Desktop and wait until it is ready.
- "Image is not built yet": run `python -m src.main sandbox-build`.