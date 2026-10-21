# Autonomous Multimodal Scientific Discovery Agent

An AI research assistant that finds recent papers on arXiv, summarises them, proposes a testable hypothesis, writes a small Python experiment, and runs it inside an isolated Docker container.

It uses free tools only: a local language model through Ollama, local search, and Docker. No API key, paid account or billing details are needed.

This project is a work in progress. What is not built yet is listed at the end.

## What it does

1. Fetches recent papers from arXiv and stores their title, abstract, authors, date, identifier and link.
2. Searches the stored papers for a research topic.
3. The Literature Reviewer agent summarises the evidence and cites papers by identifier.
4. The Hypothesis Generator agent proposes a testable hypothesis based on that evidence.
5. The Code Executor agent writes a Python experiment for the hypothesis.
6. The sandbox runs the experiment in a restricted Docker container and captures the output.
7. If the code fails, the error is sent back to the model for a fix, a limited number of times.
8. Every run is saved as a record with the sources, code, attempts, output and warnings.

## How it works

- **Ingestion:** the arXiv API is queried and papers are stored in a local SQLite database.
- **Search:** each paper is turned into a vector and the closest ones to a topic are returned. The default method compares shared words. Semantic embeddings through Ollama are optional.
- **Agents:** three LangChain prompt chains. The language model is a single setting, so another provider can be added later.
- **Citation checks:** agents must cite retrieved papers as `[arxiv:ID]`. If an answer cites none, the model is asked once more. Citations to papers that were never retrieved are flagged. These checks catch obvious problems but do not prove a claim is true.
- **Sandbox:** generated code runs only inside Docker, never directly on your computer.
- **Repair loop:** after a failed run the error goes back to the model, up to a set number of times. It is a plain retry loop, not reinforcement learning.
- **Test mode:** without a language model the project uses a rule-based mock that prints fixed placeholder text. It is labelled TEST MODE and does not reason.

Paper text and model output are treated as untrusted.

## Requirements

- Windows 10 or 11 (commands below are for PowerShell) and Python 3.11 or 3.12
- Ollama, for a local language model: https://ollama.com/download
- Docker Desktop, for the sandbox: https://www.docker.com/products/docker-desktop/ (check Docker's licence terms for your situation)
- About 8 GB of RAM is enough for a small model. Run `python -m src.main check-system` to see what your PC can handle.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Download a language model that fits your PC:

```powershell
ollama pull llama3.2:1b
```

In `.env`, set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=llama3.2:1b`, then check it:

```powershell
python -m src.main check-llm
```

Start Docker Desktop, wait until its engine is running, then build and verify the sandbox once:

```powershell
python -m src.main sandbox-build
python -m src.main sandbox-status
python -m src.main sandbox-test
```

On a low-memory PC you can limit Docker's memory by creating the file `.wslconfig` in your user folder containing `[wsl2]`, `memory=2GB` and `processors=4`, then running `wsl --shutdown`.

## Usage

```powershell
python -m src.main ingest --query "graph neural networks" --max-results 10
python -m src.main run --topic "graph neural networks" -k 3
```

Other commands:

- `search --query "..."` searches the stored papers.
- `run --topic "..." --provider test` runs everything with the mock model.
- `run --topic "..." --no-execute` writes the code but does not run it.
- `status` shows the model, stored papers and sandbox state.
- `reindex` rebuilds the search index after changing the embedding setting.
- `check-system` and `check-llm` check your hardware and the Ollama connection.

Each run is saved as a JSON file in `storage/runs/`. Logs are in `storage/logs/app.log`.

## Main settings

Settings are read from `.env` (see `.env.example`).

- `LLM_PROVIDER`: `ollama` for a real local model, `test` for the mock.
- `OLLAMA_MODEL`: the model name used with `ollama pull`.
- `LLM_MAX_TOKENS`: maximum length of an answer. Raise it if generated code is cut off.
- `EMBEDDING_BACKEND`: `hashing` (default) or `ollama` for semantic search.
- `CITATION_RETRIES`: extra tries when an answer cites no retrieved paper.
- `MAX_REPAIR_RETRIES`: repair attempts after a failed run.
- `SANDBOX_TIMEOUT_SECONDS` and `SANDBOX_MEMORY_MB`: limits for generated code.

## The sandbox

Generated code is sent to the container through standard input, and no folder from your computer is shared with it. The container has:

- no network access
- a read-only filesystem, with a small writable `/tmp`
- a non-root user with all Linux capabilities removed
- limits on memory, CPU and number of processes
- a time limit, after which the container is killed and removed
- a cap on how much output is captured

Only Python and numpy are installed, so code that needs other libraries fails with a clear message.

Run `python -m src.main sandbox-test` on your own machine. It runs scripts that try forbidden things, such as using the network, writing system files, using too much memory, or running forever, and checks that each one is stopped. A pass is evidence that the restrictions work, not a guarantee.

## Tests

```powershell
python -m pytest -q
```

Most sandbox tests use a small fake `docker` program, which tests the project's own logic but gives no isolation. The tests marked `docker` use the real engine and are skipped when Docker or the sandbox image is not available.

## Project structure

```
config/settings.py        settings read from environment variables
src/main.py               command-line interface
src/workflow.py           runs the steps in order
src/agents/               literature reviewer, hypothesis generator, code executor
src/ingestion/            arXiv client and paper storage
src/retrieval/            embeddings, vector index, search, citation checks
src/llm/                  model selection, mock model, Ollama check
src/execution/            Docker sandbox, repair loop, sandbox self-test
src/utils/                logging and hardware check
sandbox/Dockerfile        the sandbox image
tests/                    automatic tests
```

## Limitations

- Output quality depends on the model. Small local models often write code that fails or describe papers inaccurately. Treat summaries and hypotheses as drafts that need human review.
- Experiment numbers are real outputs of model-written code. They are not verified research findings.
- The default search matches shared words, not meaning.
- There is no reinforcement learning. The repair loop only shows the model an error and asks for new code. Nothing is trained or remembered between runs.
- Only arXiv works. PubMed is a placeholder.

## Not built yet

- Extracting charts and figures from paper PDFs, with a vision-language model
- A critic agent with statistical checks such as p-values and confidence intervals
- Research draft generation from verified results
- A monitoring dashboard
- A reinforcement-learning component
- PubMed ingestion

## Troubleshooting

- "Cannot reach Ollama": start Ollama from the Start menu, then run `check-llm`.
- "Docker command not found": install Docker Desktop and reopen the terminal.
- "Engine is not running": start Docker Desktop and wait until the engine is up.
- "Image is not built yet": run `python -m src.main sandbox-build`.
- Generated code is cut off: raise `LLM_MAX_TOKENS` in `.env`.
- Slow or out of memory: close other apps, use a smaller model, or limit Docker's memory as described above.
