# Autonomous Multimodal Scientific Discovery Agent

An AI research assistant that finds recent papers on arXiv, summarises them, proposes a testable hypothesis, designs a small experiment, runs it inside an isolated Docker container, and checks the result with statistics.

It uses free tools only: a local language model through Ollama, local search, and Docker. No API key, paid account or billing details are needed.

This project is a work in progress. The limits and what is not built yet are listed near the end. Please read them: the experiments are small checks on toy datasets, not research findings.

## What it does

1. Fetches recent papers from arXiv and stores their title, abstract, authors, date, identifier and link.
2. Searches the stored papers for a research topic.
3. The Literature Reviewer agent summarises the evidence and cites papers by identifier.
4. The Hypothesis Generator agent proposes a testable hypothesis based on that evidence.
5. The Code Executor agent designs a small experiment for the hypothesis. By default the model only fills in a form (dataset, metric, two methods or settings) and tested code writes the script.
6. The sandbox runs the experiment in a restricted Docker container and captures the output.
7. A statistical critic checks the numbers: it reports the difference, a confidence interval and a p-value, and says when the result is inconclusive.
8. Every run is saved as a record with the sources, design, code, output and warnings. A short note about earlier runs on similar topics is shown to the model in later runs.
9. Optionally, captions and page images of figures can be taken from a paper's PDF and added to the evidence.
10. The `draft` command turns one saved run into a short draft (Markdown and LaTeX). If the run failed, it writes a failed run report instead.

## How it works

- **Ingestion:** the arXiv API is queried and papers are stored in a local SQLite database.
- **Search:** each paper is turned into a vector and the closest ones to a topic are returned. Without any settings file the program compares shared words (hashing). The example settings file uses semantic embeddings through Ollama.
- **Agents:** three LangChain prompt chains. The language model is a single setting, so another provider can be added later.
- **Citation checks:** agents must cite retrieved papers as `[arxiv:ID]`. If an answer cites none, the model is asked once more. Citations to papers that were never retrieved are flagged. These checks catch obvious problems but do not prove a claim is true.
- **Sandbox:** generated code runs only inside Docker, never directly on your computer.
- **Experiment design:** in the default `spec` mode the model fills in a form with a dataset (iris, wine, breast_cancer or digits), a metric, and two things to compare. The form is checked, and a vetted template writes the script, so the model cannot write unsafe or broken code. The two things must differ in exactly one respect, such as 10 trees against 100 trees, or with and without feature scaling. If the model's form is invalid after a few tries, fixed keyword rules build a design from the hypothesis sentence, and the output says so. A plain check then warns when the design does not seem to match the hypothesis.
- **Statistical critic:** rule-based, no language model. It uses a corrected resampled t-test (Nadeau and Bengio) because repeated splits of the same data are not independent. It reports a mean difference with a 95% confidence interval and never calls a non-significant result proof of no effect.
- **Memory:** earlier designs and results are shown to the model as a short note. Earlier hypothesis text is left out on purpose, because a small model copied it into new hypotheses. This is context only, not learning.
- **Figures:** `figures` downloads a stored paper's PDF, finds captions that start a line with `Figure N:`, and saves the pages as images. Captions are added to the evidence in later runs. An optional vision model setting exists but has not been tested on small hardware.
- **Repair loop:** only used in `free` mode, where the model writes the code itself. After a failed run the error goes back to the model, up to a set number of times. It is a plain retry loop, not reinforcement learning.
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

Download a language model that fits your PC. `llama3.2:3b` is a good small choice; `llama3.2:1b` needs less memory.

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

`nomic-embed-text` is for semantic search. For figure extraction also run `pip install -r requirements-figures.txt`.

In `.env`, set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=llama3.2:3b`, then check it:

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
python -m src.main ingest --query "random forest" --max-results 30 --category cs.LG
python -m src.main ingest --source pubmed --query "asthma lung sounds" --max-results 20
python -m src.main run --topic "random forest number of trees" -k 3
```

Other commands:

- `search --query "..."` searches the stored papers.
- `run --topic "..." --provider test` runs everything with the mock model.
- `run --topic "..." --no-execute` writes the code but does not run it.
- `run --topic "..." --no-memory` runs without showing or saving earlier attempts.
- `figures --paper arxiv:ID` extracts figure captions and page images from a stored paper (use `--describe` only if you set up a vision model).
- `memory` shows what earlier runs the system remembers.
- `summary` counts the saved runs: results, citations and critic verdicts.
- `draft` writes a draft from the newest run with a real model (or `--run FILE`) into `storage/drafts/`, as `.md` and `.tex` (`--format md` or `tex` for one). The draft is filled from the saved record without a language model: the critic's lines are copied unchanged, model-written text is labelled, and a run with no verified result gets a failed run report instead of a paper.
- `sandbox-status` and `sandbox-test` check the sandbox.
- `dashboard` opens a read-only monitoring page in the browser (needs `pip install -r requirements-dashboard.txt`). It shows each saved run's steps, the literature citations and the time and attempts it used.
- `evaluate` takes recent arXiv papers that are not in the index, runs the agent on their titles, and writes a rating sheet (`storage/evaluation/`) for a person to score. `evaluate --dry-run` only lists the papers, and `evaluate --ratings FILE` summarizes a filled-in sheet.
- `status` shows the model, stored papers and sandbox state.
- `reindex` rebuilds the search index after changing the embedding setting.
- `check-system` and `check-llm` check your hardware and the Ollama connection.

Each run is saved as a JSON file in `storage/runs/`. Logs are in `storage/logs/app.log`.

## Keeping the index up to date

Save the searches you want to repeat, then run `update` whenever you like. It stores and indexes only papers that are new.

```powershell
python -m src.main watch add --source arxiv --query "random forest" --category cs.LG
python -m src.main watch add --source pubmed --query "asthma lung sounds"
python -m src.main watch list
python -m src.main update
```

`watch remove NUMBER` deletes a saved search. To refresh regularly, schedule `update` with Windows Task Scheduler (program: the `python.exe` inside `.venv\Scripts`, arguments: `-m src.main update`, start in: the project folder). `ARXIV_CONTACT` is also sent to PubMed as the contact email.

## Main settings

Settings are read from `.env` (see `.env.example`).

- `LLM_PROVIDER`: `ollama` for a real local model, `test` for the mock.
- `OLLAMA_MODEL`: the model name used with `ollama pull`.
- `LLM_MAX_TOKENS`: maximum length of an answer. Raise it if generated code is cut off.
- `CODE_MODE`: `spec` (default) or `free`. In `free` mode the model writes the whole script and the repair loop applies.
- `CRITIC_ALPHA`: significance level used by the critic (default 0.05).
- `MEMORY_CHAR_BUDGET`: size of the note about earlier runs.
- `UNLOAD_MODEL_BEFORE_RUN`: frees the language model's memory before the sandbox starts.
- `EMBEDDING_BACKEND`: `ollama` in `.env.example` (semantic search, needs `ollama pull nomic-embed-text`); `hashing` is the built-in default when no `.env` exists.
- `CITATION_RETRIES`: extra tries when an answer cites no retrieved paper.
- `MAX_REPAIR_RETRIES`: repair attempts after a failed run.
- `SANDBOX_TIMEOUT_SECONDS` (default 120) and `SANDBOX_MEMORY_MB`: limits for the experiment.

## The sandbox

Generated code is sent to the container through standard input, and no folder from your computer is shared with it. The container has:

- no network access
- a read-only filesystem, with a small writable `/tmp`
- a non-root user with all Linux capabilities removed
- limits on memory, CPU and number of processes
- a time limit, after which the container is killed and removed
- a cap on how much output is captured

Only Python, numpy, scipy, pandas and scikit-learn are installed, so code that needs other libraries fails with a clear message.

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
src/experiments/          the checked experiment form and the script template
src/critic/               statistical critic
src/memory/               note about earlier runs
src/ingestion/            arXiv and PubMed clients, saved searches, paper storage, figure extraction
src/retrieval/            embeddings, vector index, search, citation checks
src/llm/                  model selection, mock model, Ollama check
src/execution/            Docker sandbox, repair loop, sandbox self-test
src/draft/                draft and failed run report from a saved run
src/dashboard/            monitoring dashboard (Streamlit)
src/evaluation/           holdout evaluation
src/utils/                logging, hardware check, run summary
sandbox/Dockerfile        the sandbox image
tests/                    automatic tests
```

## Limitations

- Output quality depends on the language model. Small local models can describe papers inaccurately, so summaries and hypotheses are drafts that need human review.
- Experiments use a few small bundled datasets and simple methods. Their results describe those datasets only and are not verified research findings.
- Search quality depends on which papers are stored. Semantic search (Ollama embeddings) works better than the default word matching.
- The repair loop sends errors back to the model, but it is a plain retry loop, not reinforcement learning. A reinforcement-learning component is only a proposal.
- Figure extraction reads captions and page images. A vision-language model for describing figures is optional and not part of the default setup.
- The holdout evaluation uses only a handful of papers and rough measures such as embedding similarity. Human ratings are needed to judge the output, so it is a check and not a benchmark.
- The dashboard shows runs once they are saved. It does not follow a run step by step while it is still running.
- The draft is a short template filled from one saved run. It is not a finished research paper.
- The index is refreshed when `update` is run. Nothing runs in the background, so regular refreshes need a scheduled task (see "Keeping the index up to date").
- PubMed records follow NCBI's documented format. Records without an abstract are skipped, and PubMed has no category option.

## Not built yet

- A reinforcement-learning component

## Troubleshooting

- "Cannot reach Ollama": start Ollama from the Start menu, then run `check-llm`.
- "Docker command not found": install Docker Desktop and reopen the terminal.
- "Engine is not running": start Docker Desktop and wait until the engine is up.
- "Image is not built yet": run `python -m src.main sandbox-build`.
- Generated code is cut off: raise `LLM_MAX_TOKENS` in `.env`.
- Slow or out of memory: close other apps, use a smaller model, or limit Docker's memory as described above.
