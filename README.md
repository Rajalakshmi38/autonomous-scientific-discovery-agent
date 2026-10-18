# Autonomous Multimodal Scientific Discovery Agent

An AI research assistant that finds recent papers on arXiv, summarises them, proposes a testable hypothesis, and prepares a Python experiment. It uses free tools only and runs a local language model through [Ollama](https://ollama.com). No API key or paid account is needed.

**Status:** work in progress. Planned next: a Docker sandbox to run the generated experiments, grounded answers with sources, a critic agent, and a dashboard.

## How it works

1. Fetch papers from arXiv and store them.
2. Search the stored papers for a research topic.
3. The Literature Reviewer agent summarises the evidence.
4. The Hypothesis Generator agent proposes a hypothesis.
5. The Code Executor agent prepares a Python experiment (not run yet).

## Setup (Windows PowerShell)

Install Python 3.12 and [Ollama](https://ollama.com/download), then:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
```

In `.env`, set:

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

With little RAM, use `llama3.2:1b` instead. Run `python -m src.main check-system` to see what your PC can handle.

## Usage

```powershell
python -m src.main check-llm
python -m src.main ingest --query "graph neural networks" --max-results 10
python -m src.main run --topic "graph neural networks"
```

Other commands: `search`, `status`, `reindex`, `check-system`.

If `LLM_PROVIDER` is left as `test`, the project runs in a labelled test mode with placeholder text instead of a real model.

## Tests

```powershell
python -m pytest -q
```

More detail is in `docs/`.