# Testing guide

## Automated tests (no internet, no LLM, no Docker needed)
```powershell
python -m pytest -q
```
Expected: `62 passed, 1 skipped` (the skipped one is the live arXiv test). These tests use fixture data, a rule-based mock model, and a small FAKE Ollama server (real Ollama is not needed): they verify parsing, retries, storage, retrieval ranking, citation checking, workflow control flow and CLI behaviour. They do **not** verify LLM quality.

## Live arXiv check (needs internet; run on your PC)
```powershell
$env:RUN_NETWORK_TESTS = "1"; python -m pytest -q -k live
python -m src.main ingest --query "graph neural networks" --max-results 5
python -m src.main status
python -m src.main search --query "graph neural networks"
python -m src.main run --topic "graph neural networks"
```
Success looks like: `Stored and indexed 5 paper(s)` with real titles/ids; a `data/raw/arxiv_*.xml` file; `run` ends with `Status: COMPLETED_WEEK1` and a `storage/runs/run_*.json` file. In test mode the run prints a TEST MODE banner and labelled placeholder text.

## Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: config` or `src` | Run from the project root with `python -m src.main ...`, venv active. |
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `arXiv request failed after 3 attempts` | No internet, proxy/firewall, or arXiv rate limiting. Wait and retry. |
| `arXiv returned no papers` | Too many keywords (all must match). Use fewer words. |
| `Index was built with embedder ...` | You changed `EMBEDDING_BACKEND`; run `python -m src.main reindex`. |
| `No relevant papers ... Run 'ingest' first` | Ingest papers whose text overlaps your topic words (hashing embedder is lexical). |
| `LLM_PROVIDER=ollama needs: pip install langchain-ollama` | Install it; make sure Ollama is running. |
| Garbled characters | Handled by UTF-8 reconfiguration; otherwise use Windows Terminal. |

## Checking the free local LLM (needs Ollama on your PC)
```powershell
python -m src.main check-system     # RAM/GPU and a model-size suggestion
python -m src.main check-llm        # with LLM_PROVIDER=ollama: server up? model pulled?
python -m src.main run --topic "graph neural networks" -k 3 --provider ollama
```
The automated tests cover the Ollama code path only against a fake server. Whether a real model produces useful text is something only this manual run can show; small models may ignore the requested format, in which case the run prints warnings (e.g. a syntax error in generated code) instead of crashing.

More troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| `Cannot reach Ollama at http://localhost:11434` | Ollama not running; start it from the Start menu. Corporate proxy? set `NO_PROXY=localhost,127.0.0.1`. |
| `model 'x' is not downloaded` | `ollama pull x` |
| Very slow / times out | Use a smaller model, lower `-k`, or raise `LLM_TIMEOUT_SECONDS`. |
| Garbled/ignored answer format | Typical of tiny models; try a larger one that your hardware supports. |
| `Ollama embedding request failed` | `ollama pull nomic-embed-text`, then `python -m src.main reindex`. |
