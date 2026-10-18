# Setup guide: Windows 10/11 + VS Code, free tools only

Nothing here needs an API key, an account with billing, or a subscription.

## 0. What each tool costs
| Tool | Role | Cost / licence note (confirm current terms yourself) |
|---|---|---|
| Python, VS Code, Git | Development | Free |
| GitHub account | Hosting the repo | Free tier is enough |
| arXiv API | Paper source | Free, no key |
| Ollama | Local LLM server | Free, open source |
| LLM model weights (e.g. Llama, Qwen) | The model | Free to download; each model has its own licence, read it if the work is commercial |
| nomic-embed-text / sentence-transformers | Embeddings (optional) | Free |
| Docker | Sandbox (Week 2 only) | **Docker Desktop is free only for personal use, education, small businesses and open source; larger companies need a paid plan.** An internship for a company may count. Free alternative: Docker Engine inside WSL2 (section 7). |

## 1. Install the base tools
Download from the official sites (python.org, code.visualstudio.com, git-scm.com). On the Python installer tick **"Add python.exe to PATH"**. Or use winget in PowerShell (if an ID is not found, use the websites):
```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Microsoft.VisualStudioCode
winget install -e --id Git.Git
```
Close and reopen PowerShell, then check: `python --version`, `git --version`.

## 2. VS Code
1. Open VS Code, Extensions (Ctrl+Shift+X), install **Python** (Microsoft) and **Pylance**.
2. File > Open Folder > the project folder `autonomous-scientific-discovery-agent`.
3. Terminal > New Terminal (PowerShell opens inside the project folder).

## 3. Project environment (in the VS Code terminal)
```powershell
py -3.12 -m venv .venv          # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1    # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Copy-Item .env.example .env
python -m pytest -q
```
Then press Ctrl+Shift+P > **Python: Select Interpreter** > choose the one in `.venv`. Expected test result: `62 passed, 1 skipped`.

## 4. Can my PC run a local LLM?
```powershell
python -m src.main check-system
```
Rough guide (the real test is step 5):
| Your PC | Realistic choice |
|---|---|
| Under ~8 GB RAM, no NVIDIA GPU | Stay in **test (mock) mode** for now; optionally try a 1B model and expect slowness |
| 8-16 GB RAM, no GPU | 1-3B models (e.g. `llama3.2:3b`). Works, modest quality, tens of seconds to minutes per step |
| 16+ GB RAM, no GPU | 3B comfortable; 7-8B possible but slow on CPU |
| NVIDIA GPU with 6+ GB VRAM | 7-8B models at usable speed |
Rules of thumb: a quantised model needs roughly 0.6-0.8 GB of memory per billion parameters plus overhead; AMD/Intel/Apple GPUs may also be used by Ollama but are not detected by `check-system`. Small models make weaker scientific reasoning and often ignore output formats; the pipeline then shows warnings rather than crashing.

## 5. Install Ollama and a model (free)
1. Install from https://ollama.com/download (or `winget install -e --id Ollama.Ollama`). It runs in the background (system tray).
2. Verify: `ollama --version`.
3. Download a model that fits step 4 (confirm tag names at https://ollama.com/library):
```powershell
ollama pull llama3.2:3b
ollama run llama3.2:3b "Say hello in five words"     # type /bye to exit
```
4. Optional, free semantic embeddings with no PyTorch: `ollama pull nomic-embed-text`.
5. Edit `.env`: `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3.2:3b` (and optionally `EMBEDDING_BACKEND=ollama`; if you change the embedding backend run `python -m src.main reindex`).
6. Check it: `python -m src.main check-llm`. Success looks like `Ollama is running and model 'llama3.2:3b' is available.`
7. Run: `python -m src.main run --topic "graph neural networks" -k 3` (smaller `-k` keeps prompts short for small models). The first call is slow while the model loads.

## 6. No local LLM yet? Use mock mode
Leave `LLM_PROVIDER=test`. Everything runs (ingestion, search, workflow, saved runs) but the agents' text comes from fixed templates, **not a language model**. Banners and every saved run record say so. You can also force it per run: `python -m src.main run --topic "..." --provider test`.

## 7. Docker for the Week 2 sandbox (not needed yet)
Option A, Docker Desktop: https://www.docker.com/products/docker-desktop/ (check the licence note above). Option B, free for everyone: install WSL2 (`wsl --install` in an admin PowerShell, restart), then install Docker Engine inside the Ubuntu distribution following Docker's official Ubuntu instructions, and run Week 2 sandbox commands from there. Verify either way with `docker run --rm hello-world`. I will adapt the Week 2 commands to whichever option you pick.

## 8. First real arXiv test
```powershell
python -m src.main ingest --query "graph neural networks" --max-results 5
python -m src.main status
python -m src.main search --query "graph neural networks"
```

## 9. Git and GitHub
```powershell
git init
git branch -M main
git config user.name "Your Name"
git config user.email "you@example.com"
```
Create an empty repo on github.com (do NOT tick README/.gitignore), then `git remote add origin https://github.com/<you>/autonomous-scientific-discovery-agent.git`. Before every commit:
```powershell
git status                    # .env, .venv, data/, storage/ must NOT appear
git check-ignore -v .env      # prints the matching .gitignore rule
git diff --cached --stat      # review exactly what is staged
```
Suggested truthful commits (after tests pass):
```powershell
git add .gitignore .env.example requirements*.txt pytest.ini config src/models.py src/utils src/ingestion
git commit -m "Add arXiv ingestion, paper store and tests"
git add src/retrieval src/llm src/agents src/workflow.py src/main.py src/__init__.py tests docs README.md CHANGELOG.md
git commit -m "Add retrieval, agents, workflow, CLI, free local LLM (Ollama) support and mock mode"
git push -u origin main
```
