# GitHub AI Agent — Phase 3

A read-only public GitHub repository inspector built with FastAPI, Streamlit, and httpx. It retrieves repository metadata and a bounded selection of text files through GitHub's REST API. Downloaded content is displayed only; the application never executes repository code.

The application supports repository analysis and natural-language Q&A. Q&A indexes bounded source snapshots in a local SQLite FTS5 database, retrieves line-addressable chunks, and passes only those chunks to the configured provider. The deterministic mock provider remains the default, and live LLM inference is disabled unless explicitly enabled in the backend environment. No API key is needed for the mock provider.

## Requirements

- Python 3.11 or newer
- Windows PowerShell is supported
- Internet access to `api.github.com` for live repository inspection

## Setup (PowerShell)

From the project folder:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

The first command matches the Python 3.13 installation on this workstation. If Python is installed elsewhere, use its executable path, or use `py -3.11 -m venv .venv` when the Python launcher is available. The execution-policy command applies only to the current PowerShell process.

## Run tests and lint

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Start FastAPI

In one activated PowerShell terminal:

```powershell
python -m uvicorn repo_agent.api.main:app --app-dir src --reload
```

The API is at `http://127.0.0.1:8000`. Interactive API documentation is at `http://127.0.0.1:8000/docs`; health check: `http://127.0.0.1:8000/health`.

## Start Streamlit

In a second activated PowerShell terminal, from the project folder:

```powershell
python -m streamlit run src/repo_agent/ui/streamlit_app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`. To use a different API URL, set `$env:REPO_AGENT_API_URL` before starting Streamlit.

## API

- `GET /health` — process health check.
- `GET /api/status` — reports the active provider type and whether live inference is enabled; it never returns credentials or model configuration.
- `POST /api/repos/inspect` — inspect a public repository.
- `POST /api/repos/analyze` — analyze the selected repository context.
- `POST /api/repos/ask` — answer a repository question from the selected context.

Example request body:

```json
{
  "repository_url": "https://github.com/owner/repository"
}
```

Each returned file includes its path, Git blob SHA, the inspected default-branch commit SHA, and a GitHub citation URL pinned to that commit.

The analysis endpoint accepts the same `repository_url` request field. The question endpoint accepts `repository_url` and `question`, for example:

```json
{
  "repository_url": "https://github.com/owner/repository",
  "question": "Where is the application entry point?"
}
```

Analysis and answer responses include `context_sufficient`, estimated context tokens, and citations. The analysis endpoint retains question-aware selection across its inspected snapshot. Q&A first retrieves relevant chunks from the local index, prioritizes explicitly named paths and exact code identifiers, and sends only those chunks to the provider. Citations must resolve to chunks included in that bounded prompt; the API generates commit- and line-pinned GitHub links and ignores provider-supplied URLs. When a requested file is missing or context is insufficient, the answer says so explicitly and does not invent a citation.

The context budget defaults to 12,000 estimated tokens using a conservative four-characters-per-token approximation. The output budget defaults to 1,000 tokens. The inspection UI retains the Phase 1 20-file display limit; Q&A independently indexes up to 40 files by default, with 128 KiB per file, 5 MiB total source, 50 GitHub requests per indexing attempt, and 1,000 chunks. Chunk size defaults to 80 lines with 15 lines of overlap, and up to 10 chunks are retrieved per question. Change these limits with the `RAG_*` settings in `.env.example`. SQLite files are local under `data/` and ignored by Git. Existing commit indexes are reused until the default-branch SHA changes.

## Enabling live inference later

Live requests are disabled by default. The mock remains active unless both `LLM_ENABLED=true` and `LLM_PROVIDER=openai_compatible` are set in the backend environment. Before enabling a hosted provider, get explicit approval for live API usage. Then place the provider key in `.env` or a backend environment variable; never put it in Streamlit code or browser configuration.

Restart FastAPI after changing these settings. Enabling a hosted provider can incur charges according to that provider's model and token pricing.

Example backend `.env` settings:

```dotenv
LLM_ENABLED=true
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-approved-model
LLM_API_KEY=your-secret-key
LLM_REQUEST_TIMEOUT_SECONDS=30
LLM_MAX_CONTEXT_TOKENS=12000
LLM_MAX_OUTPUT_TOKENS=1000
```

The same provider works with OpenAI-compatible services by changing `LLM_BASE_URL`, `LLM_MODEL`, and, when required, `LLM_API_KEY`. Remote endpoints must use HTTPS; local HTTP endpoints are allowed only on localhost. The API key is held as a secret setting and is sent only in the backend authorization header. The app does not log request headers or provider response bodies. The provider uses the OpenAI-compatible `/chat/completions` route and expects JSON in the message content containing `answer`, `context_sufficient`, and `citation_paths`. A `citations` field containing exact file paths or objects with a `path` is also normalized. The API ignores any URL returned by a model and resolves citations only against source paths sent in the prompt.

## Read-only and size safeguards

- Only HTTPS repository URLs hosted on `github.com` or `www.github.com` are accepted.
- GitHub requests use only read endpoints; no clone, push, issue, or other write operations are performed.
- The default limits are 20 files, 64 KiB per file, 512 KiB total source, and a 15-second timeout per GitHub request.
- Binary and non-UTF-8 files are skipped, and file extensions are filtered to likely text formats.
- Repository contents are never executed.
- GitHub public unauthenticated API access is used. GitHub's current published REST API limit is 60 requests per hour for unauthenticated requests.

Limits can be changed with the `GITHUB_REQUEST_TIMEOUT_SECONDS`, `GITHUB_MAX_FILES`, `GITHUB_MAX_FILE_BYTES`, and `GITHUB_MAX_TOTAL_BYTES` environment variables. `.env.example` lists the defaults.

## Project map

- `src/repo_agent/github/` validates repository URLs and performs bounded GitHub API reads.
- `src/repo_agent/api/` exposes FastAPI routes and user-safe error responses.
- `src/repo_agent/ui/` contains the Streamlit interface.
- `src/repo_agent/analysis/` builds bounded context and validates source citations.
- `src/repo_agent/providers/` contains the provider contract, default mock, and optional OpenAI-compatible HTTP provider.
- `src/repo_agent/rag/` chunks files, maintains a local SQLite FTS5 index, retrieves relevant source, and verifies citations.
- `tests/unit/` tests URL parsing, retrieval limits, API behavior, bounded context, providers, and Q&A using mock providers and HTTP transports.

