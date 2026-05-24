# JarvisChat — Long-Term Instructions

This file contains standing instructions from the project owner. Every coding agent **must** read this file at the start of every session.

---

## Core Architecture

- **Backend**: Python / FastAPI in `backend/main.py`
- **Frontend**: Single static HTML file at `frontend/index.html`
- **No build step**: Frontend is pure HTML/CSS/JS — no npm, no bundler
- **Server start**: `cd backend && uvicorn main:app --reload` (requires `OLLAMA_API_KEY` env var)

## Agent Architecture

- The backend uses a LangChain ReAct agent (`create_react_agent` + `AgentExecutor`)
- The agent must use the `todo_list` tool (planning tool) as its first step for any non-trivial request
- Simple greetings bypass the agent and go directly to the LLM
- The `/chat/stream` endpoint sends SSE events so the frontend can show each agent step in real-time

## UI Rules

- Black background (`#000`), clean sans-serif typography
- **No emojis, no icons, no decorative elements** — text only
- Show agent thinking/tool steps in the UI as they happen (SSE streaming)
- Steps appear as indented, italicized text with a left border before the final answer

## Security

- `OLLAMA_API_KEY` lives only in backend env vars — never in frontend code or source control
- No secrets in committed files

## Testing

- Tests live in `tests/` directory and use pytest
- Run tests: `cd /path/to/repo && python -m pytest tests/ -v`
- Tests must pass before merging any PR
- CI runs tests automatically via `.github/workflows/test.yml`

## Code Style

- Python: follow existing patterns in `main.py` (type hints, docstrings, section headers)
- Frontend JS: vanilla ES5 style, no frameworks, IIFE pattern

## What NOT to Do

- Do not add emojis or fancy decorations to the UI
- Do not hardcode API keys
- Do not add npm/webpack/build tooling to the frontend
- Do not remove or weaken existing tests
