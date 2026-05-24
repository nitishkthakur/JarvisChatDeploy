# Copilot Instructions for JarvisChat

## Mandatory Memory Protocol

Before writing or modifying any code in this repository, you **must**:

1. **Read all files in the `memory/` folder** at the start of every session:
   - `memory/instructions.md` — standing rules and constraints
   - `memory/discoveries.md` — technical findings and gotchas
   - `memory/progress.md` — current project status

2. **Update `memory/progress.md`** after completing any task:
   - Move finished items to the "Completed" section
   - Add new planned work to "Planned / Future"
   - Mark work-in-progress items in "In Progress"

3. **Update `memory/discoveries.md`** when you discover:
   - A new technical gotcha or workaround
   - An important decision about architecture or dependencies
   - A pattern or convention that future agents should know about

## Testing Requirements

- All tests live in `tests/` and use pytest
- Run tests with: `python -m pytest tests/ -v`
- Tests **must** pass before committing changes
- When adding new backend features, add corresponding tests
- CI runs automatically via `.github/workflows/test.yml`

## Code Change Workflow

1. Read `memory/` folder
2. Understand the existing code before changing it
3. Make minimal, focused changes
4. Run `python -m pytest tests/ -v` to verify
5. Update `memory/progress.md`
6. Update `memory/discoveries.md` if applicable
7. Commit and push

## Architecture Quick Reference

- Backend: `backend/main.py` — FastAPI + LangChain ReAct agent
- Frontend: `frontend/index.html` — static HTML/CSS/JS (no build step)
- Tests: `tests/test_todo_list.py`, `tests/test_helpers.py`, `tests/test_api.py`
- Start server: `OLLAMA_API_KEY=<key> uvicorn backend.main:app --reload`

## UI Rules

- Black background, clean sans-serif typography
- No emojis, no icons, no decorative elements
- Show agent steps in real-time via SSE streaming
