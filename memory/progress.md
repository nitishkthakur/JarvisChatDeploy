# JarvisChat — Progress Tracker

This file tracks current project status. Coding agents **must** update this file after completing any task.

---

## Completed

- [x] Backend: FastAPI app with `/chat` endpoint, OllamaCloudLLM, TodoListTool, ReAct agent
- [x] Frontend: Minimal black-background chat UI with model picker, no emojis
- [x] Backend: `/chat/stream` SSE endpoint for real-time agent step display
- [x] Frontend: SSE streaming integration — shows "Thinking...", "Calling planning tool", tool results
- [x] Tests: Unit tests for TodoListState and TodoListTool (`tests/test_todo_list.py`)
- [x] Tests: Unit tests for helpers — `_is_trivial`, `_history_context`, `_to_api_messages`, `_parse_agent_steps` (`tests/test_helpers.py`)
- [x] Tests: Integration tests for `/health`, `/chat`, `/chat/stream` endpoints (`tests/test_api.py`)
- [x] CI: GitHub Actions workflow to run pytest on push/PR (`.github/workflows/test.yml`)
- [x] Memory: Long-term memory folder with `instructions.md`, `discoveries.md`, `progress.md`
- [x] Copilot: `.github/copilot-instructions.md` mandating memory reads and progress updates
- [x] Deployment config: `Procfile`, `render.yaml`, `.env.example`
- [x] Documentation: `README.md` with setup/deploy instructions

## In Progress

(none)

## Planned / Future

- [ ] Streaming LLM token output (word-by-word response rendering)
- [ ] Conversation persistence (database instead of in-memory dict)
- [ ] User authentication
- [ ] Additional agent tools (web search, calculator, code execution)
- [ ] Mobile client (Android / iOS)
- [ ] Rate limiting and abuse protection
