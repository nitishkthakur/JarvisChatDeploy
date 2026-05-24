# JarvisChat — Discoveries

This file records technical discoveries, gotchas, and decisions made during development. Coding agents should read this to avoid repeating mistakes or re-discovering known issues.

---

## 2026-05-24 — Initial Build

### LangChain Version Notes

- The project uses `langchain-classic` (v1.0.7) for `AgentExecutor` and `create_react_agent`
- These were moved out of the main `langchain` package in v1.x; `langchain_classic.agents` is the correct import path
- `langchain-core` provides `BaseChatModel`, `BaseTool`, message types, and `PromptTemplate`

### OllamaCloudLLM

- Custom `BaseChatModel` subclass that calls the Ollama Cloud HTTPS API at `/v1/chat/completions`
- Uses the OpenAI-compatible endpoint format (same request/response schema)
- Auth via `Authorization: Bearer <key>` header
- Timeout set to 120 seconds for slow model responses
- Response parsing handles both `choices[0].message.content` (OpenAI format) and `message.content` (Ollama native)

### TodoListTool

- Action format: `create|task1|task2`, `complete|N`, `add|description`, `view`
- State is per-conversation, keyed by `conv_id` in the global `todo_states` dict
- Whitespace in actions is stripped

### Agent Step Streaming

- The `/chat/stream` endpoint returns SSE events with types: `step`, `answer`, `error`
- Agent intermediate steps are parsed via `_parse_agent_steps()` which extracts Thought/Action/Observation from the ReAct agent's logs
- `AgentExecutor.return_intermediate_steps = True` is used to capture steps
- The frontend reads these SSE events and renders each step sequentially

### Testing

- `pytest` with `TestClient` from FastAPI for integration tests
- LLM calls are mocked with `unittest.mock.patch` on `OllamaCloudLLM._generate`
- Global state (`conversations`, `todo_states`) is cleared in `autouse` fixtures
- Tests run via: `python -m pytest tests/ -v`
