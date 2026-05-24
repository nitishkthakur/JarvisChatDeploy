"""
JarvisChat backend — FastAPI + LangChain agentic flow with Ollama Cloud.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_API_KEY: str = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "https://api.ollama.com")

# ---------------------------------------------------------------------------
# In-memory stores (replace with a database for production)
# ---------------------------------------------------------------------------

conversations: Dict[str, List[Dict[str, str]]] = {}
todo_states: Dict[str, "TodoListState"] = {}


# ---------------------------------------------------------------------------
# Todo-list state
# ---------------------------------------------------------------------------


class TodoListState:
    """Tracks a mutable todo list for a single conversation."""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def create(self, tasks: List[str]) -> str:
        self.items = [
            {"id": i + 1, "task": t, "done": False} for i, t in enumerate(tasks)
        ]
        return self.format()

    def complete(self, item_id: int) -> str:
        for item in self.items:
            if item["id"] == item_id:
                item["done"] = True
                break
        return self.format()

    def add(self, task: str) -> str:
        new_id = max((item["id"] for item in self.items), default=0) + 1
        self.items.append({"id": new_id, "task": task, "done": False})
        return self.format()

    def format(self) -> str:
        if not self.items:
            return "Todo list is empty."
        lines = ["Current todo list:"]
        for item in self.items:
            status = "[done]" if item["done"] else "[todo]"
            lines.append(f"  {item['id']}. {status} {item['task']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangChain TodoListTool
# ---------------------------------------------------------------------------


class TodoListTool(BaseTool):
    """LangChain tool for managing a per-conversation todo list."""

    name: str = "todo_list"
    description: str = (
        "Manage a todo list to plan your work step by step. "
        "Use this tool for any complex or multi-step task before producing an answer. "
        "Input formats (choose exactly one per call):\n"
        "  create|task 1|task 2|task 3  — create a new todo list\n"
        "  complete|N                    — mark task N as done\n"
        "  add|task description          — add a task to the list\n"
        "  view                          — view the current todo list"
    )
    conv_id: str = ""

    def _run(self, action: str) -> str:  # type: ignore[override]
        state = todo_states.setdefault(self.conv_id, TodoListState())
        action = action.strip()

        if action.startswith("create|"):
            tasks = [t.strip() for t in action[7:].split("|") if t.strip()]
            if not tasks:
                return "Error: provide at least one task. Example: create|step 1|step 2"
            return state.create(tasks)

        if action.startswith("complete|"):
            try:
                return state.complete(int(action[9:].strip()))
            except ValueError:
                return "Error: invalid task number. Example: complete|1"

        if action.startswith("add|"):
            task = action[4:].strip()
            if not task:
                return "Error: provide a task description. Example: add|summarize findings"
            return state.add(task)

        if action == "view":
            return state.format()

        return (
            "Unknown action. Valid actions: "
            "create|t1|t2, complete|N, add|task, view"
        )

    async def _arun(self, action: str) -> str:  # type: ignore[override]
        return self._run(action)


# ---------------------------------------------------------------------------
# Custom LangChain LLM — calls Ollama Cloud over HTTPS
# ---------------------------------------------------------------------------


class OllamaCloudLLM(BaseChatModel):
    """
    LangChain BaseChatModel that talks to the Ollama Cloud HTTPS API.

    Uses the OpenAI-compatible /v1/chat/completions endpoint so that no
    local Ollama installation is required on the server.
    """

    model_name: str = Field(default="qwen3.5:397b-cloud")
    base_url: str = Field(default="https://api.ollama.com")
    api_key: str = Field(default="")
    temperature: float = Field(default=0.7)

    @property
    def _llm_type(self) -> str:
        return "ollama-cloud"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_api_messages(
        self, messages: List[Any]
    ) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": str(msg.content)})
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": str(msg.content)})
        return result

    def _call_api(
        self,
        api_messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
    ) -> str:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        if "message" in data:
            return data["message"]["content"]
        return str(data)

    # ------------------------------------------------------------------
    # BaseChatModel interface
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._call_api(self._to_api_messages(messages), stop)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


# ---------------------------------------------------------------------------
# ReAct agent prompt
# ---------------------------------------------------------------------------

REACT_PROMPT = PromptTemplate.from_template(
    """You are JarvisChat, a knowledgeable and precise AI assistant.

For any complex, multi-step, or research-oriented request, use the todo_list
tool first to create a structured plan, then work through it step by step,
marking each item as done before moving to the next.

For simple greetings or single-step factual questions, answer directly without
using any tools.

You have access to the following tools:

{tools}

Use this exact format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRIVIAL_STARTERS = (
    "hello",
    "hi ",
    "hi,",
    "hey",
    "bye",
    "goodbye",
    "thanks",
    "thank you",
    "how are you",
    "what is your name",
    "who are you",
    "what can you do",
)


def _is_trivial(message: str) -> bool:
    lower = message.lower().strip()
    return lower in ("hello", "hi", "hey", "bye", "goodbye") or any(
        lower.startswith(p) for p in _TRIVIAL_STARTERS
    )


def _build_agent_executor(model: str, conv_id: str) -> AgentExecutor:
    llm = OllamaCloudLLM(
        model_name=model,
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
    )
    tools: List[BaseTool] = [TodoListTool(conv_id=conv_id)]
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=10,
        handle_parsing_errors=True,
    )


def _history_context(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for turn in history[-8:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="JarvisChat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    model: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not request.model.strip():
        raise HTTPException(status_code=400, detail="Model must be specified.")

    conv_id = request.conversation_id or str(uuid.uuid4())
    history = conversations.get(conv_id, [])

    # Build full input for the agent (includes prior context)
    context = _history_context(history)
    full_input = (
        f"{context}\n\nUser: {request.message}" if context else request.message
    )

    try:
        if _is_trivial(request.message):
            # Skip agent overhead for simple greetings
            llm = OllamaCloudLLM(
                model_name=request.model,
                base_url=OLLAMA_BASE_URL,
                api_key=OLLAMA_API_KEY,
            )
            result = llm._generate(
                [
                    SystemMessage(content="You are JarvisChat, a helpful AI assistant."),
                    HumanMessage(content=request.message),
                ]
            )
            response_text: str = result.generations[0].message.content
        else:
            executor = _build_agent_executor(request.model, conv_id)
            agent_result = executor.invoke({"input": full_input})
            response_text = agent_result.get("output", "No response generated.")

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama API error: {exc.response.status_code} {exc.response.text[:200]}",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))

    # Persist conversation turn
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": response_text})
    conversations[conv_id] = history

    return ChatResponse(response=response_text, conversation_id=conv_id)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "JarvisChat"}


# ---------------------------------------------------------------------------
# Serve frontend static files (when deployed as a single service)
# ---------------------------------------------------------------------------

_frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_path):
    app.mount(
        "/",
        StaticFiles(directory=_frontend_path, html=True),
        name="frontend",
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
