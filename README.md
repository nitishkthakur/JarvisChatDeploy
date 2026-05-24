# JarvisChat

A minimal chat application that lets you converse with Ollama Cloud models through an agentic backend built with FastAPI and LangChain.

---

## Features

- Clean, distraction-free chat interface with a black background
- Six Ollama Cloud model options selectable from a dropdown
- Agentic backend: uses a LangChain ReAct agent with a `TodoListTool` for structured, multi-step planning
- No local Ollama installation required — communicates directly with the Ollama Cloud HTTPS API
- API key stays server-side only; never exposed to the browser

---

## Project structure

```
JarvisChat/
├── backend/
│   ├── main.py            # FastAPI app + LangChain agent
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html         # Single-page chat UI
├── Procfile               # Render / Railway start command
├── render.yaml            # Render deployment config
└── README.md
```

---

## Local development

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Set environment variables

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set your OLLAMA_API_KEY
```

### 3. Start the server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The server serves the frontend at `http://localhost:8000` and exposes:

| Endpoint      | Method | Description              |
|---------------|--------|--------------------------|
| `/chat`       | POST   | Send a chat message      |
| `/health`     | GET    | Health check             |

### Chat request format

```json
{
  "message": "Explain quantum computing",
  "model": "qwen3.5:397b-cloud",
  "conversation_id": "optional-uuid"
}
```

---

## Deployment on Render

1. Push this repository to GitHub.
2. Create a new **Web Service** on [Render](https://render.com) pointing to the repository.
3. Render will detect `render.yaml` automatically.
4. Set the `OLLAMA_API_KEY` secret in the Render dashboard under **Environment**.

---

## Deployment on Railway

1. Push this repository to GitHub.
2. Create a new project on [Railway](https://railway.app) and connect the repository.
3. Railway will detect the `Procfile` and use it as the start command.
4. Add `OLLAMA_API_KEY` (and optionally `OLLAMA_BASE_URL`) as environment variables in the Railway dashboard.

---

## Environment variables

| Variable          | Required | Default                      | Description                    |
|-------------------|----------|------------------------------|--------------------------------|
| `OLLAMA_API_KEY`  | Yes      | —                            | Ollama Cloud API key           |
| `OLLAMA_BASE_URL` | No       | `https://api.ollama.com`     | Ollama Cloud base URL          |
| `PORT`            | No       | `8000`                       | Port the server listens on     |

---

## Supported models

| Model                    |
|--------------------------|
| minimax-m2.7:cloud       |
| nemotron-3-super:cloud   |
| qwen3.5:397b-cloud       |
| glm-5.1:cloud            |
| kimi-k2.6:cloud          |
| deepseek-v4-flash:cloud  |