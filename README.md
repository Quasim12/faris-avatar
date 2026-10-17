# Faris AI Avatar (Python)

A realtime talking avatar. Two Python processes plus a browser page:

- `server.py` (FastAPI): answers questions using OpenAI + your knowledge files (RAG), creates LiveKit tokens, and serves the web page from `static/`.
- `agent.py` (LiveKit worker): listens to the microphone (Deepgram), sends each final transcript to `server.py`, and speaks the answer through a Synthesia avatar with a Cartesia voice.
- `static/index.html`: the browser page (plain JavaScript).

## Setup

Requires Python 3.12 (3.10 to 3.14 should work).

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env.local         # then fill in your own keys
```

Put what the avatar should know into `knowledge/*.md`.

## Free model (no OpenAI payment)

`server.py` talks to any OpenAI-compatible API, so only `.env.local` changes:

| Provider | `LLM_BASE_URL` | `LLM_CHAT_MODEL` | `EMBED_MODEL` |
|---|---|---|---|
| Google Gemini (free tier, default in `.env.example`) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` | `gemini-embedding-001` |
| Ollama (runs on your PC, fully free) | `http://localhost:11434/v1` | e.g. `llama3.1` | e.g. `nomic-embed-text` |
| OpenAI (paid) | leave empty | `gpt-4o-mini` | `text-embedding-3-small` |

Use `LLM_API_KEY=ollama` (any text) for Ollama. Free tiers have rate limits and can change; if a model name is rejected, check the provider's current model list. For faster replies try `gemini-2.5-flash-lite`. Chat and embeddings can use different providers via `EMBED_API_KEY` and `EMBED_BASE_URL`.

## Run

Use two terminals from this folder, both with the venv active.

```bash
python server.py        # terminal 1, serves http://localhost:8080
python agent.py dev     # terminal 2, wait until it registers with LiveKit
```

Open http://localhost:8080/, click **Start Conversation**, allow the microphone, and ask a question.

Test the backend alone:

```bash
curl -X POST http://localhost:8080/api/avatar/ask \
  -H 'Content-Type: application/json' -d '{"question":"Tum kaun ho?"}'
```

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/avatar/ask` | `{"question", "sessionId"?}` returns `{"answer"}`; the same `sessionId` keeps conversation history |
| `POST /api/avatar/ask/stream` | same answer as server-sent events |
| `POST /api/livekit/token` | returns `serverUrl`, `token`, `roomName` for the browser |

## How it works

1. Browser joins a LiveKit room with the Synthesia avatar (the face).
2. `agent.py` turns speech into text (Deepgram) and sends it to `server.py` with the room name as `sessionId`.
3. `server.py` embeds the question, finds the 4 most similar chunks from `knowledge/` (the knowledge), adds the system prompt and the conversation history, and asks OpenAI (the brain).
4. The answer goes back to `agent.py`, which speaks it with a Cartesia voice (the voice) while Synthesia lip-syncs it.

History is kept in memory per conversation (last `HISTORY_MESSAGES` messages) and is lost when `server.py` restarts.

## Notes

- Start `agent.py` before opening the browser page.
- Edited `knowledge/` files or `.env.local`? Restart the process.
- A 404 on `/api/livekit/token` means the page is not being served by `server.py`; open it via port 8080.
- Not tested live: no real LiveKit, OpenAI, or Synthesia sessions were run when this was written.
