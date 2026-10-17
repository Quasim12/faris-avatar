
"""Knowledge loading (RAG) and answer generation for the Faris avatar."""
import os
import time
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env.local")

OWNER = os.getenv("OWNER_NAME", "Faris")
ASSISTANT = os.getenv("ASSISTANT_NAME", "Alex")

# Any OpenAI-compatible provider works (Gemini, Groq, OpenRouter, Ollama, OpenAI...).
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None  # None = OpenAI
CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Embeddings may come from a different provider than chat (defaults to the same one).
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or LLM_API_KEY
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL") or LLM_BASE_URL

TOP_K = 4
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50
MAX_HISTORY = int(os.getenv("HISTORY_MESSAGES", "12"))

SYSTEM_PROMPT = f"""You are a MALE AI assistant of {OWNER}, named {ASSISTANT}.
You have a male persona. Always use masculine Hindi/Hinglish expressions
when referring to yourself.

For example, say:
- "Main aapki baat sun raha hoon."
- "Main aapki madad kar sakta hoon."
- "Main check kar raha hoon."
- "Main samajh gaya."

Do not use feminine expressions such as:
- "Main sun rahi hoon."
- "Main kar sakti hoon."
- "Main samajh gayi."

Your job is to answer questions using {OWNER}'s retrieved knowledge.

Speak in natural Hinglish using Roman script, while keeping technical terms
such as Spring Boot, Docker, JVM, RAG, Vector Database, API, Dependency
Injection, etc. in English.

Keep answers short, conversational and easy to understand. Prefer around 1-2
spoken sentences unless the user explicitly asks for a detailed explanation.
Do not use bullet points, markdown or emojis, because your answer is spoken aloud.

Use the retrieved knowledge as the source of truth for anything related to
{OWNER}'s work, teaching, examples, analogies or views.

If the retrieved knowledge does not contain something specifically about
{OWNER} (for example personal details such as friends, hobbies or places), do
not guess or invent it. Say clearly that you do not have that information.
For general technical questions you may explain in simple words, but never
attribute your own explanation to {OWNER}. Never create facts about {OWNER}.

Use the earlier messages of this conversation to understand follow-up questions.

VERY IMPORTANT: when the user says "okay bye {ASSISTANT}" or "okay goodbye
{ASSISTANT}", reply with a warm goodbye: thank them for talking to you and
wish them a great day.
"""

_chat_client = None
_embed_client = None
_enc = tiktoken.get_encoding("cl100k_base")
_chunks: list[dict] = []
_vectors = np.zeros((0, 1), dtype=np.float32)
_history: dict[str, list[dict]] = {}


def _load_providers() -> list[dict]:
    """Main LLM first, then optional fallbacks LLM_2_* ... LLM_6_*."""
    providers = [
        {
            "id": 1,
            "key": LLM_API_KEY,
            "base_url": LLM_BASE_URL,
            "model": CHAT_MODEL,
        }
    ]

    for i in range(2, 7):
        model = os.getenv(f"LLM_{i}_MODEL")
        if not model:
            continue

        key = os.getenv(f"LLM_{i}_API_KEY")
        base_url = os.getenv(f"LLM_{i}_BASE_URL")

        if base_url and base_url != LLM_BASE_URL and not key:
            print(
                f"WARNING: LLM_{i}_BASE_URL is set without "
                f"LLM_{i}_API_KEY, skipping fallback {i}."
            )
            continue

        providers.append(
            {
                "id": i,
                "key": key or LLM_API_KEY,
                "base_url": base_url or LLM_BASE_URL,
                "model": model,
            }
        )

    return providers


PROVIDERS = _load_providers()
_clients: dict[tuple, AsyncOpenAI] = {}
_cooldown: dict[int, float] = {}


def client_for(provider: dict) -> AsyncOpenAI:
    cache_key = (provider["key"], provider["base_url"])

    if cache_key not in _clients:
        # Fail fast and let the next provider answer.
        _clients[cache_key] = AsyncOpenAI(
            api_key=provider["key"],
            base_url=provider["base_url"],
            max_retries=0,
            timeout=30.0,
        )

    return _clients[cache_key]


def chat_client() -> AsyncOpenAI:
    return client_for(PROVIDERS[0])


async def _chat(messages: list[dict], stream: bool = False):
    """Try each LLM in order; move to the next one when a call fails."""
    now = time.monotonic()
    order = [
        p for p in PROVIDERS
        if _cooldown.get(p["id"], 0) <= now
    ] or PROVIDERS

    last_error = None

    for provider in order:
        try:
            result = await client_for(provider).chat.completions.create(
                model=provider["model"],
                messages=messages,
                stream=stream,
            )

            _cooldown.pop(provider["id"], None)

            if provider["id"] != 1:
                print(
                    f"LLM: answered by fallback {provider['id']} "
                    f"({provider['model']})"
                )

            return result

        except (openai.APIStatusError, openai.APIConnectionError) as error:
            status = getattr(error, "status_code", None)

            seconds = (
                600 if status == 429
                else 3600 if status in (401, 403, 404)
                else 60
            )

            _cooldown[provider["id"]] = time.monotonic() + seconds

            print(
                f"LLM {provider['id']} ({provider['model']}) failed with "
                f"{status or type(error).__name__}; "
                f"skipping it for {max(seconds // 60, 1)} min"
            )

            last_error = error

    raise last_error


def embed_client() -> AsyncOpenAI:
    global _embed_client

    if _embed_client is None:
        _embed_client = AsyncOpenAI(
            api_key=EMBED_API_KEY,
            base_url=EMBED_BASE_URL,
        )

    return _embed_client


def _split(text: str) -> list[str]:
    tokens = _enc.encode(text)
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    pieces = []

    for start in range(0, len(tokens), step):
        piece = _enc.decode(
            tokens[start:start + CHUNK_TOKENS]
        ).strip()

        if piece and (start == 0 or len(piece) >= 100):
            pieces.append(piece)

        if start + CHUNK_TOKENS >= len(tokens):
            break

    return pieces


async def _embed(texts: list[str]) -> np.ndarray:
    vectors = []

    for i in range(0, len(texts), 100):
        res = await embed_client().embeddings.create(
            model=EMBED_MODEL,
            input=texts[i:i + 100],
        )
        vectors.extend(d.embedding for d in res.data)

    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


async def load_knowledge() -> None:
    """Read knowledge/*.md, split into chunks, embed, and keep in memory."""
    global _chunks, _vectors

    chunks = []

    for path in sorted((BASE_DIR / "knowledge").glob("*.md")):
        for piece in _split(path.read_text(encoding="utf-8")):
            chunks.append(
                {
                    "source": path.name,
                    "text": piece,
                }
            )

    if chunks:
        _vectors = await _embed([c["text"] for c in chunks])

    _chunks = chunks
    print(f"Loaded {len(chunks)} knowledge chunks.")


async def _retrieve(question: str) -> list[dict]:
    if not _chunks:
        return []

    query = (await _embed([question]))[0]
    scores = _vectors @ query
    best = np.argsort(scores)[::-1][:TOP_K]

    return [_chunks[i] for i in best]


def _remember(
    session_id: str | None,
    question: str,
    answer: str,
) -> None:
    """Store the exchange so the model can answer follow-up questions."""
    if not session_id or not answer:
        return

    if session_id not in _history and len(_history) >= 200:
        _history.pop(next(iter(_history)))

    messages = _history.setdefault(session_id, [])
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    del messages[:-MAX_HISTORY]


async def _messages(
    question: str,
    session_id: str | None,
) -> list[dict]:
    hits = await _retrieve(question)
    context = "\n\n---\n\n".join(
        h["text"] for h in hits
    ) or "(no knowledge found)"

    return [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nRetrieved knowledge:\n{context}",
        },
        *_history.get(session_id or "", []),
        {"role": "user", "content": question},
    ]


async def ask(
    question: str,
    session_id: str | None = None,
) -> str:
    res = await _chat(await _messages(question, session_id))
    answer = res.choices[0].message.content or ""

    _remember(session_id, question, answer)
    return answer


async def ask_stream(
    question: str,
    session_id: str | None = None,
):
    stream = await _chat(
        await _messages(question, session_id),
        stream=True,
    )

    parts = []

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            parts.append(text)
            yield text

    _remember(session_id, question, "".join(parts))
