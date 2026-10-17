"""FastAPI backend: answers, LiveKit tokens, and the web frontend."""
import os
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import openai
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from livekit import api
from pydantic import BaseModel

import rag

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    await rag.load_knowledge()
    yield


app = FastAPI(title="Faris Avatar Backend", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    sessionId: str | None = None  # same id = same conversation


class AskResponse(BaseModel):
    answer: str


class TokenRequest(BaseModel):
    roomName: str | None = None


@app.post("/api/avatar/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        answer = await rag.ask(request.question, request.sessionId)
    except openai.OpenAIError as error:
        # All configured LLMs failed. One readable line instead of a long traceback.
        print(f"LLM ERROR (all models failed) {getattr(error, 'status_code', '')}: {str(error)[:400]}")
        raise HTTPException(503, "The language model is unavailable right now.")
    return AskResponse(answer=answer)


@app.post("/api/avatar/ask/stream")
async def ask_stream(request: AskRequest):
    async def events():
        async for piece in rag.ask_stream(request.question, request.sessionId):
            for line in piece.split("\n"):
                yield f"data: {line}\n"
            yield "\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/livekit/token")
async def create_token(request: TokenRequest):
    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    if not (url and key and secret):
        raise HTTPException(500, "LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set")

    room_name = (request.roomName or "").strip() or f"avatar-{uuid.uuid4()}"
    token = (
        api.AccessToken(key, secret)
        .with_identity(f"user-{uuid.uuid4()}")
        .with_name("Faris Avatar User")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )
    return {"serverUrl": url, "token": token, "roomName": room_name}


# Must be last so it does not shadow the /api routes.
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))