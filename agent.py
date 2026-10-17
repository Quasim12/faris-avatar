"""LiveKit voice + Synthesia avatar worker. Answers come from server.py."""
import asyncio
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import synthesia

load_dotenv(Path(__file__).parent / ".env.local")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080/api/avatar/ask")
AVATAR_ID = os.environ["SYNTHESIA_AVATAR_ID"]
VOICE_ID = os.environ["CARTESIA_VOICE_ID"]

server = AgentServer()
HTTP = httpx.AsyncClient(timeout=60.0)  # one shared client, reused for every question



async def process_question(session: AgentSession, question: str, session_id: str):
    print("USER:", question)
    try:
        response = await HTTP.post(BACKEND_URL, json={"question": question, "sessionId": session_id})
        response.raise_for_status()
        answer = response.json()["answer"]
        print("ANSWER:", answer)
        session.say(answer, allow_interruptions=True)
    except Exception as error:
        print("ERROR:", error)
        session.say("Sorry, abhi answer generate karne me problem aa rhi hai.")


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        tts=inference.TTS(model="cartesia/sonic-3.6", voice=VOICE_ID, language="hi"),
    )

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):
        print("HEARD:", "(final)" if event.is_final else "(partial)", event.transcript)
        if event.is_final and event.transcript.strip():
            asyncio.create_task(process_question(session, event.transcript, ctx.room.name))

    print("SYNTHESIA AVATAR ID:", AVATAR_ID)
    try:
        avatar = synthesia.AvatarSession(synthesia.AvatarConfig(avatar_ids=[AVATAR_ID]))
        await avatar.start(session, room=ctx.room)
    except Exception as error:
        # Keep going with voice only so STT, the backend and TTS can still be tested.
        print("WARNING: avatar could not start, continuing with voice only:", error)

    await session.start(
        agent=Agent(
            instructions=(
                "You are the realtime voice adapter for the Faris AI avatar. "
                "Do not generate answers yourself. The backend service is "
                "responsible for all AI reasoning and knowledge."
            )
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)