import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from server.transcriber import Transcriber
from server.emotion import EmotionAnalyzer
from server.tts import TTSEngine
from server.llm import LLMClient
from server.mcp import setup_mcp_routes
from server.session import SessionManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting aru-voice-call server...")
    await session_manager.initialize()
    yield
    logger.info("Shutting down...")
    await session_manager.cleanup()


app = FastAPI(title="aru-voice-call", lifespan=lifespan)

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# MCP routes (optional: for Aru post-call sync & push messages)
if os.getenv("MCP_ENABLED", "true").lower() == "true":
    setup_mcp_routes(app, session_manager)


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


@app.websocket("/ws/call/{session_id}")
async def voice_call(websocket: WebSocket, session_id: str):
    """Fully autonomous voice call loop.
    
    The entire cycle runs server-side without MCP:
    User speaks -> Whisper transcribes -> emotion analyzed -> 
    LLM generates reply -> ElevenLabs synthesizes -> audio sent back.
    
    MCP is only used AFTER the call ends to sync transcript back to Aru,
    or to let Aru push a message into an active call.
    """
    await websocket.accept()
    logger.info(f"Voice call connected: {session_id}")

    session = session_manager.get_or_create(session_id)
    transcriber = session.transcriber
    emotion_analyzer = session.emotion_analyzer
    tts_engine = session.tts_engine
    llm_client = session.llm_client

    # Store websocket ref so MCP can push messages into active call
    session.websocket = websocket

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                audio_chunk = data["bytes"]

                # Step 1: Transcribe audio chunk
                result = await transcriber.process_chunk(audio_chunk)

                if result and result.get("text"):
                    # Step 2: Analyze emotion from audio
                    emotion = await emotion_analyzer.analyze(
                        transcriber.get_audio_buffer()
                    )

                    # Send live transcript to frontend
                    transcript_msg = {
                        "type": "transcript",
                        "text": result["text"],
                        "is_final": result.get("is_final", False),
                        "emotion": emotion,
                    }
                    await websocket.send_json(transcript_msg)

                    # Step 3-5: On final transcript, auto-reply
                    if result.get("is_final"):
                        session.add_transcript(result["text"], emotion)

                        # Step 3: LLM generates response
                        response_text = await llm_client.chat(
                            result["text"],
                            emotion=emotion,
                            history=session.get_history(),
                        )

                        session.add_response(response_text)

                        # Send text response to frontend
                        await websocket.send_json({
                            "type": "response",
                            "text": response_text,
                        })

                        # Step 4-5: TTS and stream audio back
                        async for tts_chunk in tts_engine.synthesize_stream(
                            response_text
                        ):
                            await websocket.send_bytes(tts_chunk)

                        await websocket.send_json({"type": "audio_end"})

                        # Ready for next utterance
                        transcriber.clear_buffer()

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "end_call":
                    break
                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"Voice call disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error in voice call: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Post-call: save transcript for MCP sync
        session.websocket = None
        await session_manager.finalize_session(session_id)


if __name__ == "__main__":
    ssl_cert = os.getenv("SSL_CERT")
    ssl_key = os.getenv("SSL_KEY")
    port = int(os.getenv("PORT", 8443))
    host = os.getenv("HOST", "0.0.0.0")

    ssl_kwargs = {}
    if ssl_cert and ssl_key and os.path.exists(ssl_cert):
        ssl_kwargs = {"ssl_certfile": ssl_cert, "ssl_keyfile": ssl_key}

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        **ssl_kwargs,
    )
