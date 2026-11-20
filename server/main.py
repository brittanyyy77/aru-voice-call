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

# MCP routes
if os.getenv("MCP_ENABLED", "true").lower() == "true":
    setup_mcp_routes(app, session_manager)


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


@app.websocket("/ws/call/{session_id}")
async def voice_call(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"Voice call connected: {session_id}")

    session = session_manager.get_or_create(session_id)
    transcriber = session.transcriber
    emotion_analyzer = session.emotion_analyzer
    tts_engine = session.tts_engine
    llm_client = session.llm_client

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                audio_chunk = data["bytes"]

                # Transcribe
                result = await transcriber.process_chunk(audio_chunk)

                if result and result.get("text"):
                    # Analyze emotion from accumulated audio
                    emotion = await emotion_analyzer.analyze(
                        transcriber.get_audio_buffer()
                    )

                    transcript_msg = {
                        "type": "transcript",
                        "text": result["text"],
                        "is_final": result.get("is_final", False),
                        "emotion": emotion,
                    }
                    await websocket.send_json(transcript_msg)

                    # If final transcript, generate response
                    if result.get("is_final"):
                        session.add_transcript(
                            result["text"], emotion
                        )

                        # Get LLM response
                        response_text = await llm_client.chat(
                            result["text"],
                            emotion=emotion,
                            history=session.get_history(),
                        )

                        session.add_response(response_text)

                        # Send text response
                        await websocket.send_json(
                            {
                                "type": "response",
                                "text": response_text,
                            }
                        )

                        # Generate and stream TTS
                        async for audio_chunk in tts_engine.synthesize_stream(
                            response_text
                        ):
                            await websocket.send_bytes(audio_chunk)

                        # Signal end of audio
                        await websocket.send_json(
                            {"type": "audio_end"}
                        )

                        # Clear audio buffer for next utterance
                        transcriber.clear_buffer()

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "end_call":
                    break

    except WebSocketDisconnect:
        logger.info(f"Voice call disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error in voice call: {e}")
        await websocket.send_json(
            {"type": "error", "message": str(e)}
        )
    finally:
        session_manager.end_session(session_id)


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
