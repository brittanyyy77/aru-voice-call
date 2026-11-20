import json
import logging
import time
from server.transcriber import Transcriber
from server.emotion import EmotionAnalyzer
from server.tts import TTSEngine
from server.llm import LLMClient

logger = logging.getLogger(__name__)


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.transcriber = Transcriber()
        self.emotion_analyzer = EmotionAnalyzer()
        self.tts_engine = TTSEngine()
        self.llm_client = LLMClient()
        self.history = []
        self.transcripts = []
        self.websocket = None  # Active WebSocket ref for MCP push
        self.started_at = time.time()
        self.ended_at = None
        self.is_active = True

    def add_transcript(self, text: str, emotion: dict):
        self.transcripts.append({
            "role": "user",
            "text": text,
            "emotion": emotion,
            "timestamp": time.time(),
        })
        self.history.append({"role": "user", "content": text})

    def add_response(self, text: str):
        self.transcripts.append({
            "role": "assistant",
            "text": text,
            "timestamp": time.time(),
        })
        self.history.append({"role": "assistant", "content": text})

    def get_history(self) -> list:
        # Keep last 20 turns
        return self.history[-20:]

    def get_latest_transcript(self) -> dict | None:
        return self.transcripts[-1] if self.transcripts else None

    def get_full_transcript(self) -> list:
        """Get complete call transcript for post-call sync."""
        return self.transcripts

    def get_call_summary(self) -> dict:
        """Summary for MCP sync after call ends."""
        duration = (self.ended_at or time.time()) - self.started_at
        emotions = [t.get("emotion", {}).get("emotion", "neutral")
                    for t in self.transcripts if t["role"] == "user"]
        return {
            "session_id": self.session_id,
            "duration_seconds": round(duration),
            "total_turns": len(self.history),
            "dominant_emotion": max(set(emotions), key=emotions.count) if emotions else "neutral",
            "transcript": self.transcripts,
        }


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.completed_sessions: dict[str, dict] = {}  # Post-call data for MCP

    async def initialize(self):
        logger.info("Session manager initialized")

    async def cleanup(self):
        self.sessions.clear()
        self.completed_sessions.clear()

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id)
        return self.sessions[session_id]

    async def finalize_session(self, session_id: str):
        """Called when a call ends. Saves transcript for MCP retrieval."""
        session = self.sessions.pop(session_id, None)
        if session:
            session.ended_at = time.time()
            session.is_active = False
            summary = session.get_call_summary()
            self.completed_sessions[session_id] = summary
            logger.info(
                f"Call {session_id} ended: {summary['duration_seconds']}s, "
                f"{summary['total_turns']} turns, "
                f"mood: {summary['dominant_emotion']}"
            )
            # Keep only last 10 completed sessions
            if len(self.completed_sessions) > 10:
                oldest = next(iter(self.completed_sessions))
                del self.completed_sessions[oldest]

    def end_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[str]:
        return list(self.sessions.keys())

    def get_completed_session(self, session_id: str) -> dict | None:
        return self.completed_sessions.get(session_id)

    def list_completed_sessions(self) -> list[str]:
        return list(self.completed_sessions.keys())
