import logging
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

    def add_transcript(self, text: str, emotion: dict):
        self.transcripts.append(
            {"text": text, "emotion": emotion}
        )
        self.history.append({"role": "user", "content": text})

    def add_response(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def get_history(self) -> list:
        # Keep last 20 turns
        return self.history[-20:]

    def get_latest_transcript(self) -> dict | None:
        return self.transcripts[-1] if self.transcripts else None


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}

    async def initialize(self):
        logger.info("Session manager initialized")

    async def cleanup(self):
        self.sessions.clear()

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id)
        return self.sessions[session_id]

    def end_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[str]:
        return list(self.sessions.keys())
