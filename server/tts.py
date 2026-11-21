import os
import logging
import httpx
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class TTSEngine:
    """Multi-provider streaming TTS.
    Supports: elevenlabs, minimax
    """

    def __init__(self):
        self.provider = os.getenv("TTS_PROVIDER", "elevenlabs")

        if self.provider == "elevenlabs":
            self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
            self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
            self.model = os.getenv("ELEVENLABS_MODEL", "eleven_v4")
            self.base_url = "https://api.elevenlabs.io/v1"
        elif self.provider == "minimax":
            self.api_key = os.getenv("MINIMAX_API_KEY", "")
            self.group_id = os.getenv("MINIMAX_GROUP_ID", "")
            self.voice_id = os.getenv("MINIMAX_VOICE_ID", "male-qn-qingse")
            self.model = os.getenv("MINIMAX_TTS_MODEL", "speech-02-turbo")
            self.base_url = "https://api.minimax.chat/v1"
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")

        logger.info(f"TTS provider: {self.provider}")

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        if self.provider == "elevenlabs":
            async for chunk in self._elevenlabs_stream(text):
                yield chunk
        elif self.provider == "minimax":
            async for chunk in self._minimax_stream(text):
                yield chunk

    async def _elevenlabs_stream(self, text: str) -> AsyncIterator[bytes]:
        url = f"{self.base_url}/text-to-speech/{self.voice_id}/stream"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model,
            "output_format": "pcm_24000",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST", url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    yield chunk

    async def _minimax_stream(self, text: str) -> AsyncIterator[bytes]:
        url = f"{self.base_url}/t2a_v2"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "text": text,
            "stream": True,
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": 1.0,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 24000,
                "format": "pcm",
            },
        }
        if self.group_id:
            params = {"GroupId": self.group_id}
        else:
            params = {}

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST", url, headers=headers, json=payload, params=params
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    # MiniMax streaming returns audio chunks directly
                    yield chunk

    async def synthesize(self, text: str) -> bytes:
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)
