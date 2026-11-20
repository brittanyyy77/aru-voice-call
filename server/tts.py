import os
import logging
import httpx
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class TTSEngine:
    """ElevenLabs streaming TTS."""

    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
        self.model = os.getenv("ELEVENLABS_MODEL", "eleven_v4")
        self.base_url = "https://api.elevenlabs.io/v1"

    async def synthesize_stream(
        self, text: str
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks."""
        url = (
            f"{self.base_url}/text-to-speech/{self.voice_id}/stream"
        )
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
                async for chunk in response.aiter_bytes(
                    chunk_size=4096
                ):
                    yield chunk

    async def synthesize(self, text: str) -> bytes:
        """Get complete TTS audio."""
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)
