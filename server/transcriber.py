import asyncio
import io
import logging
import numpy as np
import os
import httpx

logger = logging.getLogger(__name__)


class Transcriber:
    """Speech-to-text with two modes:
    - cloud: Groq/Deepgram API (recommended for small VPS)
    - local: faster-whisper (needs GPU or beefy CPU)
    """

    def __init__(self):
        self.mode = os.getenv("STT_MODE", "cloud")  # cloud or local
        self.language = os.getenv("LANGUAGE", "zh")
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        self.silence_threshold = 0.01
        self.silence_duration = 0.8
        self.last_speech_time = 0
        self.chunk_count = 0
        self.min_audio_length = 0.5

        if self.mode == "local":
            self._init_local()
        else:
            self._init_cloud()

    def _init_local(self):
        from faster_whisper import WhisperModel

        model_size = os.getenv("WHISPER_MODEL", "base")
        device = "cuda" if self._has_gpu() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(f"Loading local whisper: {model_size} on {device}")
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def _init_cloud(self):
        self.stt_provider = os.getenv("STT_PROVIDER", "groq")  # groq or deepgram
        self.stt_api_key = os.getenv("STT_API_KEY", "")

        if self.stt_provider == "groq":
            self.stt_url = "https://api.groq.com/openai/v1/audio/transcriptions"
            self.stt_model = os.getenv("STT_MODEL", "whisper-large-v3")
        elif self.stt_provider == "deepgram":
            self.stt_url = "https://api.deepgram.com/v1/listen"
            self.stt_model = os.getenv("STT_MODEL", "nova-2")
        else:
            raise ValueError(f"Unknown STT provider: {self.stt_provider}")

        logger.info(f"Using cloud STT: {self.stt_provider} ({self.stt_model})")

    def _has_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def process_chunk(self, audio_bytes: bytes) -> dict | None:
        self.audio_buffer.extend(audio_bytes)
        self.chunk_count += 1

        audio_data = self._bytes_to_numpy(self.audio_buffer)
        duration = len(audio_data) / self.sample_rate

        if duration < self.min_audio_length:
            return None

        recent_samples = audio_data[-int(self.sample_rate * 0.3):]
        rms = np.sqrt(np.mean(recent_samples**2))

        if rms < self.silence_threshold:
            self.last_speech_time += 0.3
        else:
            self.last_speech_time = 0

        if self.last_speech_time >= self.silence_duration or duration >= 3.0:
            text = await self._transcribe(audio_data)
            if text:
                is_final = self.last_speech_time >= self.silence_duration
                return {"text": text, "is_final": is_final}

        return None

    async def _transcribe(self, audio_data: np.ndarray) -> str:
        if self.mode == "local":
            return await self._transcribe_local(audio_data)
        else:
            return await self._transcribe_cloud(audio_data)

    async def _transcribe_local(self, audio_data: np.ndarray) -> str:
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            ),
        )
        text = "".join(s.text for s in segments).strip()
        return text

    async def _transcribe_cloud(self, audio_data: np.ndarray) -> str:
        # Convert to WAV bytes for API upload
        import soundfile as sf

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, self.sample_rate, format="WAV")
        wav_bytes = wav_buffer.getvalue()

        try:
            if self.stt_provider == "groq":
                return await self._transcribe_groq(wav_bytes)
            elif self.stt_provider == "deepgram":
                return await self._transcribe_deepgram(wav_bytes)
        except Exception as e:
            logger.error(f"Cloud STT error: {e}")
            return ""

    async def _transcribe_groq(self, wav_bytes: bytes) -> str:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                self.stt_url,
                headers={"Authorization": f"Bearer {self.stt_api_key}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={
                    "model": self.stt_model,
                    "language": self.language,
                    "response_format": "json",
                },
            )
            resp.raise_for_status()
            return resp.json().get("text", "").strip()

    async def _transcribe_deepgram(self, wav_bytes: bytes) -> str:
        params = {
            "model": self.stt_model,
            "language": self.language,
            "smart_format": "true",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                self.stt_url,
                headers={
                    "Authorization": f"Token {self.stt_api_key}",
                    "Content-Type": "audio/wav",
                },
                params=params,
                content=wav_bytes,
            )
            resp.raise_for_status()
            data = resp.json()
            channels = data.get("results", {}).get("channels", [])
            if channels:
                alts = channels[0].get("alternatives", [])
                if alts:
                    return alts[0].get("transcript", "").strip()
            return ""

    def _bytes_to_numpy(self, audio_bytes: bytearray) -> np.ndarray:
        try:
            import soundfile as sf

            audio_io = io.BytesIO(bytes(audio_bytes))
            audio_data, sr = sf.read(audio_io, dtype="float32")
            if sr != self.sample_rate:
                ratio = self.sample_rate / sr
                n_samples = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), n_samples),
                    np.arange(len(audio_data)),
                    audio_data,
                )
            return audio_data
        except Exception:
            audio_data = np.frombuffer(
                bytes(audio_bytes), dtype=np.int16
            ).astype(np.float32)
            audio_data /= 32768.0
            return audio_data

    def get_audio_buffer(self) -> bytes:
        return bytes(self.audio_buffer)

    def clear_buffer(self):
        self.audio_buffer = bytearray()
        self.chunk_count = 0
        self.last_speech_time = 0
