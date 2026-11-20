import asyncio
import io
import logging
import numpy as np
from faster_whisper import WhisperModel
import os
import soundfile as sf

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self):
        model_size = os.getenv("WHISPER_MODEL", "large-v3")
        device = "cuda" if self._has_gpu() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            f"Loading whisper model: {model_size} on {device}"
        )
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        self.language = os.getenv("LANGUAGE", "zh")
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        self.silence_threshold = 0.01
        self.silence_duration = 0.8  # seconds
        self.last_speech_time = 0
        self.chunk_count = 0
        self.min_audio_length = 0.5  # minimum seconds before transcribing

    def _has_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def process_chunk(self, audio_bytes: bytes) -> dict | None:
        self.audio_buffer.extend(audio_bytes)
        self.chunk_count += 1

        # Convert to numpy for analysis
        audio_data = self._bytes_to_numpy(self.audio_buffer)
        duration = len(audio_data) / self.sample_rate

        if duration < self.min_audio_length:
            return None

        # Check for silence (end of utterance)
        recent_samples = audio_data[-int(self.sample_rate * 0.3):]
        rms = np.sqrt(np.mean(recent_samples**2))

        if rms < self.silence_threshold:
            self.last_speech_time += 0.3
        else:
            self.last_speech_time = 0

        # Transcribe on silence detection or every 3 seconds
        if self.last_speech_time >= self.silence_duration or duration >= 3.0:
            text = await self._transcribe(audio_data)
            if text:
                is_final = self.last_speech_time >= self.silence_duration
                return {"text": text, "is_final": is_final}

        return None

    async def _transcribe(self, audio_data: np.ndarray) -> str:
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

    def _bytes_to_numpy(self, audio_bytes: bytearray) -> np.ndarray:
        try:
            audio_io = io.BytesIO(bytes(audio_bytes))
            audio_data, sr = sf.read(audio_io, dtype="float32")
            if sr != self.sample_rate:
                # Simple resampling
                ratio = self.sample_rate / sr
                n_samples = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), n_samples),
                    np.arange(len(audio_data)),
                    audio_data,
                )
            return audio_data
        except Exception:
            # Fallback: treat as raw 16-bit PCM
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
