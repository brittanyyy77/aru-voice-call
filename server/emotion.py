import asyncio
import io
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """Analyze emotion from audio features using librosa."""

    EMOTIONS = [
        "neutral",
        "happy",
        "sad",
        "angry",
        "anxious",
        "tender",  # 撒娇
        "tired",
    ]

    def __init__(self):
        self.sample_rate = 16000

    async def analyze(self, audio_bytes: bytes) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._analyze_sync, audio_bytes
        )

    def _analyze_sync(self, audio_bytes: bytes) -> dict:
        try:
            import librosa

            # Convert bytes to numpy
            audio_data = np.frombuffer(
                audio_bytes, dtype=np.int16
            ).astype(np.float32)
            audio_data /= 32768.0

            if len(audio_data) < self.sample_rate * 0.5:
                return {"emotion": "neutral", "confidence": 0.5, "features": {}}

            # Extract features
            # Pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, sr=self.sample_rate
            )
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            mean_pitch = (
                float(np.mean(pitch_values)) if len(pitch_values) > 0 else 200.0
            )
            pitch_std = (
                float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
            )

            # Energy / volume
            rms = librosa.feature.rms(y=audio_data)[0]
            mean_energy = float(np.mean(rms))
            energy_std = float(np.std(rms))

            # Speech rate (zero crossing rate as proxy)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            mean_zcr = float(np.mean(zcr))

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            mean_centroid = float(np.mean(spectral_centroid))

            features = {
                "mean_pitch": round(mean_pitch, 1),
                "pitch_variation": round(pitch_std, 1),
                "mean_energy": round(mean_energy, 4),
                "energy_variation": round(energy_std, 4),
                "speech_rate": round(mean_zcr, 4),
                "spectral_centroid": round(mean_centroid, 1),
            }

            emotion = self._classify(features)
            return emotion

        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return {"emotion": "neutral", "confidence": 0.5, "features": {}}

    def _classify(self, features: dict) -> dict:
        """Rule-based emotion classification from audio features."""
        pitch = features["mean_pitch"]
        pitch_var = features["pitch_variation"]
        energy = features["mean_energy"]
        energy_var = features["energy_variation"]
        rate = features["speech_rate"]

        scores = {
            "neutral": 0.3,
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "anxious": 0.0,
            "tender": 0.0,
            "tired": 0.0,
        }

        # High pitch + high energy + high variation = happy/excited
        if pitch > 250 and energy > 0.05 and pitch_var > 50:
            scores["happy"] += 0.6
        # High pitch + soft energy + high variation = tender/撒娇
        if pitch > 250 and energy < 0.04 and pitch_var > 40:
            scores["tender"] += 0.6
        # Low pitch + low energy + low variation = sad
        if pitch < 180 and energy < 0.03 and pitch_var < 30:
            scores["sad"] += 0.6
        # High energy + fast rate + high pitch var = angry
        if energy > 0.06 and rate > 0.1 and energy_var > 0.02:
            scores["angry"] += 0.6
        # High rate + medium pitch var = anxious
        if rate > 0.08 and pitch_var > 30 and energy < 0.05:
            scores["anxious"] += 0.5
        # Low energy + low rate = tired
        if energy < 0.02 and rate < 0.05:
            scores["tired"] += 0.6

        top_emotion = max(scores, key=scores.get)
        confidence = min(scores[top_emotion] + 0.3, 1.0)

        return {
            "emotion": top_emotion,
            "confidence": round(confidence, 2),
            "features": features,
        }
