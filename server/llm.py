import os
import logging
import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client for standalone mode (without Aru MCP)."""

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv(
            "LLM_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = os.getenv("LLM_MODEL", "gpt-4o")
        self.system_prompt = os.getenv(
            "SYSTEM_PROMPT",
            "You are a caring AI companion. Respond naturally and warmly.",
        )

    async def chat(
        self,
        user_text: str,
        emotion: dict | None = None,
        history: list | None = None,
    ) -> str:
        messages = [{"role": "system", "content": self._build_system(emotion)}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_text})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.8,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return (
                    data["choices"][0]["message"]["content"].strip()
                )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "抱歉，我没听清，你再说一遍呢。"

    def _build_system(self, emotion: dict | None) -> str:
        prompt = self.system_prompt
        if emotion and emotion.get("emotion") != "neutral":
            prompt += (
                f"\n\n[Voice emotion detected: {emotion['emotion']} "
                f"(confidence: {emotion.get('confidence', 0)})]"
            )
        return prompt
