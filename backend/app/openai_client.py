import os
import json
import requests
from typing import List, Dict, Any


class OpenRouterClient:
    """Client for OpenRouter (OpenAI compatible) API.

    It implements the subset of methods used by the FastAPI backend:
    * list_models – returns a placeholder list for health checks.
    * chat_with_model – returns the full response text.
    * chat_stream_with_model – yields partial response chunks.
    """

    def __init__(self, api_key: str = None, model: str = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided via argument or OPENROUTER_API_KEY env var")
        self.model = model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
            "X-Title": os.getenv("OPENROUTER_X_TITLE", "LLM Deploy Demo"),
            "Content-Type": "application/json",
        })

    def _post(self, endpoint: str, json_data: Dict[str, Any], stream: bool = False):
        url = f"{self.base_url}{endpoint}"
        return self.session.post(url, json=json_data, stream=stream, timeout=30)

    def list_models(self) -> List[str]:
        """Return a placeholder list of models for health check."""
        return [self.model]

    def chat_with_model(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Non-streaming chat request.

        Returns the assistant's reply as a plain string.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = self._post("/chat/completions", payload)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def chat_stream_with_model(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024):
        """Streaming chat request.

        Yields partial content strings as they arrive.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        response = self._post("/chat/completions", payload, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    decoded = decoded[6:]
                if decoded.strip() == "[DONE]":
                    break
                chunk = json.loads(decoded)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
            except Exception:
                continue