import requests
import json
from typing import List, Dict, Any, Optional

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "smollm2:135m"):
        self.host = host
        self.model = model
        self.session = requests.Session()

    def list_models(self) -> List[str]:
        """List available models from the Ollama server."""
        try:
            response = self.session.get(f"{self.host}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            raise Exception(f"Failed to list models: {str(e)}")
        
    def chat_with_model(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Chat with the Ollama model using conversation history."""
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        try:
            response = self.session.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise Exception(f"Failed to chat with model: {str(e)}")
        
    def chat_stream_with_model(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024):
        """Stream chat response from the Ollama model."""
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": True
        }
        try:
            with self.session.post(f"{self.host}/api/chat", json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise Exception(f"Failed to stream chat with model: {str(e)}")