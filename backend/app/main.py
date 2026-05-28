from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from .ollama_client import OllamaClient
from .openai_client import OpenRouterClient
import os

app = FastAPI(title="Ollama API", description="API to interact with Ollama language models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine which LLM provider to use. Default is Ollama.
provider = os.getenv("LLM_PROVIDER", "ollama").lower()
if provider == "openrouter":
    # OpenRouter configuration via environment variables
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    print("OpenRouter API Key:", openrouter_api_key)
    openrouter_model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.5-air:free")
    client = OpenRouterClient(api_key=openrouter_api_key, model=openrouter_model)
else:
    # Ollama configuration
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "tinyllama:latest")
    client = OllamaClient(host=ollama_host, model=ollama_model)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024

class GenerateResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    """Root endpoint to verify Ollama API is running."""
    return {"message": "Ollama API is running."}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        models = client.list_models()
        return {"status": "healthy", "models": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama server is unavailable: {str(e)}")
    
@app.post("/chat")
async def chat(request: ChatRequest):
    """Stream a response from the Ollama model based on chat messages."""
    try:
        messages_dict = [msg.dict() for msg in request.messages]

        def token_generator():
            for chunk in client.chat_stream_with_model(
                messages=messages_dict,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield chunk
        return StreamingResponse(token_generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")