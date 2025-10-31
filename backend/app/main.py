from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from .ollama_client import OllamaClient
import os

app = FastAPI(title="Ollama API", description="API to interact with Ollama language models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "tinyllama:latest")
ollama_client = OllamaClient(host=ollama_host, model=ollama_model)

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
        models = ollama_client.list_models()
        return {"status": "healthy", "models": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama server is unavailable: {str(e)}")
    
@app.post("/chat")
async def chat(request: ChatRequest):
    """Stream a response from the Ollama model based on chat messages."""
    try:
        messages_dict = [msg.dict() for msg in request.messages]

        def token_generator():
            for chunk in ollama_client.chat_stream_with_model(
                messages=messages_dict,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield chunk
        return StreamingResponse(token_generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")