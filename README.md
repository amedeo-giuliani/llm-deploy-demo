# Ollama + FastAPI + Streamlit MVP

A production-like MVP demonstrating LLM integration with a web interface.

---

## Features

- **Backend:** FastAPI exposing `/generate` endpoint for LLM inference.
- **LLM runtime:** Ollama (e.g., Qwen3:1.7B) running inside the same container (CPU-only).
- **Frontend:** Streamlit GUI for interacting with the LLM, deployable via Railway or Streamlit Cloud.
- **Deployment-ready:** Dockerized services with `docker-compose.yml` for local testing or Railway deployment.
- **Modular architecture:** Swap Ollama with other LLM providers by changing environment variables only.
