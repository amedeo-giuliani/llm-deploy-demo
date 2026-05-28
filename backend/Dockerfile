FROM python:3.11-slim

# Build argument to choose provider (default Ollama)
ARG LLM_PROVIDER=ollama
ENV LLM_PROVIDER=${LLM_PROVIDER}

WORKDIR /app

RUN apt update && apt install -y \
    curl zstd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama only when the provider is Ollama
RUN if [ "$LLM_PROVIDER" = "ollama" ]; then \
    curl -fsSL https://ollama.com/install.sh | sh; \
fi

COPY . .

EXPOSE 8000

# Create startup script conditionally
RUN cat <<'EOF' > /app/start.sh
#!/bin/bash
set -e

if [ "$LLM_PROVIDER" = "ollama" ]; then
    # Start Ollama in the background
    ollama serve &
    echo "Waiting for Ollama to start..."
    until curl -s http://localhost:11434/api/tags > /dev/null; do
        sleep 1
    done
    echo "Ollama is ready!"
    # Pull the model if not already present
    if ! ollama list | grep -q "${MODEL_NAME:-tinyllama:latest}"; then
        echo "Pulling model ${MODEL_NAME:-tinyllama:latest}..."
        ollama pull ${MODEL_NAME:-tinyllama:latest}
    fi
fi

echo "Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
EOF
RUN chmod +x /app/start.sh

# 9. The Final Command
CMD ["/app/start.sh"]