FROM python:3.11-slim

WORKDIR /app

RUN apt update && apt install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY . .

EXPOSE 8000

RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start Ollama in the background\n\
ollama serve &\n\
\n\
# Wait for Ollama to be ready\n\
echo "Waiting for Ollama to start..."\n\
until curl -s http://localhost:11434/api/tags > /dev/null; do\n\
    sleep 1\n\
done\n\
echo "Ollama is ready!"\n\
\n\
# Pull the model if not already present\n\
if ! ollama list | grep -q "${MODEL_NAME:-smollm2:135m}"; then\n\
    echo "Pulling model ${MODEL_NAME:-smollm2:135m}..."\n\
    ollama pull ${MODEL_NAME:-smollm2:135m}\n\
fi\n\
\n\
# Start the FastAPI server\n\
echo "Starting FastAPI server..."\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000' > /app/start.sh && chmod +x /app/start.sh

# 9. The Final Command
CMD ["/app/start.sh"]