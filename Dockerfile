FROM python:3.11-slim

# Minimal system deps for torch / transformers native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying the rest of the code
# so this layer is cached unless requirements.txt changes.
COPY triage_interview_app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application source (model weights and .env are excluded via .dockerignore
# and injected at runtime via volumes / environment variables).
COPY agentic_pipeline/ /app/agentic_pipeline/
COPY triage_interview_app/ /app/triage_interview_app/

# Both services are run from this directory.
WORKDIR /app/triage_interview_app
