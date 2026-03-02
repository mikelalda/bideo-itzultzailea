# =============================================================================
# Bideo Itzultzailea - Orchestrator Dockerfile
# =============================================================================

FROM python:3.11-slim-bookworm

LABEL maintainer="Bideo Itzultzailea"
LABEL description="Video translation system ES ↔ EU"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install ffmpeg and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download translation models at build time
RUN python -c "\
from transformers import MarianMTModel, MarianTokenizer; \
MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-eu'); \
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-eu'); \
MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-eu-es'); \
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-eu-es'); \
print('Models downloaded successfully')"

# Copy application code
COPY app/ /app/app/
COPY static/ /app/static/

# Symlink templates for Jinja2
RUN ln -sf /app/app/templates /app/templates

# Create work directories
RUN mkdir -p /app/uploads /app/outputs /app/temp

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
