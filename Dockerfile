# =============================================================================
# Bideo Itzultzailea - CPU irudia (Hugging Face Spaces eta tokikoa)
#
#   STT: NVIDIA NeMo Conformer-Transducer (es: nvidia, eu: HiTZ v2)
#   MT : HiTZ/mt-hitz-es-eu eta HiTZ/mt-hitz-eu-es (MarianMT)
#   TTS: Pocket TTS ONNX (eu: itzune/pocket-tts-basque-onnx, es: Kyutai spanish)
#
# Prozesu bakarra (uvicorn) 7860 atakan. Ereduak eraikitzean deskargatzen dira.
# HF Spaces-ek 1000 UID-a duen erabiltzailearekin exekutatzen du edukiontzia.
# =============================================================================
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VOICE_CACHE_DIR=/opt/models/voices \
    DATA_DIR=/app/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

# torch CPU bertsioa (CUDA gabe, irudia askoz txikiagoa)
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Ereduak deskargatu + gaztelaniazko ahots-erreferentziak sortu. Behar diren
# moduluak bakarrik kopiatzen dira lehenik, UI-ko aldaketek (main.py, static/)
# geruza hau ez baliogabetzeko eta ereduak berriz ez deskargatzeko.
COPY app/__init__.py app/config.py app/textnorm.py app/translator.py app/tts.py /app/app/
COPY app/vendor/ /app/app/vendor/
COPY scripts/ /app/scripts/
RUN python /app/scripts/prepare_models.py \
    && mkdir -p /app/data \
    && chown -R user:user /app/data /opt/models

COPY app/ /app/app/
COPY static/ /app/static/

USER user
ENV HOME=/home/user \
    HF_HUB_OFFLINE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
