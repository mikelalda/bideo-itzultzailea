# =============================================================================
# Bideo Itzultzailea - Konfigurazioa
# =============================================================================
# Eredu guztiak CPU-n exekutatzen dira. Ereduen bertsioak (revision) finkatuta
# daude, irudia berreraikitzean emaitza berdinak lortzeko.
# =============================================================================

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Direktorioak
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
TEMP_DIR = DATA_DIR / "temp"

APP_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = APP_ROOT / "static"
TEMPLATES_DIR = APP_ROOT / "app" / "templates"

# Gaztelaniazko ahotsen klon-erreferentziak (eraikitzean sortzen dira)
VOICE_CACHE_DIR = Path(os.getenv("VOICE_CACHE_DIR", "/opt/models/voices"))

# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
NUM_THREADS = int(os.getenv("NUM_THREADS", str(os.cpu_count() or 2)))
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "1") == "1"

# ---------------------------------------------------------------------------
# STT - NVIDIA NeMo Conformer-Transducer (hizkuntza bakoitzeko bat)
# (repo, fitxategia, revision)
# ---------------------------------------------------------------------------
STT_MODELS = {
    # NGC-ko stt_es_conformer_transducer_large bera, HF-n argitaratua
    "es": (
        "nvidia/stt_es_conformer_transducer_large",
        "stt_es_conformer_transducer_large.nemo",
        "690e0ca5918972f131adfda2745fc6cef3cecafe",
    ),
    "eu": (
        "HiTZ/stt_eu_conformer_transducer_large_v2",
        "stt_eu_conformer_transducer_large_v2.nemo",
        "3037d48dba04505601f1c6ff7a9516a9a6688e44",
    ),
}
STT_BATCH_SIZE = int(os.getenv("STT_BATCH_SIZE", "4"))

# VAD (silero) - audioa esalditan zatitzeko
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "350"))
VAD_MAX_SEGMENT_S = float(os.getenv("VAD_MAX_SEGMENT_S", "15"))
VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "120"))

# ---------------------------------------------------------------------------
# MT - HiTZ MarianMT
# ---------------------------------------------------------------------------
MT_MODELS = {
    ("es", "eu"): ("HiTZ/mt-hitz-es-eu", "87ea37bd573f5c268207d967b6357a057949c151"),
    ("eu", "es"): ("HiTZ/mt-hitz-eu-es", "9a3819054a43d2b7c4b8a1f4ce62647722f622c7"),
}
MT_BATCH_SIZE = int(os.getenv("MT_BATCH_SIZE", "8"))
MT_NUM_BEAMS = int(os.getenv("MT_NUM_BEAMS", "4"))
MT_LENGTH_PENALTY = float(os.getenv("MT_LENGTH_PENALTY", "0.6"))

# ---------------------------------------------------------------------------
# TTS - Pocket TTS (ONNX, onnxruntime)
# ---------------------------------------------------------------------------
TTS_EU_REPO = "itzune/pocket-tts-basque-onnx"
TTS_EU_REVISION = "96344d1108004e22864036a6adb6e6f5b90d484c"
TTS_EU_PATTERNS = [
    "*int8.onnx", "mimi_encoder.onnx", "text_conditioner.onnx",
    "*.json", "*.npy", "*.model", "voices/*",
]
TTS_EU_TEMPERATURE = float(os.getenv("TTS_EU_TEMPERATURE", "0.3"))

TTS_ES_REPO = "KevinAHM/pocket-tts-onnx"
TTS_ES_REVISION = "58a6d00cf13d239b6748cb0769f35c580a8f606c"
TTS_ES_LANGUAGE = "spanish"
TTS_ES_PATTERNS = [
    f"onnx/{TTS_ES_LANGUAGE}/*int8.onnx",
    f"onnx/{TTS_ES_LANGUAGE}/mimi_encoder.onnx",
    f"onnx/{TTS_ES_LANGUAGE}/text_conditioner.onnx",
    f"onnx/{TTS_ES_LANGUAGE}/*.json",
    f"onnx/{TTS_ES_LANGUAGE}/*.npy",
    f"onnx/{TTS_ES_LANGUAGE}/*.model",
]
TTS_ES_TEMPERATURE = float(os.getenv("TTS_ES_TEMPERATURE", "0.5"))

TTS_SAMPLE_RATE = 24000

# Generoa -> ahotsa. Gaztelaniaz ere Maider eta Antton erabiltzen dira:
# euskal ereduarekin erreferentzia-klip bat sortu eta gaztelaniazko ereduak
# klip horretatik klonatzen du ahotsa (zero-shot).
VOICES = {"female": "Maider", "male": "Antton"}

# Gaztelaniazko klonaziorako erreferentzia-esaldia (euskaraz, ~6-8 s)
VOICE_PROMPT_TEXT = (
    "Kaixo, zer moduz zaude? Gaur eguraldi ona dago, eta arratsaldean "
    "lagunekin paseatzera joango naiz hondartzara."
)

# ---------------------------------------------------------------------------
# Generoa (ahots-tonua, F0) - esaldi bakoitzeko ahotsa erabakitzeko
# ---------------------------------------------------------------------------
# F0 mediana muga honen gainetik -> emakumea (Maider), azpitik -> gizona
F0_THRESHOLD_HZ = float(os.getenv("F0_THRESHOLD_HZ", "165"))

# ---------------------------------------------------------------------------
# Denbora-doiketa
# ---------------------------------------------------------------------------
MAX_SPEEDUP = float(os.getenv("MAX_SPEEDUP", "1.6"))
