# =============================================================================
# TTS - Pocket TTS (ONNX) euskararako eta gaztelaniarako
# =============================================================================
#   eu: itzune/pocket-tts-basque-onnx  (Maider eta Antton ahotsak barnean)
#   es: KevinAHM/pocket-tts-onnx, Kyutai-ren "spanish" bundle-a
#
# Gaztelaniaz ere Maider/Antton entzuteko: euskal ereduarekin erreferentzia-
# klip labur bat sortzen da ahots bakoitzarekin, eta gaztelaniazko ereduak
# klip horretatik klonatzen du ahotsa (Mimi encoder-a, zero-shot).
# =============================================================================

import logging
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

from app import config
from app.textnorm import normalize
from app.vendor.pocket_tts_onnx import PocketTTSOnnx

logger = logging.getLogger(__name__)

SR = config.TTS_SAMPLE_RATE


def eu_bundle_dir() -> str:
    return snapshot_download(
        config.TTS_EU_REPO, revision=config.TTS_EU_REVISION,
        allow_patterns=config.TTS_EU_PATTERNS,
    )


def es_bundle_dir() -> str:
    root = snapshot_download(
        config.TTS_ES_REPO, revision=config.TTS_ES_REVISION,
        allow_patterns=config.TTS_ES_PATTERNS,
    )
    return str(Path(root) / "onnx")


def trim_silence(audio: np.ndarray, threshold: float = 0.01, margin_s: float = 0.05) -> np.ndarray:
    idx = np.where(np.abs(audio) > threshold)[0]
    if len(idx) == 0:
        return audio
    margin = int(margin_s * SR)
    return audio[max(0, idx[0] - margin): min(len(audio), idx[-1] + margin)]


class SpeechSynthesizer:
    def __init__(self):
        self._engines = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- engines
    def _make_engine(self, lang: str) -> PocketTTSOnnx:
        if lang == "eu":
            engine = PocketTTSOnnx(
                models_dir=eu_bundle_dir(), precision="int8",
                temperature=config.TTS_EU_TEMPERATURE, device="cpu",
            )
            bundle = Path(engine.bundle_dir)
            for name in config.VOICES.values():
                state = engine.prepare_voice_state(str(bundle / "voices" / f"{name}.safetensors"))
                engine._voice_state_cache[f"eu:{name}"] = state
            return engine

        if lang == "es":
            engine = PocketTTSOnnx(
                models_dir=es_bundle_dir(), language=config.TTS_ES_LANGUAGE,
                precision="int8", temperature=config.TTS_ES_TEMPERATURE, device="cpu",
            )
            for name in config.VOICES.values():
                prompt = self.voice_prompt(name)
                embeddings = engine.encode_voice(prompt)
                engine._voice_state_cache[f"es:{name}"] = engine.prepare_voice_state(embeddings)
            return engine

        raise ValueError(f"TTS hizkuntza ez onartua: {lang}")

    def load(self, lang: str) -> PocketTTSOnnx:
        with self._lock:
            if lang not in self._engines:
                logger.info(f"TTS eredua kargatzen ({lang})")
                # gaztelaniak euskal motorra behar du erreferentzia-klipak sortzeko
                if lang == "es" and "eu" not in self._engines and not self._prompts_ready():
                    self._engines["eu"] = self._make_engine("eu")
                self._engines[lang] = self._make_engine(lang)
                logger.info(f"TTS eredua prest ({lang})")
            return self._engines[lang]

    def is_loaded(self, lang: str) -> bool:
        return lang in self._engines

    # ------------------------------------------------- voice clone prompts
    @staticmethod
    def _prompt_path(name: str) -> Path:
        for base in (config.VOICE_CACHE_DIR, config.DATA_DIR / "voices"):
            p = base / f"{name}_prompt.wav"
            if p.exists():
                return p
        return config.VOICE_CACHE_DIR / f"{name}_prompt.wav"

    def _prompts_ready(self) -> bool:
        return all(self._prompt_path(n).exists() for n in config.VOICES.values())

    def voice_prompt(self, name: str) -> str:
        """Ahotsaren erreferentzia-klipa (euskaraz sortua), gaztelaniaz klonatzeko."""
        path = self._prompt_path(name)
        if path.exists():
            return str(path)
        engine = self._engines.get("eu") or self._make_engine("eu")
        self._engines.setdefault("eu", engine)
        np.random.seed(1234)
        audio = trim_silence(engine.generate(config.VOICE_PROMPT_TEXT, f"eu:{name}"))
        for base in (config.VOICE_CACHE_DIR, config.DATA_DIR / "voices"):
            try:
                base.mkdir(parents=True, exist_ok=True)
                path = base / f"{name}_prompt.wav"
                sf.write(str(path), audio, SR, subtype="PCM_16")
                logger.info(f"Ahots-erreferentzia sortuta: {path} ({len(audio) / SR:.1f}s)")
                return str(path)
            except OSError:
                continue
        raise RuntimeError("Ezin izan da ahots-erreferentzia gorde")

    # ---------------------------------------------------------- synthesize
    def synthesize(self, text: str, lang: str, gender: str) -> np.ndarray:
        """Testua -> 24 kHz float32 audioa. gender: 'female' | 'male'."""
        engine = self.load(lang)
        name = config.VOICES.get(gender, config.VOICES["male"])
        clean = normalize(text, lang)
        if not clean:
            return np.zeros(0, dtype=np.float32)

        # Pocket TTS-ek batzuetan (~%1-2) erreferentzia-klipa jarraitzen du
        # testua irakurri beharrean: iraupen arraroa bada, berriz saiatu.
        expected = len(clean) / 14.0
        best = None
        for _ in range(3):
            audio = trim_silence(engine.generate(clean, f"{lang}:{name}"))
            dur = len(audio) / SR
            if best is None or abs(dur - expected) < abs(len(best) / SR - expected):
                best = audio
            if 0.35 * expected - 0.5 <= dur <= 2.5 * expected + 1.5:
                break
            logger.warning(f"TTS iraupen susmagarria ({dur:.1f}s vs ~{expected:.1f}s), berriz")
        return best.astype(np.float32)


synthesizer = SpeechSynthesizer()
