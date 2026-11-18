# =============================================================================
# STT - silero-VAD bidezko esaldi-zatiketa + NeMo Conformer-Transducer
# =============================================================================
# Ereduek letra xehez eta puntuaziorik gabe transkribatzen dute. Audioa VAD
# bidez esalditan zatitzen da (isiluneetan), eta zati bakoitza bereiz
# transkribatzen da: horrela esaldi bakoitzak bere denbora-marka du, eta
# ondoren bakoitzari ahotsa (Maider/Antton) eslei dakioke.
# =============================================================================

import logging
import threading
from typing import Callable, Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from app import config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _hyp_text(item) -> str:
    """NeMo bertsioaren arabera, transcribe()-k str edo Hypothesis itzultzen du."""
    if isinstance(item, str):
        return item
    text = getattr(item, "text", None)
    if text is None and isinstance(item, (list, tuple)) and item:
        return _hyp_text(item[0])
    return text or ""


def _tidy(text: str) -> str:
    """Letra larria hasieran eta puntua amaieran (MT-ak hobeto itzultzen du)."""
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1].isalnum():
        text += "."
    return text


class SpeechRecognizer:
    def __init__(self):
        self._models = {}
        self._vad = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def load_vad(self):
        with self._lock:
            if self._vad is None:
                from silero_vad import load_silero_vad
                self._vad = load_silero_vad()
            return self._vad

    def load(self, lang: str):
        with self._lock:
            if lang in self._models:
                return self._models[lang]
            import nemo.collections.asr as nemo_asr

            repo, filename, revision = config.STT_MODELS[lang]
            logger.info(f"STT eredua kargatzen ({lang}): {repo}")
            path = hf_hub_download(repo, filename, revision=revision)
            model = nemo_asr.models.ASRModel.restore_from(path, map_location="cpu")
            model.eval()
            # Beroketa: lehen inferentzia motela da (grafoak, memoria-erreserbak)
            with torch.inference_mode():
                model.transcribe([np.zeros(SAMPLE_RATE, dtype=np.float32)], batch_size=1, verbose=False)
            self._models[lang] = model
            logger.info(f"STT eredua prest ({lang})")
            return model

    def is_loaded(self, lang: str) -> bool:
        return lang in self._models

    # ------------------------------------------------------------- segment
    def segment(self, audio: np.ndarray) -> list:
        """Ahots-zatiak [(hasiera, amaiera)] lagin-tan itzultzen ditu."""
        from silero_vad import get_speech_timestamps

        vad = self.load_vad()
        stamps = get_speech_timestamps(
            torch.from_numpy(audio),
            vad,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
            max_speech_duration_s=config.VAD_MAX_SEGMENT_S,
            speech_pad_ms=config.VAD_SPEECH_PAD_MS,
        )
        return [(int(s["start"]), int(s["end"])) for s in stamps]

    # ---------------------------------------------------------- transcribe
    def transcribe(
        self,
        audio: np.ndarray,
        lang: str,
        progress: Optional[Callable[[float], None]] = None,
    ) -> list:
        """
        16 kHz mono audio osoa transkribatu. Itzultzen du:
        [{"start": s, "end": s, "samples": (a, b), "text": str}, ...]
        """
        model = self.load(lang)
        segments = self.segment(audio)
        logger.info(f"VAD: {len(segments)} ahots-zati")

        results = []
        batch = config.STT_BATCH_SIZE
        with torch.inference_mode():
            for i in range(0, len(segments), batch):
                group = segments[i:i + batch]
                clips = [audio[a:b].astype(np.float32) for a, b in group]
                out = model.transcribe(clips, batch_size=len(clips), verbose=False)
                if isinstance(out, tuple):  # (best, all) bertsio zaharretan
                    out = out[0]
                for (a, b), hyp in zip(group, out):
                    text = _tidy(_hyp_text(hyp))
                    if text:
                        results.append({
                            "start": a / SAMPLE_RATE,
                            "end": b / SAMPLE_RATE,
                            "samples": (a, b),
                            "text": text,
                        })
                if progress:
                    progress(min(1.0, (i + len(group)) / max(len(segments), 1)))
        return results


recognizer = SpeechRecognizer()
