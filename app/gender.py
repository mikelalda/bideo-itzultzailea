# =============================================================================
# Esaldi bakoitzeko ahotsa erabaki: ahots-tonuaren (F0) mediana
# =============================================================================
# Helduen F0 tipikoa: gizonak ~85-155 Hz, emakumeak ~165-255 Hz.
# Mediana config.F0_THRESHOLD_HZ-tik gora badago -> emakumea (Maider),
# bestela gizona (Antton). F0 fidagarririk ez duten zatiek (oso laburrak,
# musika...) aurreko/hurrengo esaldiaren ahotsa hartzen dute.
# =============================================================================

from typing import Optional

import librosa
import numpy as np

from app import config

SAMPLE_RATE = 16000


def median_f0(clip: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[float]:
    if len(clip) < int(0.3 * sr):
        return None
    # 8 kHz-ra jaitsi: F0-rako nahikoa da eta pyin bi aldiz azkarrago doa
    clip8 = librosa.resample(clip.astype(np.float32), orig_sr=sr, target_sr=8000)
    f0, voiced, _ = librosa.pyin(
        clip8, fmin=65.0, fmax=400.0, sr=8000,
        frame_length=512, hop_length=160,
    )
    f0 = f0[voiced & ~np.isnan(f0)]
    if len(f0) < 5:
        return None
    return float(np.median(f0))


def classify(f0: Optional[float]) -> Optional[str]:
    if f0 is None:
        return None
    return "female" if f0 >= config.F0_THRESHOLD_HZ else "male"


def fill_unknown(genders: list, default: str = "male") -> list:
    """None balioak ondoko esaldien generoarekin bete."""
    out = list(genders)
    last = None
    for i, g in enumerate(out):
        if g is None:
            out[i] = last
        else:
            last = g
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return [g or default for g in out]
