"""
Irudia eraikitzean exekutatzen da: eredu guztiak deskargatu (bertsio
finkoekin) eta gaztelaniazko Maider/Antton ahotsen erreferentzia-klipak sortu.
Horrela Space-a abiatzean ez da ezer deskargatu behar.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download  # noqa: E402

from app import config  # noqa: E402
from app.translator import translator  # noqa: E402
from app.tts import synthesizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    for lang, (repo, filename, revision) in config.STT_MODELS.items():
        logging.info(f"STT ({lang}): {repo}")
        hf_hub_download(repo, filename, revision=revision)

    for src, tgt in config.MT_MODELS:
        translator.load(src, tgt)

    synthesizer.load("eu")
    synthesizer.load("es")  # erreferentzia-klipak VOICE_CACHE_DIR-en sortzen ditu

    for name in config.VOICES.values():
        path = config.VOICE_CACHE_DIR / f"{name}_prompt.wav"
        if not path.exists():
            raise SystemExit(f"Ahots-erreferentzia falta da: {path}")
    logging.info("Eredu guztiak prest.")


if __name__ == "__main__":
    main()
