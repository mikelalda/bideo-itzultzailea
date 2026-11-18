# =============================================================================
# Itzulpena - HiTZ MarianMT (es->eu: HiTZ/mt-hitz-es-eu, eu->es: HiTZ/mt-hitz-eu-es)
# =============================================================================

import logging
import threading

import torch
from transformers import MarianMTModel, MarianTokenizer

from app import config

logger = logging.getLogger(__name__)


class TextTranslator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self._lock = threading.Lock()

    def load(self, source_lang: str, target_lang: str):
        key = (source_lang, target_lang)
        if key not in config.MT_MODELS:
            raise ValueError(
                f"{source_lang}->{target_lang} itzulpena ez dago onartuta. "
                f"Aukerak: {list(config.MT_MODELS)}"
            )
        with self._lock:
            if key not in self.models:
                name, revision = config.MT_MODELS[key]
                logger.info(f"MT eredua kargatzen: {name}")
                self.tokenizers[key] = MarianTokenizer.from_pretrained(name, revision=revision)
                model = MarianMTModel.from_pretrained(name, revision=revision)
                # HiTZ-en checkpoint-ak transformers 4.12-rekin gorde ziren: orduan
                # lm_head beti `shared` embedding-ari lotzen zitzaion, eta fitxategiko
                # lm_head.weight zaharkituta dago. transformers 5-ek fitxategikoa
                # kargatzen du (eta zaborra itzultzen du), beraz lotura berrezarri.
                model.lm_head.weight = model.model.shared.weight
                model.eval()
                self.models[key] = model
                logger.info(f"MT eredua prest: {name}")
        return self.tokenizers[key], self.models[key]

    def is_loaded(self, source_lang: str, target_lang: str) -> bool:
        return (source_lang, target_lang) in self.models

    def translate_batch(self, texts: list, source_lang: str, target_lang: str) -> list:
        tokenizer, model = self.load(source_lang, target_lang)
        results = [""] * len(texts)
        todo = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
        bs = config.MT_BATCH_SIZE
        with torch.inference_mode():
            for start in range(0, len(todo), bs):
                group = todo[start:start + bs]
                inputs = tokenizer(
                    [t for _, t in group],
                    return_tensors="pt", padding=True, truncation=True, max_length=512,
                )
                # length_penalty < 1: Marian NMT-ren normalizazioa (0.6). HF-ren
                # 1.0 lehenetsiarekin es->eu ereduak jarraipen asmatuak gehitzen ditu.
                generated = model.generate(
                    **inputs,
                    num_beams=config.MT_NUM_BEAMS,
                    length_penalty=config.MT_LENGTH_PENALTY,
                    max_new_tokens=int(inputs["input_ids"].shape[1] * 2 + 16),
                )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for (i, _), out in zip(group, decoded):
                    results[i] = out.strip()
        return results

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return self.translate_batch([text], source_lang, target_lang)[0]


translator = TextTranslator()
