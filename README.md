---
title: Bideo Itzultzailea
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Bideoen bikoizketa Castellano ↔ Euskera, CPU-n
models:
  - nvidia/stt_es_conformer_transducer_large
  - HiTZ/stt_eu_conformer_transducer_large_v2
  - HiTZ/mt-hitz-es-eu
  - HiTZ/mt-hitz-eu-es
  - itzune/pocket-tts-basque-onnx
  - KevinAHM/pocket-tts-onnx
---

# Bideo Itzultzailea

Bideoen itzulpen eta bikoizketa automatikoa, **gaztelania ↔ euskara**, CPU
batean osorik exekutatzen dena.

| Urratsa | ES → EU | EU → ES |
|---|---|---|
| Esaldi-zatiketa | silero-VAD | silero-VAD |
| Transkripzioa (STT) | [nvidia/stt_es_conformer_transducer_large](https://huggingface.co/nvidia/stt_es_conformer_transducer_large) (NGC-ko eredu bera) | [HiTZ/stt_eu_conformer_transducer_large_v2](https://huggingface.co/HiTZ/stt_eu_conformer_transducer_large_v2) |
| Itzulpena (MT) | [HiTZ/mt-hitz-es-eu](https://huggingface.co/HiTZ/mt-hitz-es-eu) | [HiTZ/mt-hitz-eu-es](https://huggingface.co/HiTZ/mt-hitz-eu-es) |
| Ahotsa (TTS) | [itzune/pocket-tts-basque-onnx](https://huggingface.co/itzune/pocket-tts-basque-onnx) | [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx) (`spanish`) |

## Nola dabil

1. Audioa bideotik atera (16 kHz) eta silero-VAD bidez esalditan zatitu.
2. Esaldi bakoitza transkribatu. Esaldi bakoitzaren **ahots-tonua (F0)**
   neurtzen da: mediana 165 Hz-tik gora bada **Maider** (emakumea), bestela
   **Antton** (gizona).
3. **Berrikuspena:** esaldi bakoitza laukitxo batean: hasiera, amaiera, ahotsa
   (Maider/Antton hautatzailea) eta testua. Esaldiak gehitu/ezabatu daitezke,
   laukitxoan klik eginda bideoa esaldi horretara doa, eta erreproduzitzean
   uneko esaldia nabarmentzen da. **SRT-a igo eta deskargatu** daiteke; ahotsa
   `[Maider] testua` / `[Antton] testua` etiketa gisa gordetzen da fitxategian.
4. Itzulpena (laukitxoetan berriz editagarria, jatorrizko esaldia azpian) →
   esaldi bakoitza bere ahotsarekin sintetizatu → jatorrizko denbora-marketan
   kokatu (behar bada 1.6× arte azkartuta).
5. Ahotsa entzun, onartu, eta MP4 finala deskargatu.

Gaztelaniaz ere Maider eta Antton entzuten dira: irudia eraikitzean euskal
ereduak erreferentzia-klip labur bat sortzen du ahots bakoitzarekin, eta
gaztelaniazko Pocket TTS ereduak klip horretatik klonatzen du ahotsa.

## CPU-n errendimendua

Ereduak txikiak dira (STT ~119M parametro, MT ~75M, TTS ONNX int8 ~160 MB).
Eredu guztiak kargatuta, edukiontziak ~3.2 GB memoria erabiltzen du;
Space-ko `cpu-basic` hardwarea (2 vCPU, 16 GB) nahikoa da.

Tokian neurtua, 2 CPU-rekin (`--cpus=2`, `NUM_THREADS=2`), 14 s-ko bideo
batekin (lau esaldi, bi hizlari):

| Urratsa | Denbora |
|---|---|
| Transkripzioa + ahotsen detekzioa | ~2 s |
| Itzulpena | ~1.5 s |
| Ahots-sintesia | ~8-12 s (bideoaren iraupenaren ~0.6-0.8×) |
| Bideo finala | ~0.5 s |

Sintesia da urratsik motelena: minutu bateko bideo batek ~1 minutu behar du gutxi
gorabehera. Abiaraztean ereduak ~1-2 minututan kargatzen dira.

## Tokian exekutatu

```bash
docker compose up --build -d
# -> http://localhost:7860
```

Aldagai erabilgarriak: `NUM_THREADS`, `F0_THRESHOLD_HZ` (lehenetsia 165),
`MAX_SPEEDUP` (1.6), `MT_NUM_BEAMS` (4), `MT_LENGTH_PENALTY` (0.6),
`TTS_EU_TEMPERATURE` (0.3), `TTS_ES_TEMPERATURE` (0.5).

## Oharrak

- HiTZ-en MT checkpoint-ak transformers 4.12-rekin gorde ziren; transformers
  5-ekin `lm_head` berriz lotu behar da `shared` embedding-ari, eta
  `length_penalty=0.6` erabili (Marian-en lehenetsia) es→eu ereduak jarraipen
  asmatuak gehi ez ditzan. Biak `app/translator.py`-n daude.
- Pocket TTS-ek zenbakiak hitzez behar ditu: `app/textnorm.py`-k euskarazko
  zenbakiak (sistema hogeitarra) eta gaztelaniazkoak (num2words) idazten ditu.
- `app/vendor/pocket_tts_onnx.py` KevinAHM-en runtime-a da (Apache-2.0).

## Lizentziak eta aitortzak

- STT: NVIDIA `stt_es_conformer_transducer_large` (CC-BY-4.0);
  HiTZ/Aholab `stt_eu_conformer_transducer_large_v2` (Apache-2.0).
- MT: HiTZ Zentroa eta IXA taldea, UPV/EHU (Apache-2.0).
- TTS: Kyutai Pocket TTS (CC-BY-4.0); itzune/pocket-tts-basque (CC-BY-4.0);
  Maider eta Antton ahotsak HiTZ-Aholab estudio-corpusekoak (CC-BY-4.0);
  BSSD datu-multzoa (CC-BY-SA-4.0).
