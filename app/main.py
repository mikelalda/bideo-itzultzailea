# =============================================================================
# Bideo Itzultzailea - Bideoen itzulpena / bikoizketa (CPU)
#
#   ES -> EU: NVIDIA stt_es_conformer_transducer_large -> HiTZ/mt-hitz-es-eu
#             -> Pocket TTS euskaraz
#   EU -> ES: HiTZ/stt_eu_conformer_transducer_large_v2 -> HiTZ/mt-hitz-eu-es
#             -> Pocket TTS gaztelaniaz
#
# Esaldi bakoitzak bere ahotsa du (Maider edo Antton), jatorrizko hizlariaren
# ahots-tonuaren arabera erabakia; SRT-an [Maider]/[Antton] etiketa aldatuz
# eskuz zuzen daiteke.
#
# Urratsak (bakoitzaren ondoren erabiltzailearen onarpenaren zain):
#   1. transkripzioa (SRT editagarria)  2. itzulpena (SRT editagarria)
#   3. ahotsa (entzun daiteke)          4. bideo finala
# =============================================================================

import asyncio
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional

from pydantic import BaseModel

from app import config
from app.audio import build_timeline, extract_audio, mux_video
from app.gender import classify, fill_unknown, median_f0
from app.stt import recognizer
from app.translator import translator
from app.tts import synthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bideo")

torch.set_num_threads(config.NUM_THREADS)

LANGS = ("es", "eu")
VOICE_MODES = ("auto", "female", "male")
JOB_MAX_AGE_S = 24 * 3600

# Prozesatze astuna (STT/MT/TTS) bata bestearen atzetik: CPU bakarra dago.
heavy_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Ereduen aurre-karga (atzealdean, /api/health berehala erantzuteko)
# ---------------------------------------------------------------------------
preload_state = {"running": False, "done": False, "error": None}


def _preload_models():
    preload_state["running"] = True
    try:
        recognizer.load_vad()
        for lang in LANGS:
            recognizer.load(lang)
        translator.load("es", "eu")
        translator.load("eu", "es")
        synthesizer.load("eu")
        synthesizer.load("es")
        # librosa.pyin-en JIT konpilazioa (~7 s) orain egin, ez lehen bideoan
        t = np.arange(16000) / 16000.0
        median_f0((0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32))
        preload_state["done"] = True
        logger.info("Eredu guztiak kargatuta.")
    except Exception as e:  # pragma: no cover - logean ikusteko
        preload_state["error"] = str(e)
        logger.exception("Ereduen aurre-kargak huts egin du")
    finally:
        preload_state["running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (config.UPLOAD_DIR, config.OUTPUT_DIR, config.TEMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if config.PRELOAD_MODELS:
        threading.Thread(target=_preload_models, daemon=True).start()
    yield


app = FastAPI(
    title="Bideo Itzultzailea",
    description="Bideoen itzulpena Castellano <-> Euskera (CPU)",
    version="2.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Job-en egoera
# ---------------------------------------------------------------------------
job_status: dict = {}


def update_job(job_id: str, state: str, progress: int, detail: str = ""):
    job_status.setdefault(job_id, {})
    job_status[job_id].update({"state": state, "progress": progress, "detail": detail})
    logger.info(f"[{job_id}] {state} ({progress}%): {detail}")


def set_job_error(job_id: str, message: str):
    job_status.setdefault(job_id, {})
    job_status[job_id].update({"state": "error", "progress": -1, "detail": message})


def cleanup_old_files():
    cutoff = time.time() - JOB_MAX_AGE_S
    for d in (config.UPLOAD_DIR, config.OUTPUT_DIR, config.TEMP_DIR):
        for p in d.glob("*"):
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Azpitituluak. Barnean: {"text", "timestamp": [s, e], "gender"}.
# APIan: {"start", "end", "text", "gender", ["source"]}. SRT fitxategiak
# nabigatzailean sortu/irakurtzen dira, "[Maider] testua" etiketekin.
# ---------------------------------------------------------------------------
GENDERS = tuple(config.VOICES)  # ("female", "male")


def _srt_timestamp(seconds) -> str:
    ms_total = int(round(max(0.0, seconds or 0.0) * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def chunks_to_srt(chunks: list) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        start, end = chunk["timestamp"]
        voice = config.VOICES.get(chunk.get("gender") or "male")
        lines += [
            str(i),
            f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}",
            f"[{voice}] {(chunk.get('text') or '').strip()}",
            "",
        ]
    return "\n".join(lines)


def chunks_to_subtitles(chunks: list) -> list:
    out = []
    for c in chunks:
        item = {
            "start": c["timestamp"][0],
            "end": c["timestamp"][1],
            "text": c.get("text", ""),
            "gender": c.get("gender") or "male",
        }
        if "source" in c:
            item["source"] = c["source"]
        out.append(item)
    return out


class Subtitle(BaseModel):
    start: float
    end: float
    text: str
    gender: str = "male"
    source: Optional[str] = None


class SubtitlesUpdate(BaseModel):
    subtitles: List[Subtitle]


def subtitles_to_chunks(subtitles: List[Subtitle]) -> list:
    chunks = []
    for i, sub in enumerate(subtitles, start=1):
        text = " ".join(sub.text.split())
        if not text:
            raise ValueError(f"{i}. azpitituluak ez du testurik")
        if sub.start < 0 or sub.end <= sub.start:
            raise ValueError(f"{i}. azpitituluaren amaiera hasieraren ondoren izan behar da")
        if sub.gender not in GENDERS:
            raise ValueError(f"{i}. azpitituluaren ahotsa baliogabea: {sub.gender}")
        chunk = {"text": text, "timestamp": [round(sub.start, 3), round(sub.end, 3)],
                 "gender": sub.gender}
        if sub.source is not None:
            chunk["source"] = sub.source
        chunks.append(chunk)
    if not chunks:
        raise ValueError("Gutxienez azpititulu bat behar da")
    chunks.sort(key=lambda c: c["timestamp"][0])
    return chunks


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "preload": preload_state,
        "models": {
            "stt": {l: recognizer.is_loaded(l) for l in LANGS},
            "mt": {
                "es-eu": translator.is_loaded("es", "eu"),
                "eu-es": translator.is_loaded("eu", "es"),
            },
            "tts": {l: synthesizer.is_loaded(l) for l in LANGS},
        },
        "busy": heavy_lock.locked(),
    }


@app.get("/api/voices")
async def get_voices():
    return {"modes": list(VOICE_MODES), "voices": config.VOICES}


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    job = job_status.get(job_id)
    if job is None:
        return {"state": "unknown", "progress": 0, "detail": "Job not found"}
    response = {k: job.get(k) for k in ("state", "progress", "detail")}
    if job.get("state") == "awaiting_transcript_review":
        response["subtitles"] = chunks_to_subtitles(job.get("chunks") or [])
    if job.get("state") == "awaiting_translation_review":
        response["subtitles"] = chunks_to_subtitles(job.get("translated_chunks") or [])
    if job.get("state") == "awaiting_synthesis_review":
        response["audio_url"] = f"/api/job/{job_id}/audio"
    if job.get("state") == "completed":
        response["result"] = job.get("result")
    return response


@app.put("/api/job/{job_id}/subtitles")
async def update_job_subtitles(job_id: str, body: SubtitlesUpdate):
    """Editatutako azpitituluak gorde (transkripzioa edo itzulpena berrikustean)."""
    job = job_status.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    state = job.get("state")
    if state == "awaiting_transcript_review":
        key = "chunks"
    elif state == "awaiting_translation_review":
        key = "translated_chunks"
    else:
        raise HTTPException(400, f"Job ez dago azpitituluak editatzeko egoeran (egoera: {state})")
    try:
        job[key] = subtitles_to_chunks(body.subtitles)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"success": True, "subtitles": chunks_to_subtitles(job[key])}


@app.post("/api/job/{job_id}/approve")
async def approve_job_step(job_id: str):
    job = job_status.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    state = job.get("state")
    if state == "awaiting_transcript_review":
        update_job(job_id, "translating", 45, "Testua itzultzen...")
        asyncio.create_task(run_translation(job_id))
    elif state == "awaiting_translation_review":
        update_job(job_id, "synthesizing", 60, "Ahotsa sintetizatzen...")
        asyncio.create_task(run_synthesis(job_id))
    elif state == "awaiting_synthesis_review":
        update_job(job_id, "finalizing", 90, "Bideo finala sortzen...")
        asyncio.create_task(run_finalize(job_id))
    else:
        raise HTTPException(400, f"Job ez dago onarpen-urrats batean (egoera: {state})")
    return {"success": True}


@app.get("/api/job/{job_id}/audio")
async def get_job_audio(job_id: str):
    job = job_status.get(job_id)
    if not job or not job.get("final_audio_path") or not Path(job["final_audio_path"]).exists():
        raise HTTPException(404, "Audioa ez dago")
    return FileResponse(job["final_audio_path"], media_type="audio/wav")


@app.post("/api/translate")
async def translate_video(
    file: UploadFile = File(..., description="Itzuli beharreko bideoa"),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    voice: str = Form("auto", description="auto | female | male"),
):
    """Bideoa igo eta transkripzioa abiarazi. Berehala itzultzen du job_id-a."""
    if source_lang not in LANGS or target_lang not in LANGS:
        raise HTTPException(400, "Hizkuntzak 'es' edo 'eu' izan behar dira")
    if source_lang == target_lang:
        raise HTTPException(400, "Jatorrizko eta helburu hizkuntzak desberdinak izan behar dira")
    if voice not in VOICE_MODES:
        voice = "auto"

    cleanup_old_files()
    job_id = uuid.uuid4().hex[:8]
    update_job(job_id, "uploading", 5, "Bideoa igotzen...")

    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):
        suffix = ".mp4"
    video_path = str(config.UPLOAD_DIR / f"{job_id}{suffix}")
    with open(video_path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    job_status[job_id].update({
        "video_path": video_path,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "voice_mode": voice,
    })
    asyncio.create_task(run_transcription(job_id))
    return JSONResponse({"success": True, "job_id": job_id})


# ---------------------------------------------------------------------------
# Pipeline urratsak
# ---------------------------------------------------------------------------
def _progress_cb(job_id: str, state: str, lo: int, hi: int, label: str):
    def cb(frac: float):
        update_job(job_id, state, lo + int((hi - lo) * frac), f"{label} ({int(frac * 100)}%)")
    return cb


async def _wait_turn(job_id: str, state: str, progress: int):
    if heavy_lock.locked():
        update_job(job_id, state, progress, "Beste lan bat amaitu arte itxaroten...")


async def run_transcription(job_id: str):
    job = job_status[job_id]
    try:
        update_job(job_id, "extracting_audio", 10, "Audioa ateratzen bideotik...")
        audio_path = str(config.TEMP_DIR / f"{job_id}_audio16k.wav")
        duration = await asyncio.to_thread(extract_audio, job["video_path"], audio_path)
        audio, _ = sf.read(audio_path, dtype="float32")
        job["video_duration"] = duration

        await _wait_turn(job_id, "transcribing", 15)
        async with heavy_lock:
            update_job(job_id, "transcribing", 15, "Audioa transkribatzen...")
            segments = await asyncio.to_thread(
                recognizer.transcribe, audio, job["source_lang"],
                _progress_cb(job_id, "transcribing", 15, 27, "Audioa transkribatzen"),
            )
            if not segments:
                raise RuntimeError("Ez da hizketarik detektatu bideoan")

            if job["voice_mode"] == "auto":
                update_job(job_id, "transcribing", 28, "Esaldi bakoitzaren ahotsa aztertzen...")

                def detect():
                    return fill_unknown([
                        classify(median_f0(audio[a:b])) for a, b in (s["samples"] for s in segments)
                    ])
                genders = await asyncio.to_thread(detect)
            else:
                genders = [job["voice_mode"]] * len(segments)

        chunks = [
            {"text": s["text"], "timestamp": [round(s["start"], 3), round(s["end"], 3)], "gender": g}
            for s, g in zip(segments, genders)
        ]
        job["chunks"] = chunks
        update_job(job_id, "awaiting_transcript_review", 30,
                   "Berrikusi transkripzioa eta ahotsak, eta onartu itzultzeko")
    except Exception as e:
        set_job_error(job_id, str(e))
        logger.exception(f"[{job_id}] Transkripzioak huts egin du")


async def run_translation(job_id: str):
    job = job_status[job_id]
    try:
        chunks = job["chunks"]
        await _wait_turn(job_id, "translating", 45)
        async with heavy_lock:
            update_job(job_id, "translating", 45, "Testua itzultzen...")
            texts = await asyncio.to_thread(
                translator.translate_batch, [c["text"] for c in chunks],
                job["source_lang"], job["target_lang"],
            )
        translated = [
            {"text": t, "timestamp": c["timestamp"], "gender": c.get("gender", "male"),
             "source": c["text"]}
            for c, t in zip(chunks, texts) if t
        ]
        job["translated_chunks"] = translated
        update_job(job_id, "awaiting_translation_review", 55,
                   "Berrikusi itzulpena eta onartu ahotsa sortzeko")
    except Exception as e:
        set_job_error(job_id, str(e))
        logger.exception(f"[{job_id}] Itzulpenak huts egin du")


async def run_synthesis(job_id: str):
    job = job_status[job_id]
    try:
        chunks = job["translated_chunks"]
        lang = job["target_lang"]
        await _wait_turn(job_id, "synthesizing", 60)
        async with heavy_lock:
            items = []
            for i, chunk in enumerate(chunks):
                voice = config.VOICES.get(chunk["gender"], "")
                update_job(job_id, "synthesizing", 60 + int(22 * i / max(len(chunks), 1)),
                           f"Esaldia sintetizatzen {i + 1}/{len(chunks)} ({voice})...")
                audio = await asyncio.to_thread(
                    synthesizer.synthesize, chunk["text"], lang, chunk["gender"]
                )
                items.append({"start": chunk["timestamp"][0], "audio": audio})

            update_job(job_id, "combining", 83, "Audio zatiak denboran kokatzen...")
            timeline = await asyncio.to_thread(
                build_timeline, items, job["video_duration"], config.MAX_SPEEDUP
            )
        final_audio_path = str(config.TEMP_DIR / f"{job_id}_final.wav")
        sf.write(final_audio_path, timeline, config.TTS_SAMPLE_RATE, subtype="PCM_16")
        job["final_audio_path"] = final_audio_path
        update_job(job_id, "awaiting_synthesis_review", 85,
                   "Ahotsa sortuta. Entzun eta onartu bideo finala sortzeko")
    except Exception as e:
        set_job_error(job_id, str(e))
        logger.exception(f"[{job_id}] Sintesiak huts egin du")


async def run_finalize(job_id: str):
    job = job_status[job_id]
    try:
        update_job(job_id, "finalizing", 92, "Bideo finala sortzen...")
        output_filename = f"{job_id}_{job['target_lang']}.mp4"
        output_path = str(config.OUTPUT_DIR / output_filename)
        await asyncio.to_thread(mux_video, job["video_path"], job["final_audio_path"], output_path)

        for p in config.TEMP_DIR.glob(f"{job_id}_*"):
            try:
                p.unlink()
            except OSError:
                pass
        job.pop("final_audio_path", None)

        job.update({
            "state": "completed",
            "progress": 100,
            "detail": "Itzulpena amaituta!",
            "result": {
                "success": True,
                "job_id": job_id,
                "download_url": f"/api/download/{output_filename}",
                "transcription": " ".join(c["text"] for c in job["chunks"]),
                "translated_text": " ".join(c["text"] for c in job["translated_chunks"]),
                "srt_translated": chunks_to_srt(job["translated_chunks"]),
            },
        })
        logger.info(f"[{job_id}] completed (100%)")
    except Exception as e:
        set_job_error(job_id, str(e))
        logger.exception(f"[{job_id}] Bideo finalak huts egin du")


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    if Path(filename).name != filename or not filename.endswith(".mp4"):
        raise HTTPException(400, "Fitxategi-izen baliogabea")
    file_path = config.OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)
