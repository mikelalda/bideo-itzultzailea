# =============================================================================
# Audio/bideo tresnak (ffmpeg + numpy)
# =============================================================================

import subprocess

import numpy as np

from app import config

SR = config.TTS_SAMPLE_RATE


def _run(cmd: list, what: str, **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, **kwargs)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace") if isinstance(result.stderr, bytes) else result.stderr
        raise RuntimeError(f"{what} huts egin du: {err[-1500:]}")
    return result


def media_duration(path: str) -> float:
    out = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        "ffprobe", text=True,
    )
    return float(out.stdout.strip())


def extract_audio(video_path: str, audio_path: str) -> float:
    """Bideotik 16 kHz mono WAV atera (STT-rako). Iraupena segundotan."""
    _run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path],
        "Audioa ateratzea",
    )
    return media_duration(video_path)


def time_stretch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Audioa azkartu (factor > 1) tonua aldatu gabe, ffmpeg atempo bidez."""
    if abs(factor - 1.0) < 0.02 or len(audio) == 0:
        return audio
    filters = []
    f = factor
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5")
        f /= 0.5
    filters.append(f"atempo={f:.4f}")
    result = _run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         "-filter:a", ",".join(filters), "-f", "f32le", "pipe:1"],
        "Abiadura-doiketa",
        input=audio.astype(np.float32).tobytes(),
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def build_timeline(items: list, total_duration: float, max_speedup: float) -> np.ndarray:
    """
    Esaldi bakoitzaren audioa bere hasiera-unean jarri.
    items: [{"start": s, "audio": np.ndarray}, ...] (hasieraren arabera ordenatuta)
    Esaldi bat hurrengoa hasi baino lehen amaitzen ez bada, azkartu egiten da
    (gehienez max_speedup aldiz); hala ere ez bada sartzen, hurrengoa atzeratu.
    """
    total = int(np.ceil(total_duration * SR))
    out = np.zeros(total, dtype=np.float32)
    cursor = 0.0
    for i, item in enumerate(items):
        audio = item["audio"]
        if audio is None or len(audio) == 0:
            continue
        start = max(float(item["start"]), cursor)
        next_start = items[i + 1]["start"] if i + 1 < len(items) else total_duration
        available = max(next_start - start, 0.3)
        duration = len(audio) / SR
        if duration > available:
            audio = time_stretch(audio, min(duration / available, max_speedup))

        a = int(start * SR)
        if a >= total:
            break
        b = min(a + len(audio), total)
        out[a:b] += audio[: b - a]
        cursor = start + len(audio) / SR + 0.05

    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:
        out *= 0.99 / peak
    return out


def mux_video(video_path: str, audio_path: str, output_path: str):
    """Jatorrizko bideoa + audio berria -> MP4. Bideoa kopiatu; ezin bada, H.264-ra."""
    base = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0"]
    tail = ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", output_path]
    try:
        _run(base + ["-c:v", "copy"] + tail, "Bideoa sortzea")
    except RuntimeError:
        _run(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] + tail,
             "Bideoa sortzea (H.264)")
