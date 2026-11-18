// =============================================================================
// Bideo Itzultzailea - Frontend Application
// =============================================================================

const API_BASE = '';
let selectedFile = null;
let pollInterval = null;

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const uploadArea = document.getElementById('uploadArea');
const videoFileInput = document.getElementById('videoFile');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const removeFileBtn = document.getElementById('removeFile');
const previewContainer = document.getElementById('previewContainer');
const videoPreview = document.getElementById('videoPreview');
const sourceLang = document.getElementById('sourceLang');
const targetLang = document.getElementById('targetLang');
const swapLangs = document.getElementById('swapLangs');
const voiceSelect = document.getElementById('voice');
const translateForm = document.getElementById('translateForm');
const submitBtn = document.getElementById('submitBtn');
const progressCard = document.getElementById('progressCard');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const resultsCard = document.getElementById('resultsCard');
const originalText = document.getElementById('originalText');
const translatedText = document.getElementById('translatedText');
const downloadBtn = document.getElementById('downloadBtn');
const newTranslationBtn = document.getElementById('newTranslation');
const errorCard = document.getElementById('errorCard');
const errorText = document.getElementById('errorText');
const retryBtn = document.getElementById('retryBtn');

// Step-by-step review cards
const transcriptReviewCard = document.getElementById('transcriptReviewCard');
const transcriptSubtitleEditor = document.getElementById('transcriptSubtitleEditor');
const approveTranscriptBtn = document.getElementById('approveTranscriptBtn');
const translationReviewCard = document.getElementById('translationReviewCard');
const translationSubtitleEditor = document.getElementById('translationSubtitleEditor');
const srtFileInput = document.getElementById('srtFileInput');
const downloadResultSrtBtn = document.getElementById('downloadResultSrtBtn');
const approveTranslationBtn = document.getElementById('approveTranslationBtn');
const synthesisReviewCard = document.getElementById('synthesisReviewCard');
const approveSynthesisBtn = document.getElementById('approveSynthesisBtn');

let currentJobId = null;
let resultSrt = '';
let srtUploadTarget = null;

// Generoa -> ahotsaren izena (backend-eko config.VOICES-en berdina)
const VOICE_NAMES = { female: 'Maider', male: 'Antton' };
const VOICE_ALIASES = {
    maider: 'female', emakumea: 'female', mujer: 'female', female: 'female', f: 'female',
    antton: 'male', gizona: 'male', hombre: 'male', male: 'male', m: 'male',
};

const synthesisAudio = document.getElementById('synthesisAudio');

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setInterval(checkHealth, 15000);
});

// ---------------------------------------------------------------------------
// Health Check
// ---------------------------------------------------------------------------
async function checkHealth() {
    const dots = {
        stt: document.getElementById('statusStt'),
        mt: document.getElementById('statusMt'),
        tts: document.getElementById('statusTts'),
    };

    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        const models = data.models || {};
        const failed = data.preload && data.preload.error;
        Object.entries(dots).forEach(([key, dot]) => {
            const loaded = Object.values(models[key] || {});
            const ready = loaded.length > 0 && loaded.every(Boolean);
            dot.className = 'status-dot ' + (ready ? 'ok' : (failed ? 'error' : 'loading'));
            dot.title = ready ? 'Prest' : (failed ? data.preload.error : 'Eredua kargatzen...');
        });
    } catch {
        Object.values(dots).forEach(d => d.className = 'status-dot error');
    }
}

// ---------------------------------------------------------------------------
// File Upload
// ---------------------------------------------------------------------------
uploadArea.addEventListener('click', () => videoFileInput.click());

uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

videoFileInput.addEventListener('change', () => {
    if (videoFileInput.files.length > 0) {
        handleFile(videoFileInput.files[0]);
    }
});

removeFileBtn.addEventListener('click', () => {
    clearFile();
});

function handleFile(file) {
    const validTypes = ['video/mp4', 'video/avi', 'video/x-msvideo', 'video/x-matroska',
                        'video/quicktime', 'video/webm', 'video/mpeg'];

    // Also accept by extension
    const ext = file.name.split('.').pop().toLowerCase();
    const validExts = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'mpeg', 'mpg'];

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
        showError('Fitxategi mota ez da onartzen. Erabili MP4, AVI, MKV, MOV edo WEBM.');
        return;
    }

    selectedFile = file;
    fileName.textContent = `${file.name} (${formatSize(file.size)})`;
    fileInfo.style.display = 'flex';
    uploadArea.style.display = 'none';

    // Video preview
    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    previewContainer.style.display = 'block';
}

function clearFile() {
    selectedFile = null;
    videoFileInput.value = '';
    fileInfo.style.display = 'none';
    uploadArea.style.display = 'block';
    previewContainer.style.display = 'none';
    videoPreview.src = '';
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ---------------------------------------------------------------------------
// Language & Voice
// ---------------------------------------------------------------------------
swapLangs.addEventListener('click', () => {
    const src = sourceLang.value;
    const tgt = targetLang.value;
    sourceLang.value = tgt;
    targetLang.value = src;
});

sourceLang.addEventListener('change', () => {
    if (sourceLang.value === targetLang.value) {
        targetLang.value = sourceLang.value === 'es' ? 'eu' : 'es';
    }
});

targetLang.addEventListener('change', () => {
    if (targetLang.value === sourceLang.value) {
        sourceLang.value = targetLang.value === 'es' ? 'eu' : 'es';
    }
});

// ---------------------------------------------------------------------------
// Form Submission (urrats-1: bideoa igo + transkribatu)
// ---------------------------------------------------------------------------
translateForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!selectedFile) {
        showError('Mesedez, hautatu bideo bat.');
        return;
    }

    if (sourceLang.value === targetLang.value) {
        showError('Jatorrizko eta helburu hizkuntzak desberdinak izan behar dira.');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('source_lang', sourceLang.value);
    formData.append('target_lang', targetLang.value);
    formData.append('voice', voiceSelect.value);

    hideError();
    resultsCard.style.display = 'none';
    transcriptReviewCard.style.display = 'none';
    translationReviewCard.style.display = 'none';
    synthesisReviewCard.style.display = 'none';
    progressCard.style.display = 'block';
    submitBtn.disabled = true;
    resetProgress();

    try {
        // Berehala itzultzen du job_id bat -- pipeline osoa ez du itxaron
        // behar (HF Spaces-en proxy-ak eskaera luzeak mozten ditu ~60s-tara).
        const res = await fetch(`${API_BASE}/api/translate`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Translation failed');
        }

        const { job_id } = await res.json();
        currentJobId = job_id;
        startPolling(job_id);

    } catch (err) {
        showError(err.message);
        progressCard.style.display = 'none';
        submitBtn.disabled = false;
    }
});

// ---------------------------------------------------------------------------
// Polling: pipeline pausuz pausu doa, eta erabiltzailearen onarpenaren zain
// gelditzen da urrats bakoitzaren ondoren (transkripzioa, itzulpena,
// sintesia). Job-aren egoerak dio zein card erakutsi.
// ---------------------------------------------------------------------------
function startPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/job/${jobId}`);
            const data = await res.json();

            updateProgress(data.state, data.progress, data.detail);

            switch (data.state) {
                case 'awaiting_transcript_review':
                    stopPolling();
                    renderSubtitleEditor(transcriptSubtitleEditor, data.subtitles || []);
                    progressCard.style.display = 'none';
                    transcriptReviewCard.style.display = 'block';
                    submitBtn.disabled = false;
                    break;

                case 'awaiting_translation_review':
                    stopPolling();
                    renderSubtitleEditor(translationSubtitleEditor, data.subtitles || []);
                    progressCard.style.display = 'none';
                    translationReviewCard.style.display = 'block';
                    break;

                case 'awaiting_synthesis_review':
                    stopPolling();
                    synthesisAudio.src = `${data.audio_url}?t=${Date.now()}`;
                    progressCard.style.display = 'none';
                    synthesisReviewCard.style.display = 'block';
                    break;

                case 'completed': {
                    stopPolling();
                    markAllStepsDone();
                    const result = data.result || {};
                    originalText.textContent = result.transcription || '(ez dago testurik)';
                    translatedText.textContent = result.translated_text || '(ez dago testurik)';
                    downloadBtn.href = result.download_url;
                    resultSrt = result.srt_translated || '';
                    setTimeout(() => {
                        progressCard.style.display = 'none';
                        resultsCard.style.display = 'block';
                    }, 800);
                    break;
                }

                case 'error':
                    stopPolling();
                    submitBtn.disabled = false;
                    showError(data.detail || 'Translation failed');
                    progressCard.style.display = 'none';
                    break;
            }
        } catch (err) {
            stopPolling();
            submitBtn.disabled = false;
            showError(err.message);
        }
    }, 1500);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// ---------------------------------------------------------------------------
// Azpititulu-editorea: esaldi bakoitza laukitxo batean (denborak, ahotsa, testua)
// ---------------------------------------------------------------------------
function createSubtitleRow(subtitle, index) {
    const row = document.createElement('div');
    row.className = 'subtitle-row';
    if (subtitle.source) row.dataset.source = subtitle.source;

    const number = document.createElement('span');
    number.className = 'subtitle-number';

    const start = document.createElement('input');
    start.type = 'number';
    start.className = 'subtitle-time subtitle-start';
    start.min = '0';
    start.step = '0.01';
    start.value = Number(subtitle.start || 0).toFixed(2);
    start.title = 'Hasiera (s)';

    const end = document.createElement('input');
    end.type = 'number';
    end.className = 'subtitle-time subtitle-end';
    end.min = '0.01';
    end.step = '0.01';
    end.value = Number(subtitle.end || 2).toFixed(2);
    end.title = 'Amaiera (s)';

    const voice = document.createElement('select');
    voice.className = 'subtitle-voice';
    Object.entries(VOICE_NAMES).forEach(([gender, name]) => {
        const opt = document.createElement('option');
        opt.value = gender;
        opt.textContent = name;
        voice.appendChild(opt);
    });
    voice.value = VOICE_NAMES[subtitle.gender] ? subtitle.gender : 'male';
    voice.title = 'Esaldi hau irakurriko duen ahotsa';
    const paintVoice = () => { row.dataset.gender = voice.value; };
    voice.addEventListener('change', paintVoice);
    paintVoice();

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'subtitle-delete';
    remove.textContent = '×';
    remove.title = 'Esaldia ezabatu';
    remove.addEventListener('click', () => {
        const editor = row.parentElement;
        row.remove();
        renumberSubtitleRows(editor);
    });

    const text = document.createElement('textarea');
    text.className = 'subtitle-text';
    text.rows = 2;
    text.value = subtitle.text || '';

    const head = document.createElement('div');
    head.className = 'subtitle-head';
    const times = document.createElement('div');
    times.className = 'subtitle-times';
    const arrow = document.createElement('span');
    arrow.className = 'subtitle-arrow';
    arrow.textContent = '→';
    times.append(start, arrow, end);
    head.append(number, times, voice, remove);
    row.append(head, text);

    if (subtitle.source) {
        const src = document.createElement('p');
        src.className = 'subtitle-source';
        src.textContent = subtitle.source;
        row.appendChild(src);
    }

    row.addEventListener('click', event => {
        if (event.target.matches('input, textarea, button, select')) return;
        if (videoPreview.src) videoPreview.currentTime = Number(start.value) || 0;
    });
    return row;
}

function renumberSubtitleRows(editor) {
    [...editor.querySelectorAll('.subtitle-row')].forEach((row, index) => {
        row.querySelector('.subtitle-number').textContent = index + 1;
    });
}

function renderSubtitleEditor(editor, subtitles) {
    editor.replaceChildren(...subtitles.map((sub, i) => createSubtitleRow(sub, i)));
    renumberSubtitleRows(editor);
}

function addSubtitle(editor) {
    const rows = editor.querySelectorAll('.subtitle-row');
    const last = rows[rows.length - 1];
    const lastEnd = last ? Number(last.querySelector('.subtitle-end').value) : 0;
    const gender = last ? last.querySelector('.subtitle-voice').value : 'male';
    editor.appendChild(createSubtitleRow({ start: lastEnd, end: lastEnd + 2, text: '', gender }, rows.length));
    renumberSubtitleRows(editor);
    editor.lastElementChild.querySelector('.subtitle-text').focus();
}

function collectSubtitles(editor) {
    const subtitles = [...editor.querySelectorAll('.subtitle-row')].map((row, index) => {
        const start = Number(row.querySelector('.subtitle-start').value);
        const end = Number(row.querySelector('.subtitle-end').value);
        const text = row.querySelector('.subtitle-text').value.trim();
        const gender = row.querySelector('.subtitle-voice').value;
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start || !text) {
            row.scrollIntoView({ behavior: 'smooth', block: 'center' });
            throw new Error(`${index + 1}. esaldia osatu eta denborak zuzendu (amaiera > hasiera).`);
        }
        const sub = { start, end, text, gender };
        if (row.dataset.source) sub.source = row.dataset.source;
        return sub;
    });
    if (!subtitles.length) throw new Error('Gutxienez esaldi bat behar da.');
    return subtitles;
}

async function saveSubtitles(editor) {
    const res = await fetch(`${API_BASE}/api/job/${currentJobId}/subtitles`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subtitles: collectSubtitles(editor) }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Azpitituluak ezin izan dira gorde');
    }
}

// ---------------------------------------------------------------------------
// SRT igo / jaitsi. Ahotsa "[Maider] testua" etiketa gisa gordetzen da.
// ---------------------------------------------------------------------------
function formatSrtTimestamp(seconds) {
    const totalMillis = Math.max(0, Math.round(seconds * 1000));
    const hours = Math.floor(totalMillis / 3600000);
    const minutes = Math.floor((totalMillis % 3600000) / 60000);
    const secs = Math.floor((totalMillis % 60000) / 1000);
    const millis = totalMillis % 1000;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')},${String(millis).padStart(3, '0')}`;
}

function subtitlesToSrt(subtitles) {
    return subtitles.map((sub, index) => [
        index + 1,
        `${formatSrtTimestamp(sub.start)} --> ${formatSrtTimestamp(sub.end)}`,
        `[${VOICE_NAMES[sub.gender] || VOICE_NAMES.male}] ${sub.text}`,
        '',
    ].join('\n')).join('\n');
}

function parseSrtTimestamp(value) {
    const match = value.trim().match(/^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})/);
    if (!match) throw new Error(`SRT denbora baliogabea: ${value}`);
    return Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3])
        + Number(match[4].padEnd(3, '0')) / 1000;
}

function srtToSubtitles(srt, previous) {
    const normalized = srt.replace(/^﻿/, '').replace(/\r\n?/g, '\n').trim();
    if (!normalized) throw new Error('SRT fitxategia hutsik dago.');

    return normalized.split(/\n\s*\n/).map((block, index) => {
        const lines = block.split('\n');
        const timeIndex = lines.findIndex(line => line.includes('-->'));
        if (timeIndex < 0) throw new Error(`${index + 1}. SRT blokeak ez du denborarik.`);
        const [startRaw, endRaw] = lines[timeIndex].split(/\s*-->\s*/);
        let text = lines.slice(timeIndex + 1).join(' ').replace(/<[^>]+>/g, '').trim();
        let gender = null;
        const tag = text.match(/^\[([^\]]+)\]\s*/);
        if (tag && VOICE_ALIASES[tag[1].trim().toLowerCase()]) {
            gender = VOICE_ALIASES[tag[1].trim().toLowerCase()];
            text = text.slice(tag[0].length).trim();
        }
        if (!text) throw new Error(`${index + 1}. SRT blokeak ez du testurik.`);
        const sub = {
            start: parseSrtTimestamp(startRaw),
            end: parseSrtTimestamp(endRaw),
            text,
            gender: gender || (previous[index] && previous[index].gender) || 'male',
        };
        if (sub.end <= sub.start) {
            throw new Error(`${index + 1}. SRT blokearen amaiera hasieraren ondoren izan behar da.`);
        }
        return sub;
    });
}

function downloadText(content, filename) {
    const blobUrl = URL.createObjectURL(new Blob([content], { type: 'application/x-subrip;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
}

function baseName() {
    return selectedFile?.name.replace(/\.[^.]+$/, '') || 'bideoa';
}

document.querySelectorAll('.srt-download-btn').forEach(button => {
    button.addEventListener('click', () => {
        try {
            const editor = document.getElementById(button.dataset.editor);
            downloadText(subtitlesToSrt(collectSubtitles(editor)), `${baseName()}_${button.dataset.suffix}.srt`);
        } catch (err) {
            showError(err.message);
        }
    });
});

document.querySelectorAll('.srt-upload-btn').forEach(button => {
    button.addEventListener('click', () => {
        srtUploadTarget = document.getElementById(button.dataset.editor);
        srtFileInput.click();
    });
});

srtFileInput.addEventListener('change', async () => {
    const file = srtFileInput.files[0];
    if (!file || !srtUploadTarget) return;
    try {
        const previous = [...srtUploadTarget.querySelectorAll('.subtitle-row')].map(row => ({
            gender: row.querySelector('.subtitle-voice').value,
        }));
        renderSubtitleEditor(srtUploadTarget, srtToSubtitles(await file.text(), previous));
        hideError();
    } catch (err) {
        showError(err.message);
    } finally {
        srtFileInput.value = '';
    }
});

document.querySelectorAll('.add-subtitle-btn').forEach(button => {
    button.addEventListener('click', () => addSubtitle(document.getElementById(button.dataset.editor)));
});

document.querySelectorAll('.bulk-voice-btn').forEach(button => {
    button.addEventListener('click', () => {
        const editor = document.getElementById(button.dataset.editor);
        editor.querySelectorAll('.subtitle-voice').forEach(select => {
            select.value = button.dataset.gender;
            select.dispatchEvent(new Event('change'));
        });
    });
});

downloadResultSrtBtn.addEventListener('click', () => {
    if (resultSrt) downloadText(resultSrt, `${baseName()}_itzulpena.srt`);
});

// Bideoa erreproduzitzean uneko esaldia nabarmendu
function syncActiveSubtitle() {
    const t = videoPreview.currentTime;
    [transcriptSubtitleEditor, translationSubtitleEditor].forEach(editor => {
        if (editor.closest('.review-card').style.display === 'none') return;
        let activeRow = null;
        editor.querySelectorAll('.subtitle-row').forEach(row => {
            const start = Number(row.querySelector('.subtitle-start').value);
            const end = Number(row.querySelector('.subtitle-end').value);
            const isActive = t >= start && t < end;
            row.classList.toggle('active', isActive);
            if (isActive) activeRow = row;
        });
        if (activeRow && !activeRow.matches(':focus-within') && !videoPreview.paused) {
            const top = activeRow.offsetTop - editor.offsetTop;
            if (top < editor.scrollTop || top + activeRow.offsetHeight > editor.scrollTop + editor.clientHeight) {
                editor.scrollTo({ top: top - editor.clientHeight / 2 + activeRow.offsetHeight / 2, behavior: 'smooth' });
            }
        }
    });
}
videoPreview.addEventListener('timeupdate', syncActiveSubtitle);
videoPreview.addEventListener('seeked', syncActiveSubtitle);

// ---------------------------------------------------------------------------
// Urrats-1 onarpena: SRT (agian editatua) bidali eta itzulpena abiarazi
// ---------------------------------------------------------------------------
approveTranscriptBtn.addEventListener('click', async () => {
    if (!currentJobId) return;
    approveTranscriptBtn.disabled = true;
    try {
        await saveSubtitles(transcriptSubtitleEditor);

        const approveRes = await fetch(`${API_BASE}/api/job/${currentJobId}/approve`, {
            method: 'POST',
        });
        if (!approveRes.ok) {
            const err = await approveRes.json().catch(() => ({ detail: approveRes.statusText }));
            throw new Error(err.detail || 'Ezin izan da itzulpena abiarazi');
        }

        transcriptReviewCard.style.display = 'none';
        progressCard.style.display = 'block';
        startPolling(currentJobId);
    } catch (err) {
        showError(err.message);
    } finally {
        approveTranscriptBtn.disabled = false;
    }
});

// ---------------------------------------------------------------------------
// Urrats-2 onarpena: itzulpena onartu, ahotsa sortu
// ---------------------------------------------------------------------------
approveTranslationBtn.addEventListener('click', async () => {
    if (!currentJobId) return;
    approveTranslationBtn.disabled = true;
    try {
        await saveSubtitles(translationSubtitleEditor);

        const approveRes = await fetch(`${API_BASE}/api/job/${currentJobId}/approve`, {
            method: 'POST',
        });
        if (!approveRes.ok) {
            const err = await approveRes.json().catch(() => ({ detail: approveRes.statusText }));
            throw new Error(err.detail || 'Ezin izan da ahotsa abiarazi');
        }

        translationReviewCard.style.display = 'none';
        progressCard.style.display = 'block';
        startPolling(currentJobId);
    } catch (err) {
        showError(err.message);
    } finally {
        approveTranslationBtn.disabled = false;
    }
});

// ---------------------------------------------------------------------------
// Urrats-3 onarpena: ahotsa onartu, bideo finala sortu
// ---------------------------------------------------------------------------
approveSynthesisBtn.addEventListener('click', async () => {
    if (!currentJobId) return;
    approveSynthesisBtn.disabled = true;
    try {
        const approveRes = await fetch(`${API_BASE}/api/job/${currentJobId}/approve`, {
            method: 'POST',
        });
        if (!approveRes.ok) {
            const err = await approveRes.json().catch(() => ({ detail: approveRes.statusText }));
            throw new Error(err.detail || 'Ezin izan da bideo finala abiarazi');
        }

        synthesisAudio.pause();
        synthesisReviewCard.style.display = 'none';
        progressCard.style.display = 'block';
        startPolling(currentJobId);
    } catch (err) {
        showError(err.message);
    } finally {
        approveSynthesisBtn.disabled = false;
    }
});

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------
const STEP_ORDER = [
    'uploading', 'extracting_audio', 'transcribing',
    'translating', 'synthesizing', 'combining', 'finalizing', 'completed'
];

function resetProgress() {
    progressBar.style.width = '5%';
    progressText.textContent = 'Hasieratzen...';
    document.querySelectorAll('.step').forEach(s => {
        s.classList.remove('active', 'done');
    });
}

function updateProgress(step, progress, detail) {
    progressBar.style.width = `${Math.max(progress, 5)}%`;
    progressText.textContent = detail || step;

    const stepIdx = STEP_ORDER.indexOf(step);
    document.querySelectorAll('.step').forEach(el => {
        const elStep = el.dataset.step;
        const elIdx = STEP_ORDER.indexOf(elStep);
        el.classList.remove('active', 'done');
        if (elIdx < stepIdx) el.classList.add('done');
        else if (elIdx === stepIdx) el.classList.add('active');
    });
}

function markAllStepsDone() {
    document.querySelectorAll('.step').forEach(s => {
        s.classList.remove('active');
        s.classList.add('done');
    });
}

// ---------------------------------------------------------------------------
// Error Handling
// ---------------------------------------------------------------------------
function showError(message) {
    errorText.textContent = message;
    errorCard.style.display = 'block';
}

function hideError() {
    errorCard.style.display = 'none';
}

retryBtn.addEventListener('click', () => {
    hideError();
});

newTranslationBtn.addEventListener('click', () => {
    resultsCard.style.display = 'none';
    transcriptReviewCard.style.display = 'none';
    translationReviewCard.style.display = 'none';
    synthesisReviewCard.style.display = 'none';
    currentJobId = null;
    clearFile();
    resetProgress();
});

// ---------------------------------------------------------------------------
// Progress reporting is now REAL, not simulated: see startPolling() above, which
// polls GET /api/job/{job_id} while the backend runs the pipeline as a
// background task. This replaces the old simulateProgress() timer-based
// fake progress bar, which was needed only because /api/translate used to
// block until the whole pipeline finished (and got killed by the HF Spaces
// proxy's ~60s timeout on long-running requests).
