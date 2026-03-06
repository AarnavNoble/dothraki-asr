"""FastAPI backend for Dothraki ASR web demo."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Resolve paths relative to project root (web/api/main.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = SYNTHETIC_DIR / "manifest.json"
DIALOGUE_PATH = PROJECT_ROOT / "data" / "dialogue" / "dothraki_dialogue.json"
LEXICON_PATH = PROJECT_ROOT / "data" / "lexicon" / "dothraki_lexicon.json"
REFERENCE_WAV = RAW_DIR / "drogo_speech_clean.wav"
SPEAK_OUTPUT_DIR = PROJECT_ROOT / "data" / "speak_cache"

# Curated demo clips — hand-picked for variety and recognizability
DEMO_CLIP_IDS = {"d0000", "d0004", "d0005", "d0010", "d0050", "d0100"}

app = FastAPI(title="Dothraki ASR", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_manifest_cache: list[dict] | None = None
_dialogue_cache: list[dict] | None = None
_lexicon_cache: list[dict] | None = None


def _load_manifest() -> list[dict]:
    global _manifest_cache
    if _manifest_cache is None:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            _manifest_cache = json.load(f)
    return _manifest_cache


def _load_dialogue() -> list[dict]:
    global _dialogue_cache
    if _dialogue_cache is None:
        with open(DIALOGUE_PATH, encoding="utf-8") as f:
            _dialogue_cache = json.load(f)
    return _dialogue_cache


def _load_lexicon() -> list[dict]:
    global _lexicon_cache
    if _lexicon_cache is None:
        with open(LEXICON_PATH, encoding="utf-8") as f:
            _lexicon_cache = json.load(f)
    return _lexicon_cache


def _find_closest_dothraki(english_text: str, top_k: int = 3) -> list[dict]:
    """Find the closest Dothraki translations for English text.

    Strategy:
    1. Check the lexicon for exact word matches (single words / short phrases)
    2. Search dialogue pairs, preferring exact word containment over fuzzy match
    """
    query = english_text.lower().strip()
    query_clean = query.translate(str.maketrans("", "", "!?,.:;\"'"))
    query_words = set(query_clean.split()) - {"the", "a", "an", "is", "to"}
    results: list[dict] = []

    # 1. Lexicon lookup (word-level, clean translations)
    lexicon = _load_lexicon()
    for entry in lexicon:
        eng = entry.get("english", "").lower()
        if not eng:
            continue
        # Strip parentheticals for matching
        eng_bare = eng.replace("(singular)", "").replace("(plural)", "").strip()
        if query == eng_bare or query == eng:
            # Exact match
            results.append({
                "dothraki": entry["word"],
                "english": entry["english"],
                "score": 1.0,
                "id": f"lex_{entry['word']}",
            })
        elif eng_bare == query or set(eng_bare.split()) == query_words:
            # Words match exactly (different order)
            results.append({
                "dothraki": entry["word"],
                "english": entry["english"],
                "score": 0.95,
                "id": f"lex_{entry['word']}",
            })
        elif query in eng_bare.split():
            # Query is one of the words in the gloss (partial match — lower score)
            results.append({
                "dothraki": entry["word"],
                "english": entry["english"],
                "score": 0.7,
                "id": f"lex_{entry['word']}",
            })

    # 2. Dialogue pair matching
    dialogue = _load_dialogue()
    scored = []
    for entry in dialogue:
        eng = entry.get("english", "").lower()
        if not eng:
            continue
        # Boost score if query words appear as whole words in the English text
        eng_words = set(eng.translate(str.maketrans("", "", "!?,.:;\"'")).split())
        overlap = len(query_words & eng_words)
        coverage = overlap / len(query_words) if query_words else 0
        fuzzy = SequenceMatcher(None, query, eng).ratio()
        # Heavy weight on word coverage — all query words matching matters most
        score = fuzzy * 0.4 + coverage * 0.6
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    for s, e in scored:
        results.append({
            "dothraki": e["dothraki"],
            "english": e["english"],
            "score": round(s, 3),
            "id": e["id"],
        })

    # Sort all results by score, with phrase preference for multi-word queries
    is_phrase = len(query_words) > 1
    results.sort(key=lambda r: (
        # For phrases, prefer sentences over single words
        (" " in r["dothraki"]) if is_phrase else True,
        r["score"],
    ), reverse=True)

    # Deduplicate by dothraki text, keep highest score
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in results:
        if r["dothraki"] not in seen:
            seen.add(r["dothraki"])
            deduped.append(r)

    return deduped[:top_k]


# Lazy-loaded TTS model
_tts_model = None
_tts_latents = None


def _get_tts():
    """Lazy-load XTTS v2 model and speaker latents."""
    global _tts_model, _tts_latents

    if _tts_model is not None:
        return _tts_model, _tts_latents

    import numpy as np
    import torch
    import torchaudio
    from TTS.api import TTS

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    model = tts.synthesizer.tts_model

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=str(REFERENCE_WAV),
    )

    _tts_model = model
    _tts_latents = (gpt_cond_latent, speaker_embedding)
    return _tts_model, _tts_latents


def _synthesize_dothraki(text: str) -> Path:
    """Synthesize Dothraki text to a WAV file using XTTS v2."""
    import numpy as np
    import torch
    import torchaudio

    model, (gpt_cond_latent, speaker_embedding) = _get_tts()

    result = model.inference(
        text=text,
        language="es",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
    )

    wav_out = result["wav"]
    if isinstance(wav_out, torch.Tensor):
        wav_array = wav_out.cpu().numpy().squeeze()
    else:
        wav_array = np.array(wav_out, dtype=np.float32).squeeze()

    tensor = torch.from_numpy(wav_array).unsqueeze(0).float()
    # XTTS outputs at 24kHz, resample to 16kHz
    resampler = torchaudio.transforms.Resample(24000, 16000)
    tensor = resampler(tensor)
    peak = tensor.abs().max()
    if peak > 0:
        tensor = tensor / peak

    SPEAK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SPEAK_OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"
    torchaudio.save(str(output_path), tensor, 16000)
    return output_path


def _serialize_result(result) -> dict:
    """Convert a PipelineResult into a JSON-safe dict."""
    out: dict = {
        "strategy": result.strategy,
        "quality": result.quality,
        "raw_dothraki": result.raw_dothraki,
        "clip_matches": result.clip_matches,
    }

    if result.transcription is not None:
        out["transcription"] = {
            "text": result.transcription.text,
            "language": result.transcription.language,
            "model": result.transcription.model,
        }

    if result.translation is not None:
        out["translation"] = {
            "text": result.translation.translation,
            "words": [
                {
                    "original": w.original,
                    "dothraki": w.dothraki,
                    "english": w.english,
                    "confidence": w.confidence,
                }
                for w in result.translation.words
            ],
        }

    if result.match_results:
        out["match_results"] = result.match_results

    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/demo-clips")
def get_demo_clips():
    """Return curated demo clips with metadata."""
    manifest = _load_manifest()
    clips = [entry for entry in manifest if entry["id"] in DEMO_CLIP_IDS]
    # Also add raw audio files as demo clips
    raw_clips = []
    for p in sorted(RAW_DIR.glob("*")):
        if p.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            continue
        raw_clips.append(
            {
                "id": f"raw_{p.stem}",
                "audio_file": p.name,
                "dothraki": "",
                "english": f"Real speech: {p.stem.replace('_', ' ')}",
                "category": "raw",
            }
        )
    for clip in clips:
        clip["category"] = "synthetic"
    # Real clips first so they appear at the top
    return {"clips": raw_clips + clips}


@app.get("/api/audio/speak/{filename}")
def get_speak_audio(filename: str):
    """Serve generated speak audio files."""
    audio_path = SPEAK_OUTPUT_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    try:
        audio_path.resolve().relative_to(SPEAK_OUTPUT_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(audio_path, media_type="audio/wav")


@app.get("/api/audio/{category}/{filename}")
def get_audio(category: str, filename: str):
    """Serve audio files from data directories."""
    if category == "synthetic":
        audio_path = SYNTHETIC_DIR / filename
    elif category == "raw":
        audio_path = RAW_DIR / filename
    else:
        raise HTTPException(status_code=400, detail="Invalid category")

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Prevent path traversal
    try:
        audio_path.resolve().relative_to(PROJECT_ROOT / "data")
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }
    media_type = media_types.get(audio_path.suffix.lower(), "audio/wav")
    return FileResponse(audio_path, media_type=media_type)


@app.post("/api/speak")
async def speak_dothraki(text: str = Form(...)):
    """Translate English text to Dothraki and synthesize speech.

    1. Find the closest Dothraki match from the 1,712 dialogue pairs
    2. Synthesize the Dothraki text with XTTS v2 (Drogo's voice)
    3. Return the audio URL + translation info
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Text too long (max 500 chars)")

    # Find closest Dothraki translation
    matches = _find_closest_dothraki(text, top_k=3)
    if not matches:
        raise HTTPException(status_code=404, detail="No translation found")

    best = matches[0]
    dothraki_text = best["dothraki"]

    # Synthesize speech
    try:
        audio_path = _synthesize_dothraki(dothraki_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    return JSONResponse({
        "dothraki": dothraki_text,
        "english": best["english"],
        "score": best["score"],
        "audio_url": f"/api/audio/speak/{audio_path.name}",
        "alternatives": matches[1:],
    })


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    strategy: str = Form("phoneme"),
    language: str | None = Form(None),
):
    """Run the ASR pipeline on an uploaded audio file."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from pipeline.config import Strategy
    from pipeline.run import Pipeline

    # Validate strategy
    try:
        Strategy(strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Choose from: {[s.value for s in Strategy]}",
        )

    # Save upload to temp file
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(audio.file, tmp)
        tmp.close()

        pipeline = Pipeline(
            strategy=strategy,
            skip_separation=True,  # Skip Demucs for speed in web demo
            whisper_model="tiny",  # Use tiny for fast responses
        )
        result = pipeline.run(tmp.name, language=language, save=False)
        return JSONResponse(_serialize_result(result))
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/transcribe-clip")
async def transcribe_clip(
    clip_id: str = Form(...),
    category: str = Form("synthetic"),
    strategy: str = Form("phoneme"),
    language: str | None = Form(None),
):
    """Run the ASR pipeline on a demo clip by ID (no upload needed)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from pipeline.config import Strategy
    from pipeline.run import Pipeline

    try:
        Strategy(strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Choose from: {[s.value for s in Strategy]}",
        )

    if category == "synthetic":
        # Find audio file from manifest
        manifest = _load_manifest()
        entry = next((e for e in manifest if e["id"] == clip_id), None)
        if entry is None:
            raise HTTPException(status_code=404, detail="Clip not found")
        audio_path = SYNTHETIC_DIR / entry["audio_file"]
    elif category == "raw":
        # clip_id format: "raw_<stem>"
        stem = clip_id.removeprefix("raw_")
        candidates = list(RAW_DIR.glob(f"{stem}.*"))
        if not candidates:
            raise HTTPException(status_code=404, detail="Clip not found")
        audio_path = candidates[0]
    else:
        raise HTTPException(status_code=400, detail="Invalid category")

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    pipeline = Pipeline(
        strategy=strategy,
        skip_separation=True,
        whisper_model="tiny",
    )
    result = pipeline.run(str(audio_path), language=language, save=False)
    return JSONResponse(_serialize_result(result))
