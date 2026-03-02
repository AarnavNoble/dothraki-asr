"""FastAPI backend for Dothraki ASR web demo."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Resolve paths relative to project root (web/api/main.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = SYNTHETIC_DIR / "manifest.json"

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


def _load_manifest() -> list[dict]:
    global _manifest_cache
    if _manifest_cache is None:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            _manifest_cache = json.load(f)
    return _manifest_cache


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
    for p in sorted(RAW_DIR.glob("*.wav")):
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
    return {"clips": clips + raw_clips[:2]}


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

    return FileResponse(audio_path, media_type="audio/wav")


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
