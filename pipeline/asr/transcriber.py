"""Zero-shot ASR using mlx-whisper for the Dothraki ASR pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.config import DEFAULT_WHISPER_MODEL, RESULTS_DIR, WHISPER_MODELS

# Maps short model names → mlx-community HuggingFace repos
_MODEL_REPOS: dict[str, str] = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


@dataclass
class Segment:
    start: float   # seconds
    end: float     # seconds
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    language: str | None   # ISO-639-1 code detected by Whisper (e.g. "tr", "ar")
    model: str
    audio_path: str        # stored as str for JSON serialisability

    def save(self, output_path: str | Path | None = None) -> Path:
        """Persist result as JSON.  Defaults to RESULTS_DIR/<audio_stem>.json."""
        if output_path is None:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            stem = Path(self.audio_path).stem
            output_path = RESULTS_DIR / f"{stem}_transcription.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, ensure_ascii=False, indent=2)

        return output_path


class Transcriber:
    """Runs mlx-whisper on a 16 kHz mono WAV and returns a TranscriptionResult."""

    def __init__(self, model: str = DEFAULT_WHISPER_MODEL):
        if model not in _MODEL_REPOS:
            raise ValueError(
                f"Unknown model '{model}'. Choose from: {list(_MODEL_REPOS)}"
            )
        self.model = model
        self._repo = _MODEL_REPOS[model]

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        task: str = "transcribe",
        save_result: bool = False,
    ) -> TranscriptionResult:
        """Transcribe a WAV file with Whisper.

        Args:
            audio_path: Path to a 16 kHz mono WAV (output of VocalSeparator).
            language: ISO-639-1 code to force (e.g. "tr" for Turkish as a
                      phonological proxy).  None = let Whisper auto-detect.
            task: "transcribe" (keep detected language) or "translate" (→ English).
            save_result: If True, write a JSON file to RESULTS_DIR.

        Returns:
            TranscriptionResult with full text, segments, and detected language.
        """
        import mlx_whisper

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        raw = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._repo,
            language=language,
            task=task,
            word_timestamps=False,              # segment-level is enough for the matcher
            condition_on_previous_text=False,   # prevent hallucination loops
            compression_ratio_threshold=1.5,    # tighter repetition detection (default 2.4)
            hallucination_silence_threshold=2.0, # skip hallucinated silent segments
        )

        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in raw.get("segments", [])
        ]

        result = TranscriptionResult(
            text=raw["text"].strip(),
            segments=segments,
            language=raw.get("language"),
            model=self.model,
            audio_path=str(audio_path),
        )

        if save_result:
            result.save()

        return result
