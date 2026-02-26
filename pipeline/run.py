"""End-to-end Dothraki ASR pipeline runner.

Usage:
    from pipeline.run import Pipeline
    result = Pipeline().run("path/to/got_scene.mp4")
    print(result.translation.translation)

CLI:
    uv run python -m pipeline.run path/to/got_scene.mp4
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.asr.transcriber import TranscriptionResult, Transcriber
from pipeline.audio.separator import VocalSeparator
from pipeline.config import DEFAULT_WHISPER_MODEL, RESULTS_DIR
from pipeline.dothraki.matcher import DothrakiMatcher
from pipeline.dothraki.translator import TranslationResult, Translator


@dataclass
class PipelineResult:
    input_path: str
    vocals_path: str
    transcription: TranscriptionResult
    match_results: list[dict]
    translation: TranslationResult
    quality: str = "good"  # "good", "low_confidence", "hallucinated", "empty"

    def save(self, output_dir: str | Path | None = None) -> Path:
        """Save all pipeline outputs to a directory."""
        if output_dir is None:
            output_dir = RESULTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(self.input_path).stem

        self.transcription.save(output_dir / f"{stem}_transcription.json")
        self.translation.save(output_dir / f"{stem}_translation.json")

        summary = {
            "input": self.input_path,
            "vocals": self.vocals_path,
            "quality": self.quality,
            "whisper_text": self.transcription.text,
            "whisper_language": self.transcription.language,
            "whisper_model": self.transcription.model,
            "translation": self.translation.translation,
            "words": [
                {
                    "heard": w.original,
                    "dothraki": w.dothraki,
                    "english": w.english,
                    "confidence": w.confidence,
                }
                for w in self.translation.words
            ],
        }
        summary_path = output_dir / f"{stem}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        return summary_path


class Pipeline:
    """Runs the full Dothraki ASR pipeline: audio → ASR → match → translate."""

    def __init__(
        self,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        min_confidence: float = 0.4,
        top_k: int = 5,
        skip_separation: bool = False,
    ):
        self.separator = None if skip_separation else VocalSeparator()
        self.transcriber = Transcriber(model=whisper_model)
        self.matcher = DothrakiMatcher()
        self.translator = Translator(min_confidence=min_confidence)
        self.top_k = top_k

    def run(
        self,
        input_path: str | Path,
        language: str | None = None,
        save: bool = True,
    ) -> PipelineResult:
        """Run the full pipeline on an audio or video file.

        Args:
            input_path: Path to audio/video with Dothraki speech.
                        If skip_separation=True, this should be a clean
                        16 kHz mono WAV (vocals only, no music/SFX).
            language: Force a Whisper language code (None = auto-detect).
            save: Write result JSONs to RESULTS_DIR.

        Returns:
            PipelineResult with all intermediate and final outputs.
        """
        input_path = Path(input_path)

        # 1. Vocal isolation (skip if already clean vocals)
        if self.separator is not None:
            vocals_path = self.separator.separate(input_path)
        else:
            vocals_path = input_path

        # 2. Zero-shot ASR
        transcription = self.transcriber.transcribe(vocals_path, language=language)

        # 3. Quality gate — skip matching/translation for garbage input
        if transcription.is_hallucination:
            quality = "hallucinated"
            match_results: list[dict] = []
            translation = TranslationResult(words=[], translation="")
        elif not transcription.text.strip():
            quality = "empty"
            match_results = []
            translation = TranslationResult(words=[], translation="")
        else:
            # 3a. Phoneme matching
            match_results = self.matcher.match_transcription(
                transcription, top_k=self.top_k
            )
            # 3b. Translation
            translation = self.translator.translate(match_results)
            # Classify quality based on translation confidence
            if translation.words:
                avg_conf = sum(w.confidence for w in translation.words) / len(translation.words)
                quality = "good" if avg_conf >= 0.4 else "low_confidence"
            else:
                quality = "low_confidence"

        result = PipelineResult(
            input_path=str(input_path),
            vocals_path=str(vocals_path),
            transcription=transcription,
            match_results=match_results,
            translation=translation,
            quality=quality,
        )

        if save:
            result.save()

        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.run <audio_or_video_path>")
        sys.exit(1)

    result = Pipeline().run(sys.argv[1])
    print(f"\nWhisper heard:  {result.transcription.text}")
    print(f"Detected lang:  {result.transcription.language}")
    print(f"\nTranslation:    {result.translation.translation}")
    print(f"\nPer-word breakdown:")
    for w in result.translation.words:
        tag = f"{w.dothraki} → {w.english}" if w.dothraki else "???"
        print(f"  {w.original:20s}  ({w.confidence:.2f})  {tag}")
