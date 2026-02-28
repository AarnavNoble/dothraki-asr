"""End-to-end Dothraki ASR pipeline runner.

Usage:
    from pipeline.run import Pipeline
    result = Pipeline().run("path/to/got_scene.mp4")
    print(result.translation.translation)

CLI:
    uv run python -m pipeline.run path/to/got_scene.mp4
    uv run python -m pipeline.run --strategy embedding data/synthetic/d0001.wav
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.config import DEFAULT_WHISPER_MODEL, RESULTS_DIR, Strategy


@dataclass
class PipelineResult:
    input_path: str
    vocals_path: str
    strategy: str
    transcription: object | None  # TranscriptionResult, None for non-phoneme strategies
    match_results: list[dict]
    translation: object | None  # TranslationResult
    quality: str = "good"
    clip_matches: list[dict] | None = None  # For embedding/dtw/finetune/ensemble
    raw_dothraki: str | None = None  # Direct Dothraki output from finetune

    def save(self, output_dir: str | Path | None = None) -> Path:
        """Save all pipeline outputs to a directory."""
        if output_dir is None:
            output_dir = RESULTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(self.input_path).stem

        if self.transcription is not None:
            self.transcription.save(output_dir / f"{stem}_transcription.json")
        if self.translation is not None:
            self.translation.save(output_dir / f"{stem}_translation.json")

        summary = {
            "input": self.input_path,
            "vocals": self.vocals_path,
            "strategy": self.strategy,
            "quality": self.quality,
        }

        if self.transcription is not None:
            summary["whisper_text"] = self.transcription.text
            summary["whisper_language"] = self.transcription.language
            summary["whisper_model"] = self.transcription.model

        if self.translation is not None:
            summary["translation"] = self.translation.translation
            summary["words"] = [
                {
                    "heard": w.original,
                    "dothraki": w.dothraki,
                    "english": w.english,
                    "confidence": w.confidence,
                }
                for w in self.translation.words
            ]

        if self.clip_matches is not None:
            summary["clip_matches"] = self.clip_matches

        if self.raw_dothraki is not None:
            summary["raw_dothraki"] = self.raw_dothraki

        summary_path = output_dir / f"{stem}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        return summary_path


class Pipeline:
    """Runs the Dothraki ASR pipeline with selectable matching strategy."""

    def __init__(
        self,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        min_confidence: float = 0.4,
        top_k: int = 5,
        skip_separation: bool = False,
        strategy: str = "phoneme",
    ):
        self.strategy = Strategy(strategy)
        self.top_k = top_k
        self._skip_separation = skip_separation
        self._whisper_model = whisper_model
        self._min_confidence = min_confidence

        # Lazy-init only the components needed for the chosen strategy
        self._separator = None
        self._transcriber = None
        self._matcher = None
        self._translator = None
        self._embedding_matcher = None
        self._sequence_matcher = None
        self._finetuned_decoder = None

        strategies_needing_phoneme = {Strategy.PHONEME, Strategy.ENSEMBLE}
        strategies_needing_embedding = {Strategy.EMBEDDING, Strategy.ENSEMBLE}
        strategies_needing_dtw = {Strategy.DTW, Strategy.ENSEMBLE}
        strategies_needing_finetune = {Strategy.FINETUNE, Strategy.ENSEMBLE}

        if not skip_separation:
            from pipeline.audio.separator import VocalSeparator
            self._separator = VocalSeparator()

        if self.strategy in strategies_needing_phoneme:
            from pipeline.asr.transcriber import Transcriber
            from pipeline.dothraki.matcher import DothrakiMatcher
            from pipeline.dothraki.translator import Translator
            self._transcriber = Transcriber(model=whisper_model)
            self._matcher = DothrakiMatcher()
            self._translator = Translator(min_confidence=min_confidence)

        if self.strategy in strategies_needing_embedding:
            from pipeline.asr.embedder import EmbeddingMatcher
            self._embedding_matcher = EmbeddingMatcher()

        if self.strategy in strategies_needing_dtw:
            from pipeline.dothraki.sequence_matcher import SequenceMatcher
            self._sequence_matcher = SequenceMatcher()

        if self.strategy in strategies_needing_finetune:
            from pipeline.asr.finetuned import FinetunedDecoder
            self._finetuned_decoder = FinetunedDecoder()

    def run(
        self,
        input_path: str | Path,
        language: str | None = None,
        save: bool = True,
    ) -> PipelineResult:
        """Run the pipeline on an audio or video file."""
        input_path = Path(input_path)

        # 1. Vocal isolation
        if self._separator is not None:
            vocals_path = self._separator.separate(input_path)
        else:
            vocals_path = input_path

        dispatch = {
            Strategy.PHONEME: self._run_phoneme,
            Strategy.EMBEDDING: self._run_embedding,
            Strategy.DTW: self._run_dtw,
            Strategy.FINETUNE: self._run_finetune,
            Strategy.ENSEMBLE: self._run_ensemble,
        }

        result = dispatch[self.strategy](input_path, vocals_path, language)

        if save:
            result.save()

        return result

    def _run_phoneme(
        self, input_path: Path, vocals_path: Path, language: str | None
    ) -> PipelineResult:
        """Original phoneme-matching pipeline."""
        from pipeline.dothraki.translator import TranslationResult

        transcription = self._transcriber.transcribe(vocals_path, language=language)

        if transcription.is_hallucination:
            quality = "hallucinated"
            match_results: list[dict] = []
            translation = TranslationResult(words=[], translation="")
        elif not transcription.text.strip():
            quality = "empty"
            match_results = []
            translation = TranslationResult(words=[], translation="")
        else:
            match_results = self._matcher.match_transcription(
                transcription, top_k=self.top_k
            )
            translation = self._translator.translate(match_results)
            if translation.words:
                avg_conf = sum(w.confidence for w in translation.words) / len(
                    translation.words
                )
                quality = "good" if avg_conf >= 0.4 else "low_confidence"
            else:
                quality = "low_confidence"

        return PipelineResult(
            input_path=str(input_path),
            vocals_path=str(vocals_path),
            strategy=self.strategy.value,
            transcription=transcription,
            match_results=match_results,
            translation=translation,
            quality=quality,
        )

    def _run_embedding(
        self, input_path: Path, vocals_path: Path, language: str | None
    ) -> PipelineResult:
        """Whisper encoder embedding similarity matching."""
        matches = self._embedding_matcher.match(vocals_path, top_k=self.top_k)

        clip_matches = [
            {
                "clip_id": m.clip_id,
                "dothraki": m.dothraki,
                "english": m.english,
                "score": m.score,
                "audio_file": m.audio_file,
            }
            for m in matches
        ]

        # Use top-1 match's English as the translation
        top_english = matches[0].english if matches else ""
        quality = "good" if matches and matches[0].score >= 0.5 else "low_confidence"

        return PipelineResult(
            input_path=str(input_path),
            vocals_path=str(vocals_path),
            strategy=self.strategy.value,
            transcription=None,
            match_results=[],
            translation=None,
            quality=quality,
            clip_matches=clip_matches,
            raw_dothraki=matches[0].dothraki if matches else None,
        )

    def _run_dtw(
        self, input_path: Path, vocals_path: Path, language: str | None
    ) -> PipelineResult:
        """DTW sequence matching on MFCC features."""
        matches = self._sequence_matcher.match(vocals_path, top_k=self.top_k)

        clip_matches = [
            {
                "clip_id": m.clip_id,
                "dothraki": m.dothraki,
                "english": m.english,
                "score": m.score,
                "dtw_cost": m.dtw_cost,
                "audio_file": m.audio_file,
            }
            for m in matches
        ]

        top_english = matches[0].english if matches else ""
        quality = "good" if matches and matches[0].score >= 0.5 else "low_confidence"

        return PipelineResult(
            input_path=str(input_path),
            vocals_path=str(vocals_path),
            strategy=self.strategy.value,
            transcription=None,
            match_results=[],
            translation=None,
            quality=quality,
            clip_matches=clip_matches,
            raw_dothraki=matches[0].dothraki if matches else None,
        )

    def _run_finetune(
        self, input_path: Path, vocals_path: Path, language: str | None
    ) -> PipelineResult:
        """Fine-tuned Whisper decoder for direct Dothraki output."""
        result = self._finetuned_decoder.decode(vocals_path)

        quality = "good" if result.text.strip() else "empty"

        return PipelineResult(
            input_path=str(input_path),
            vocals_path=str(vocals_path),
            strategy=self.strategy.value,
            transcription=None,
            match_results=[],
            translation=None,
            quality=quality,
            raw_dothraki=result.text,
        )

    def _run_ensemble(
        self, input_path: Path, vocals_path: Path, language: str | None
    ) -> PipelineResult:
        """Run all strategies and merge results."""
        phoneme_result = self._run_phoneme(input_path, vocals_path, language)
        embedding_result = self._run_embedding(input_path, vocals_path, language)
        dtw_result = self._run_dtw(input_path, vocals_path, language)
        finetune_result = self._run_finetune(input_path, vocals_path, language)

        # Merge clip matches from embedding and DTW (dedupe by clip_id, keep highest score)
        seen: dict[str, dict] = {}
        for match_list in [
            embedding_result.clip_matches or [],
            dtw_result.clip_matches or [],
        ]:
            for m in match_list:
                cid = m["clip_id"]
                if cid not in seen or m["score"] > seen[cid]["score"]:
                    seen[cid] = m

        merged_matches = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

        # Prefer finetune output for raw_dothraki if non-empty
        raw_dothraki = finetune_result.raw_dothraki
        if not raw_dothraki or not raw_dothraki.strip():
            raw_dothraki = embedding_result.raw_dothraki

        # Quality: best of all strategies
        qualities = [
            phoneme_result.quality,
            embedding_result.quality,
            dtw_result.quality,
            finetune_result.quality,
        ]
        quality = "good" if "good" in qualities else "low_confidence"

        return PipelineResult(
            input_path=str(input_path),
            vocals_path=str(vocals_path),
            strategy=self.strategy.value,
            transcription=phoneme_result.transcription,
            match_results=phoneme_result.match_results,
            translation=phoneme_result.translation,
            quality=quality,
            clip_matches=merged_matches,
            raw_dothraki=raw_dothraki,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.run [--strategy STRATEGY] <audio_or_video_path>")
        sys.exit(1)

    # Simple arg parsing for CLI
    strategy = "phoneme"
    path_arg = sys.argv[1]
    if sys.argv[1] == "--strategy" and len(sys.argv) >= 4:
        strategy = sys.argv[2]
        path_arg = sys.argv[3]

    result = Pipeline(strategy=strategy, skip_separation=True).run(path_arg)

    print(f"\nStrategy: {result.strategy}")
    if result.transcription is not None:
        print(f"Whisper heard:  {result.transcription.text}")
        print(f"Detected lang:  {result.transcription.language}")
    if result.translation is not None:
        print(f"\nTranslation:    {result.translation.translation}")
        print(f"\nPer-word breakdown:")
        for w in result.translation.words:
            tag = f"{w.dothraki} → {w.english}" if w.dothraki else "???"
            print(f"  {w.original:20s}  ({w.confidence:.2f})  {tag}")
    if result.raw_dothraki:
        print(f"\nDothraki output: {result.raw_dothraki}")
    if result.clip_matches:
        print(f"\nTop clip matches:")
        for m in result.clip_matches[:5]:
            print(f"  {m['clip_id']}: {m['dothraki']!r} ({m['score']:.3f})")
