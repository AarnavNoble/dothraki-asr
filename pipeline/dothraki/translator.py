"""Dothraki → English translation from phoneme matcher output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.config import RESULTS_DIR


@dataclass
class TranslatedWord:
    original: str            # Word as Whisper heard it
    dothraki: str | None     # Best-match Dothraki word (None if below threshold)
    english: str | None      # English gloss (None if below threshold)
    confidence: float        # Match score 0→1


@dataclass
class TranslationResult:
    words: list[TranslatedWord]
    translation: str         # Assembled English sentence (low-confidence words kept as-is)

    def save(self, output_path: str | Path | None = None) -> Path:
        if output_path is None:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = RESULTS_DIR / "translation.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, ensure_ascii=False, indent=2)
        return output_path


class Translator:
    """Assembles an English translation from DothrakiMatcher results.

    Words whose best match falls below *min_confidence* are kept as-is
    (bracketed) in the translation so the user can see where the model
    was uncertain.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def translate(self, match_results: list[dict]) -> TranslationResult:
        """Turn per-word match dicts into a TranslationResult.

        Args:
            match_results: Output of DothrakiMatcher.match_text() or
                           .match_transcription().  Each dict has keys
                           "word", "ipa", "matches".
        """
        words: list[TranslatedWord] = []
        parts: list[str] = []

        for entry in match_results:
            original = entry["word"]

            # Skip single-character words (Whisper artifacts like "V", "L")
            if len(original.strip(".,!?;:\"'")) <= 1:
                words.append(TranslatedWord(
                    original=original, dothraki=None, english=None, confidence=0.0,
                ))
                continue

            matches = entry.get("matches", [])

            if matches and matches[0].score >= self.min_confidence:
                best = matches[0]
                english = best.english

                # Deduplicate consecutive identical translations
                if parts and parts[-1] == english:
                    words.append(TranslatedWord(
                        original=original, dothraki=best.word,
                        english=english, confidence=best.score,
                    ))
                    continue

                tw = TranslatedWord(
                    original=original,
                    dothraki=best.word,
                    english=english,
                    confidence=best.score,
                )
                parts.append(english)
            else:
                score = matches[0].score if matches else 0.0
                tw = TranslatedWord(
                    original=original,
                    dothraki=None,
                    english=None,
                    confidence=score,
                )
                parts.append(f"[{original}]")

            words.append(tw)

        return TranslationResult(
            words=words,
            translation=" ".join(parts),
        )
