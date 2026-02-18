"""Dothraki phoneme matching engine.

Matches IPA strings produced by phonemizer.py against the Dothraki lexicon
using normalised edit distance.  Also supports direct orthographic matching
(useful when Whisper happens to output a phonetically plausible Dothraki word).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from pipeline.config import LEXICON_FILE, TOP_K_MATCHES
from pipeline.dothraki.phonemizer import phonemize_text, whisper_lang_to_gruut


@dataclass
class MatchCandidate:
    word: str           # Dothraki orthographic form
    ipa: str            # Dothraki IPA
    english: str        # English gloss
    part_of_speech: str
    score: float        # Normalised similarity 0.0 (none) → 1.0 (perfect)
    edit_distance: int  # Raw IPA edit distance


def _normalize_ipa(ipa: str) -> str:
    """Lowercase and strip stress / length markers."""
    return (
        ipa.replace("ˈ", "")
           .replace("ˌ", "")
           .replace("ː", "")
           .lower()
    )


def _levenshtein(a: str, b: str) -> int:
    """Character-level Levenshtein distance, O(m·n) time, O(n) space."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


class DothrakiMatcher:
    """Matches IPA (or orthographic) queries against the Dothraki lexicon."""

    def __init__(self):
        with open(LEXICON_FILE, encoding="utf-8") as fh:
            raw = json.load(fh)

        # Pre-normalise IPA for every entry at load time
        self._entries: list[dict] = [
            {**entry, "_norm_ipa": _normalize_ipa(entry["ipa"])}
            for entry in raw
            if entry.get("ipa")
        ]

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def match_ipa(
        self, query_ipa: str, top_k: int = TOP_K_MATCHES
    ) -> list[MatchCandidate]:
        """Return the top_k closest lexicon entries for an IPA query."""
        query_norm = _normalize_ipa(query_ipa)

        scored: list[tuple[int, dict]] = [
            (_levenshtein(query_norm, entry["_norm_ipa"]), entry)
            for entry in self._entries
        ]
        scored.sort(key=lambda x: x[0])

        results = []
        for dist, entry in scored[:top_k]:
            max_len = max(len(query_norm), len(entry["_norm_ipa"]), 1)
            results.append(
                MatchCandidate(
                    word=entry["word"],
                    ipa=entry["ipa"],
                    english=entry["english"],
                    part_of_speech=entry["part_of_speech"],
                    score=round(1.0 - dist / max_len, 4),
                    edit_distance=dist,
                )
            )
        return results

    def match_word(
        self, word: str, lang: str = "en-us", top_k: int = TOP_K_MATCHES
    ) -> list[MatchCandidate]:
        """Phonemize a single word then match against the lexicon.

        Falls back to orthographic edit distance if gruut can't phonemize it.
        """
        pairs = phonemize_text(word, lang=lang)
        if pairs and pairs[0][1]:
            return self.match_ipa(pairs[0][1], top_k=top_k)

        # Fallback: compare Whisper word directly against Dothraki orthography
        query = word.lower()
        scored: list[tuple[int, dict]] = [
            (_levenshtein(query, entry["word"].lower()), entry)
            for entry in self._entries
        ]
        scored.sort(key=lambda x: x[0])
        results = []
        for dist, entry in scored[:top_k]:
            max_len = max(len(query), len(entry["word"]), 1)
            results.append(
                MatchCandidate(
                    word=entry["word"],
                    ipa=entry["ipa"],
                    english=entry["english"],
                    part_of_speech=entry["part_of_speech"],
                    score=round(1.0 - dist / max_len, 4),
                    edit_distance=dist,
                )
            )
        return results

    def match_text(
        self,
        text: str,
        lang: str = "en-us",
        top_k: int = TOP_K_MATCHES,
    ) -> list[dict]:
        """Phonemize all words in text and match each against the lexicon.

        Returns a list of per-word dicts::

            [
                {"word": "dothrak", "ipa": "doθɾæk", "matches": [MatchCandidate, ...]},
                ...
            ]
        """
        results = []
        for word, ipa in phonemize_text(text, lang=lang):
            if ipa:
                matches = self.match_ipa(ipa, top_k=top_k)
            else:
                matches = self.match_word(word, lang=lang, top_k=top_k)
            results.append({"word": word, "ipa": ipa, "matches": matches})
        return results

    def match_transcription(
        self, result, top_k: int = TOP_K_MATCHES
    ) -> list[dict]:
        """Match all words in a TranscriptionResult against the lexicon.

        Uses Whisper's detected language to guide phonemization.
        """
        lang = whisper_lang_to_gruut(result.language)
        return self.match_text(result.text, lang=lang, top_k=top_k)
