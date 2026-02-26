"""Dothraki phoneme matching engine.

Matches IPA strings produced by phonemizer.py against the Dothraki lexicon
using normalised edit distance.  Also supports direct orthographic matching
(useful when Whisper happens to output a phonetically plausible Dothraki word).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from pipeline.config import LEXICON_FILE, TOP_K_MATCHES
from pipeline.dothraki.phonemizer import ENGLISH_STOP_WORDS, phonemize_text, whisper_lang_to_gruut


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


# Minimum IPA length — queries shorter than this are too ambiguous to match
MIN_IPA_LENGTH = 3

# Minimum similarity score — matches below this are pure noise
MIN_MATCH_SCORE = 0.25

# Articulatory-weighted substitution costs for IPA characters.
# Pairs of sounds that are phonetically close get reduced cost.
_VOICING_PAIRS = {
    frozenset(p) for p in [
        ("p", "b"), ("t", "d"), ("k", "g"), ("f", "v"), ("s", "z"),
        ("θ", "ð"), ("ʃ", "ʒ"), ("tʃ", "dʒ"),
    ]
}
_PLACE_PAIRS = {
    frozenset(p) for p in [
        ("t", "s"), ("d", "z"), ("n", "l"), ("n", "r"), ("l", "r"),
        ("ʃ", "s"), ("ʒ", "z"),
    ]
}
_VOWEL_HEIGHT_PAIRS = {
    frozenset(p) for p in [
        ("a", "ɛ"), ("ɛ", "e"), ("e", "i"), ("o", "ɔ"), ("ɔ", "u"),
        ("a", "æ"), ("ɛ", "ɪ"), ("ʊ", "u"), ("ə", "ɛ"), ("ə", "a"),
    ]
}


def _sub_cost(c1: str, c2: str) -> float:
    """Articulatory-weighted substitution cost between two IPA chars."""
    if c1 == c2:
        return 0.0
    pair = frozenset((c1, c2))
    if pair in _VOICING_PAIRS:
        return 0.3
    if pair in _VOWEL_HEIGHT_PAIRS:
        return 0.3
    if pair in _PLACE_PAIRS:
        return 0.5
    return 1.0


def _weighted_levenshtein(a: str, b: str) -> float:
    """Weighted Levenshtein with articulatory substitution costs."""
    m, n = len(a), len(b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = _sub_cost(a[i - 1], b[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,      # deletion
                dp[i][j - 1] + 1.0,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n]


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

        # Quality gate: skip queries that are too short to match meaningfully
        if len(query_norm) < MIN_IPA_LENGTH:
            return []

        scored: list[tuple[float, dict]] = [
            (_weighted_levenshtein(query_norm, entry["_norm_ipa"]), entry)
            for entry in self._entries
        ]
        scored.sort(key=lambda x: x[0])

        results = []
        for dist, entry in scored[:top_k]:
            max_len = max(len(query_norm), len(entry["_norm_ipa"]), 1)
            score = round(1.0 - dist / max_len, 4)
            # Quality gate: skip matches below minimum score
            if score < MIN_MATCH_SCORE:
                continue
            results.append(
                MatchCandidate(
                    word=entry["word"],
                    ipa=entry["ipa"],
                    english=entry["english"],
                    part_of_speech=entry["part_of_speech"],
                    score=score,
                    edit_distance=int(dist),
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
        if len(query) < MIN_IPA_LENGTH:
            return []
        scored: list[tuple[float, dict]] = [
            (_weighted_levenshtein(query, entry["word"].lower()), entry)
            for entry in self._entries
        ]
        scored.sort(key=lambda x: x[0])
        results = []
        for dist, entry in scored[:top_k]:
            max_len = max(len(query), len(entry["word"]), 1)
            score = round(1.0 - dist / max_len, 4)
            if score < MIN_MATCH_SCORE:
                continue
            results.append(
                MatchCandidate(
                    word=entry["word"],
                    ipa=entry["ipa"],
                    english=entry["english"],
                    part_of_speech=entry["part_of_speech"],
                    score=score,
                    edit_distance=int(dist),
                )
            )
        return results

    def match_text(
        self,
        text: str,
        lang: str = "en-us",
        top_k: int = TOP_K_MATCHES,
        whisper_lang: str | None = None,
    ) -> list[dict]:
        """Phonemize all words in text and match each against the lexicon.

        Returns a list of per-word dicts::

            [
                {"word": "dothrak", "ipa": "doθɾæk", "matches": [MatchCandidate, ...]},
                ...
            ]
        """
        results = []
        for word, ipa in phonemize_text(text, lang=lang, whisper_lang=whisper_lang):
            clean = word.lower().strip(".,!?;:\"'")
            if ipa:
                matches = self.match_ipa(ipa, top_k=top_k)
            elif clean in ENGLISH_STOP_WORDS or len(clean) <= 1:
                # Skip stop words and single chars entirely
                matches = []
            else:
                # Orthographic fallback for non-stop, non-trivial words
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
        return self.match_text(
            result.text, lang=lang, top_k=top_k, whisper_lang=result.language
        )
