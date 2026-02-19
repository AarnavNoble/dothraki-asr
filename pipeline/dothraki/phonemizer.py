"""Text-to-IPA phonemization for Whisper output preprocessing.

Primary backend: gruut (supports ~11 languages).
Fallback backend: espeak-ng via subprocess (supports 100+ languages).
If neither can handle a word, returns None so the matcher can fall back
to orthographic comparison.
"""

from __future__ import annotations

import shutil
import subprocess

import gruut

# Whisper ISO-639-1 codes → gruut language tags (best-effort mapping)
_WHISPER_TO_GRUUT: dict[str, str] = {
    "en": "en-us",
    "de": "de-de",
    "fr": "fr-fr",
    "es": "es-es",
    "nl": "nl",
    "pt": "pt-br",
    "ru": "ru-ru",
    "sv": "sv-se",
    "sw": "sw",
    "uk": "uk",
    "zh": "zh-cn",
}
_DEFAULT_LANG = "en-us"

_ESPEAK_BIN: str | None = shutil.which("espeak-ng") or shutil.which("espeak")


def whisper_lang_to_gruut(whisper_lang: str | None) -> str:
    """Convert a Whisper language code to a gruut language tag.

    Falls back to en-us for unsupported languages (e.g. "tr" for Turkish,
    which Whisper commonly detects when hearing Dothraki).
    """
    if not whisper_lang:
        return _DEFAULT_LANG
    return _WHISPER_TO_GRUUT.get(whisper_lang, _DEFAULT_LANG)


def _gruut_phonemize(text: str, lang: str) -> list[tuple[str, str | None]]:
    """Try gruut first. Returns list of (word, ipa|None)."""
    results: list[tuple[str, str | None]] = []
    for sentence in gruut.sentences(text, lang=lang):
        for word in sentence:
            if word.is_major_break or word.is_minor_break:
                continue
            ipa = "".join(word.phonemes) if word.phonemes else None
            results.append((word.text, ipa))
    return results


def _espeak_phonemize(text: str, lang: str) -> list[tuple[str, str | None]]:
    """Fallback: use espeak-ng subprocess for languages gruut doesn't support.

    espeak-ng accepts ISO-639-1 codes directly (e.g. "tr", "ar", "hi").
    """
    if not _ESPEAK_BIN:
        return [(w, None) for w in text.split()]

    # Strip the gruut region suffix (e.g. "en-us" → "en") for espeak
    espeak_lang = lang.split("-")[0] if "-" in lang else lang

    try:
        result = subprocess.run(
            [_ESPEAK_BIN, "--ipa", "-q", f"--sep= ", f"-v{espeak_lang}", text],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return [(w, None) for w in text.split()]

        # espeak outputs space-separated IPA phonemes, one line per clause
        ipa_tokens = result.stdout.strip().split()
        words = text.split()

        # Best-effort alignment: pair words with IPA tokens positionally
        results: list[tuple[str, str | None]] = []
        for i, word in enumerate(words):
            ipa = ipa_tokens[i] if i < len(ipa_tokens) else None
            results.append((word, ipa))
        return results

    except (subprocess.TimeoutExpired, OSError):
        return [(w, None) for w in text.split()]


def phonemize_text(
    text: str,
    lang: str = "en-us",
    whisper_lang: str | None = None,
) -> list[tuple[str, str | None]]:
    """Phonemize all words in text, trying gruut then espeak-ng.

    Args:
        text: Raw text (Whisper transcription output).
        lang: gruut language tag (e.g. "en-us").
        whisper_lang: Original Whisper ISO-639-1 code.  Used for espeak-ng
                      fallback when gruut doesn't support the language.

    Returns:
        List of (word, ipa) tuples.  ipa is None for words that couldn't
        be phonemized by any backend.
    """
    if not text or not text.strip():
        return []

    # Try gruut first
    try:
        results = _gruut_phonemize(text, lang)
        # Check if gruut actually produced any IPA (it might return all Nones
        # if the language data isn't really installed)
        if results and any(ipa for _, ipa in results):
            return results
    except Exception:
        pass

    # Fallback to espeak-ng with the original Whisper language code
    espeak_lang = whisper_lang or lang.split("-")[0]
    return _espeak_phonemize(text, espeak_lang)
