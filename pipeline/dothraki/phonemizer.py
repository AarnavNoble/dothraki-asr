"""Text-to-IPA phonemization using gruut, for Whisper output preprocessing."""

from __future__ import annotations

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


def whisper_lang_to_gruut(whisper_lang: str | None) -> str:
    """Convert a Whisper language code to a gruut language tag.

    Falls back to en-us for unsupported languages (e.g. "tr" for Turkish,
    which Whisper commonly detects when hearing Dothraki).
    """
    if not whisper_lang:
        return _DEFAULT_LANG
    return _WHISPER_TO_GRUUT.get(whisper_lang, _DEFAULT_LANG)


def phonemize_text(text: str, lang: str = "en-us") -> list[tuple[str, str | None]]:
    """Phonemize all words in text using gruut.

    Args:
        text: Raw text (Whisper transcription output).
        lang: gruut language tag (e.g. "en-us").

    Returns:
        List of (word, ipa) tuples.  ipa is None for words gruut can't handle.
    """
    results: list[tuple[str, str | None]] = []
    try:
        for sentence in gruut.sentences(text, lang=lang):
            for word in sentence:
                if word.is_major_break or word.is_minor_break:
                    continue
                ipa = "".join(word.phonemes) if word.phonemes else None
                results.append((word.text, ipa))
    except Exception:
        # If gruut fails entirely (unsupported lang, empty input), return
        # raw words with no IPA so the caller can still attempt matching.
        for w in text.split():
            results.append((w, None))
    return results
