"""Fine-tuned Whisper decoder for Dothraki speech.

Loads base Whisper model from HuggingFace, applies fine-tuned decoder weights,
and performs greedy decoding to produce Dothraki text directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from pipeline.config import FINETUNE_MODEL_DIR

# Whisper special tokens
SOT = 50258
EOT = 50257
NO_TIMESTAMPS = 50363

MAX_DECODE_TOKENS = 80


@dataclass
class FinetuneResult:
    raw_text: str       # Full decoder output before truncation
    text: str           # Cleaned text after repetition truncation
    audio_path: str


class FinetunedDecoder:
    """Greedy decoder using fine-tuned Whisper weights for Dothraki."""

    def __init__(self, model_dir: str | Path | None = None):
        self._model_dir = Path(model_dir) if model_dir else FINETUNE_MODEL_DIR
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load base model and apply fine-tuned weights."""
        if self._model is not None:
            return

        from mlx_whisper.load_models import load_model
        from mlx_whisper.tokenizer import get_tokenizer

        # Read config to find base model repo
        config = json.loads((self._model_dir / "config.json").read_text())
        base_repo = config["base_repo"]

        self._model = load_model(base_repo)
        self._tokenizer = get_tokenizer(multilingual=True)

        # Apply fine-tuned weights on top of base model
        weights_path = self._model_dir / "weights.npz"
        self._model.load_weights(str(weights_path))

    def decode(self, audio_path: str | Path) -> FinetuneResult:
        """Decode an audio file to Dothraki text.

        Args:
            audio_path: Path to a 16kHz mono WAV.

        Returns:
            FinetuneResult with raw and cleaned text.
        """
        self._load()

        from mlx_whisper.audio import (
            N_FRAMES,
            N_SAMPLES,
            log_mel_spectrogram,
            pad_or_trim,
        )

        # Prepare mel spectrogram
        mel = log_mel_spectrogram(
            str(audio_path), n_mels=self._model.dims.n_mels, padding=N_SAMPLES
        )
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float32)
        mel = mx.expand_dims(mel, axis=0)  # (1, 3000, 80)

        # Encode audio
        audio_features = self._model.encoder(mel)

        # Greedy decode: SOT → NO_TIMESTAMPS → argmax until EOT or max tokens
        tokens = [SOT, NO_TIMESTAMPS]

        for _ in range(MAX_DECODE_TOKENS):
            token_input = mx.array([tokens])
            logits = self._model.decoder(token_input, audio_features)[0]
            # logits shape: (1, seq_len, vocab) — take last position
            next_token = mx.argmax(logits[0, -1, :]).item()

            if next_token == EOT:
                break
            tokens.append(next_token)

        mx.eval(mx.array(0))  # flush any pending evals

        # Decode tokens to text (skip SOT and NO_TIMESTAMPS)
        text_tokens = tokens[2:]
        raw_text = self._tokenizer.decode(text_tokens)
        text = self._truncate_repetitions(raw_text)

        return FinetuneResult(
            raw_text=raw_text,
            text=text,
            audio_path=str(audio_path),
        )

    @staticmethod
    def _truncate_repetitions(text: str) -> str:
        """Stop at the first repeated sentence/phrase.

        The fine-tuned model sometimes doesn't learn EOT well and repeats
        output. This detects the first repetition and truncates there.
        """
        text = text.strip()
        if not text:
            return text

        # Split into sentences on . , ! ? or whitespace runs of 3+
        parts = re.split(r'[.!?]+|\s{3,}', text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            return text

        # Keep sentences until we see a repeat
        seen = set()
        kept = []
        for part in parts:
            normalized = part.lower().strip()
            if normalized in seen:
                break
            seen.add(normalized)
            kept.append(part)

        return " ".join(kept)
