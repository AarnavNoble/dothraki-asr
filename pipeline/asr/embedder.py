"""Whisper encoder embedding matcher for audio fingerprinting.

Extracts fixed-size embeddings from Whisper's encoder (bypassing the decoder)
and matches audio by cosine similarity against a pre-computed index of
known Dothraki clips.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from pipeline.config import FEATURES_DIR, SYNTHETIC_DIR


@dataclass
class EmbeddingMatch:
    clip_id: str
    dothraki: str
    english: str
    score: float        # cosine similarity 0→1
    audio_file: str


class EmbeddingMatcher:
    """Matches audio against a pre-built embedding index using cosine similarity."""

    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self._index: np.ndarray | None = None
        self._metadata: list[dict] = []
        self._model = None

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            from mlx_whisper.load_models import load_model

            repo_map = {
                "tiny": "mlx-community/whisper-tiny-mlx",
                "base": "mlx-community/whisper-base-mlx",
                "small": "mlx-community/whisper-small-mlx",
            }
            self._model = load_model(repo_map[self.model_name])
        return self._model

    def _audio_to_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Extract a fixed-size embedding from an audio file.

        Returns a 1-D numpy array (n_audio_state,) — e.g. (384,) for tiny.
        """
        from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES, N_SAMPLES

        model = self._load_model()
        mel = log_mel_spectrogram(
            str(audio_path), n_mels=model.dims.n_mels, padding=N_SAMPLES
        )
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float32)
        mel = mx.expand_dims(mel, axis=0)  # (1, 3000, 80)

        emb = model.embed_audio(mel)       # (1, 1500, n_audio_state)
        pooled = mx.mean(emb[0], axis=0)   # (n_audio_state,)
        mx.eval(pooled)
        return np.array(pooled)

    def build_index(self, manifest_path: str | Path | None = None) -> Path:
        """Pre-compute embeddings for all synthetic clips.

        Args:
            manifest_path: Path to manifest.json. Defaults to SYNTHETIC_DIR/manifest.json.

        Returns:
            Path to the saved .npz index file.
        """
        if manifest_path is None:
            manifest_path = SYNTHETIC_DIR / "manifest.json"
        manifest = json.loads(Path(manifest_path).read_text())

        embeddings = []
        metadata = []

        for i, entry in enumerate(manifest):
            audio_path = SYNTHETIC_DIR / entry["audio_file"]
            if not audio_path.exists():
                continue

            emb = self._audio_to_embedding(audio_path)
            embeddings.append(emb)
            metadata.append({
                "clip_id": entry["id"],
                "dothraki": entry["dothraki"],
                "english": entry["english"],
                "audio_file": entry["audio_file"],
                "ipa": entry.get("ipa_clean", entry.get("ipa", "")),
                "scene": entry.get("scene", ""),
            })

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(manifest)}] embeddings computed")

        embeddings_arr = np.stack(embeddings)  # (N, n_audio_state)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings_arr = embeddings_arr / norms

        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        index_path = FEATURES_DIR / f"embedding_index_{self.model_name}.npz"
        np.savez(
            str(index_path),
            embeddings=embeddings_arr,
            metadata=json.dumps(metadata),
        )

        print(f"Saved {len(embeddings)} embeddings to {index_path}")
        print(f"Shape: {embeddings_arr.shape}, dtype: {embeddings_arr.dtype}")
        return index_path

    def load_index(self, index_path: str | Path | None = None):
        """Load a pre-built embedding index."""
        if index_path is None:
            index_path = FEATURES_DIR / f"embedding_index_{self.model_name}.npz"
        data = np.load(str(index_path), allow_pickle=False)
        self._index = data["embeddings"]
        self._metadata = json.loads(str(data["metadata"]))

    def match(
        self, audio_path: str | Path, top_k: int = 5
    ) -> list[EmbeddingMatch]:
        """Match an audio file against the embedding index.

        Args:
            audio_path: Path to a 16kHz mono WAV.
            top_k: Number of top matches to return.

        Returns:
            List of EmbeddingMatch sorted by cosine similarity (highest first).
        """
        if self._index is None:
            self.load_index()

        query = self._audio_to_embedding(audio_path)
        query = query / max(np.linalg.norm(query), 1e-8)

        # Cosine similarity (index is already normalized)
        similarities = self._index @ query  # (N,)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            meta = self._metadata[idx]
            results.append(EmbeddingMatch(
                clip_id=meta["clip_id"],
                dothraki=meta["dothraki"],
                english=meta["english"],
                score=float(similarities[idx]),
                audio_file=meta["audio_file"],
            ))
        return results
