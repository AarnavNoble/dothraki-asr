"""Sequence-level matching using Dynamic Time Warping on MFCC features.

Instead of word-by-word phoneme matching, this compares the full audio
sequence against all known Dothraki clips using DTW on MFCC feature
sequences. Think of it like Shazam for Dothraki.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from pipeline.config import FEATURES_DIR, SAMPLE_RATE, SYNTHETIC_DIR


N_MFCC = 13  # MFCC coefficients
HOP_LENGTH = 512  # ~32ms at 16kHz

# Duration tolerance for pre-filtering (skip DTW on clips too different in length)
DURATION_TOLERANCE = 2.0  # allow ±2x duration ratio


@dataclass
class DTWMatch:
    clip_id: str
    dothraki: str
    english: str
    score: float        # normalized similarity (1 - cost/max_cost)
    dtw_cost: float     # raw DTW alignment cost
    audio_file: str


class SequenceMatcher:
    """Matches audio against known Dothraki clips using DTW on MFCCs."""

    def __init__(self):
        self._features: list[np.ndarray] | None = None
        self._metadata: list[dict] = []
        self._durations: np.ndarray | None = None

    @staticmethod
    def _extract_mfcc(audio_path: str | Path) -> np.ndarray:
        """Extract MFCC features from an audio file.

        Returns array of shape (n_mfcc, n_frames).
        """
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
        )
        # Delta features for better temporal representation
        delta = librosa.feature.delta(mfcc)
        return np.vstack([mfcc, delta])  # (2*n_mfcc, n_frames)

    def build_features(self, manifest_path: str | Path | None = None) -> Path:
        """Pre-compute MFCC features for all synthetic clips.

        Returns:
            Path to the saved features file.
        """
        if manifest_path is None:
            manifest_path = SYNTHETIC_DIR / "manifest.json"
        manifest = json.loads(Path(manifest_path).read_text())

        features = []
        metadata = []
        durations = []

        for i, entry in enumerate(manifest):
            audio_path = SYNTHETIC_DIR / entry["audio_file"]
            if not audio_path.exists():
                continue

            mfcc = self._extract_mfcc(audio_path)
            y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE)
            duration = len(y) / SAMPLE_RATE

            features.append(mfcc)
            durations.append(duration)
            metadata.append({
                "clip_id": entry["id"],
                "dothraki": entry["dothraki"],
                "english": entry["english"],
                "audio_file": entry["audio_file"],
                "ipa": entry.get("ipa_clean", entry.get("ipa", "")),
                "scene": entry.get("scene", ""),
            })

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(manifest)}] MFCC features computed")

        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        features_path = FEATURES_DIR / "mfcc_features.npz"

        # Store as object array since features have variable-length time axes
        np.savez(
            str(features_path),
            n_features=len(features),
            durations=np.array(durations),
            metadata=json.dumps(metadata),
            **{f"feat_{i}": f for i, f in enumerate(features)},
        )

        print(f"Saved {len(features)} MFCC feature sets to {features_path}")
        return features_path

    def load_features(self, features_path: str | Path | None = None):
        """Load pre-computed MFCC features."""
        if features_path is None:
            features_path = FEATURES_DIR / "mfcc_features.npz"
        data = np.load(str(features_path), allow_pickle=False)

        n = int(data["n_features"])
        self._features = [data[f"feat_{i}"] for i in range(n)]
        self._durations = data["durations"]
        self._metadata = json.loads(str(data["metadata"]))

    def match(
        self, audio_path: str | Path, top_k: int = 5
    ) -> list[DTWMatch]:
        """Match an audio file against all stored clips using DTW.

        Pre-filters by duration similarity to avoid expensive DTW on
        clips that are way too long or short.

        Args:
            audio_path: Path to a 16kHz mono WAV.
            top_k: Number of top matches to return.

        Returns:
            List of DTWMatch sorted by similarity (highest first).
        """
        if self._features is None:
            self.load_features()

        query_mfcc = self._extract_mfcc(audio_path)
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE)
        query_duration = len(y) / SAMPLE_RATE

        # Pre-filter by duration
        duration_ratios = self._durations / max(query_duration, 0.1)
        candidates = np.where(
            (duration_ratios > 1.0 / DURATION_TOLERANCE) &
            (duration_ratios < DURATION_TOLERANCE)
        )[0]

        if len(candidates) == 0:
            # No duration-similar clips — fall back to all clips
            candidates = np.arange(len(self._features))

        # DTW against each candidate
        costs = []
        for idx in candidates:
            ref_mfcc = self._features[idx]
            D, wp = librosa.sequence.dtw(
                X=query_mfcc, Y=ref_mfcc, metric="cosine"
            )
            # Normalized cost: total path cost / path length
            path_cost = D[-1, -1]
            path_len = max(query_mfcc.shape[1], ref_mfcc.shape[1])
            norm_cost = path_cost / max(path_len, 1)
            costs.append((idx, norm_cost, path_cost))

        # Sort by normalized cost (lower = better match)
        costs.sort(key=lambda x: x[1])

        # Convert to similarity scores (invert cost)
        max_cost = max(c[1] for c in costs) if costs else 1.0
        results = []
        for idx, norm_cost, raw_cost in costs[:top_k]:
            meta = self._metadata[idx]
            score = max(0.0, 1.0 - norm_cost / max(max_cost, 1e-8))
            results.append(DTWMatch(
                clip_id=meta["clip_id"],
                dothraki=meta["dothraki"],
                english=meta["english"],
                score=round(score, 4),
                dtw_cost=round(float(raw_cost), 4),
                audio_file=meta["audio_file"],
            ))

        return results
