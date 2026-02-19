"""Vocal isolation using Demucs for the Dothraki ASR pipeline."""

from pathlib import Path
import subprocess
import tempfile

import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

from pipeline.config import PROCESSED_AUDIO_DIR, SAMPLE_RATE

_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


class VocalSeparator:
    """Isolates vocals from mixed audio/video using Demucs htdemucs."""

    def __init__(self, model_name: str = "htdemucs"):
        from demucs.pretrained import get_model

        self.model_name = model_name
        self.model = get_model(model_name)
        self.model.eval()

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self.model.to(self._device)

    def separate(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Isolate vocals from an audio or video file.

        Args:
            input_path: Path to audio (wav/mp3/flac) or video (mp4/mkv/mov).
            output_path: Destination for the vocals WAV. Defaults to
                         PROCESSED_AUDIO_DIR/<stem>_vocals.wav.

        Returns:
            Path to the saved 16kHz mono vocals WAV.
        """
        input_path = Path(input_path)

        if output_path is None:
            PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            output_path = PROCESSED_AUDIO_DIR / f"{input_path.stem}_vocals.wav"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_path = self._ensure_audio(input_path)
        is_temp = audio_path != input_path

        try:
            waveform, sr = self._load_audio(audio_path)
            vocals = self._run_demucs(waveform, sr)
            vocals_16k = self._resample(vocals, self.model.samplerate, SAMPLE_RATE)
            self._save(vocals_16k, output_path)
        finally:
            if is_temp:
                audio_path.unlink(missing_ok=True)

        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_audio(self, path: Path) -> Path:
        """Return an audio-only path, extracting from video via ffmpeg if needed."""
        if path.suffix.lower() not in _VIDEO_EXTS:
            return path

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        cmd = [
            "ffmpeg", "-y",
            "-i", str(path),
            "-vn",                  # drop video stream
            "-ar", "44100",
            "-ac", "2",
            "-f", "wav",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed extracting audio from {path}:\n"
                + result.stderr.decode()
            )

        return Path(tmp.name)

    def _load_audio(self, path: Path) -> tuple[torch.Tensor, int]:
        """Load audio file, return (waveform [C, T], sample_rate)."""
        waveform, sr = torchaudio.load(str(path))
        return waveform, sr

    def _run_demucs(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Run Demucs and return the vocals stem as a [C, T] tensor."""
        from demucs.apply import apply_model

        # Resample to model's native rate if the input differs
        if sr != self.model.samplerate:
            waveform = self._resample(waveform, sr, self.model.samplerate)

        # Demucs expects stereo; upmix mono if needed
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        mix = waveform.to(self._device).unsqueeze(0)  # [1, 2, T]

        with torch.no_grad():
            sources = apply_model(self.model, mix, progress=True)
        # sources: [batch, num_sources, channels, time]

        vocal_idx = self.model.sources.index("vocals")
        return sources[0, vocal_idx]  # [2, T]

    def _resample(
        self, waveform: torch.Tensor, orig_sr: int, target_sr: int
    ) -> torch.Tensor:
        if orig_sr == target_sr:
            return waveform
        return T.Resample(orig_sr, target_sr)(waveform.cpu())

    def _save(self, waveform: torch.Tensor, path: Path) -> None:
        """Save waveform as 16kHz mono WAV."""
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        sf.write(str(path), waveform.squeeze().numpy(), SAMPLE_RATE)
