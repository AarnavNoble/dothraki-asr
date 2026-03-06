"""
Synthesize Dothraki audio using XTTS v2 voice cloning.

Clones voice from a clean Khal Drogo speech sample and synthesizes
all dialogue entries with natural, actor-like delivery. Uses raw
Dothraki text with Spanish as the language hint (phonetically close
open vowel patterns).

Replaces the legacy espeak-ng approach (synthesize_audio.py) which
produced robotic formant-based output.

Prerequisites:
    pip install coqui-tts

Usage:
    python scripts/synthesize_xtts.py                    # synthesize all
    python scripts/synthesize_xtts.py --limit 50         # first 50 entries
    python scripts/synthesize_xtts.py --ids d0000 d0004  # specific entries
    python scripts/synthesize_xtts.py --device mps       # use Apple Silicon GPU

Output: data/synthetic/ (one WAV per entry + manifest.json)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from TTS.api import TTS

PROJECT_ROOT = Path(__file__).parent.parent
DIALOGUE_PATH = PROJECT_ROOT / "data" / "dialogue" / "dothraki_dialogue.json"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
MANIFEST_PATH = SYNTHETIC_DIR / "manifest.json"
REFERENCE_WAV = PROJECT_ROOT / "data" / "raw" / "drogo_speech_clean.wav"

TARGET_SAMPLE_RATE = 16000


def select_device(requested: str) -> str:
    """Pick the best available device."""
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested in ("cuda", "mps"):
        print(f"Warning: {requested} not available, falling back to cpu")
    return "cpu"


def resample_and_save(wav_array: np.ndarray, orig_sr: int, output_path: Path) -> None:
    """Resample a numpy audio array to 16kHz mono and save as WAV."""
    tensor = torch.from_numpy(wav_array).unsqueeze(0).float()

    if orig_sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_sr, TARGET_SAMPLE_RATE)
        tensor = resampler(tensor)

    # Normalize to [-1, 1]
    peak = tensor.abs().max()
    if peak > 0:
        tensor = tensor / peak

    torchaudio.save(str(output_path), tensor, TARGET_SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize Dothraki audio with XTTS v2 voice cloning"
    )
    parser.add_argument("--limit", type=int, help="Max entries to synthesize")
    parser.add_argument("--ids", nargs="*", help="Specific entry IDs to synthesize")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Compute device (default: cpu)"
    )
    parser.add_argument(
        "--language", default="es",
        help="Language hint for XTTS (default: es — phonetically close to Dothraki)"
    )
    parser.add_argument(
        "--reference", type=Path, default=REFERENCE_WAV,
        help="Reference WAV for voice cloning"
    )
    args = parser.parse_args()

    # Validate reference audio
    if not args.reference.exists():
        print(f"Error: reference audio not found: {args.reference}")
        sys.exit(1)

    # Load dialogue entries
    entries = json.loads(DIALOGUE_PATH.read_text())
    print(f"Loaded {len(entries)} dialogue entries")

    # Filter
    if args.ids:
        entries = [e for e in entries if e["id"] in args.ids]
        print(f"Filtered to {len(entries)} entries by ID")
    elif args.limit:
        entries = entries[: args.limit]
        print(f"Limited to first {args.limit} entries")

    if not entries:
        print("No entries to synthesize.")
        sys.exit(0)

    # Create output directory
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    # Load XTTS v2 model
    device = select_device(args.device)
    print(f"Loading XTTS v2 model on {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    model = tts.synthesizer.tts_model
    print("Model loaded.")

    # Precompute speaker conditioning latents ONCE (this is the expensive part)
    print(f"Computing speaker latents from {args.reference.name}...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=str(args.reference),
    )
    print("Speaker latents cached — reusing for all clips.")

    # Synthesize
    manifest = []
    success = 0
    failed = 0
    start_time = time.time()

    for i, entry in enumerate(entries):
        entry_id = entry["id"]
        text = entry["dothraki"]
        output_path = SYNTHETIC_DIR / f"{entry_id}.wav"

        try:
            result = model.inference(
                text=text,
                language=args.language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
            )
            wav_out = result["wav"]
            if isinstance(wav_out, torch.Tensor):
                wav_array = wav_out.cpu().numpy().squeeze()
            else:
                wav_array = np.array(wav_out, dtype=np.float32).squeeze()

            # XTTS v2 outputs at 24kHz
            resample_and_save(wav_array, 24000, output_path)

            manifest.append({
                "id": entry_id,
                "audio_file": output_path.name,
                "dothraki": text,
                "ipa": entry["ipa"],
                "english": entry["english"],
                "scene": entry["scene"],
                "source": entry["source"],
                "type": "synthetic_xtts",
            })
            success += 1

        except Exception as e:
            print(f"  Failed {entry_id}: {e}")
            failed += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta_s = (len(entries) - i - 1) / rate if rate > 0 else 0
            eta_m = eta_s / 60
            print(
                f"  [{i + 1}/{len(entries)}] "
                f"{success} ok, {failed} failed "
                f"({rate:.1f} clips/s, ETA {eta_m:.0f}m)"
            )

    # Save manifest
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s: {success}/{len(entries)} entries synthesized")
    print(f"Audio files: {SYNTHETIC_DIR}/")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
