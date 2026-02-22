"""
Synthesize Dothraki audio from IPA transcriptions using espeak-ng.

Takes the parsed dialogue entries and generates WAV audio files
for each entry using espeak-ng's IPA input mode.

This gives us hundreds of audio samples with perfect ground truth
alignment for testing the ASR pipeline.

Usage:
    python scripts/synthesize_audio.py                    # synthesize all
    python scripts/synthesize_audio.py --limit 50         # first 50 entries
    python scripts/synthesize_audio.py --ids d0004 d0005  # specific entries

Output: data/synthetic/ (one WAV per entry + manifest.json)
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DIALOGUE_PATH = PROJECT_ROOT / "data" / "dialogue" / "dothraki_dialogue.json"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
MANIFEST_PATH = SYNTHETIC_DIR / "manifest.json"

# The dialogue PDFs use X-SAMPA-like notation for IPA.
# espeak-ng accepts IPA in double brackets: [[ipa here]]
# We need to convert X-SAMPA -> Unicode IPA first.
XSAMPA_TO_IPA = {
    "S": "ʃ",
    "Z": "ʒ",
    "T": "θ",
    "D": "ð",
    "R": "ɾ",
    "N": "ŋ",
    "G": "ɣ",
    "O": "ɔ",
    "E": "ɛ",
    "?": "ʔ",
}

# Characters to strip from the IPA before synthesis
# » is the stress marker used in the PDFs (maps to IPA ˈ)
# . is syllable boundary
STRIP_CHARS = {
    "»": "ˈ",  # convert to IPA primary stress
    ".": "",    # remove syllable dots (espeak handles syllabification)
}


def clean_ipa_for_espeak(raw_ipa: str) -> str:
    """Convert X-SAMPA IPA from the dialogue PDFs to Unicode IPA for espeak-ng."""
    result = []
    for char in raw_ipa:
        if char in STRIP_CHARS:
            replacement = STRIP_CHARS[char]
            if replacement:
                result.append(replacement)
        elif char in XSAMPA_TO_IPA:
            result.append(XSAMPA_TO_IPA[char])
        else:
            result.append(char)

    cleaned = "".join(result)
    # Remove any double spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def synthesize_entry(entry: dict, output_dir: Path, rate: int = 130) -> Path | None:
    """
    Synthesize a single dialogue entry to WAV using espeak-ng.

    Args:
        entry: dialogue entry dict with 'id' and 'ipa' fields
        output_dir: directory to write WAV files
        rate: speech rate (words per minute), lower = slower/clearer

    Returns:
        Path to the generated WAV file, or None on failure
    """
    entry_id = entry["id"]
    raw_ipa = entry["ipa"]

    # Convert to clean IPA
    ipa = clean_ipa_for_espeak(raw_ipa)
    if not ipa:
        return None

    output_path = output_dir / f"{entry_id}.wav"

    # espeak-ng IPA mode: double brackets [[...]]
    # Using a neutral voice — espeak doesn't have a Dothraki voice,
    # but the IPA input bypasses language-specific processing
    cmd = [
        "espeak-ng",
        "-v", "en",              # base voice (phonemes override it)
        "-s", str(rate),         # speech rate
        "-w", str(output_path),  # output WAV
        f"[[{ipa}]]",           # IPA input in double brackets
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            # espeak-ng may warn but still produce output
            if not output_path.exists():
                return None

        # Resample to 16kHz mono (pipeline expects this)
        resampled_path = output_dir / f"{entry_id}_16k.wav"
        resample_cmd = [
            "ffmpeg", "-y", "-i", str(output_path),
            "-ar", "16000", "-ac", "1",
            str(resampled_path),
        ]
        subprocess.run(resample_cmd, capture_output=True, timeout=10)

        if resampled_path.exists():
            # Replace original with resampled version
            resampled_path.rename(output_path)
            return output_path
        else:
            return output_path  # keep original if resample fails

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Error synthesizing {entry_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Synthesize Dothraki audio from IPA")
    parser.add_argument("--limit", type=int, help="Max entries to synthesize")
    parser.add_argument("--ids", nargs="*", help="Specific entry IDs to synthesize")
    parser.add_argument("--rate", type=int, default=130, help="Speech rate (default: 130)")
    args = parser.parse_args()

    # Load dialogue entries
    entries = json.loads(DIALOGUE_PATH.read_text())
    print(f"Loaded {len(entries)} dialogue entries")

    # Filter
    if args.ids:
        entries = [e for e in entries if e["id"] in args.ids]
        print(f"Filtered to {len(entries)} entries by ID")
    elif args.limit:
        entries = entries[:args.limit]
        print(f"Limited to first {args.limit} entries")

    # Create output directory
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    # Check ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except FileNotFoundError:
        print("Warning: ffmpeg not found, audio won't be resampled to 16kHz")

    # Synthesize
    manifest = []
    success = 0
    for i, entry in enumerate(entries):
        wav_path = synthesize_entry(entry, SYNTHETIC_DIR, rate=args.rate)
        if wav_path:
            manifest.append({
                "id": entry["id"],
                "audio_file": wav_path.name,
                "dothraki": entry["dothraki"],
                "ipa": entry["ipa"],
                "ipa_clean": clean_ipa_for_espeak(entry["ipa"]),
                "english": entry["english"],
                "scene": entry["scene"],
                "source": entry["source"],
                "type": "synthetic",
            })
            success += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(entries)} ({success} successful)")

    # Save manifest
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone: {success}/{len(entries)} entries synthesized")
    print(f"Audio files: {SYNTHETIC_DIR}/")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
