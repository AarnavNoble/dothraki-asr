"""
Download Dothraki audio clips from YouTube URLs.

Usage:
    python scripts/download_clips.py <url> [<url> ...]
    python scripts/download_clips.py --file clips.txt

The clips.txt file should have one URL per line. Lines starting with #
are treated as comments, and you can add a label after the URL:

    # Khal Drogo's iron throne speech
    https://youtube.com/watch?v=xxx  drogo_iron_throne
    https://youtube.com/watch?v=yyy  drogo_wedding

Downloads are saved as WAV (16kHz mono) to data/raw/ — ready for the pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_clip(url: str, label: str | None = None) -> Path | None:
    """Download a single clip from a URL, convert to 16kHz mono WAV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Use label as filename, or let yt-dlp generate one
    if label:
        output_template = str(RAW_DIR / f"{label}.%(ext)s")
    else:
        output_template = str(RAW_DIR / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
        "--output", output_template,
        "--no-playlist",
        url,
    ]

    print(f"\nDownloading: {url}")
    if label:
        print(f"  Label: {label}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  Error: {result.stderr.strip()}")
            return None

        # Find the downloaded file
        if label:
            wav_path = RAW_DIR / f"{label}.wav"
        else:
            # yt-dlp prints the output filename
            for line in result.stdout.splitlines():
                if "Destination:" in line and ".wav" in line:
                    wav_path = Path(line.split("Destination:")[-1].strip())
                    break
            else:
                # Fallback: find most recent wav
                wavs = sorted(RAW_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
                wav_path = wavs[-1] if wavs else None

        if wav_path and wav_path.exists():
            size_mb = wav_path.stat().st_size / (1024 * 1024)
            print(f"  Saved: {wav_path.name} ({size_mb:.1f} MB)")
            return wav_path
        else:
            print("  Warning: download succeeded but WAV file not found")
            return None

    except subprocess.TimeoutExpired:
        print("  Error: download timed out (120s)")
        return None
    except FileNotFoundError:
        print("  Error: yt-dlp not found. Install with: brew install yt-dlp")
        sys.exit(1)


def parse_clips_file(path: str) -> list[tuple[str, str | None]]:
    """Parse a clips file with URLs and optional labels."""
    clips = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            url = parts[0]
            label = parts[1] if len(parts) > 1 else None
            clips.append((url, label))
    return clips


def main():
    parser = argparse.ArgumentParser(description="Download Dothraki audio clips")
    parser.add_argument("urls", nargs="*", help="YouTube URLs to download")
    parser.add_argument("--file", "-f", help="Text file with URLs (one per line)")
    args = parser.parse_args()

    clips = []

    if args.file:
        clips = parse_clips_file(args.file)
    elif args.urls:
        clips = [(url, None) for url in args.urls]
    else:
        parser.print_help()
        print("\nExample:")
        print("  python scripts/download_clips.py https://youtube.com/watch?v=xxx")
        print("  python scripts/download_clips.py --file scripts/clips.txt")
        sys.exit(1)

    print(f"Downloading {len(clips)} clip(s) to {RAW_DIR}/")

    downloaded = []
    for url, label in clips:
        path = download_clip(url, label)
        if path:
            downloaded.append(path)

    print(f"\nDone: {len(downloaded)}/{len(clips)} clips downloaded")
    for p in downloaded:
        print(f"  {p}")


if __name__ == "__main__":
    main()
