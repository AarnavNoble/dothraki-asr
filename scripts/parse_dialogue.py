"""
Parse the Master Dialogue PDFs into structured JSON.

Extracts every Dothraki dialogue entry with:
- dothraki: the Dothraki text (with stress marks)
- ipa: IPA phonetic transcription
- gloss: interlinear gloss (word-by-word breakdown)
- english: English translation
- scene: scene description from section header
- source: which PDF / season

Sources:
- data/dialogue/master_dialogue_s1s2.pdf (Seasons 1-2, 197 pages)
- data/dialogue/master_dialogue_s3s8.pdf (Seasons 3-8, 393 pages)

Output: data/dialogue/dothraki_dialogue.json
"""

import json
import re
import sys
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError:
    print("pymupdf not installed. Run: uv pip install pymupdf")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
DIALOGUE_DIR = PROJECT_ROOT / "data" / "dialogue"
OUTPUT_PATH = DIALOGUE_DIR / "dothraki_dialogue.json"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF, skipping headers/footers."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text("text")
        # Remove the running header
        text = re.sub(
            r"Master Dialogue Document for Dothraki—David J\. Peterson \d+\n?",
            "",
            text,
        )
        pages_text.append(text)
    doc.close()
    return "\n".join(pages_text)


def parse_dialogue_entries(full_text: str, source_label: str) -> list[dict]:
    """
    Parse dialogue entries from extracted PDF text.

    The format is:
        Section Header (blue bar): scene description
        Dothraki text (bold/italic, may span multiple lines)
            [IPA transcription in brackets, may span multiple lines]
            /interlinear gloss in slashes, may span multiple lines/
        "English translation in quotes"
        Notes: optional commentary
    """
    entries = []
    current_scene = ""

    # Split into lines for processing
    lines = full_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Detect scene headers — they contain episode/character info
        # Patterns like: "From the 7/12 Script: ILLYRIO's Conversation..."
        # or "Episode 101:" or "Scene:" etc.
        if _is_scene_header(line):
            current_scene = line
            i += 1
            continue

        # Skip known non-dialogue lines
        if _should_skip(line):
            i += 1
            continue

        # Try to detect a Dothraki dialogue entry.
        # Dothraki lines contain accented characters and are followed by [IPA]
        # Look ahead to see if brackets follow within the next few lines
        if _looks_like_dothraki(line, lines, i):
            entry, new_i = _parse_entry(lines, i, current_scene, source_label)
            if entry:
                entries.append(entry)
            i = new_i
        else:
            i += 1

    return entries


def _is_scene_header(line: str) -> bool:
    """Check if a line is a scene/section header."""
    patterns = [
        r"^From the .+ Script:",
        r"^Episode \d+",
        r"^Requested Translation:",
        r"^Season \d+",
        r"^GoT\s+SEASON",
        r"^Postproduction",
    ]
    return any(re.match(p, line) for p in patterns)


def _should_skip(line: str) -> bool:
    """Check if a line should be skipped."""
    if not line:
        return True
    if line.startswith("Notes:"):
        return True
    if line.startswith("Legend"):
        return True
    if re.match(r"^\d+$", line):  # page numbers
        return True
    if line.startswith("•"):  # bullet points in version history
        return True
    return False


def _looks_like_dothraki(line: str, lines: list[str], idx: int) -> bool:
    """
    Heuristic: a Dothraki line has accented Latin chars and is followed
    by an IPA line in brackets within the next 4 lines.
    """
    if not line or len(line) < 3:
        return False

    # Must contain at least one accented character typical of Dothraki stress marking
    has_accent = bool(re.search(r"[áéíóúÁÉÍÓÚ]", line))
    if not has_accent:
        return False

    # Look ahead for IPA brackets
    for lookahead in range(1, 6):
        if idx + lookahead >= len(lines):
            break
        next_line = lines[idx + lookahead].strip()
        if next_line.startswith("[") or next_line.startswith("["):
            return True

    return False


def _parse_entry(
    lines: list[str], start: int, scene: str, source: str
) -> tuple[dict | None, int]:
    """
    Parse a complete dialogue entry starting at the given line index.
    Returns (entry_dict, next_line_index).
    """
    i = start

    # Collect Dothraki text (may span multiple lines until we hit '[')
    dothraki_lines = []
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("["):
            break
        dothraki_lines.append(line)
        i += 1

    if i >= len(lines):
        return None, i

    dothraki_text = " ".join(dothraki_lines).strip()

    # Collect IPA (starts with '[', may span multiple lines until ']')
    ipa_lines = []
    while i < len(lines):
        line = lines[i].strip()
        ipa_lines.append(line)
        i += 1
        if "]" in line:
            break

    ipa_text = " ".join(ipa_lines).strip()
    # Clean IPA — extract content between brackets
    ipa_match = re.search(r"\[(.+?)\]", ipa_text, re.DOTALL)
    ipa_clean = ipa_match.group(1).strip() if ipa_match else ipa_text

    # Collect interlinear gloss (starts with '/', may span until '/')
    gloss_lines = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("/") or (gloss_lines and not line.startswith('"')):
            gloss_lines.append(line)
            i += 1
            if line.endswith("/"):
                break
        else:
            break

    gloss_text = " ".join(gloss_lines).strip()
    # Clean gloss — extract content between slashes
    gloss_match = re.search(r"/(.+)/", gloss_text, re.DOTALL)
    gloss_clean = gloss_match.group(1).strip() if gloss_match else gloss_text

    # Collect English translation (in double quotes)
    english_lines = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('"') or english_lines:
            english_lines.append(line)
            i += 1
            # Check if we've closed the quotes
            combined = " ".join(english_lines)
            # Count quotes — when we have at least 2, the translation is complete
            if combined.count('"') >= 2:
                break
        else:
            break

    english_text = " ".join(english_lines).strip()
    # Clean English — extract content between quotes
    english_match = re.search(r'"(.+?)"', english_text, re.DOTALL)
    english_clean = english_match.group(1).strip() if english_match else english_text

    # Skip entries that are too short or clearly not dialogue
    if len(dothraki_text) < 3 or not ipa_clean:
        return None, i

    entry = {
        "dothraki": dothraki_text,
        "ipa": ipa_clean,
        "gloss": gloss_clean,
        "english": english_clean,
        "scene": scene,
        "source": source,
    }

    # Skip past any Notes section
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Notes:"):
            # Skip until next empty line or next entry
            i += 1
            while i < len(lines) and lines[i].strip():
                # Stop if we hit what looks like a new Dothraki entry or scene header
                if _is_scene_header(lines[i].strip()):
                    break
                if _looks_like_dothraki(lines[i].strip(), lines, i):
                    break
                i += 1
            break
        elif not line:
            i += 1
        else:
            break

    return entry, i


def build_dialogue_dataset():
    """Parse both dialogue PDFs and output structured JSON."""
    all_entries = []

    for pdf_name, label in [
        ("master_dialogue_s1s2.pdf", "seasons_1_2"),
        ("master_dialogue_s3s8.pdf", "seasons_3_8"),
    ]:
        pdf_path = DIALOGUE_DIR / pdf_name
        if not pdf_path.exists():
            print(f"Skipping {pdf_name} (not found)")
            continue

        print(f"Parsing {pdf_name}...")
        text = extract_text_from_pdf(pdf_path)
        entries = parse_dialogue_entries(text, label)
        print(f"  Found {len(entries)} dialogue entries")
        all_entries.extend(entries)

    # Add sequential IDs
    for idx, entry in enumerate(all_entries):
        entry["id"] = f"d{idx:04d}"

    # Save
    OUTPUT_PATH.write_text(
        json.dumps(all_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nTotal entries: {len(all_entries)}")
    print(f"Saved to {OUTPUT_PATH}")

    # Print a few samples
    print("\nSample entries:")
    for e in all_entries[:3]:
        print(f"  [{e['id']}] {e['dothraki'][:60]}...")
        print(f"    IPA: {e['ipa'][:60]}...")
        print(f"    EN:  {e['english'][:60]}...")
        print()


if __name__ == "__main__":
    build_dialogue_dataset()
