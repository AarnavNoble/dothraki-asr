"""
Build the Dothraki lexicon JSON from the official dictionary PDF.

Source: https://docs.dothraki.org/Dothraki.pdf (ver 3.11, 25 pages)
Uses pymupdf (fitz) for reliable text extraction including proper
column separation and Unicode handling.

The PDF uses X-SAMPA-like notation for IPA in its font encoding.
This script converts those to proper Unicode IPA symbols.

Output: data/lexicon/dothraki_lexicon.json
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

try:
    import requests
except ImportError:
    requests = None

PROJECT_ROOT = Path(__file__).parent.parent
LEXICON_DIR = PROJECT_ROOT / "data" / "lexicon"
PDF_PATH = LEXICON_DIR / "Dothraki.pdf"
OUTPUT_PATH = LEXICON_DIR / "dothraki_lexicon.json"

# X-SAMPA / ASCII-IPA to Unicode IPA mapping
# The PDF font renders IPA using ASCII substitutions
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
    ">": "",     # artifact from PDF extraction
    "\x11": "",  # control character artifact
    "\x00": "",
}

POS_MAP = {
    "na": "noun.animate",
    "ni": "noun.inanimate",
    "n": "noun",
    "v": "verb",
    "vtr": "verb.transitive",
    "vin": "verb.intransitive",
    "v.aux": "verb.auxiliary",
    "adj": "adjective",
    "adv": "adverb",
    "conj": "conjunction",
    "intj": "interjection",
    "pn": "pronoun",
    "prep": "preposition",
    "num": "numeral",
    "aux": "auxiliary",
    "phrase": "phrase",
}

# Regex for a dictionary entry start. Matches lines like:
#   acchakat: [attSakat] DP vtr. to silence
#   affa (1): [affa] DP intj. whoa! (horse command)
#   arakh: [arax] M,DP ni. curved sword
ENTRY_START = re.compile(
    r"^(?P<word>[a-zA-Z][a-zA-Z' ]*?(?:\s*\(\d+\))?)\s*:"  # word:
    r"\s*\[(?P<ipa>[^\]]+)\]"                                 # [ipa]
    r"\s+(?P<rest>.+)$"                                        # rest of line
)

# Parse the "rest" after the IPA bracket for POS and definition
REST_PATTERN = re.compile(
    r"^(?:[A-Z][A-Z,]*\s+)?"  # source tags (DP, M, M,DP) — optional
    r"(?P<pos>na|ni|n|ni-B|na-B|n-B|vtr|vin|v\.aux|v|adj|adv|conj|intj|pn|prep|num|aux|phrase|gen)"
    r"[.\s]+"
    r"(?P<english>.+)$"
)


def download_pdf():
    """Download the dictionary PDF if not already present."""
    if PDF_PATH.exists():
        print(f"PDF already exists at {PDF_PATH}")
        return

    if requests is None:
        print(
            f"PDF not found at {PDF_PATH}.\n"
            "Either place it there manually or install requests:\n"
            "  uv pip install requests"
        )
        sys.exit(1)

    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://docs.dothraki.org/Dothraki.pdf"
    print(f"Downloading dictionary from {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    PDF_PATH.write_bytes(resp.content)
    print(f"Saved to {PDF_PATH} ({len(resp.content)} bytes)")


def xsampa_to_ipa(text: str) -> str:
    """Convert X-SAMPA-like ASCII IPA from the PDF to Unicode IPA."""
    result = []
    for char in text:
        result.append(XSAMPA_TO_IPA.get(char, char))
    return "".join(result)


def extract_all_text(doc: fitz.Document) -> str:
    """Extract text from all dictionary pages, skipping intro/appendix."""
    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Skip pages that don't have dictionary entries
        # (intro pages and changelog/appendix at the end)
        if "The Dictionary" in text and page_num < 3:
            # Include from "The Dictionary" header onward
            idx = text.index("The Dictionary")
            text = text[idx:]

        all_text.append(text)

    return "\n".join(all_text)


def parse_entries(full_text: str) -> list[dict]:
    """Parse dictionary entries from extracted text, handling multi-line definitions."""
    entries = []
    current_entry = None
    seen = set()

    # Section headers to skip
    skip_patterns = [
        re.compile(r"^[A-Z]$"),          # Letter headers (A, B, C...)
        re.compile(r"^The Dictionary$"),
        re.compile(r"^\d+$"),             # Page numbers
        re.compile(r"^Appendix"),
        re.compile(r"^Numbers"),
        re.compile(r"^Version History"),
        re.compile(r"^>$"),              # PDF artifacts
    ]

    for line in full_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Skip section headers and artifacts
        if any(p.match(line) for p in skip_patterns):
            continue

        # Try to match a new entry
        entry_match = ENTRY_START.match(line)
        if entry_match:
            # Save the previous entry if we have one
            if current_entry:
                _finalize_entry(current_entry, entries, seen)

            word = entry_match.group("word").strip()
            ipa_raw = entry_match.group("ipa").strip()
            rest = entry_match.group("rest").strip()

            current_entry = {
                "word_raw": word,
                "ipa_raw": ipa_raw,
                "rest": rest,
            }
        elif current_entry:
            # Continuation line — append to the rest
            current_entry["rest"] += " " + line

    # Don't forget the last entry
    if current_entry:
        _finalize_entry(current_entry, entries, seen)

    return entries


def _finalize_entry(raw: dict, entries: list, seen: set):
    """Process a raw entry and add it to the entries list."""
    word = raw["word_raw"]
    ipa = xsampa_to_ipa(raw["ipa_raw"])
    rest = raw["rest"]

    # Clean up extra whitespace in rest
    rest = re.sub(r"\s+", " ", rest).strip()

    # Fix hyphenated line breaks from PDF column wrapping (e.g. "power- ful" -> "powerful")
    rest = re.sub(r"- ", "", rest)

    rest_match = REST_PATTERN.match(rest)
    if not rest_match:
        return

    pos_raw = rest_match.group("pos")
    english = rest_match.group("english").strip()

    # Handle Type B marker
    is_type_b = pos_raw.endswith("-B")
    pos_key = pos_raw.replace("-B", "")
    pos = POS_MAP.get(pos_key, pos_key)

    # Clean word — extract base form without numbering
    base_word = re.sub(r"\s*\(\d+\)", "", word).strip()

    # Deduplicate
    key = (base_word.lower(), pos, english[:40])
    if key in seen:
        return
    seen.add(key)

    entry = {
        "word": base_word,
        "ipa": ipa,
        "part_of_speech": pos,
        "english": english,
    }
    if word != base_word:
        entry["sense"] = int(re.search(r"\((\d+)\)", word).group(1))
    if is_type_b:
        entry["type_b"] = True

    entries.append(entry)


def build_lexicon():
    """Build the complete lexicon from the dictionary PDF."""
    download_pdf()

    print("Opening PDF with pymupdf...")
    doc = fitz.open(PDF_PATH)
    print(f"PDF has {len(doc)} pages")

    print("Extracting text...")
    full_text = extract_all_text(doc)
    doc.close()

    print("Parsing entries...")
    entries = parse_entries(full_text)

    # Sort alphabetically
    entries.sort(key=lambda e: (e["word"].lower(), e.get("sense", 0)))

    # Save
    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print stats
    pos_counts = {}
    for e in entries:
        pos_counts[e["part_of_speech"]] = pos_counts.get(e["part_of_speech"], 0) + 1

    print(f"\nLexicon saved to {OUTPUT_PATH}")
    print(f"Total entries: {len(entries)}")
    print(f"\nBy part of speech:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        print(f"  {pos}: {count}")


if __name__ == "__main__":
    build_lexicon()
