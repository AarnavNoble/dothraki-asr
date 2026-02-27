"""Pre-compute Whisper encoder embeddings for all synthetic Dothraki clips.

Usage:
    python scripts/build_embedding_index.py              # default: tiny
    python scripts/build_embedding_index.py --model small

Output: data/features/embedding_index_{model}.npz
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asr.embedder import EmbeddingMatcher


def main():
    parser = argparse.ArgumentParser(description="Build embedding index")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small"],
                        help="Whisper model size for embeddings")
    args = parser.parse_args()

    print(f"Building embedding index with whisper-{args.model}...")
    start = time.time()

    matcher = EmbeddingMatcher(model_name=args.model)
    index_path = matcher.build_index()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    main()
