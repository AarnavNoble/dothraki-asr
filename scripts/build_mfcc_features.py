"""Pre-compute MFCC features for all synthetic Dothraki clips.

Usage:
    python scripts/build_mfcc_features.py

Output: data/features/mfcc_features.npz
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dothraki.sequence_matcher import SequenceMatcher


def main():
    print("Building MFCC features for all synthetic clips...")
    start = time.time()

    matcher = SequenceMatcher()
    features_path = matcher.build_features()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Features saved to: {features_path}")


if __name__ == "__main__":
    main()
