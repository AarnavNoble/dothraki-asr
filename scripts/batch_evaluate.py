"""
Batch evaluation of the pipeline on synthetic Dothraki audio.

Runs the full pipeline on synthetic clips and collects statistics:
- What languages Whisper detects
- Whisper output text vs ground truth
- Phoneme matching accuracy
- Per-model comparison (if multiple models tested)

Usage:
    python scripts/batch_evaluate.py                  # all clips
    python scripts/batch_evaluate.py --limit 100      # first 100
    python scripts/batch_evaluate.py --model medium    # specific model

Output: data/results/batch_evaluation.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.run import Pipeline

SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
MANIFEST_PATH = SYNTHETIC_DIR / "manifest.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def evaluate_batch(entries: list[dict], model: str) -> list[dict]:
    """Run the pipeline on a batch of entries and collect results."""
    p = Pipeline(whisper_model=model, skip_separation=True)
    results = []

    start_time = time.time()
    for i, entry in enumerate(entries):
        audio_path = SYNTHETIC_DIR / entry["audio_file"]
        if not audio_path.exists():
            continue

        try:
            result = p.run(str(audio_path), save=False)
            results.append({
                "id": entry["id"],
                "gt_dothraki": entry["dothraki"],
                "gt_ipa": entry["ipa"],
                "gt_english": entry["english"],
                "scene": entry["scene"],
                "whisper_lang": result.transcription.language,
                "whisper_text": result.transcription.text,
                "is_hallucination": result.transcription.is_hallucination,
                "quality": result.quality,
                "num_segments": len(result.transcription.segments),
                "pipeline_translation": (
                    result.translation.translation if result.translation else ""
                ),
                "model": model,
            })
        except Exception as e:
            results.append({
                "id": entry["id"],
                "gt_dothraki": entry["dothraki"],
                "gt_english": entry["english"],
                "error": str(e),
                "model": model,
            })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(entries) - i - 1) / rate
            print(f"  [{i+1}/{len(entries)}] {rate:.1f} clips/sec, ETA: {eta:.0f}s")

    return results


def compute_stats(results: list[dict]) -> dict:
    """Compute summary statistics from evaluation results."""
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    # Language detection distribution
    lang_counts = {}
    for r in valid:
        lang = r["whisper_lang"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    # Hallucination count
    hallucinations = sum(1 for r in valid if r.get("is_hallucination", False))

    # Quality distribution
    quality_counts: dict[str, int] = {}
    for r in valid:
        q = r.get("quality", "unknown")
        quality_counts[q] = quality_counts.get(q, 0) + 1

    # Empty transcription rate
    empty = sum(1 for r in valid if not r["whisper_text"].strip())
    nonempty = len(valid) - empty

    # Average transcription length
    avg_whisper_len = (
        sum(len(r["whisper_text"]) for r in valid) / len(valid)
        if valid else 0
    )
    avg_gt_len = (
        sum(len(r["gt_dothraki"]) for r in valid) / len(valid)
        if valid else 0
    )

    # Translation coverage (non-empty pipeline translations)
    has_translation = sum(
        1 for r in valid if r["pipeline_translation"].strip()
    )

    return {
        "total_clips": len(results),
        "successful": len(valid),
        "errors": len(errors),
        "hallucinations": hallucinations,
        "hallucination_rate": hallucinations / len(valid) if valid else 0,
        "quality_distribution": dict(
            sorted(quality_counts.items(), key=lambda x: -x[1])
        ),
        "empty_transcriptions": empty,
        "nonempty_transcriptions": nonempty,
        "empty_rate": empty / len(valid) if valid else 0,
        "language_distribution": dict(
            sorted(lang_counts.items(), key=lambda x: -x[1])
        ),
        "avg_whisper_output_length": round(avg_whisper_len, 1),
        "avg_gt_dothraki_length": round(avg_gt_len, 1),
        "translation_coverage": has_translation,
        "translation_coverage_rate": (
            has_translation / len(valid) if valid else 0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate pipeline")
    parser.add_argument("--limit", type=int, help="Max clips to evaluate")
    parser.add_argument("--model", default="small", help="Whisper model size")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    entries = manifest[:args.limit] if args.limit else manifest
    print(f"Evaluating {len(entries)} clips with whisper-{args.model}")

    results = evaluate_batch(entries, args.model)
    stats = compute_stats(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model,
        "num_clips": len(entries),
        "stats": stats,
        "results": results,
    }
    output_path = RESULTS_DIR / f"batch_eval_{args.model}.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY (whisper-{args.model})")
    print(f"{'='*50}")
    print(f"Total clips:              {stats['total_clips']}")
    print(f"Successful:               {stats['successful']}")
    print(f"Errors:                   {stats['errors']}")
    print(f"Hallucinations filtered:  {stats['hallucinations']} ({stats['hallucination_rate']:.1%})")
    print(f"Empty transcriptions:     {stats['empty_transcriptions']} ({stats['empty_rate']:.1%})")
    print(f"Avg Whisper output len:   {stats['avg_whisper_output_length']} chars")
    print(f"Avg GT Dothraki len:      {stats['avg_gt_dothraki_length']} chars")
    print(f"Translation coverage:     {stats['translation_coverage_rate']:.1%}")
    print(f"\nQuality distribution:")
    for q, count in stats["quality_distribution"].items():
        pct = count / stats["successful"] * 100
        print(f"  {q:18s}: {count:4d} ({pct:.1f}%)")
    print(f"\nDetected languages:")
    for lang, count in stats["language_distribution"].items():
        pct = count / stats["successful"] * 100
        print(f"  {lang:5s}: {count:4d} ({pct:.1f}%)")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
