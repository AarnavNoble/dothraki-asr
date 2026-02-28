"""
Batch evaluation of the pipeline on synthetic Dothraki audio.

Runs the full pipeline on synthetic clips and collects statistics:
- What languages Whisper detects
- Whisper output text vs ground truth
- Phoneme matching accuracy
- Per-model comparison (if multiple models tested)
- Exact-match accuracy for embedding/dtw/finetune strategies

Usage:
    python scripts/batch_evaluate.py                              # all clips, phoneme
    python scripts/batch_evaluate.py --limit 100                  # first 100
    python scripts/batch_evaluate.py --model medium               # specific model
    python scripts/batch_evaluate.py --strategy embedding         # embedding strategy
    python scripts/batch_evaluate.py --strategy finetune --limit 20

Output: data/results/batch_eval_{strategy}_{model}.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import Strategy
from pipeline.run import Pipeline

SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
MANIFEST_PATH = SYNTHETIC_DIR / "manifest.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def evaluate_batch(entries: list[dict], model: str, strategy: str) -> list[dict]:
    """Run the pipeline on a batch of entries and collect results."""
    p = Pipeline(whisper_model=model, skip_separation=True, strategy=strategy)
    results = []

    start_time = time.time()
    for i, entry in enumerate(entries):
        audio_path = SYNTHETIC_DIR / entry["audio_file"]
        if not audio_path.exists():
            continue

        try:
            result = p.run(str(audio_path), save=False)

            record = {
                "id": entry["id"],
                "gt_dothraki": entry["dothraki"],
                "gt_ipa": entry.get("ipa", ""),
                "gt_english": entry["english"],
                "scene": entry.get("scene", ""),
                "strategy": strategy,
                "quality": result.quality,
                "model": model,
            }

            # Phoneme-specific fields
            if result.transcription is not None:
                record["whisper_lang"] = result.transcription.language
                record["whisper_text"] = result.transcription.text
                record["is_hallucination"] = result.transcription.is_hallucination
                record["num_segments"] = len(result.transcription.segments)

            if result.translation is not None:
                record["pipeline_translation"] = (
                    result.translation.translation if result.translation else ""
                )

            # Clip match fields (embedding/dtw/ensemble)
            if result.clip_matches is not None:
                record["clip_matches"] = result.clip_matches
                if result.clip_matches:
                    top = result.clip_matches[0]
                    record["top_match_id"] = top["clip_id"]
                    record["top_match_dothraki"] = top["dothraki"]
                    record["top_match_english"] = top["english"]
                    record["top_match_score"] = top["score"]

            # Finetune-specific fields
            if result.raw_dothraki is not None:
                record["raw_dothraki"] = result.raw_dothraki

            # Exact match: does the output Dothraki match ground truth?
            output_dothraki = ""
            if result.raw_dothraki:
                output_dothraki = result.raw_dothraki.strip().lower()
            elif result.clip_matches:
                output_dothraki = result.clip_matches[0]["dothraki"].strip().lower()

            gt_lower = entry["dothraki"].strip().lower()
            record["exact_match"] = output_dothraki == gt_lower
            record["output_dothraki"] = output_dothraki

            results.append(record)

        except Exception as e:
            results.append({
                "id": entry["id"],
                "gt_dothraki": entry["dothraki"],
                "gt_english": entry["english"],
                "error": str(e),
                "model": model,
                "strategy": strategy,
            })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(entries) - i - 1) / rate
            print(f"  [{i+1}/{len(entries)}] {rate:.1f} clips/sec, ETA: {eta:.0f}s")

    return results


def compute_stats(results: list[dict], strategy: str) -> dict:
    """Compute summary statistics from evaluation results."""
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    # Quality distribution
    quality_counts: dict[str, int] = {}
    for r in valid:
        q = r.get("quality", "unknown")
        quality_counts[q] = quality_counts.get(q, 0) + 1

    stats: dict = {
        "total_clips": len(results),
        "successful": len(valid),
        "errors": len(errors),
        "quality_distribution": dict(
            sorted(quality_counts.items(), key=lambda x: -x[1])
        ),
    }

    # Phoneme-specific stats
    if strategy in ("phoneme", "ensemble"):
        phoneme_valid = [r for r in valid if "whisper_text" in r]

        lang_counts: dict[str, int] = {}
        for r in phoneme_valid:
            lang = r["whisper_lang"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        hallucinations = sum(
            1 for r in phoneme_valid if r.get("is_hallucination", False)
        )
        empty = sum(1 for r in phoneme_valid if not r["whisper_text"].strip())

        has_translation = sum(
            1 for r in valid if r.get("pipeline_translation", "").strip()
        )

        avg_whisper_len = (
            sum(len(r["whisper_text"]) for r in phoneme_valid) / len(phoneme_valid)
            if phoneme_valid
            else 0
        )
        avg_gt_len = (
            sum(len(r["gt_dothraki"]) for r in valid) / len(valid) if valid else 0
        )

        stats.update({
            "hallucinations": hallucinations,
            "hallucination_rate": hallucinations / len(phoneme_valid) if phoneme_valid else 0,
            "empty_transcriptions": empty,
            "empty_rate": empty / len(phoneme_valid) if phoneme_valid else 0,
            "language_distribution": dict(
                sorted(lang_counts.items(), key=lambda x: -x[1])
            ),
            "avg_whisper_output_length": round(avg_whisper_len, 1),
            "avg_gt_dothraki_length": round(avg_gt_len, 1),
            "translation_coverage": has_translation,
            "translation_coverage_rate": (
                has_translation / len(valid) if valid else 0
            ),
        })

    # Exact match stats (for all strategies)
    exact_matches = sum(1 for r in valid if r.get("exact_match", False))
    stats["exact_matches"] = exact_matches
    stats["exact_match_rate"] = exact_matches / len(valid) if valid else 0

    # Score stats for clip-match strategies
    if strategy in ("embedding", "dtw", "ensemble"):
        scores = [r["top_match_score"] for r in valid if "top_match_score" in r]
        if scores:
            stats["avg_top_score"] = round(sum(scores) / len(scores), 4)
            stats["min_top_score"] = round(min(scores), 4)
            stats["max_top_score"] = round(max(scores), 4)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate pipeline")
    parser.add_argument("--limit", type=int, help="Max clips to evaluate")
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument(
        "--strategy",
        default="phoneme",
        choices=[s.value for s in Strategy],
        help="Matching strategy",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    entries = manifest[: args.limit] if args.limit else manifest
    print(
        f"Evaluating {len(entries)} clips with strategy={args.strategy}, "
        f"model=whisper-{args.model}"
    )

    results = evaluate_batch(entries, args.model, args.strategy)
    stats = compute_stats(results, args.strategy)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "strategy": args.strategy,
        "model": args.model,
        "num_clips": len(entries),
        "stats": stats,
        "results": results,
    }
    output_path = RESULTS_DIR / f"batch_eval_{args.strategy}_{args.model}.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY ({args.strategy}, whisper-{args.model})")
    print(f"{'='*50}")
    print(f"Total clips:              {stats['total_clips']}")
    print(f"Successful:               {stats['successful']}")
    print(f"Errors:                   {stats['errors']}")
    print(f"Exact matches:            {stats['exact_matches']} ({stats['exact_match_rate']:.1%})")

    if "hallucinations" in stats:
        print(f"Hallucinations filtered:  {stats['hallucinations']} ({stats['hallucination_rate']:.1%})")
        print(f"Empty transcriptions:     {stats['empty_transcriptions']} ({stats['empty_rate']:.1%})")
        print(f"Translation coverage:     {stats['translation_coverage_rate']:.1%}")

    if "avg_top_score" in stats:
        print(f"Avg top score:            {stats['avg_top_score']:.4f}")
        print(f"Score range:              [{stats['min_top_score']:.4f}, {stats['max_top_score']:.4f}]")

    print(f"\nQuality distribution:")
    for q, count in stats["quality_distribution"].items():
        pct = count / stats["successful"] * 100 if stats["successful"] else 0
        print(f"  {q:18s}: {count:4d} ({pct:.1f}%)")

    if "language_distribution" in stats:
        print(f"\nDetected languages:")
        for lang, count in stats["language_distribution"].items():
            pct = count / stats["successful"] * 100 if stats["successful"] else 0
            print(f"  {lang:5s}: {count:4d} ({pct:.1f}%)")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
