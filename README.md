# Dothraki ASR

Zero-shot automatic speech recognition for Dothraki — a constructed language from Game of Thrones — using open-source multilingual ASR models.

## What is this?

This project explores whether modern multilingual ASR models (Whisper, wav2vec2) can transcribe and translate a language they've never seen. Dothraki was invented by David J. Peterson for HBO's Game of Thrones and has a documented phonology and lexicon, but zero representation in any training dataset.

The pipeline:
1. **Vocal Isolation** — Strip music/SFX from scene audio (Demucs)
2. **Zero-Shot ASR** — Run Whisper on Dothraki speech and see what it outputs
3. **Phoneme Extraction** — Convert ASR output to IPA phonemes
4. **Dothraki Matching** — Custom engine to map phonemes to real Dothraki words
5. **Translation** — Dictionary lookup from Dothraki to English

## Project Structure

```
notebooks/       # Research notebooks walking through each step
pipeline/        # Core ML pipeline (Python)
scripts/         # Data processing & utility scripts
web/             # Interactive demo website (Next.js)
data/            # Lexicon data and cached results
```

## Tech Stack

- **ASR**: mlx-whisper (Apple Silicon optimized)
- **Vocal Isolation**: Demucs
- **Phoneme Processing**: espeak-ng, gruut
- **Frontend**: Next.js 14, Tailwind CSS, Framer Motion, wavesurfer.js
- **Compute**: Runs locally on Apple M-series (no GPU required)

## Status

🚧 Work in progress
