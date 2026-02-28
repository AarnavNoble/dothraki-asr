# Three ML Improvements — Embedding Matcher, DTW Sequence Matching, Fine-Tuning

## Context
Current pipeline gets 20% "good" quality on synthetic Dothraki clips. Whisper has zero Dothraki training data, so it hallucinates or guesses wrong languages. The pipeline improvements (hallucination filtering, weighted matching, stop words) cleaned up garbage output but didn't improve the fundamental accuracy. We need three deeper ML approaches to get presentation-ready results.

**Constraint:** M2 MacBook Pro 16GB, all free/open-source, Python 3.11 venv with mlx 0.30.6, mlx-whisper, librosa 0.11.0, torch 2.10.0.

---

## Implementation Order

### Phase 1: Embedding Matcher (audio fingerprinting)
**Why first:** No training needed, fastest to implement, immediate results.

**New file: `pipeline/asr/embedder.py`**
- Class `EmbeddingMatcher`:
  - `build_index(manifest, model="tiny")` — loads each synthetic WAV, computes mel spectrogram, runs through `model.embed_audio(mel)`, mean-pools over time axis → 384-dim vector per clip. Saves index to `data/features/embedding_index_{model}.npz`.
  - `match(audio_path, top_k=5)` — encodes input audio the same way, computes cosine similarity against all stored embeddings, returns top-k clips with scores.
- Uses `mlx_whisper.load_models.load_model()` to get the Whisper model, then `model.embed_audio(mel)` for encoder-only inference.
- Mel computation: use `mlx_whisper.audio.log_mel_spectrogram()` (same preprocessing Whisper uses internally).
- Output shape: `(1500, 384)` for tiny → mean pool → `(384,)` per clip.

**New script: `scripts/build_embedding_index.py`**
- CLI to pre-compute embeddings for all 1,712 synthetic clips.
- Saves: numpy `.npz` with `embeddings` array (1712, 384), `ids` array, `metadata` dict.
- Estimated time: ~5 min on M2 (1,712 clips × ~0.2s each).

### Phase 2: DTW Sequence Matcher
**Why second:** Builds on same mel infrastructure, different matching approach.

**New file: `pipeline/dothraki/sequence_matcher.py`**
- Class `SequenceMatcher`:
  - `build_features(manifest)` — extracts MFCC features (13 coefficients) for all synthetic clips using `librosa.feature.mfcc()`. Saves to `data/features/mfcc_features.npz`.
  - `match(audio_path, top_k=5)` — extracts MFCCs for input, runs `librosa.sequence.dtw()` against all stored features, returns top-k by DTW cost.
- Uses MFCCs instead of full mel (13-dim vs 80-dim) for faster DTW alignment.
- DTW metric: cosine distance on MFCC frames.
- **Speed concern:** DTW is O(n*m) per comparison × 1,712 clips. Mitigation: pre-filter by clip duration (±50% tolerance), only DTW against duration-similar clips. Should reduce to ~200-400 comparisons per query.

**New script: `scripts/build_mfcc_features.py`**
- CLI to pre-compute MFCCs for all synthetic clips.
- Estimated time: ~2 min on M2.

### Phase 3: Fine-Tune Whisper on Synthetic Dothraki
**Why third:** Most impactful but requires training loop, benefits from having eval infrastructure ready.

**New script: `scripts/finetune_whisper.py`**
- Base model: whisper-tiny (39M params) — fits easily in 16GB.
- **Strategy:** Freeze encoder, fine-tune decoder only. The encoder already produces good audio features — we just need the decoder to output Dothraki text instead of English/Thai/Polish.
- Training data: 1,712 clips → 1,370 train / 342 val (80/20 split).
- Training loop (pure MLX):
  1. Load audio → `log_mel_spectrogram()` → pad to 30s.
  2. Tokenize Dothraki text target using Whisper's multilingual BPE tokenizer (`whisper.tokenizer`).
  3. Run `model.embed_audio(mel)` to get encoder features (frozen, no grad).
  4. Teacher-force decoder: `model.logits(tokens[:-1], audio_features)` → cross-entropy vs `tokens[1:]`.
  5. `mx.value_and_grad` → `optimizer.update()`.
- Hyperparameters:
  - Optimizer: AdamW, LR 1e-4 with cosine decay
  - Batch size: 8 (fits in 16GB with frozen encoder)
  - Epochs: 10
  - Save best checkpoint by val loss
- Output: fine-tuned weights in `models/whisper-tiny-dothraki/`
- Estimated training time: ~20-40 min on M2.

**Modify `pipeline/asr/transcriber.py`:**
- Add support for loading fine-tuned local weights alongside HuggingFace repos.
- New model key: `"tiny-dothraki"` pointing to `models/whisper-tiny-dothraki/`.

### Phase 4: Ensemble + Integration

**Modify `pipeline/config.py`:**
- Add `FEATURES_DIR = DATA_DIR / "features"`, `MODELS_DIR = ROOT_DIR / "models"`.

**Modify `pipeline/run.py`:**
- Add `strategy` parameter: `"phoneme"` (existing), `"embedding"`, `"dtw"`, `"finetune"`, `"ensemble"`.
- Ensemble mode: run all three methods, combine scores:
  - Embedding: top-k clip IDs with cosine similarity scores
  - DTW: top-k clip IDs with DTW cost (inverted to similarity)
  - Fine-tuned: direct Dothraki text output (highest priority if confidence is high)
  - Weighted vote: if fine-tuned output matches an embedding/DTW top-k candidate, boost confidence.

**Modify `scripts/batch_evaluate.py`:**
- Add `--strategy` flag to select matching approach.
- Track per-strategy metrics.

### Phase 5: Evaluation Notebook

**New file: `notebooks/07_method_comparison.ipynb`**
- Run all 4 strategies (phoneme, embedding, DTW, fine-tuned) on same 200 clips.
- Head-to-head comparison charts: accuracy, coverage, speed.
- Confusion analysis per method.
- Ensemble vs individual method performance.
- Key findings and discussion.

---

## Files Summary

**New files (7):**
- `pipeline/asr/embedder.py` — Whisper encoder embedding matcher
- `pipeline/dothraki/sequence_matcher.py` — DTW-based sequence matcher
- `scripts/finetune_whisper.py` — Fine-tuning training loop
- `scripts/build_embedding_index.py` — Pre-compute embedding index
- `scripts/build_mfcc_features.py` — Pre-compute MFCC features
- `notebooks/07_method_comparison.ipynb` — Comparison notebook
- `models/whisper-tiny-dothraki/` — Fine-tuned weights (directory)

**Modified files (4):**
- `pipeline/config.py` — Add FEATURES_DIR, MODELS_DIR
- `pipeline/asr/transcriber.py` — Support fine-tuned model loading
- `pipeline/run.py` — Add strategy parameter + ensemble logic
- `scripts/batch_evaluate.py` — Add --strategy flag

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Mel spectrogram mismatch between inference methods | Use `mlx_whisper.audio.log_mel_spectrogram()` consistently everywhere |
| DTW too slow (1,712 comparisons per query) | Pre-filter by duration ±50%, use MFCCs (13-dim) not full mel |
| Fine-tuning overfits on synthetic audio | Freeze encoder, 80/20 split, monitor val loss, early stopping |
| Whisper tokenizer can't handle Dothraki text | Whisper's multilingual BPE handles arbitrary Unicode — verified |
| Memory overflow during training | Batch size 8 + frozen encoder = <2GB, well within 16GB M2 |

## Verification
1. Build embedding index → run `batch_evaluate.py --strategy embedding` → check accuracy
2. Build MFCC features → run `batch_evaluate.py --strategy dtw` → check accuracy
3. Fine-tune → run `batch_evaluate.py --strategy finetune` → check accuracy
4. Run `batch_evaluate.py --strategy ensemble` → compare all
5. Notebook 07 executes end-to-end, produces comparison charts
6. Commit after each phase
