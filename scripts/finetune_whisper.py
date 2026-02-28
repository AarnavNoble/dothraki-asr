"""Fine-tune Whisper decoder on synthetic Dothraki audio.

Freezes the encoder (audio features are already good) and trains
the decoder to output Dothraki text instead of English/Thai/Polish.

Usage:
    python scripts/finetune_whisper.py                    # defaults
    python scripts/finetune_whisper.py --epochs 20        # more training
    python scripts/finetune_whisper.py --model base       # larger model

Output: models/whisper-{model}-dothraki/weights.npz
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import MODELS_DIR, SYNTHETIC_DIR

# Whisper special tokens
SOT = 50258
EOT = 50257
NO_TIMESTAMPS = 50363


def clean_target_text(text: str) -> str:
    """Strip source annotations like (fs_2.mp3) from training targets."""
    return re.sub(r'\s*\([^)]*\)', '', text).strip()


def load_dataset(manifest_path: Path, val_ratio: float = 0.2):
    """Load and split the synthetic Dothraki dataset."""
    manifest = json.loads(manifest_path.read_text())

    # Shuffle deterministically
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(manifest))

    split = int(len(manifest) * (1 - val_ratio))
    train_entries = [manifest[i] for i in indices[:split]]
    val_entries = [manifest[i] for i in indices[split:]]

    return train_entries, val_entries


def prepare_batch(entries, tokenizer, model, synthetic_dir: Path):
    """Prepare a batch of (mel, tokens) for training.

    Returns:
        mel: (batch, n_frames, n_mels) padded mel spectrograms
        tokens_in: (batch, seq_len) decoder input tokens (SOT + text)
        tokens_out: (batch, seq_len) decoder target tokens (text + EOT)
    """
    from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES, N_SAMPLES

    mels = []
    all_tokens = []

    for entry in entries:
        audio_path = synthetic_dir / entry["audio_file"]
        if not audio_path.exists():
            continue

        # Mel spectrogram
        mel = log_mel_spectrogram(
            str(audio_path), n_mels=model.dims.n_mels, padding=N_SAMPLES
        )
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float32)
        mels.append(mel)

        # Tokenize target text: [SOT, NO_TIMESTAMPS, ...text_tokens..., EOT]
        text_tokens = tokenizer.encode(clean_target_text(entry["dothraki"]))
        full_tokens = [SOT, NO_TIMESTAMPS] + text_tokens + [EOT]
        all_tokens.append(full_tokens)

    if not mels:
        return None, None, None

    # Stack mels
    mel_batch = mx.stack(mels)  # (batch, n_frames, n_mels)

    # Pad token sequences to same length
    max_len = max(len(t) for t in all_tokens)
    padded_tokens = np.full((len(all_tokens), max_len), EOT, dtype=np.int32)
    for i, t in enumerate(all_tokens):
        padded_tokens[i, :len(t)] = t

    tokens_in = mx.array(padded_tokens[:, :-1])   # decoder input (without last)
    tokens_out = mx.array(padded_tokens[:, 1:])   # decoder target (without first)

    return mel_batch, tokens_in, tokens_out


def compute_loss(model, mel, tokens_in, tokens_out):
    """Cross-entropy loss on decoder output."""
    # Encode audio (frozen — no gradients flow here)
    audio_features = model.encoder(mel)
    audio_features = mx.stop_gradient(audio_features)

    # Decode
    logits = model.decoder(tokens_in, audio_features)[0]  # (batch, seq_len, vocab)

    # Cross-entropy loss (ignore padding = EOT token)
    # Reshape for cross_entropy
    batch, seq_len, vocab = logits.shape
    logits_flat = logits.reshape(-1, vocab)
    targets_flat = tokens_out.reshape(-1)

    # Mask: don't penalize predictions where target is EOT (padding)
    mask = (targets_flat != EOT).astype(mx.float32)

    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
    loss = (loss * mask).sum() / mx.maximum(mask.sum(), mx.array(1.0))

    return loss


def evaluate(model, val_entries, tokenizer, synthetic_dir, batch_size):
    """Compute validation loss."""
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(val_entries), batch_size):
        batch = val_entries[i:i + batch_size]
        mel, tokens_in, tokens_out = prepare_batch(batch, tokenizer, model, synthetic_dir)
        if mel is None:
            continue

        loss = compute_loss(model, mel, tokens_in, tokens_out)
        mx.eval(loss)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Dothraki")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base"],
                        help="Base Whisper model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    # Load model
    from mlx_whisper.load_models import load_model
    from mlx_whisper.tokenizer import get_tokenizer

    repo_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
    }

    print(f"Loading whisper-{args.model}...")
    model = load_model(repo_map[args.model])
    tokenizer = get_tokenizer(multilingual=True)

    # Cast decoder to float32 for training stability
    # (MLX Whisper models load as float16 which overflows during optimizer updates)
    def to_float32(params):
        if isinstance(params, mx.array):
            return params.astype(mx.float32)
        elif isinstance(params, dict):
            return {k: to_float32(v) for k, v in params.items()}
        elif isinstance(params, list):
            return [to_float32(v) for v in params]
        return params

    model.decoder.update(to_float32(model.decoder.parameters()))
    print("Cast decoder to float32 for training stability")

    # Freeze encoder (stays float16 — no gradients needed)
    model.encoder.freeze()
    trainable_params = dict(nn.utils.tree_flatten(model.trainable_parameters()))
    all_params = dict(nn.utils.tree_flatten(model.parameters()))
    trainable = sum(v.size for v in trainable_params.values())
    total = sum(v.size for v in all_params.values())
    print(f"Parameters: {total:,} total, {trainable:,} trainable (decoder only)", flush=True)

    # Load dataset
    manifest_path = SYNTHETIC_DIR / "manifest.json"
    train_entries, val_entries = load_dataset(manifest_path, args.val_ratio)
    print(f"Dataset: {len(train_entries)} train, {len(val_entries)} val", flush=True)

    # Optimizer with cosine decay
    n_steps = (len(train_entries) // args.batch_size) * args.epochs
    warmup_steps = min(100, n_steps // 10)
    schedule = opt.cosine_decay(args.lr, n_steps, 1e-6)
    optimizer = opt.AdamW(learning_rate=schedule)

    # Training loop
    loss_and_grad = nn.value_and_grad(model, compute_loss)
    best_val_loss = float("inf")
    output_dir = MODELS_DIR / f"whisper-{args.model}-dothraki"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs, {n_steps} steps...", flush=True)
    print(f"Output: {output_dir}\n", flush=True)

    global_step = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Shuffle training data each epoch
        rng = np.random.RandomState(epoch)
        shuffled = [train_entries[i] for i in rng.permutation(len(train_entries))]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(shuffled), args.batch_size):
            batch = shuffled[i:i + args.batch_size]
            mel, tokens_in, tokens_out = prepare_batch(
                batch, tokenizer, model, SYNTHETIC_DIR
            )
            if mel is None:
                continue

            loss, grads = loss_and_grad(model, mel, tokens_in, tokens_out)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 10 == 0:
                avg = epoch_loss / n_batches
                print(f"  step {global_step}/{n_steps}: loss={avg:.4f}", flush=True)

        # Epoch stats
        train_loss = epoch_loss / max(n_batches, 1)
        val_loss = evaluate(model, val_entries, tokenizer, SYNTHETIC_DIR, args.batch_size)
        elapsed = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"time={elapsed:.0f}s")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            weights_path = output_dir / "weights.npz"
            model.save_weights(str(weights_path))
            # Also save config
            config = {
                "base_model": args.model,
                "base_repo": repo_map[args.model],
                "epochs_trained": epoch + 1,
                "best_val_loss": best_val_loss,
                "train_size": len(train_entries),
                "val_size": len(val_entries),
                "lr": args.lr,
                "batch_size": args.batch_size,
            }
            (output_dir / "config.json").write_text(
                json.dumps(config, indent=2)
            )
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
