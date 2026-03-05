#!/usr/bin/env python3
"""
Fine-tune BAAI/bge-reranker-base (or any CrossEncoder) on the compliance golden dataset.

Training data produced by:
  python3 scripts/prepare_golden_for_training.py --format reranker

Each row: { query, passage, score }
  score 1.0  → Fully Addressed (correct match)
  score 0.7  → Partially Addressed, multi-control passage
  score 0.5  → Partially Addressed, single-control
  score 0.0  → Not Addressed / hard negative

Usage (local CPU, quick smoke-test):
  python3 scripts/finetune_reranker.py --train data/07_golden_mapping/training_data/train.json \
    --dev data/07_golden_mapping/training_data/dev.json \
    --output models/compliance-reranker --epochs 1 --batch-size 8

Usage (Colab GPU, full training):
  python3 scripts/finetune_reranker.py --train data/07_golden_mapping/training_data/train.json \
    --dev data/07_golden_mapping/training_data/dev.json \
    --output models/compliance-reranker --epochs 5 --batch-size 16 --warmup-ratio 0.1
"""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from the project root when run as scripts/finetune_reranker.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return rows


def make_examples(rows, hard_negative_weight: float = 1.5, label_smoothing: float = 0.1):
    """
    Convert training rows to (sentence_pair, score) tuples.

    Label smoothing: prevents the model from predicting extreme 0/1 scores,
    which collapses the ranking (Spearman) even while binary accuracy improves.
      1.0  →  1.0 - ε   (e.g. 0.90)
      0.0  →  ε          (e.g. 0.10)
    Mid-range scores (0.5, 0.7) are kept as-is.

    Hard negatives are duplicated to upweight them during training.
    Default weight is 1.5 (one extra half-copy on average) rather than 2.0
    to avoid over-suppressing the model's positive signal.
    """
    from sentence_transformers import InputExample

    eps = label_smoothing

    def smooth(score: float) -> float:
        if score >= 0.9:
            return 1.0 - eps
        if score <= 0.05:
            return eps
        return score

    examples = []
    for r in rows:
        query = r["query"]
        passage = r["passage"]
        raw_score = float(r.get("score", 0.0))
        score = smooth(raw_score)
        is_hard_neg = r.get("is_hard_negative", False)

        examples.append(InputExample(texts=[query, passage], label=score))

        # Duplicate hard negatives; fractional weight handled by probabilistic copy
        if is_hard_neg and hard_negative_weight > 1.0:
            full_copies = int(hard_negative_weight) - 1
            for _ in range(full_copies):
                examples.append(InputExample(texts=[query, passage], label=score))
            # Fractional part: add one more copy with that probability
            import random
            frac = hard_negative_weight - int(hard_negative_weight)
            if frac > 0 and random.random() < frac:
                examples.append(InputExample(texts=[query, passage], label=score))

    return examples


def evaluate(model, dev_rows: list) -> dict:
    """Compute mean absolute error and correlation on dev set."""
    import statistics

    pairs = [[r["query"], r["passage"]] for r in dev_rows]
    gold = [float(r.get("score", 0.0)) for r in dev_rows]

    pred = model.predict(pairs, show_progress_bar=False)

    errors = [abs(p - g) for p, g in zip(pred, gold)]
    mae = statistics.mean(errors)

    # Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(pred, gold)
    except ImportError:
        corr = float("nan")

    # Accuracy at binary threshold 0.5
    binary_correct = sum(
        1 for p, g in zip(pred, gold) if (p >= 0.5) == (g >= 0.5)
    )
    accuracy = binary_correct / len(gold) if gold else 0.0

    return {"mae": mae, "spearman": corr, "binary_accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune reranker on compliance golden data")
    parser.add_argument("--train", required=True, help="Path to train.json")
    parser.add_argument("--dev", required=True, help="Path to dev.json")
    parser.add_argument("--output", default="models/compliance-reranker", help="Output directory")
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base",
                        help="Base cross-encoder model (default: BAAI/bge-reranker-base)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Fraction of steps for linear LR warmup (default: 0.1)")
    parser.add_argument("--hard-neg-weight", type=float, default=1.5,
                        help="Hard negative duplication multiplier (default: 1.5; lower = better Spearman)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length for query+passage (default: 512)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing: shifts 1.0→(1-ε) and 0.0→ε to prevent score collapse (default: 0.1)")
    args = parser.parse_args()

    try:
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
        from torch.utils.data import DataLoader
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading training data from {args.train}")
    train_rows = load_rows(args.train)
    dev_rows = load_rows(args.dev)
    print(f"  Train: {len(train_rows)} rows | Dev: {len(dev_rows)} rows")

    # Label distribution summary
    score_buckets = {"positive(1.0)": 0, "soft_pos(0.5-0.7)": 0, "negative(0.0)": 0}
    for r in train_rows:
        s = float(r.get("score", 0.0))
        if s >= 0.9:
            score_buckets["positive(1.0)"] += 1
        elif s > 0.0:
            score_buckets["soft_pos(0.5-0.7)"] += 1
        else:
            score_buckets["negative(0.0)"] += 1
    print(f"  Score distribution: {score_buckets}")

    hard_negs = sum(1 for r in train_rows if r.get("is_hard_negative"))
    print(f"  Hard negatives: {hard_negs} (duplication weight ×{args.hard_neg_weight})")

    # ── Build examples ───────────────────────────────────────────────────────
    train_examples = make_examples(
        train_rows,
        hard_negative_weight=args.hard_neg_weight,
        label_smoothing=args.label_smoothing,
    )
    print(f"  Effective training examples after hard-neg duplication: {len(train_examples)}")
    print(f"  Label smoothing ε={args.label_smoothing}  (1.0→{1.0-args.label_smoothing:.2f}, 0.0→{args.label_smoothing:.2f})")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading base model: {args.base_model}")
    model = CrossEncoder(args.base_model, max_length=args.max_length, num_labels=1)

    # ── Baseline evaluation ──────────────────────────────────────────────────
    print("\nBaseline (before fine-tuning):")
    baseline = evaluate(model, dev_rows)
    print(f"  MAE={baseline['mae']:.4f}  Spearman={baseline['spearman']:.4f}  "
          f"BinaryAcc={baseline['binary_accuracy']:.4f}")

    # ── Train ─────────────────────────────────────────────────────────────────
    loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    warmup_steps = int(len(loader) * args.epochs * args.warmup_ratio)

    output_path = str(Path(args.output).resolve())
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print(f"\nFine-tuning for {args.epochs} epochs "
          f"(batch={args.batch_size}, warmup={warmup_steps} steps)…")

    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        save_best_model=True,
    )

    # ── Explicitly save the final model ──────────────────────────────────────
    # sentence-transformers v3 fit() only saves best checkpoints during training;
    # it does NOT guarantee the final in-memory model is on disk afterwards.
    # Calling model.save() here ensures the weights are always persisted.
    print(f"\nSaving final model to {output_path} ...")
    model.save(output_path)

    # Verify the save actually produced files
    import os
    saved_files = []
    for root, dirs, files in os.walk(output_path):
        for f in files:
            saved_files.append(os.path.relpath(os.path.join(root, f), output_path))
    if saved_files:
        print(f"  ✓ {len(saved_files)} file(s) written:")
        for sf in saved_files[:10]:
            print(f"    {sf}")
        if len(saved_files) > 10:
            print(f"    ... and {len(saved_files) - 10} more")
    else:
        print("  WARNING: No files found after save — check disk space / permissions.")

    # ── Post-training evaluation ─────────────────────────────────────────────
    print("\nAfter fine-tuning:")
    trained = evaluate(model, dev_rows)
    print(f"  MAE={trained['mae']:.4f}  Spearman={trained['spearman']:.4f}  "
          f"BinaryAcc={trained['binary_accuracy']:.4f}")

    delta_mae = baseline["mae"] - trained["mae"]
    delta_acc = trained["binary_accuracy"] - baseline["binary_accuracy"]
    print(f"\n  Δ MAE  = {delta_mae:+.4f}  (positive = improvement)")
    print(f"  Δ Acc  = {delta_acc:+.4f}")

    print(f"\nModel saved → {output_path}")
    print("\nTo use the fine-tuned model in the pipeline, set:")
    print(f"  export RERANKER_MODEL={output_path}")
    print("  python3 quick_start_compliance.py")


if __name__ == "__main__":
    main()
