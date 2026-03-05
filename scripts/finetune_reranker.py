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


def make_examples(rows, hard_negative_weight: float = 2.0):
    """
    Convert training rows to (sentence_pair, score) tuples.
    Hard negatives are duplicated to upweight them during training.
    """
    from sentence_transformers import InputExample

    examples = []
    for r in rows:
        query = r["query"]
        passage = r["passage"]
        score = float(r.get("score", 0.0))
        is_hard_neg = r.get("is_hard_negative", False)

        examples.append(InputExample(texts=[query, passage], label=score))

        # Duplicate hard negatives so the model sees them more often
        if is_hard_neg and hard_negative_weight >= 2:
            for _ in range(int(hard_negative_weight) - 1):
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
    parser.add_argument("--hard-neg-weight", type=float, default=2.0,
                        help="Hard negative duplication multiplier (default: 2)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length for query+passage (default: 512)")
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
    print(f"  Hard negatives: {hard_negs} (will be duplicated ×{int(args.hard_neg_weight)})")

    # ── Build examples ───────────────────────────────────────────────────────
    train_examples = make_examples(train_rows, hard_negative_weight=args.hard_neg_weight)
    print(f"  Effective training examples after hard-neg duplication: {len(train_examples)}")

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
