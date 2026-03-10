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

Loss modes
----------
--loss pairwise  (DEFAULT) MarginMSE ranking loss.
    Builds (query, positive, negative) triplets and trains the model to satisfy
    score(q, pos) - score(q, neg) ≈ 1.0.
    Directly optimises ranking (preserves Spearman) while still learning binary boundaries.
    Fixes the Spearman collapse seen with pointwise MSE training.

--loss mse       Pointwise regression on raw scores (legacy, kept for comparison).
    Trains the model to output exact scores (1.0 / 0.5 / 0.0).
    Known issue: tends to collapse to binary predictions, hurting Spearman.

Usage (GPU, full training):
  python3 scripts/finetune_reranker.py \\
      --train data/07_golden_mapping/training_data/train.json \\
      --dev   data/07_golden_mapping/training_data/dev.json \\
      --output models/compliance-reranker \\
      --loss pairwise --epochs 5 --batch-size 16
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Pairwise (MarginMSE) helpers ──────────────────────────────────────────────

def make_triplets(rows: list, max_negs_per_pos: int = 5) -> list:
    """
    Group rows by query and build (query, positive_passage, negative_passage) triplets.

    max_negs_per_pos caps how many negatives each positive is paired with.
    Hard negatives are prioritised; remaining slots filled with random negatives.
    Default=5 keeps total triplets manageable for CPU (~3,500 vs 41,000+).
    """
    groups = defaultdict(lambda: {"pos": [], "neg": [], "hard": []})
    for r in rows:
        q = r["query"]
        p = r["passage"]
        s = float(r.get("score", 0.0))
        if s >= 0.5:
            groups[q]["pos"].append(p)
        elif r.get("is_hard_negative"):
            groups[q]["hard"].append(p)
        else:
            groups[q]["neg"].append(p)

    triplets = []
    for q, g in groups.items():
        # Hard negatives first, then random negatives, capped per positive
        all_negs = g["hard"] + g["neg"]
        if not all_negs:
            continue
        for pos in g["pos"]:
            sampled = all_negs[:max_negs_per_pos]   # hard negs already first
            for neg in sampled:
                triplets.append((q, pos, neg))

    random.shuffle(triplets)
    n_pos = sum(len(g["pos"]) for g in groups.values())
    n_neg = sum(len(g["neg"]) + len(g["hard"]) for g in groups.values())
    print(f"  Triplets: {len(triplets)}  "
          f"(from {n_pos} positives × up to {max_negs_per_pos} negatives each; "
          f"{n_neg} total negatives available)")
    return triplets


def train_pairwise(cross_encoder, triplets: list, epochs: int,
                   batch_size: int, warmup_steps: int, output_path: str):
    """
    MarginMSE pairwise training loop.

    For each triplet (q, pos, neg), we compute:
        loss = MSE(score(q, pos) - score(q, neg),  target=1.0)

    Pos and neg pairs are concatenated into ONE forward pass of size 2×batch_size
    to halve memory vs two separate passes. Mixed precision (fp16) is used
    automatically when a CUDA GPU is available.
    """
    import torch
    import torch.nn as nn
    from transformers import get_linear_schedule_with_warmup

    hf_model = cross_encoder.model
    tokenizer = cross_encoder.tokenizer
    max_len   = cross_encoder.max_length
    device    = next(hf_model.parameters()).device

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    optimizer = torch.optim.AdamW(hf_model.parameters(), lr=5e-6, weight_decay=0.01)
    total_steps = (len(triplets) // batch_size + 1) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # BPR loss: -log(sigmoid(logit_pos - logit_neg))
    # Maximises P(pos ranks above neg).  Numerically stable — no large values.
    # Better than MSE on raw logits (which explodes when logits are ±10+).
    def bpr_loss(pos_logits, neg_logits):
        return -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-8).mean()

    def score_pairs_combined(queries, pos_passages, neg_passages):
        """Single forward pass for both pos and neg — halves peak GPU memory.

        Returns RAW LOGITS (not sigmoid) so the MarginMSE diff target of 1.0
        is achievable without forcing extreme weight values.  With sigmoid,
        the max achievable diff is ~0.76 in normal logit range, which drives
        the model to push weights to ±10 and destroys ranking quality.
        """
        all_q = queries + queries                    # [q…q, q…q]
        all_p = pos_passages + neg_passages          # [pos…, neg…]
        enc = tokenizer(
            all_q, all_p,
            padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = hf_model(**enc).logits.squeeze(-1)   # (2B,)  raw logits
        B = len(queries)
        return logits[:B], logits[B:]   # pos_logits, neg_logits

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    best_loss = float("inf")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    steps_per_epoch = (len(triplets) + batch_size - 1) // batch_size
    log_every = max(1, steps_per_epoch // 10)   # print ~10 times per epoch

    for epoch in range(epochs):
        random.shuffle(triplets)
        hf_model.train()
        epoch_loss = 0.0
        steps = 0

        batches = range(0, len(triplets), batch_size)
        if use_tqdm:
            batches = tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for i in batches:
            batch = triplets[i : i + batch_size]
            queries   = [t[0] for t in batch]
            pos_pass  = [t[1] for t in batch]
            neg_pass  = [t[2] for t in batch]

            pos_logits, neg_logits = score_pairs_combined(queries, pos_pass, neg_pass)
            loss = bpr_loss(pos_logits, neg_logits)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(hf_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(hf_model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            steps += 1

            if use_tqdm:
                batches.set_postfix(loss=f"{loss.item():.4f}")
            elif steps % log_every == 0:
                print(f"  Epoch {epoch+1}/{epochs}  step {steps}/{steps_per_epoch}  "
                      f"loss={loss.item():.4f}  avg={epoch_loss/steps:.4f}", flush=True)

        avg = epoch_loss / max(steps, 1)
        print(f"  Epoch {epoch+1}/{epochs} complete  avg_loss={avg:.4f}", flush=True)

        if avg < best_loss:
            best_loss = avg
            cross_encoder.save(output_path)
            print(f"    → checkpoint saved (best so far)", flush=True)

    print(f"\nBest pairwise loss: {best_loss:.4f}")


# ── Pointwise (MSE) helpers ───────────────────────────────────────────────────

def make_examples(rows, hard_negative_weight=1.5, label_smoothing=0.0):
    """Pointwise (query, passage, score) InputExamples for legacy MSE training."""
    from sentence_transformers import InputExample

    eps = label_smoothing

    def smooth(s):
        if s >= 0.9:
            return 1.0 - eps
        if s <= 0.05:
            return eps
        return s

    examples = []
    for r in rows:
        q = r["query"]
        p = r["passage"]
        s = smooth(float(r.get("score", 0.0)))
        is_hard = r.get("is_hard_negative", False)

        examples.append(InputExample(texts=[q, p], label=s))

        if is_hard and hard_negative_weight > 1.0:
            for _ in range(int(hard_negative_weight) - 1):
                examples.append(InputExample(texts=[q, p], label=s))
            frac = hard_negative_weight - int(hard_negative_weight)
            if frac > 0 and random.random() < frac:
                examples.append(InputExample(texts=[q, p], label=s))

    return examples


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, dev_rows: list) -> dict:
    import statistics

    pairs = [[r["query"], r["passage"]] for r in dev_rows]
    gold  = [float(r.get("score", 0.0)) for r in dev_rows]
    pred  = model.predict(pairs, show_progress_bar=False)

    mae = statistics.mean(abs(p - g) for p, g in zip(pred, gold))

    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(pred, gold)
    except ImportError:
        corr = float("nan")

    binary_correct = sum(1 for p, g in zip(pred, gold) if (p >= 0.5) == (g >= 0.5))
    accuracy = binary_correct / len(gold) if gold else 0.0

    return {"mae": mae, "spearman": corr, "binary_accuracy": accuracy}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune reranker on compliance golden data")
    parser.add_argument("--train",      required=True)
    parser.add_argument("--dev",        required=True)
    parser.add_argument("--output",     default="models/compliance-reranker")
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--lora",       action="store_true",
                        help="Use LoRA adapters (requires pip install peft). "
                             "Trains only ~2M params instead of 278M — 10× faster on CPU, "
                             "prevents catastrophic forgetting. Recommended for CPU runs.")
    parser.add_argument("--lora-r",     type=int, default=16,
                        help="LoRA rank (default=16; lower=faster/smaller, higher=more capacity)")
    parser.add_argument("--loss",       default="pairwise", choices=["pairwise", "mse"],
                        help="pairwise=MarginMSE ranking loss (default, fixes Spearman collapse); "
                             "mse=pointwise regression (legacy)")
    parser.add_argument("--epochs",         type=int,   default=3,
                        help="Training epochs (default=3; more than 3 risks catastrophic forgetting "
                             "on small datasets like this one)")
    parser.add_argument("--batch-size",     type=int,   default=8,
                        help="Per-step triplet batch (pairwise mode runs 2× this through GPU; "
                             "default=8 → 16 pairs per forward pass, safe on 15 GiB GPU)")
    parser.add_argument("--warmup-ratio",   type=float, default=0.1)
    parser.add_argument("--hard-neg-weight",type=float, default=1.5,
                        help="For mse mode: hard-neg duplication multiplier.")
    parser.add_argument("--max-negs-per-pos", type=int, default=5,
                        help="Pairwise mode: max negatives paired with each positive. "
                             "Default=5 → ~3,500 triplets, feasible on CPU in ~20-30 min. "
                             "Increase for GPU runs (e.g. 20 → ~14,000 triplets).")
    parser.add_argument("--max-length",     type=int,   default=512)
    parser.add_argument("--label-smoothing",type=float, default=0.0,
                        help="For mse mode only. 0.0 = no smoothing (default).")
    args = parser.parse_args()

    try:
        from sentence_transformers import CrossEncoder
        from torch.utils.data import DataLoader
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading training data from {args.train}")
    train_rows = load_rows(args.train)
    dev_rows   = load_rows(args.dev)
    print(f"  Train: {len(train_rows)} rows | Dev: {len(dev_rows)} rows")

    score_buckets = {"positive(≥0.5)": 0, "negative(0.0)": 0}
    for r in train_rows:
        s = float(r.get("score", 0.0))
        if s >= 0.5:
            score_buckets["positive(≥0.5)"] += 1
        else:
            score_buckets["negative(0.0)"] += 1
    hard_negs = sum(1 for r in train_rows if r.get("is_hard_negative"))
    print(f"  Score distribution: {score_buckets}  |  Hard negatives: {hard_negs}")
    print(f"  Loss mode: {args.loss.upper()}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading base model: {args.base_model}")
    model = CrossEncoder(args.base_model, max_length=args.max_length, num_labels=1)

    # ── Optional LoRA wrapping ────────────────────────────────────────────────
    if args.lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            print("LoRA requires the peft library: pip install peft")
            sys.exit(1)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
            bias="none",
        )
        model.model = get_peft_model(model.model, peft_config)
        trainable, total = model.model.get_nb_trainable_parameters() if hasattr(
            model.model, "get_nb_trainable_parameters") else (
            sum(p.numel() for p in model.model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.model.parameters()))
        print(f"\nLoRA enabled  rank={args.lora_r}  "
              f"trainable={trainable:,} / {total:,} params "
              f"({100 * trainable / total:.2f}%)")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\nBaseline (before fine-tuning):")
    baseline = evaluate(model, dev_rows)
    print(f"  MAE={baseline['mae']:.4f}  Spearman={baseline['spearman']:.4f}  "
          f"BinaryAcc={baseline['binary_accuracy']:.4f}")

    output_path = str(Path(args.output).resolve())
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.loss == "pairwise":
        print(f"\nBuilding triplets for pairwise MarginMSE training...")
        triplets = make_triplets(train_rows, max_negs_per_pos=args.max_negs_per_pos)

        total_steps  = (len(triplets) // args.batch_size + 1) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        print(f"\nFine-tuning for {args.epochs} epochs  "
              f"(batch={args.batch_size}, warmup={warmup_steps} steps, "
              f"total={total_steps} steps)…")

        train_pairwise(model, triplets, args.epochs, args.batch_size,
                       warmup_steps, output_path)

    else:  # mse
        print(f"\nBuilding pointwise examples for MSE training...")
        train_examples = make_examples(
            train_rows,
            hard_negative_weight=args.hard_neg_weight,
            label_smoothing=args.label_smoothing,
        )
        print(f"  Examples: {len(train_examples)}  "
              f"(label_smoothing ε={args.label_smoothing})")

        loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        warmup_steps = int(len(loader) * args.epochs * args.warmup_ratio)
        print(f"\nFine-tuning for {args.epochs} epochs  "
              f"(batch={args.batch_size}, warmup={warmup_steps} steps)…")

        model.fit(
            train_dataloader=loader,
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
            save_best_model=True,
        )

    # ── Explicit final save ───────────────────────────────────────────────────
    print(f"\nSaving final model to {output_path} ...")
    if args.lora:
        # Merge LoRA adapters back into base weights before saving so the
        # output is a standard CrossEncoder loadable without peft installed.
        print("  Merging LoRA adapters into base weights...")
        model.model = model.model.merge_and_unload()
    model.save(output_path)

    saved_files = []
    for root, _, files in os.walk(output_path):
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

    # ── Post-training evaluation ──────────────────────────────────────────────
    print("\nAfter fine-tuning:")
    trained = evaluate(model, dev_rows)
    print(f"  MAE={trained['mae']:.4f}  Spearman={trained['spearman']:.4f}  "
          f"BinaryAcc={trained['binary_accuracy']:.4f}")

    print(f"\n  Δ MAE      = {baseline['mae'] - trained['mae']:+.4f}  (positive = improvement)")
    print(f"  Δ Spearman = {trained['spearman'] - baseline['spearman']:+.4f}  (positive = improvement)")
    print(f"  Δ Acc      = {trained['binary_accuracy'] - baseline['binary_accuracy']:+.4f}")

    print(f"\nModel saved → {output_path}")
    print("\nTo use the fine-tuned model in the pipeline, set:")
    print(f"  export RERANKER_MODEL={output_path}")
    print("  python3 quick_start_compliance.py")


if __name__ == "__main__":
    main()
