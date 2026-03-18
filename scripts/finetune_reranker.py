#!/usr/bin/env python3
"""
Fine-tune ms-marco-MiniLM-L-6-v2 (or any CrossEncoder) on the compliance golden dataset.

Training data produced by:
  python3 scripts/prepare_golden_for_training.py --format reranker

Each row: { query, passage, score }
  score 1.0  → Fully Addressed (correct match)
  score 0.7  → Partially Addressed, multi-control passage
  score 0.5  → Partially Addressed, single-control
  score 0.0  → Not Addressed / hard negative

Loss
----
BPR (Bayesian Personalised Ranking) pairwise loss:
    loss = -log(sigmoid(pos_logit - neg_logit))

Only cares about relative ordering (pos > neg) — never pushes logits toward a
specific value, preventing score collapse.  A small pos_penalty regulariser
keeps positive pair logits above 0, preventing them from drifting negative.

Early stopping monitors Spearman correlation on the dev set after each epoch
and stops training (restoring the best checkpoint) if Spearman doesn't improve
for `--patience` consecutive epochs.

GPU run (~10 min on T4):
  python3 scripts/finetune_reranker.py \\
      --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
      --loss bpr --max-negs-per-pos 3 --epochs 3 --lr 2e-5 \\
      --batch-size 16 --early-stopping \\
      --output models/reranker-finetuned-v3

CPU run with LoRA (~45 min):
  python3 scripts/finetune_reranker.py \\
      --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
      --loss bpr --lora --lora-r 8 --max-negs-per-pos 3 \\
      --epochs 3 --lr 2e-5 --batch-size 8 --max-length 128 \\
      --early-stopping --output models/reranker-finetuned-v3

Eval only (epochs=0):
  python3 scripts/finetune_reranker.py \\
      --base-model models/reranker-finetuned-v3 \\
      --epochs 0 --output /tmp/eval_only
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


# ── Triplet building ──────────────────────────────────────────────────────────

def make_triplets(rows: list, max_negs_per_pos: int = 3) -> list:
    """
    Build (query, positive_passage, negative_passage) triplets.

    Negative selection priority per query:
      1. Hard negatives (is_hard_negative=True) — domain-relevant wrong controls
      2. Random negatives from the same query group
      3. Cross-query hard negatives if local supply is exhausted

    max_negs_per_pos=3 is the recommended default.  Using more (e.g. 10) causes
    score collapse: the gradient pressure to push neg scores down overwhelms the
    pressure to keep pos scores up, and the model learns to output low logits for
    everything.
    """
    groups = defaultdict(lambda: {"pos": [], "hard": [], "neg": []})
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

    # Collect all hard negatives for cross-query fallback
    all_hard_negs = [p for g in groups.values() for p in g["hard"]]

    triplets = []
    for q, g in groups.items():
        if not g["pos"]:
            continue
        # Ordered preference: hard → random → cross-query hard
        ordered_negs = g["hard"] + g["neg"]
        if len(ordered_negs) < max_negs_per_pos and all_hard_negs:
            extra = [h for h in all_hard_negs if h not in g["hard"]]
            random.shuffle(extra)
            ordered_negs = ordered_negs + extra

        for pos in g["pos"]:
            for neg in ordered_negs[:max_negs_per_pos]:
                triplets.append((q, pos, neg))

    random.shuffle(triplets)
    n_pos  = sum(len(g["pos"])  for g in groups.values())
    n_neg  = sum(len(g["hard"]) + len(g["neg"]) for g in groups.values())
    n_hard = sum(len(g["hard"]) for g in groups.values())
    print(f"  Triplets: {len(triplets)}  "
          f"(from {n_pos} positives × up to {max_negs_per_pos} negatives each; "
          f"{n_neg} local negatives, {n_hard} hard)")
    return triplets


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, dev_rows: list, sample_n: int = None) -> dict:
    """
    Evaluate ranking quality on dev set.

    Binary threshold is 0.0 (logit > 0 = positive) which is appropriate for
    raw-logit cross-encoder models (not sigmoid-activated outputs).
    """
    import statistics
    rows = dev_rows
    if sample_n and len(rows) > sample_n:
        rows = random.sample(rows, sample_n)

    pairs = [[r["query"], r["passage"]] for r in rows]
    gold  = [float(r.get("score", 0.0)) for r in rows]
    pred  = model.predict(pairs, show_progress_bar=False)

    mae = statistics.mean(abs(p - g) for p, g in zip(pred, gold))

    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(pred, gold)
    except ImportError:
        corr = float("nan")

    # Use logit threshold 0.0 (not 0.5): raw logits are centred near 0, not 0-1.
    binary_correct = sum(1 for p, g in zip(pred, gold) if (p >= 0.0) == (g >= 0.5))
    accuracy = binary_correct / len(gold) if gold else 0.0

    pos_logits = [p for p, g in zip(pred, gold) if g >= 0.5]
    neg_logits = [p for p, g in zip(pred, gold) if g < 0.5]
    avg_pos = sum(pos_logits) / len(pos_logits) if pos_logits else float("nan")
    avg_neg = sum(neg_logits) / len(neg_logits) if neg_logits else float("nan")

    return {
        "mae": mae,
        "spearman": corr,
        "binary_accuracy": accuracy,
        "avg_pos_logit": avg_pos,
        "avg_neg_logit": avg_neg,
    }


# ── BPR training loop ─────────────────────────────────────────────────────────

def train_bpr(cross_encoder, triplets: list, dev_rows: list,
              epochs: int, batch_size: int, warmup_steps: int,
              output_path: str, lr: float = 2e-5,
              pos_penalty_weight: float = 0.1,
              early_stopping: bool = True, patience: int = 2):
    """
    BPR (Bayesian Personalised Ranking) pairwise training loop.

    loss = -log(sigmoid(pos_logit - neg_logit))
         + pos_penalty_weight * relu(-pos_logit).mean()

    The pos_penalty term prevents score collapse: it penalises the model when
    positive pair logits go negative, keeping them above 0 without forcing them
    to a specific value.

    Best checkpoint is saved based on dev Spearman (not train loss).
    """
    import torch
    import torch.nn as nn
    from transformers import get_linear_schedule_with_warmup

    hf_model = cross_encoder.model
    tokenizer = cross_encoder.tokenizer
    max_len   = cross_encoder.max_length
    device    = next(hf_model.parameters()).device

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    optimizer = torch.optim.AdamW(hf_model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(triplets) // batch_size + 1) * epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    def bpr_loss_fn(pos_logits, neg_logits):
        return -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-8).mean()

    def score_combined(queries, pos_passages, neg_passages):
        all_q = queries + queries
        all_p = pos_passages + neg_passages
        enc = tokenizer(
            all_q, all_p,
            padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = hf_model(**enc).logits.squeeze(-1)
        B = len(queries)
        return logits[:B], logits[B:]

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    Path(output_path).mkdir(parents=True, exist_ok=True)

    best_spearman   = float("-inf")
    no_improve_cnt  = 0
    steps_per_epoch = (len(triplets) + batch_size - 1) // batch_size
    log_every       = max(1, steps_per_epoch // 10)

    for epoch in range(epochs):
        random.shuffle(triplets)
        hf_model.train()
        epoch_loss = 0.0
        steps = 0

        batches = range(0, len(triplets), batch_size)
        if use_tqdm:
            batches = tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for i in batches:
            batch      = triplets[i : i + batch_size]
            queries    = [t[0] for t in batch]
            pos_pass   = [t[1] for t in batch]
            neg_pass   = [t[2] for t in batch]

            pos_logits, neg_logits = score_combined(queries, pos_pass, neg_pass)

            loss = bpr_loss_fn(pos_logits, neg_logits)

            # Pos-penalty: penalise when positive logits go below 0
            if pos_penalty_weight > 0:
                pos_penalty = torch.relu(-pos_logits).mean()
                loss = loss + pos_penalty_weight * pos_penalty

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
                print(f"  step {steps}/{steps_per_epoch}  "
                      f"loss={loss.item():.4f}  avg={epoch_loss/steps:.4f}", flush=True)

        avg_loss = epoch_loss / max(steps, 1)

        # Per-epoch Spearman on a dev sample (fast)
        hf_model.eval()
        dev_metrics = evaluate(cross_encoder, dev_rows, sample_n=200)
        sp = dev_metrics["spearman"]
        avg_pos = dev_metrics["avg_pos_logit"]
        avg_neg = dev_metrics["avg_neg_logit"]

        print(f"  Epoch {epoch+1}/{epochs}  "
              f"loss={avg_loss:.4f}  spearman={sp:.4f}  "
              f"binary_acc={dev_metrics['binary_accuracy']:.4f}  "
              f"avg_pos={avg_pos:.2f}  avg_neg={avg_neg:.2f}", flush=True)

        if sp > best_spearman:
            best_spearman = sp
            no_improve_cnt = 0
            cross_encoder.save(output_path)
            print(f"    → checkpoint saved (best spearman={sp:.4f})", flush=True)
        else:
            no_improve_cnt += 1
            print(f"    → no improvement ({no_improve_cnt}/{patience})", flush=True)
            if early_stopping and no_improve_cnt >= patience:
                print(f"  Early stopping triggered — restoring best checkpoint.", flush=True)
                break

    print(f"\nBest dev Spearman: {best_spearman:.4f}")
    return best_spearman


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune reranker on compliance golden data")
    parser.add_argument("--train",      required=True)
    parser.add_argument("--dev",        required=True)
    parser.add_argument("--output",     default="models/compliance-reranker")
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--lora",       action="store_true",
                        help="Use LoRA adapters (requires pip install peft). "
                             "Recommended for CPU runs.")
    parser.add_argument("--lora-r",     type=int, default=8)
    parser.add_argument("--loss",       default="bpr", choices=["bpr", "pairwise", "mse"],
                        help="bpr=BPR ranking loss (default, prevents score collapse); "
                             "pairwise=alias for bpr; "
                             "mse=pointwise regression (legacy)")
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--warmup-ratio",   type=float, default=0.1)
    parser.add_argument("--pos-penalty",    type=float, default=0.1,
                        help="Weight for positive-logit regulariser (keeps pos logits > 0). "
                             "Default=0.1. Set 0 to disable.")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Stop training if dev Spearman does not improve for --patience epochs.")
    parser.add_argument("--patience",       type=int,   default=2,
                        help="Early stopping patience (default=2 epochs).")
    parser.add_argument("--hard-neg-weight",type=float, default=1.5,
                        help="For mse mode: hard-neg duplication multiplier.")
    parser.add_argument("--max-negs-per-pos", type=int, default=3,
                        help="Max negatives per positive for BPR triplets. "
                             "Default=3 prevents score collapse. "
                             "Hard negatives are prioritised.")
    parser.add_argument("--max-length",     type=int,   default=256)
    parser.add_argument("--label-smoothing",type=float, default=0.0,
                        help="For mse mode only.")
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
    if args.lora and args.epochs > 0:
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
        trainable, total = (
            model.model.get_nb_trainable_parameters()
            if hasattr(model.model, "get_nb_trainable_parameters")
            else (
                sum(p.numel() for p in model.model.parameters() if p.requires_grad),
                sum(p.numel() for p in model.model.parameters()),
            )
        )
        print(f"\nLoRA enabled  rank={args.lora_r}  "
              f"trainable={trainable:,} / {total:,} params "
              f"({100 * trainable / total:.2f}%)")
    elif args.lora and args.epochs == 0:
        print("  (--lora ignored in eval-only mode)")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\nBaseline (before fine-tuning):")
    baseline = evaluate(model, dev_rows)
    print(f"  Spearman={baseline['spearman']:.4f}  "
          f"BinaryAcc={baseline['binary_accuracy']:.4f}  "
          f"avg_pos_logit={baseline['avg_pos_logit']:.2f}  "
          f"avg_neg_logit={baseline['avg_neg_logit']:.2f}")

    if args.epochs == 0:
        print("\n(eval-only mode — no training)")
        return

    output_path = str(Path(args.output).resolve())
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.loss in ("bpr", "pairwise"):
        print(f"\nBuilding BPR triplets  (max_negs_per_pos={args.max_negs_per_pos})...")
        triplets = make_triplets(train_rows, max_negs_per_pos=args.max_negs_per_pos)

        total_steps  = (len(triplets) // args.batch_size + 1) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        print(f"\nFine-tuning for {args.epochs} epochs  "
              f"(batch={args.batch_size}, lr={args.lr:.0e}, "
              f"warmup={warmup_steps}, total={total_steps})…")
        if args.early_stopping:
            print(f"  Early stopping: patience={args.patience} epochs (metric=Spearman)")
        if args.pos_penalty > 0:
            print(f"  Pos-penalty weight: {args.pos_penalty} (keeps positive logits > 0)")

        train_bpr(
            model, triplets, dev_rows,
            epochs=args.epochs,
            batch_size=args.batch_size,
            warmup_steps=warmup_steps,
            output_path=output_path,
            lr=args.lr,
            pos_penalty_weight=args.pos_penalty,
            early_stopping=args.early_stopping,
            patience=args.patience,
        )

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

    # ── Final save ────────────────────────────────────────────────────────────
    # For BPR mode the best checkpoint is already saved inside train_bpr().
    # For MSE mode model.fit() handles saving. We do a final explicit save here
    # to ensure the merged (non-PEFT) weights are always on disk.
    print(f"\nSaving final model to {output_path} ...")
    if args.lora:
        print("  Merging LoRA adapters into base weights...")
        model.model = model.model.merge_and_unload()
    model.save(output_path)

    saved_files = []
    for root, _, files in os.walk(output_path):
        for f in files:
            saved_files.append(os.path.relpath(os.path.join(root, f), output_path))
    if saved_files:
        print(f"  ✓ {len(saved_files)} file(s) saved:")
        for sf in saved_files[:10]:
            print(f"    {sf}")
        if len(saved_files) > 10:
            print(f"    ... and {len(saved_files) - 10} more")
    else:
        print("  WARNING: no files found — check disk space / permissions.")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nAfter fine-tuning (full dev set):")
    trained = evaluate(model, dev_rows)
    print(f"  Spearman={trained['spearman']:.4f}  "
          f"BinaryAcc={trained['binary_accuracy']:.4f}  "
          f"avg_pos_logit={trained['avg_pos_logit']:.2f}  "
          f"avg_neg_logit={trained['avg_neg_logit']:.2f}")

    print(f"\n  Δ Spearman   = {trained['spearman'] - baseline['spearman']:+.4f}  "
          f"(positive = improvement)")
    print(f"  Δ BinaryAcc  = {trained['binary_accuracy'] - baseline['binary_accuracy']:+.4f}")
    print(f"  Δ avg_pos    = {trained['avg_pos_logit'] - baseline['avg_pos_logit']:+.2f}  "
          f"(positive = pos logits moved up)")

    print(f"\nModel saved → {output_path}")
    print("\nTo use in the pipeline:")
    print(f"  export RERANKER_MODEL={output_path}")
    print("  python3 quick_start_compliance.py")


if __name__ == "__main__":
    main()
