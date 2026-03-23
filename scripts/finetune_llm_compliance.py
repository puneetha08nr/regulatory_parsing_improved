#!/usr/bin/env python3
"""
Fine-tune Llama 3.2 (1B or 3B) as a UAE IA compliance classifier using LoRA SFT.

Trains on data/07_golden_mapping/training_data/train.json and evaluates on dev.json.
Output: a single-word classifier (FA / PA / NA) using completion-only loss.

Model: meta-llama/Llama-3.2-3B-Instruct  (fallback: Llama-3.2-1B-Instruct)
LoRA: r=8, target=[q_proj, v_proj], ~0.06% trainable params
Output: models/compliance-llm-judge/

Class balancing:
  FA weight = 5.0    (rare — ~10% of data)
  PA weight = 15.0   (very rare — ~3% of data; raised from 5x to counteract PA starvation)
  NA weight = 1.0    (majority class)

Completion-only loss: gradient is computed ONLY on the response token (FA/PA/NA)
and the end-of-turn token. The full prompt (system + user) is masked as -100.

Usage:
  # GPU (recommended, ~30-60 min)
  python3 scripts/finetune_llm_compliance.py \\
      --base-model meta-llama/Llama-3.2-3B-Instruct \\
      --device cuda --batch-size 4 --epochs 3

  # Quick smoke test
  python3 scripts/finetune_llm_compliance.py --limit 50 --epochs 1

  # Use after training
  python3 scripts/llm_judge.py \\
      --use-finetuned --finetuned-model models/compliance-llm-judge \\
      --mappings single_policy_e2e/output/mappings.json
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a UAE IA compliance classifier.\n"
    "Classify if the policy passage addresses the control.\n"
    "Answer with exactly one word: FA, PA, or NA.\n"
    "FA = Fully Addressed\n"
    "PA = Partially Addressed\n"
    "NA = Not Addressed"
)

LABEL_FROM_STATUS = {
    "Fully Addressed":    "FA",
    "Fully addressed":    "FA",
    "Partially Addressed": "PA",
    "Partially addressed": "PA",
    "Not Addressed":      "NA",
    "Not addressed":      "NA",
}

LABEL_FROM_SCORE = {1.0: "FA", 0.7: "PA", 0.5: "PA", 0.0: "NA"}

CLASS_WEIGHTS = {"FA": 5.0, "PA": 15.0, "NA": 1.0}

STATUS_FROM_LABEL = {
    "FA": "Fully Addressed",
    "PA": "Partially Addressed",
    "NA": "Not Addressed",
}


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_rows(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def row_to_label(row: dict) -> str:
    label = LABEL_FROM_STATUS.get((row.get("label") or "").strip())
    if label:
        return label
    score = float(row.get("score", -1))
    return LABEL_FROM_SCORE.get(score, "NA")


def build_user_message(row: dict) -> str:
    control_id = row.get("control_id", "")
    query      = (row.get("query") or "").strip()
    passage    = (row.get("passage") or "").strip()
    return (
        f"Control {control_id}: {query[:200]}\n\n"
        f"Passage: {passage[:300]}"
    )


def render_messages(tokenizer, messages: list, add_generation_prompt: bool) -> str:
    """Render chat messages to a text prompt.

    Uses tokenizer chat template when available; otherwise falls back to a
    simple tagged format. This keeps the script compatible with models like
    Phi-* where not all tokenizers expose apply_chat_template.
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    # Minimal fallback format: keep system/user/assistant ordering deterministic.
    parts = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}\n")
        elif role == "user":
            parts.append(f"[USER]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}\n")
        else:
            parts.append(f"[{role.upper()}]\n{content}\n")

    if add_generation_prompt:
        parts.append("[ASSISTANT]\n")
    return "\n".join(parts).strip() + "\n"


# ── Dataset ───────────────────────────────────────────────────────────────────

class ComplianceDataset:
    """Tokenised training examples with completion-only label masking.

    For each row we build:
      prompt_text = chat_template([system, user], add_generation_prompt=True)
      full_text   = chat_template([system, user, assistant(label)], add_generation_prompt=False)

    input_ids = tokenise(full_text)
    labels    = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

    This ensures loss is computed only on the response token(s), not the prompt.
    """

    def __init__(self, rows: list, tokenizer, max_length: int = 512):
        import torch
        self.examples = []
        skipped = 0

        for row in rows:
            label = row_to_label(row)
            user_msg = build_user_message(row)

            prompt_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ]
            full_msgs = prompt_msgs + [{"role": "assistant", "content": label}]

            prompt_text = render_messages(
                tokenizer, prompt_msgs, add_generation_prompt=True
            )
            full_text = render_messages(
                tokenizer, full_msgs, add_generation_prompt=False
            )

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)

            if len(full_ids) > max_length:
                skipped += 1
                continue

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

            self.examples.append({
                "input_ids":      torch.tensor(full_ids,   dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                "labels":         torch.tensor(labels,     dtype=torch.long),
                "class_label":    label,
            })

        print(f"  Dataset: {len(self.examples)} examples  ({skipped} skipped — too long)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch: list) -> dict:
    """Pad a batch of variable-length sequences."""
    import torch
    max_len = max(ex["input_ids"].shape[0] for ex in batch)

    input_ids  = []
    attn_masks = []
    labels_out = []

    for ex in batch:
        pad_len = max_len - ex["input_ids"].shape[0]
        input_ids.append(torch.cat([
            ex["input_ids"],
            torch.zeros(pad_len, dtype=torch.long),
        ]))
        attn_masks.append(torch.cat([
            ex["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long),
        ]))
        labels_out.append(torch.cat([
            ex["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),
        ]))

    return {
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attn_masks),
        "labels":         torch.stack(labels_out),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, dev_rows: list, device, max_length: int = 512) -> dict:
    """Run greedy inference on dev set; return per-class P/R/F1 and accuracy."""
    import torch

    model.eval()
    # Some models (e.g., Phi-3) can error in generate() when KV-cache is used
    # with PEFT/wrappers. Force cache off for stable greedy decoding.
    if hasattr(model, "config"):
        model.config.use_cache = False
    preds, golds = [], []

    for row in dev_rows:
        gold = row_to_label(row)
        user_msg = build_user_message(row)

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        prompt = render_messages(tokenizer, msgs, add_generation_prompt=True)
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        pred_text  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()

        # Extract first word token
        pred = "NA"
        for tok in pred_text.split():
            if tok in ("FA", "PA", "NA"):
                pred = tok
                break

        preds.append(pred)
        golds.append(gold)

    # Per-class metrics
    classes = ["FA", "PA", "NA"]
    report = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(preds, golds) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(preds, golds) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(preds, golds) if p != cls and g == cls)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        report[cls] = {"precision": prec, "recall": recall, "f1": f1,
                       "support": sum(1 for g in golds if g == cls)}

    accuracy = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds) if golds else 0.0
    report["accuracy"] = accuracy
    report["predictions"] = Counter(preds)
    report["gold_dist"]   = Counter(golds)

    model.train()
    return report


def print_eval(report: dict):
    print(f"\n  {'Class':<6}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}  {'Support':>7}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*7}")
    for cls in ("FA", "PA", "NA"):
        r = report[cls]
        print(f"  {cls:<6}  {r['precision']:>9.3f}  {r['recall']:>7.3f}  "
              f"{r['f1']:>6.3f}  {r['support']:>7}")
    print(f"\n  Accuracy  : {report['accuracy']:.3f}")
    print(f"  Predicted : {dict(report['predictions'])}")
    print(f"  Gold dist : {dict(report['gold_dist'])}")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            get_linear_schedule_with_warmup,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install transformers peft accelerate")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = args.train
    if args.clean:
        train_path = args.train.replace("train.json", "train_clean.json")
        print(f"--clean: using {train_path}")
    print(f"Loading training data from {train_path} ...")
    train_rows = load_rows(train_path)
    dev_rows   = load_rows(args.dev)

    if args.limit:
        train_rows = train_rows[:args.limit]
        print(f"  Limited to {args.limit} examples (--limit)")

    dist = Counter(row_to_label(r) for r in train_rows)
    print(f"  Train: {len(train_rows)} rows  dist={dict(dist)}")
    print(f"  Dev  : {len(dev_rows)} rows")

    # ── Load model & tokenizer (with gated-model fallbacks) ───────────────
    hf_token   = os.environ.get("HF_TOKEN")
    device     = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    # Use {"": 0} instead of "auto" to force all layers onto cuda:0.
    # "auto" can shard the model across CPU+GPU which breaks batch.to(device).
    device_map = {"": 0} if device.type == "cuda" else None

    models_to_try = [args.base_model]
    if args.base_model_fallback:
        models_to_try.append(args.base_model_fallback)
    if args.extra_fallback_model:
        models_to_try.append(args.extra_fallback_model)

    last_err = None
    tokenizer = None
    model = None

    for m in models_to_try:
        try:
            print(f"\nTrying base model: {m}")
            print(f"  Loading tokenizer: {m} ...")
            tokenizer = AutoTokenizer.from_pretrained(
                m, token=hf_token, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            print(f"  Loading model: {m}  (device={device}) ...")
            load_kwargs: dict = dict(token=hf_token, trust_remote_code=True)
            if device.type == "cuda":
                load_kwargs["torch_dtype"] = torch.float16
                if device_map:
                    load_kwargs["device_map"] = device_map
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    print("  4-bit quantisation enabled")
                except Exception:
                    pass
            else:
                load_kwargs["torch_dtype"] = torch.float32

            model = AutoModelForCausalLM.from_pretrained(m, **load_kwargs)
            if device.type == "cpu" or device_map is None:
                model = model.to(device)
            model.config.use_cache = False

            print(f"  ✓ Loaded model successfully: {m}")
            args.base_model = m  # for metadata & downstream use
            break
        except Exception as e:
            last_err = e
            print(f"  ✗ Failed loading {m}: {type(e).__name__}: {e}")

    if model is None or tokenizer is None:
        raise RuntimeError(
            "Could not load any candidate base model. Last error:\n"
            f"{last_err}"
        )

    def detect_lora_target_modules(mdl) -> list[str]:
        """Detect attention projection module names for LoRA injection.

        PEFT matches `target_modules` against module names (typically suffixes).
        Different model families use different naming conventions (e.g. Llama
        uses q_proj/v_proj; Phi-3 often uses qkv_proj).
        """
        # Candidates ordered by preference (keep small).
        candidates = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "qkv_proj",
            # extra common names
            "query_proj", "value_proj",
        ]
        module_names = [name for name, _ in mdl.named_modules()]

        present = []
        for c in candidates:
            if any(n.endswith(c) or f".{c}." in n or n.endswith(f"/{c}") for n in module_names):
                present.append(c)

        # Prefer q_proj/v_proj pair when both exist.
        if "q_proj" in present and "v_proj" in present:
            return ["q_proj", "v_proj"]
        # Otherwise prefer qkv_proj if available.
        if "qkv_proj" in present:
            return ["qkv_proj"]
        # Fall back to whatever we found (at least 1).
        if present:
            # Keep list short to reduce trainable params.
            # Prefer query/key/value over output.
            preferred = [x for x in present if x in ("q_proj", "k_proj", "v_proj", "qkv_proj")]
            return preferred[:2] if preferred else present[:2]

        raise RuntimeError(
            "Could not auto-detect LoRA target modules for this base model. "
            "Try passing --lora-target-modules explicitly."
        )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    if args.lora_target_modules:
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    else:
        target_modules = detect_lora_target_modules(model)

    print(f"LoRA target_modules={target_modules}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = (
        model.get_nb_trainable_parameters()
        if hasattr(model, "get_nb_trainable_parameters")
        else (
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()),
        )
    )
    print(f"LoRA r={args.lora_r}  "
          f"trainable={trainable:,} / {total:,}  ({100*trainable/max(total,1):.3f}%)")

    # ── Dataset + WeightedRandomSampler ───────────────────────────────────────
    print("\nBuilding dataset ...")
    dataset = ComplianceDataset(train_rows, tokenizer, max_length=args.max_length)

    sample_weights = [CLASS_WEIGHTS.get(ex["class_label"], 1.0)
                      for ex in dataset.examples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    print(f"  Sampler weights — FA={CLASS_WEIGHTS['FA']}× PA={CLASS_WEIGHTS['PA']}× "
          f"NA={CLASS_WEIGHTS['NA']}×")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps  = len(loader) * args.epochs // args.grad_accum
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_fp16  = (device.type == "cuda")
    scaler    = torch.amp.GradScaler("cuda") if use_fp16 else None

    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nFine-tuning for {args.epochs} epochs  "
          f"(batch={args.batch_size}  grad_accum={args.grad_accum}  "
          f"lr={args.lr}  warmup={warmup_steps}/{total_steps} steps)")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_fa_recall  = -1.0
    best_epoch      = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        steps = 0

        try:
            from tqdm import tqdm
            batches = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        except ImportError:
            batches = loader

        for step_idx, batch in enumerate(batches):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_fp16:
                with torch.amp.autocast("cuda"):
                    out  = model(**batch)
                    loss = out.loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                out  = model(**batch)
                loss = out.loss / args.grad_accum
                loss.backward()

            epoch_loss += loss.item() * args.grad_accum

            if (step_idx + 1) % args.grad_accum == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1

            if hasattr(batches, "set_postfix"):
                batches.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}")

        avg_loss = epoch_loss / max(len(loader), 1)
        ckpt_dir = output_path / f"checkpoint-epoch{epoch+1}"
        ckpt_dir.mkdir(exist_ok=True)

        # Per-epoch evaluation
        print(f"\n  Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")
        print(f"  Evaluating on dev set ({len(dev_rows)} rows) ...")
        report = evaluate(model, tokenizer, dev_rows, device, args.max_length)
        print_eval(report)

        fa_recall = report["FA"]["recall"]
        if fa_recall >= best_fa_recall:
            best_fa_recall = fa_recall
            best_epoch     = epoch + 1
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"    → checkpoint saved (best FA recall={fa_recall:.3f})")

    # ── Save final model (best checkpoint merged) ─────────────────────────────
    print(f"\nBest epoch: {best_epoch}  FA recall={best_fa_recall:.3f}")
    best_ckpt = output_path / f"checkpoint-epoch{best_epoch}"

    print(f"Merging LoRA adapters from best checkpoint and saving to {output_path} ...")
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=hf_token, trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    peft_model  = PeftModel.from_pretrained(base_model, str(best_ckpt))
    merged      = peft_model.merge_and_unload()
    merged.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Write a flag file so llm_judge.py knows this is a single-word classifier
    (output_path / "judge_format.json").write_text(
        json.dumps({"format": "single_word", "labels": ["FA", "PA", "NA"],
                    "base_model": args.base_model}, indent=2)
    )
    print(f"  ✓ Model saved → {output_path}")
    print(f"\nTo use as judge:")
    print(f"  python3 scripts/llm_judge.py \\")
    print(f"      --use-finetuned \\")
    print(f"      --finetuned-model {output_path} \\")
    print(f"      --mappings single_policy_e2e/output/mappings.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune Llama 3.2 as a UAE IA compliance classifier (FA/PA/NA)"
    )
    ap.add_argument("--train",      default="data/07_golden_mapping/training_data/train.json")
    ap.add_argument("--clean",      action="store_true",
                    help="Use train_clean.json (deduped + NA-capped) instead of train.json. "
                         "Run scripts/fix_dataset.py first to generate it.")
    ap.add_argument("--dev",        default="data/07_golden_mapping/training_data/dev.json")
    ap.add_argument("--output",     default="models/compliance-llm-judge")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct",
                    help="HF model ID. Fallback: meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--base-model-fallback", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="Fallback if the primary model is gated/unavailable.")
    ap.add_argument("--extra-fallback-model", default="microsoft/Phi-3-mini-4k-instruct",
                    help="Final fallback if Meta-Llama access is not available.")
    ap.add_argument("--device",     default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto = use CUDA if available (default)")
    ap.add_argument("--epochs",     type=int,   default=3)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--grad-accum", type=int,   default=4,
                    help="Gradient accumulation steps (effective batch = batch × accum)")
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r",     type=int,   default=8)
    ap.add_argument(
        "--lora-target-modules",
        default="",
        help="Comma-separated module name suffixes for LoRA injection "
             "(e.g. q_proj,v_proj). If empty, auto-detects from the model."
    )
    ap.add_argument("--max-length", type=int,   default=512)
    ap.add_argument("--limit",      type=int,   default=None,
                    help="Use only first N training rows (for quick tests)")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
