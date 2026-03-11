#!/usr/bin/env python3
"""
Phase 3B — Fine-tune LLaMA 3.2 1B as a UAE IA Compliance Judge
===============================================================
Uses PEFT LoRA + TRL SFTTrainer to instruction-tune Llama-3.2-1B-Instruct
on (control, passage) → FA/PA/NA verdict pairs.

After fine-tuning, the model is saved to models/compliance-llm-judge/ and
can be loaded directly from HuggingFace (no Ollama required) via llm_judge.py
with the --use-finetuned flag.

Model requirements:
  - HuggingFace model: meta-llama/Llama-3.2-1B-Instruct  (~2.5 GB download)
  - RAM: ~4 GB (LoRA keeps only ~1M params trainable)
  - HuggingFace token required for gated model access:
      huggingface-cli login
      OR set HF_TOKEN env var

Training data:
  - Real golden pairs: data/07_golden_mapping/golden_mapping_dataset.json
  - Synthetic pairs (optional): data/07_golden_mapping/synthetic_pairs.json

Training format (Alpaca-style instruction tuning):
  {
    "instruction": "UAE IA compliance auditor system prompt",
    "input":  "Control T2.2.1: <text>\\n\\nPolicy passage:\\n<text>",
    "output": "Fully Addressed | The passage explicitly states..."
  }

LoRA configuration:
  - Rank: 8  (targets q_proj, v_proj only — ~800K trainable params)
  - Training 0.06% of total model parameters
  - Merged back into base weights before saving

Usage:
  # Install dependencies first
  pip install transformers peft trl datasets accelerate bitsandbytes

  # Full training on real + synthetic data (~4 hrs CPU, ~30 min GPU)
  python3 scripts/finetune_llm_compliance.py

  # Quick test on 50 examples
  python3 scripts/finetune_llm_compliance.py --limit 50 --epochs 1

  # GPU run (faster)
  python3 scripts/finetune_llm_compliance.py --device cuda --batch-size 4

  # Use fine-tuned model for judging (after training)
  python3 scripts/llm_judge.py --use-finetuned \\
      --finetuned-model models/compliance-llm-judge

  # Alternative: use a locally available model (no HF token needed)
  python3 scripts/finetune_llm_compliance.py \\
      --base-model microsoft/Phi-3-mini-4k-instruct
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── System prompt (identical to llm_judge.py CoT style) ──────────────────────

SYSTEM_PROMPT = """You are a compliance auditor specialising in UAE Information Assurance \
(UAE IA) Regulation. Your task is to assess whether a policy passage addresses a specific \
compliance control.

Label definitions:
- Fully Addressed    : passage explicitly and completely covers the control — no gaps.
- Partially Addressed: passage is relevant but leaves requirements unmet or only implied.
- Not Addressed      : passage is off-topic or too generic to satisfy the control.

Rules:
- Vague statements ("we follow best practices") = Not Addressed.
- Topic mentioned but no HOW/WHAT details = Partially Addressed at best.
- When in doubt between Fully and Partially → choose Partially Addressed.

Respond with the verdict in this format:
  LABEL | one-sentence reason"""

# ── Training example templates ────────────────────────────────────────────────

def make_user_input(control_id: str, control_text: str, passage_text: str) -> str:
    return (
        f"Control ID: {control_id}\n"
        f"Control requirement:\n{control_text[:600]}\n\n"
        f"Policy passage:\n{passage_text[:1200]}"
    )


def status_to_label(status: str) -> str:
    """Normalise compliance_status to one of the three canonical labels."""
    s = status.strip().lower()
    if "fully" in s:
        return "Fully Addressed"
    if "partially" in s:
        return "Partially Addressed"
    return "Not Addressed"


def make_target_output(status: str, reason: str = "") -> str:
    label = status_to_label(status)
    if reason:
        return f"{label} | {reason}"
    # Generate a minimal reason if none is recorded
    defaults = {
        "Fully Addressed":    "The passage explicitly and completely satisfies the control requirement.",
        "Partially Addressed": "The passage is relevant but does not fully address all control requirements.",
        "Not Addressed":      "The passage does not address the control requirement.",
    }
    return f"{label} | {defaults[label]}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_golden(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_examples(records: list, is_synthetic: bool = False) -> list:
    """Convert golden/synthetic records into instruction-tuning examples."""
    examples = []
    for r in records:
        status = (r.get("compliance_status") or "").strip()
        if not status:
            continue
        ctrl_id   = r.get("control_id", "")
        ctrl_text = (r.get("control_text_snippet") or "").strip()
        passage   = (r.get("policy_text_snippet") or "").strip()
        reason    = (r.get("evidence_or_notes") or r.get("comments") or "").strip()

        if not ctrl_text or not passage:
            continue
        if len(passage.split()) < 8:
            continue

        examples.append({
            "instruction": SYSTEM_PROMPT,
            "input":       make_user_input(ctrl_id, ctrl_text, passage),
            "output":      make_target_output(status, reason),
            "control_id":  ctrl_id,
            "status":      status_to_label(status),
            "is_synthetic": is_synthetic,
        })
    return examples


def format_prompt(example: dict, tokenizer) -> str:
    """Format as a chat template if the tokenizer supports it, else Alpaca."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system",    "content": example["instruction"]},
            {"role": "user",      "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    # Alpaca fallback
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, examples: list):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers peft trl datasets accelerate")
        sys.exit(1)

    print(f"\nLoading tokenizer: {args.base_model} ...")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {args.base_model} ...")
    device_map = "auto" if args.device == "cuda" else "cpu"
    load_kwargs = dict(
        token=hf_token,
        trust_remote_code=True,
        device_map=device_map,
    )
    # Use 4-bit quantization on GPU to fit in ~8 GB VRAM
    if args.device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("  4-bit quantization enabled (GPU)")
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters() if hasattr(
        model, "get_nb_trainable_parameters") else (
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters()))
    print(f"LoRA rank={args.lora_r}  "
          f"trainable={trainable:,} / {total:,} ({100*trainable/max(total,1):.3f}%)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    formatted = [format_prompt(ex, tokenizer) for ex in examples]
    dataset = Dataset.from_dict({"text": formatted})
    print(f"\nDataset: {len(dataset)} examples")
    print(f"Sample prompt (first 300 chars):\n{formatted[0][:300]}\n...")

    # ── Training arguments ────────────────────────────────────────────────────
    output_path = str(Path(args.output).resolve())
    Path(output_path).mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=(args.device == "cuda"),
        max_seq_length=args.max_length,
        dataset_text_field="text",
        report_to="none",
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    print(f"Training for {args.epochs} epochs  "
          f"(batch={args.batch_size}, lr={args.lr}, device={args.device}) ...")
    trainer.train()

    # ── Save (merge LoRA into base weights) ───────────────────────────────────
    print(f"\nMerging LoRA adapters and saving to {output_path} ...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Write model card
    card = Path(output_path) / "MODEL_CARD.md"
    card.write_text(
        f"# UAE IA Compliance LLM Judge\n\n"
        f"Fine-tuned from `{args.base_model}` on UAE IA compliance mapping pairs.\n\n"
        f"## Training\n"
        f"- LoRA rank: {args.lora_r}\n"
        f"- Epochs: {args.epochs}\n"
        f"- Examples: {len(examples)}\n"
        f"- Format: Alpaca instruction tuning (FA/PA/NA verdicts)\n\n"
        f"## Usage\n"
        f"```bash\n"
        f"python3 scripts/llm_judge.py --use-finetuned --finetuned-model {output_path}\n"
        f"```\n"
    )
    print(f"  Saved to: {output_path}")

    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune LLM as a UAE IA compliance judge (LoRA SFT)"
    )
    ap.add_argument("--golden",      default="data/07_golden_mapping/golden_mapping_dataset.json",
                    help="Human-annotated golden pairs")
    ap.add_argument("--synthetic",   default=None,
                    help="Synthetic pairs JSON (from generate_synthetic_pairs.py). "
                         "If provided, added to training data.")
    ap.add_argument("--output",      default="models/compliance-llm-judge",
                    help="Output directory for the fine-tuned model")
    ap.add_argument("--base-model",  default="meta-llama/Llama-3.2-1B-Instruct",
                    help="HuggingFace model ID or local path. "
                         "Alternatives: microsoft/Phi-3-mini-4k-instruct (no gating), "
                         "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (smaller)")
    ap.add_argument("--device",      default="cpu", choices=["cpu", "cuda"],
                    help="Device to train on (default: cpu)")
    ap.add_argument("--epochs",      type=int,   default=3)
    ap.add_argument("--batch-size",  type=int,   default=1,
                    help="Per-device batch size (default: 1 for CPU)")
    ap.add_argument("--lr",          type=float, default=2e-4,
                    help="Learning rate (default: 2e-4)")
    ap.add_argument("--lora-r",      type=int,   default=8,
                    help="LoRA rank (default: 8 — ~800K trainable params)")
    ap.add_argument("--max-length",  type=int,   default=512,
                    help="Max token length per example (default: 512)")
    ap.add_argument("--real-weight", type=int,   default=3,
                    help="Duplicate real golden rows this many times (default: 3)")
    ap.add_argument("--limit",       type=int,   default=None,
                    help="Only use first N examples (for quick testing)")
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # ── Load and convert data ─────────────────────────────────────────────────
    print(f"Loading golden data from {args.golden} ...")
    golden_records = load_golden(args.golden)
    real_examples = build_examples(golden_records, is_synthetic=False)
    print(f"  {len(golden_records)} records → {len(real_examples)} training examples")

    from collections import Counter
    status_dist = Counter(ex["status"] for ex in real_examples)
    print(f"  Status distribution: {dict(status_dist)}")

    # Weight real examples so they dominate over synthetic
    all_examples = real_examples * args.real_weight

    if args.synthetic:
        print(f"\nLoading synthetic data from {args.synthetic} ...")
        syn_records = load_golden(args.synthetic)
        syn_examples = build_examples(syn_records, is_synthetic=True)
        print(f"  {len(syn_records)} records → {len(syn_examples)} synthetic examples")
        syn_dist = Counter(ex["status"] for ex in syn_examples)
        print(f"  Status distribution: {dict(syn_dist)}")
        all_examples = all_examples + syn_examples

    random.shuffle(all_examples)

    if args.limit:
        all_examples = all_examples[:args.limit]
        print(f"\nLimited to {args.limit} examples (--limit)")

    print(f"\nTotal training examples: {len(all_examples)}")
    print(f"  Real × {args.real_weight}: {len(real_examples) * args.real_weight}")
    if args.synthetic:
        print(f"  Synthetic:              {len(syn_examples)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    output_path = train(args, all_examples)

    print(f"\n{'='*60}")
    print(f"LLM compliance judge fine-tuning complete!")
    print(f"  Model saved to: {output_path}")
    print(f"\nTo use the fine-tuned judge:")
    print(f"  python3 scripts/llm_judge.py \\")
    print(f"      --use-finetuned \\")
    print(f"      --finetuned-model {output_path}")
    print(f"\nTo evaluate after judging:")
    print(f"  python3 scripts/llm_judge.py \\")
    print(f"      --use-finetuned \\")
    print(f"      --finetuned-model {output_path} \\")
    print(f"      --evaluate")


if __name__ == "__main__":
    main()
