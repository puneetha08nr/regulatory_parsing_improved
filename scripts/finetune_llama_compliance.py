#!/usr/bin/env python3
"""
Fine-tune Llama 3.2 (1B or 3B) on the compliance classification dataset.

Uses Unsloth for 4-bit QLoRA — fits on a free Colab T4 GPU.
Only trains LoRA adapters (~1% of parameters).

Requirements (install on Colab first):
  pip install unsloth trl datasets scipy -q

Usage (Colab, T4 GPU, ~45-60 min):
  python3 scripts/finetune_llama_compliance.py \
    --train data/07_golden_mapping/llm_training_data/train.jsonl \
    --dev   data/07_golden_mapping/llm_training_data/dev.jsonl \
    --output models/llama-compliance

If llm_training_data/ doesn't exist yet, run first:
  python3 scripts/prepare_llm_training_data.py \
    --train data/07_golden_mapping/training_data/train.json \
    --dev   data/07_golden_mapping/training_data/dev.json \
    --output data/07_golden_mapping/llm_training_data

Model size options:
  --base-model unsloth/Llama-3.2-1B-Instruct   (1B, fits on T4, ~45 min)
  --base-model unsloth/Llama-3.2-3B-Instruct   (3B, needs A100, ~90 min)
  --base-model unsloth/Llama-3.2-1B-Instruct-bnb-4bit  (pre-quantized, loads faster)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SYSTEM_PROMPT = (
    "You are a compliance analyst specialising in the UAE Information Assurance (UAE IA) "
    "regulatory framework. Given a regulatory control and a policy passage, determine "
    "whether the passage addresses the control requirement. "
    "Respond with exactly one of: Fully Addressed, Partially Addressed, Not Addressed. "
    "Follow with a single sentence explaining your reasoning."
)


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_conversation(row: dict, tokenizer) -> dict:
    """Apply the model's chat template so loss is only on assistant tokens."""
    messages = []
    for turn in row["conversations"]:
        role = "assistant" if turn["from"] == "gpt" else turn["from"]
        messages.append({"role": role, "content": turn["value"]})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def evaluate_classification(model, tokenizer, dev_rows: list, n_samples: int = 50) -> dict:
    """Quick label accuracy on a sample of dev examples."""
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    correct = 0
    label_map = {
        "fully": "Fully Addressed",
        "partially": "Partially Addressed",
        "not": "Not Addressed",
    }

    samples = dev_rows[:n_samples]
    for row in samples:
        gold_label = row["metadata"]["label"]
        convo = row["conversations"]
        messages = [
            {"role": "system" if t["from"] == "system" else
             ("user" if t["from"] == "human" else "skip"), "content": t["value"]}
            for t in convo if t["from"] != "gpt"
        ]
        messages = [m for m in messages if m["role"] != "skip"]

        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=20, temperature=0.01,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip().lower()

        predicted = None
        for key, label in label_map.items():
            if response.startswith(key):
                predicted = label
                break

        if predicted == gold_label:
            correct += 1

    FastLanguageModel.for_training(model)
    return {"label_accuracy": correct / len(samples), "n_samples": len(samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument("--dev", required=True, help="Path to dev.jsonl")
    parser.add_argument("--output", default="models/llama-compliance")
    parser.add_argument(
        "--base-model",
        default="unsloth/Llama-3.2-1B-Instruct",
        help="Unsloth model ID (default: Llama-3.2-1B-Instruct)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (default: 4, use 2 if OOM)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch*accum)")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank r (16 = good balance of size/quality)")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    # ── Check dependencies ───────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        import torch
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install unsloth trl datasets -q")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading train data: {args.train}")
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)

    label_counts = {}
    for r in train_rows:
        lbl = r["metadata"]["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"  Train: {len(train_rows)} rows | Dev: {len(dev_rows)} rows")
    print(f"  Label distribution: {label_counts}")

    # ── Load model (4-bit QLoRA) ─────────────────────────────────────────────
    print(f"\nLoading {args.base_model} in 4-bit …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # Attach LoRA adapters — only these are trained (~1% of params)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    # ── Format data with chat template ───────────────────────────────────────
    def _fmt(row):
        return format_conversation(row, tokenizer)

    train_dataset = Dataset.from_list(train_rows).map(_fmt, remove_columns=["conversations", "metadata"])
    dev_dataset = Dataset.from_list(dev_rows).map(_fmt, remove_columns=["conversations", "metadata"])

    # ── Baseline accuracy ─────────────────────────────────────────────────────
    print("\nBaseline label accuracy (before fine-tuning, 50 dev samples):")
    baseline = evaluate_classification(model, tokenizer, dev_rows, n_samples=50)
    print(f"  {baseline['label_accuracy']*100:.1f}%")

    # ── Train ─────────────────────────────────────────────────────────────────
    out_dir = str(Path(args.output).resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    warmup_steps = max(1, int(len(train_dataset) / args.batch_size * 0.1))
    eff_batch = args.batch_size * args.grad_accum

    print(f"\nTraining: {args.epochs} epochs, "
          f"batch={args.batch_size} × accum={args.grad_accum} (effective={eff_batch}), "
          f"warmup={warmup_steps} steps …")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            seed=42,
        ),
    )

    trainer_stats = trainer.train()
    print(f"\nTraining complete. "
          f"Steps: {trainer_stats.global_step}, "
          f"Loss: {trainer_stats.training_loss:.4f}")

    # ── Post-training accuracy ────────────────────────────────────────────────
    print("\nPost-training label accuracy (50 dev samples):")
    trained_eval = evaluate_classification(model, tokenizer, dev_rows, n_samples=50)
    delta = trained_eval["label_accuracy"] - baseline["label_accuracy"]
    print(f"  {trained_eval['label_accuracy']*100:.1f}%  (Δ {delta*100:+.1f}pp vs baseline)")

    # ── Save LoRA adapters ────────────────────────────────────────────────────
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nLoRA adapters saved → {out_dir}")
    print("Size: ~50–100 MB (adapters only — base model is loaded from HuggingFace at inference)")
    print("\nTo run inference:")
    print(f"  from unsloth import FastLanguageModel")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{out_dir}', load_in_4bit=True)")


if __name__ == "__main__":
    main()
