#!/usr/bin/env python3
"""
Per-atom binary compliance judge — unsloth 4-bit QLoRA fine-tune (GPU required).

Trains a small Llama-3.2 model to answer: "Does this passage satisfy the obligation?"
(yes / no), one atom at a time. Outputs are aggregated to pair-level FA/PA/NA to
match grade_judge.py's expected format.

Designed to run in Google Colab (T4 is enough for 1B; A10G for 3B).

SETUP (Colab cell 1):
    !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install -q trl peft accelerate bitsandbytes

USAGE:
    python scripts/finetune_atom_compliance.py --train data/12_atoms/atoms_train.jsonl \
        --test  data/12_atoms/atoms_test.jsonl \
        --pairs data/12_atoms/pairs_eval.json \
        --out   data/10_judge/answer_key.judged_llama.json \
        [--model "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"]
        [--epochs 3]
        [--save_path /content/atom_judge_adapter]

After training the adapter is saved to --save_path.  Re-run inference-only with:
    python scripts/finetune_atom_compliance.py --infer_only \
        --adapter /content/atom_judge_adapter ...
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset

FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"
MIN_HYP_LEN = 25   # filter degenerate atom hypotheses


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def make_prompt(premise, hypothesis):
    """Instruction prompt for the binary atom coverage task."""
    return (
        "You are a compliance analyst.\n"
        "Read the policy passage below and decide whether it satisfies the stated obligation.\n\n"
        f"### Passage\n{premise[:1800]}\n\n"
        f"### Obligation\n{hypothesis}\n\n"
        "### Question\nDoes the passage satisfy this obligation? Answer yes or no.\n\n"
        "### Answer\n"
    )


def format_row(row, eos):
    """Return (prompt_str, full_str) for SFT training."""
    label = "yes" if row["covered"] else "no"
    prompt = make_prompt(row["premise"], row["hypothesis"])
    return prompt, prompt + label + eos


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,         # auto (bfloat16 on Ampere+, float16 otherwise)
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    eos = tokenizer.eos_token or ""
    rows = [r for r in load_jsonl(args.train) if len(r["hypothesis"]) >= MIN_HYP_LEN]
    print(f"Training on {len(rows)} atoms (of {sum(1 for _ in load_jsonl(args.train))} total; "
          f"{sum(r['covered'] for r in rows)} covered)")

    # class-balance: oversample covered (minority class ~22%)
    pos = [r for r in rows if r["covered"]]
    neg = [r for r in rows if not r["covered"]]
    factor = max(1, len(neg) // max(len(pos), 1))
    balanced = pos * factor + neg
    print(f"Balanced: {len(pos)} pos x{factor} + {len(neg)} neg = {len(balanced)} rows")

    texts = [format_row(r, eos)[1] for r in balanced]
    train_dataset = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            output_dir="output_atom_judge",
            report_to="none",
        ),
    )
    trainer.train()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"\nAdapter saved -> {args.save_path}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(model, tokenizer, rows, batch_size=8):
    """Returns {(pair_id, atom_key): covered_bool} for all rows."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    results = {}
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        prompts = [make_prompt(r["premise"], r["hypothesis"]) for r in chunk]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=2048).to(model.device)
        out = model.generate(**inputs, max_new_tokens=3, temperature=1.0, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
        for j, r in enumerate(chunk):
            gen = tokenizer.decode(out[j, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip().lower()
            results[(r["pair_id"], r["atom_key"])] = gen.startswith("yes")
    return results


def aggregate_to_pairs(atom_results, pairs):
    """Roll per-atom boolean coverage up to pair-level FA/PA/NA."""
    labels = {}
    for p in pairs:
        pid = p["pair_id"]
        if not p["atoms"]:
            # atomless: use _whole if present, else skip
            cov = atom_results.get((pid, "_whole"), False)
            labels[pid] = PA if cov else NA
        else:
            covered_count = sum(
                1 for a in p["atoms"] if atom_results.get((pid, a["key"]), False)
            )
            n = len(p["atoms"])
            labels[pid] = FA if covered_count == n else (PA if covered_count > 0 else NA)
    return labels


def evaluate_and_write(atom_results, pairs, out_path):
    pair_labels = aggregate_to_pairs(atom_results, pairs)
    results = []
    for p in pairs:
        pid = p["pair_id"]
        results.append({
            "control_id": p["control_id"],
            "passage_id": p["passage_id"],
            "human_label": p["v2_label"],
            "ai_label": pair_labels.get(pid, NA),
            "ai_reason": "per-atom Llama binary coverage",
        })

    # quick metrics
    POS = {FA, PA}
    test_pairs = [p for p in pairs if p["split"] == "test"]
    test_set = {p["pair_id"] for p in test_pairs}
    test_res = [r for r in results if f"{r['control_id']}__{r['passage_id']}" in test_set]
    agree = sum(1 for r in test_res if r["human_label"] == r["ai_label"])
    tp = sum(1 for r in test_res if r["human_label"] in POS and r["ai_label"] in POS)
    fp = sum(1 for r in test_res if r["human_label"] not in POS and r["ai_label"] in POS)
    fn = sum(1 for r in test_res if r["human_label"] in POS and r["ai_label"] not in POS)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"\nTEST n={len(test_res)}  agree={agree/len(test_res):.1%}  posF1={f1:.2f} (P={prec:.2f}/R={rec:.2f})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": "llama-atom-judge", "results": results},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"-> {out_path}")
    print("  grade: python scripts/grade_judge.py --judged", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/12_atoms/atoms_train.jsonl")
    ap.add_argument("--test",  default="data/12_atoms/atoms_test.jsonl")
    ap.add_argument("--pairs", default="data/12_atoms/pairs_eval.json")
    ap.add_argument("--out",   default="data/10_judge/answer_key.judged_llama.json")
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--save_path", default="atom_judge_adapter")
    ap.add_argument("--infer_only", action="store_true",
                    help="skip training, load from --adapter path")
    ap.add_argument("--adapter", default=None,
                    help="path to saved adapter for --infer_only mode")
    args = ap.parse_args()

    pairs = json.load(open(args.pairs, encoding="utf-8"))

    if args.infer_only:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter or args.save_path,
            max_seq_length=2048, dtype=None, load_in_4bit=True,
        )
    else:
        model, tokenizer = train(args)

    # run inference on test atoms (for evaluation)
    test_rows = [r for r in load_jsonl(args.test) if len(r["hypothesis"]) >= MIN_HYP_LEN]
    train_rows = [r for r in load_jsonl(args.train) if len(r["hypothesis"]) >= MIN_HYP_LEN]
    all_rows = train_rows + test_rows
    print(f"\nRunning inference on {len(all_rows)} atoms ...")
    atom_results = infer(model, tokenizer, all_rows)
    evaluate_and_write(atom_results, pairs, args.out)


if __name__ == "__main__":
    main()
