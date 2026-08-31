#!/usr/bin/env python3
"""
Fine-tune Llama on Work 3 healthcare atom compliance pairs.

Takes the gold_pairs.json (3,800 pairs, 329 positives) and trains a QLoRA adapter.
After training, runs inference on the full set and reports posF1 vs human gold.

Designed for Lightning.ai / Colab (T4 is sufficient for 1B).

SETUP (run once):
    !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install -q trl peft accelerate bitsandbytes datasets

USAGE (Lightning.ai / Colab):
    python3 scripts/finetune_atom_compliance_healthcare.py \
        --pairs   data/14_healthcare_gold/gold_pairs.json \
        --records data/13_healthcare/incidents/records.json \
        --out     data/14_healthcare_gold/llama_healthcare_predictions.json \
        --save_path atom_judge_adapter_healthcare

    # Inference-only with a saved adapter:
    python3 scripts/finetune_atom_compliance_healthcare.py --infer_only \
        --adapter atom_judge_adapter_healthcare \
        --pairs   data/14_healthcare_gold/gold_pairs.json \
        --records data/13_healthcare/incidents/records.json \
        --out     data/14_healthcare_gold/llama_healthcare_predictions.json
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset

BASE_MODEL  = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
MIN_HYP_LEN = 25


def make_prompt(premise: str, hypothesis: str) -> str:
    return (
        "You are a compliance analyst.\n"
        "Read the incident record below and decide whether it satisfies the stated obligation.\n\n"
        f"### Incident Record\n{premise[:1800]}\n\n"
        f"### Obligation\n{hypothesis}\n\n"
        "### Question\nDoes the incident record satisfy this obligation? Answer yes or no.\n\n"
        "### Answer\n"
    )


def build_rows(pairs, records):
    """Convert gold pairs + records into (premise, hypothesis, covered) dicts."""
    rows = []
    for p in pairs:
        if len(p["atom_claim"]) < MIN_HYP_LEN:
            continue
        r = records[p["incident_id"]]
        premise = f"{r['observed_behaviour']} {r['actions_taken']} {r['outcome']}"
        rows.append({
            "pair_id":    p["incident_id"] + "__" + p["atom_full_id"],
            "incident_id": p["incident_id"],
            "atom_full_id": p["atom_full_id"],
            "premise":    premise,
            "hypothesis": p["atom_claim"],
            "covered":    p["label"] > 0,   # Y=1, P=0.5→True, N=0→False
            "standard":   p["standard"],
            "attack_category": p["attack_category"],
        })
    return rows


def metrics(gold, pred):
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))
    tn = sum(g == 0 and p == 0 for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round((tp + tn) / len(gold), 4) if gold else 0}


def train(args, rows, model=None, tokenizer=None):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    if model is None:
        print(f"Loading base model: {args.model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            dtype=None,
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

    # 80/20 train/val split stratified by label
    rng = random.Random(args.seed)
    pos = [r for r in rows if r["covered"]]
    neg = [r for r in rows if not r["covered"]]
    rng.shuffle(pos); rng.shuffle(neg)

    val_pos  = pos[:max(1, len(pos) // 5)]
    val_neg  = neg[:max(1, len(neg) // 5)]
    train_pos = pos[len(val_pos):]
    train_neg = neg[len(val_neg):]

    # Oversample positives in training (minority class ~8.7%)
    factor = max(1, len(train_neg) // max(len(train_pos), 1))
    balanced = train_pos * factor + train_neg
    rng.shuffle(balanced)

    print(f"Train: {len(train_pos)} pos ×{factor} + {len(train_neg)} neg = {len(balanced)} rows")
    print(f"Val:   {len(val_pos)} pos + {len(val_neg)} neg = {len(val_pos)+len(val_neg)} rows")

    def to_text(r):
        label = "yes" if r["covered"] else "no"
        prompt = make_prompt(r["premise"], r["hypothesis"])
        return prompt + label + eos

    train_dataset = Dataset.from_dict({"text": [to_text(r) for r in balanced]})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
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
            output_dir="output_healthcare_judge",
            report_to="none",
            dataset_text_field="text",
            max_seq_length=2048,
        ),
    )
    trainer.train()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"\nAdapter saved -> {args.save_path}")
    return model, tokenizer, val_pos + val_neg


def infer(model, tokenizer, rows):
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    results = []
    for i, r in enumerate(rows):
        prompt = make_prompt(r["premise"], r["hypothesis"])
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=3,
            temperature=1.0, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()
        pred = 1 if answer.startswith("yes") else 0
        results.append({**r, "llama_answer": answer, "llama_pred": pred})

        if (i + 1) % 200 == 0:
            so_far_gold = [int(x["covered"]) for x in results]
            so_far_pred = [x["llama_pred"] for x in results]
            m = metrics(so_far_gold, so_far_pred)
            print(f"  {i+1}/{len(rows)}  running F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    return results


def report(results, label, adapter_name):
    gold = [int(r["covered"]) for r in results]
    pred = [r["llama_pred"] for r in results]
    m    = metrics(gold, pred)

    def breakdown(key):
        groups = {}
        for r in results:
            k = r.get(key, "?")
            groups.setdefault(k, {"gold": [], "pred": []})
            groups[k]["gold"].append(int(r["covered"]))
            groups[k]["pred"].append(r["llama_pred"])
        return {k: metrics(v["gold"], v["pred"]) for k, v in groups.items()}

    print(f"\n{'='*50}")
    print(f"{label} (n={len(results)}, adapter={adapter_name})")
    print(f"  Precision:  {m['precision']:.4f}")
    print(f"  Recall:     {m['recall']:.4f}")
    print(f"  posF1:      {m['f1']:.4f}")
    print(f"  Accuracy:   {m['accuracy']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

    bd_std = breakdown("standard")
    print(f"\n  By standard:")
    for s, bm in bd_std.items():
        print(f"    {s:12s}  P={bm['precision']:.3f}  R={bm['recall']:.3f}  F1={bm['f1']:.3f}")

    bd_cat = breakdown("attack_category")
    print(f"\n  By attack category:")
    for c, bm in sorted(bd_cat.items()):
        print(f"    {c:12s}  P={bm['precision']:.3f}  R={bm['recall']:.3f}  F1={bm['f1']:.3f}")

    return {"overall": m, "by_standard": bd_std, "by_attack_category": bd_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",      default="data/14_healthcare_gold/gold_pairs.json")
    ap.add_argument("--records",    default="data/13_healthcare/incidents/records.json")
    ap.add_argument("--out",        default="data/14_healthcare_gold/llama_healthcare_predictions.json")
    ap.add_argument("--save_path",  default="atom_judge_adapter_healthcare")
    ap.add_argument("--model",      default=BASE_MODEL)
    ap.add_argument("--epochs",     type=int, default=3)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--infer_only", action="store_true")
    ap.add_argument("--adapter",    default=None, help="adapter path for --infer_only")
    args = ap.parse_args()

    pairs   = json.load(open(args.pairs))
    records = {r["incident_id"]: r for r in json.load(open(args.records))}
    rows    = build_rows(pairs, records)
    print(f"Pairs: {len(rows)}  positives: {sum(r['covered'] for r in rows)}")

    if args.infer_only:
        from unsloth import FastLanguageModel
        adapter = args.adapter or args.save_path
        print(f"Loading adapter for inference: {adapter}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        val_rows = rows  # run on all pairs
    else:
        model, tokenizer, val_rows = train(args, rows)

    print(f"\nRunning inference on {len(rows)} pairs...")
    all_results = infer(model, tokenizer, rows)

    # Val split metrics (only valid after training, not infer_only)
    if not args.infer_only:
        val_ids = {r["pair_id"] for r in val_rows}
        val_results = [r for r in all_results if r["pair_id"] in val_ids]
        report(val_results, "VAL SPLIT (20%)", args.save_path)

    full_report = report(all_results, "ALL PAIRS", args.adapter or args.save_path)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(args.out, "w"), indent=2)

    report_path = args.out.replace("_predictions.json", "_report.json")
    json.dump({
        "adapter":    args.adapter or args.save_path,
        "n_pairs":    len(all_results),
        "n_positives": sum(int(r["covered"]) for r in all_results),
        **full_report,
    }, open(report_path, "w"), indent=2)

    print(f"\nWrote: {args.out}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
