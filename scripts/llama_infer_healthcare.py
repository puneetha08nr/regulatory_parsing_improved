#!/usr/bin/env python3
"""
Llama adapter inference on Work 3 healthcare gold pairs.

Runs the existing atom_judge_adapter (trained on UAE IA) on the 3,800
healthcare (incident_record, atom) pairs.  This gives the zero-shot
transfer baseline before healthcare fine-tuning.

Designed for Lightning.ai / Colab (T4 or better).

SETUP (run once in Colab/Lightning cell):
    !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install -q peft accelerate bitsandbytes

USAGE:
    # Transfer baseline (UAE IA adapter → healthcare data):
    python3 scripts/llama_infer_healthcare.py \
        --adapter atom_judge_adapter \
        --pairs   data/14_healthcare_gold/gold_pairs.json \
        --records data/13_healthcare/incidents/records.json \
        --out     data/14_healthcare_gold/llama_transfer_predictions.json

    # After healthcare fine-tune, run again with new adapter:
    python3 scripts/llama_infer_healthcare.py \
        --adapter atom_judge_adapter_healthcare \
        --pairs   data/14_healthcare_gold/gold_pairs.json \
        --records data/13_healthcare/incidents/records.json \
        --out     data/14_healthcare_gold/llama_healthcare_predictions.json
"""

import argparse
import json
from pathlib import Path


def make_prompt(premise: str, hypothesis: str) -> str:
    return (
        "You are a compliance analyst.\n"
        "Read the incident record below and decide whether it satisfies the stated obligation.\n\n"
        f"### Incident Record\n{premise[:1800]}\n\n"
        f"### Obligation\n{hypothesis}\n\n"
        "### Question\nDoes the incident record satisfy this obligation? Answer yes or no.\n\n"
        "### Answer\n"
    )


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter",  default="atom_judge_adapter")
    ap.add_argument("--pairs",    default="data/14_healthcare_gold/gold_pairs.json")
    ap.add_argument("--records",  default="data/13_healthcare/incidents/records.json")
    ap.add_argument("--out",      default="data/14_healthcare_gold/llama_transfer_predictions.json")
    ap.add_argument("--batch",    type=int, default=8)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    # Load base model + adapter
    print(f"Loading adapter: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Build premise lookup: incident_id -> full text
    records = {r["incident_id"]: r for r in json.load(open(args.records))}
    def premise(inc_id):
        r = records[inc_id]
        return f"{r['observed_behaviour']} {r['actions_taken']} {r['outcome']}"

    pairs = json.load(open(args.pairs))
    print(f"Pairs: {len(pairs)}  positives: {sum(p['label'] for p in pairs)}")

    # Inference
    results = []
    for i, pair in enumerate(pairs):
        prompt = make_prompt(premise(pair["incident_id"]), pair["atom_claim"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=3,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip().lower()
        pred = 1 if answer.startswith("yes") else 0
        results.append({**pair, "llama_answer": answer, "llama_pred": pred})

        if (i + 1) % 100 == 0:
            done = results
            m = metrics([int(r["label"]) for r in done], [r["llama_pred"] for r in done])
            print(f"  {i+1}/{len(pairs)}  running F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    # Final metrics
    gold = [int(r["label"]) for r in results]
    pred = [r["llama_pred"] for r in results]
    all_m = metrics(gold, pred)

    def breakdown(key):
        groups = {}
        for r in results:
            k = r[key]
            groups.setdefault(k, {"gold": [], "pred": []})
            groups[k]["gold"].append(int(r["label"]))
            groups[k]["pred"].append(r["llama_pred"])
        return {k: metrics(v["gold"], v["pred"]) for k, v in groups.items()}

    report = {
        "adapter": args.adapter,
        "n_pairs": len(results),
        "n_positives": int(sum(gold)),
        "all_pairs":  all_m,
        "by_standard": breakdown("standard"),
        "by_attack_category": breakdown("attack_category"),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    report_path = args.out.replace("_predictions.json", "_report.json")
    json.dump(report, open(report_path, "w"), indent=2)

    print(f"\n{'='*50}")
    print(f"Adapter:    {args.adapter}")
    print(f"Pairs:      {len(results)}")
    print(f"Precision:  {all_m['precision']:.4f}")
    print(f"Recall:     {all_m['recall']:.4f}")
    print(f"posF1:      {all_m['f1']:.4f}")
    print(f"Accuracy:   {all_m['accuracy']:.4f}")
    print(f"\nBy standard:")
    for s, m in report["by_standard"].items():
        print(f"  {s:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"\nBy attack category:")
    for c, m in sorted(report["by_attack_category"].items()):
        print(f"  {c:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"\nWrote: {args.out}")
    print(f"Wrote: {report_path}")
    print(f"\nNext: python3 scripts/finetune_atom_compliance_healthcare.py")


if __name__ == "__main__":
    main()
