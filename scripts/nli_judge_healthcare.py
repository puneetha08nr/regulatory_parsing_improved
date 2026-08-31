#!/usr/bin/env python3
"""
Zero-shot NLI judge for Work 3 healthcare atom compliance.

For each (incident_record, atom) pair in the gold set:
  premise   = observed_behaviour + " " + actions_taken + " " + outcome
  hypothesis = atom_claim
  verdict   = 1 if P(entailment) >= threshold else 0

Threshold is calibrated on a random 70% of the 3,800 pairs; results reported
on the held-out 30% and on all pairs.

Model: cross-encoder/nli-deberta-v3-small  (runs on CPU, ~400 MB)
       Falls back to typeform/distilbert-base-uncased-mnli if not cached.

Run:
    python3 scripts/nli_judge_healthcare.py
    python3 scripts/nli_judge_healthcare.py --threshold 0.5  # skip calibration
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GOLD_PAIRS   = "data/14_healthcare_gold/gold_pairs.json"
RECORDS_PATH = "data/13_healthcare/incidents/records.json"
OUT_PATH     = "data/14_healthcare_gold/nli_predictions.json"
REPORT_PATH  = "data/14_healthcare_gold/nli_report.json"

MODELS = [
    "cross-encoder/nli-deberta-v3-small",
    "typeform/distilbert-base-uncased-mnli",
    "cross-encoder/nli-MiniLM2-L6-H768",
]


def load_model():
    for name in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            mdl.eval()
            labels = [mdl.config.id2label[i].lower() for i in range(len(mdl.config.id2label))]
            ent_idx = next(i for i, l in enumerate(labels) if "entail" in l)
            print(f"Loaded: {name}")
            return tok, mdl, ent_idx, name
        except Exception as e:
            print(f"  {name}: {e}")
    raise SystemExit("No NLI model found. Run: pip install transformers torch\n"
                     "Then download a model: python -c \"from transformers import pipeline; "
                     "pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-small')\"")


@torch.no_grad()
def entail_scores(tok, mdl, ent_idx, pairs, batch=16):
    scores = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        enc = tok(
            [p for p, _ in chunk], [h for _, h in chunk],
            return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        probs = torch.softmax(mdl(**enc).logits, -1)[:, ent_idx]
        scores.extend(probs.tolist())
        if (i // batch) % 20 == 0:
            print(f"  {i+len(chunk)}/{len(pairs)} pairs scored", end="\r")
    print()
    return scores


def metrics(gold, pred):
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))
    tn = sum(g == 0 and p == 0 for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc  = (tp + tn) / len(gold) if gold else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed threshold (skip calibration)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load gold pairs
    pairs = json.load(open(GOLD_PAIRS))
    print(f"Gold pairs: {len(pairs)}  positives: {sum(p['label'] for p in pairs)}")

    # Build full incident text for each record
    records = {r["incident_id"]: r for r in json.load(open(RECORDS_PATH))}
    def premise(inc_id):
        r = records[inc_id]
        return f"{r['observed_behaviour']} {r['actions_taken']} {r['outcome']}"

    # Score all pairs
    tok, mdl, ent_idx, model_name = load_model()
    print(f"Scoring {len(pairs)} pairs (batch={args.batch})...")
    ph = [(premise(p["incident_id"]), p["atom_claim"]) for p in pairs]
    scores = entail_scores(tok, mdl, ent_idx, ph, batch=args.batch)

    for p, s in zip(pairs, scores):
        p["nli_score"] = round(s, 4)

    # Calibrate threshold on 70% split
    rng = random.Random(args.seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    cal_idx  = set(indices[:int(0.7 * len(indices))])
    test_idx = set(indices) - cal_idx

    if args.threshold is None:
        print("Calibrating threshold on 70% split...")
        best_t, best_f1 = 0.5, -1
        cal_gold  = [int(pairs[i]["label"]) for i in cal_idx]
        cal_scores = [pairs[i]["nli_score"] for i in cal_idx]
        for t in [x / 100 for x in range(10, 90, 5)]:
            cal_pred = [1 if s >= t else 0 for s in cal_scores]
            m = metrics(cal_gold, cal_pred)
            if m["f1"] > best_f1:
                best_f1, best_t = m["f1"], t
        threshold = best_t
        print(f"Best threshold: {threshold:.2f}  (cal F1={best_f1:.4f})")
    else:
        threshold = args.threshold
        print(f"Using fixed threshold: {threshold}")

    # Apply threshold
    for p in pairs:
        p["nli_pred"] = 1 if p["nli_score"] >= threshold else 0

    # Evaluate on test split
    test_gold = [int(pairs[i]["label"]) for i in test_idx]
    test_pred = [pairs[i]["nli_pred"] for i in test_idx]
    test_m    = metrics(test_gold, test_pred)

    all_gold  = [int(p["label"]) for p in pairs]
    all_pred  = [p["nli_pred"] for p in pairs]
    all_m     = metrics(all_gold, all_pred)

    # Per-standard breakdown
    def breakdown(key):
        groups = {}
        for p in pairs:
            k = p[key]
            groups.setdefault(k, {"gold": [], "pred": []})
            groups[k]["gold"].append(int(p["label"]))
            groups[k]["pred"].append(p["nli_pred"])
        return {k: metrics(v["gold"], v["pred"]) for k, v in groups.items()}

    report = {
        "model": model_name,
        "threshold": threshold,
        "n_pairs": len(pairs),
        "n_positives": int(sum(all_gold)),
        "test_split": test_m,
        "all_pairs":  all_m,
        "by_standard": breakdown("standard"),
        "by_attack_category": breakdown("attack_category"),
    }

    json.dump(pairs,  open(OUT_PATH,    "w"), indent=2)
    json.dump(report, open(REPORT_PATH, "w"), indent=2)

    print(f"\n{'='*50}")
    print(f"Model:      {model_name}")
    print(f"Threshold:  {threshold}")
    print(f"\nTest split (30%, n={len(test_idx)}):")
    print(f"  Precision: {test_m['precision']:.4f}")
    print(f"  Recall:    {test_m['recall']:.4f}")
    print(f"  F1:        {test_m['f1']:.4f}")
    print(f"  Accuracy:  {test_m['accuracy']:.4f}")
    print(f"\nAll pairs (n={len(pairs)}):")
    print(f"  Precision: {all_m['precision']:.4f}")
    print(f"  Recall:    {all_m['recall']:.4f}")
    print(f"  F1:        {all_m['f1']:.4f}")
    print(f"\nBy standard:")
    for s, m in report["by_standard"].items():
        print(f"  {s:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"\nWrote: {OUT_PATH}")
    print(f"Wrote: {REPORT_PATH}")
    print(f"\nNext: python3 scripts/grade_healthcare_judge.py")


if __name__ == "__main__":
    main()
