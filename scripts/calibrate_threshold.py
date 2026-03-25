#!/usr/bin/env python3
"""
Calibrate reranker score threshold to maximise F1 on the golden set.

Reads:
  - single_policy_e2e/output/mappings.json   (reranker output with entailment_score)
  - single_policy_e2e/output/golden_filtered.json  (ground truth for this policy)

For each threshold T in [1.0, 10.0] step 0.5:
  predicted_positive = {(control_id, passage_id) : score >= T}
  TP = predicted_positive ∩ golden_positive
  FP = predicted_positive - golden_positive
  FN = golden_positive - predicted_positive
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 * P * R / (P + R)

Prints a table and highlights best F1 and best Recall rows.

Usage:
  python3 scripts/calibrate_threshold.py
  python3 scripts/calibrate_threshold.py \
      --mappings single_policy_e2e/output/mappings.json \
      --golden   single_policy_e2e/output/golden_filtered.json \
      --min-threshold 0.0 --max-threshold 10.0 --step 0.5
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_mappings(path: str) -> list:
    data = json.load(open(path))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def load_golden(path: str):
    """Return sets of positive and negative (control_id, passage_id) keys."""
    data = json.load(open(path))
    positives = set()
    negatives = set()
    for r in data:
        cid = r.get("corrected_control_id") or r.get("control_id", "")
        pid = r.get("policy_passage_id", "")
        if not cid or not pid:
            continue
        status = r.get("compliance_status", "")
        if status in ("Fully Addressed", "Partially Addressed"):
            positives.add((cid, pid))
        elif status == "Not Addressed":
            negatives.add((cid, pid))
    return positives, negatives


def evaluate_threshold(mappings: list, golden_pos: set, golden_neg: set, threshold: float):
    predicted_pos = set()
    for m in mappings:
        score = m.get("entailment_score", m.get("score", 0.0))
        if score >= threshold:
            cid = m.get("source_control_id") or m.get("control_id", "")
            pid = m.get("target_policy_id") or m.get("policy_passage_id", "")
            if cid and pid:
                predicted_pos.add((cid, pid))

    all_golden = golden_pos | golden_neg
    # Only evaluate against pairs that appear in the golden set
    predicted_in_golden = predicted_pos & all_golden

    tp = len(predicted_in_golden & golden_pos)
    fp = len(predicted_in_golden & golden_neg)
    fn = len(golden_pos - predicted_pos)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "preds":     len(predicted_pos),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mappings",       default=str(ROOT / "single_policy_e2e/output/mappings.json"))
    ap.add_argument("--golden",         default=str(ROOT / "single_policy_e2e/output/golden_filtered.json"))
    ap.add_argument("--min-threshold",  type=float, default=1.0)
    ap.add_argument("--max-threshold",  type=float, default=10.0)
    ap.add_argument("--step",           type=float, default=0.5)
    args = ap.parse_args()

    mappings = load_mappings(args.mappings)
    golden_pos, golden_neg = load_golden(args.golden)

    print(f"Mappings loaded  : {len(mappings)}")
    print(f"Score field      : entailment_score")
    scores = [m.get("entailment_score", m.get("score", 0.0)) for m in mappings]
    print(f"Score range      : {min(scores):.3f} – {max(scores):.3f}")
    print(f"Golden positives : {len(golden_pos)}")
    print(f"Golden negatives : {len(golden_neg)}")
    print()

    # Build threshold list
    thresholds = []
    t = args.min_threshold
    while t <= args.max_threshold + 1e-9:
        thresholds.append(round(t, 2))
        t += args.step

    results = [evaluate_threshold(mappings, golden_pos, golden_neg, t) for t in thresholds]

    # Find best rows
    best_f1_row     = max(results, key=lambda r: r["f1"])
    best_recall_row = max(results, key=lambda r: (r["recall"], r["f1"]))

    # Print table
    header = f"{'Threshold':>9} | {'Preds':>5} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}"
    sep    = "-" * len(header)
    print(header)
    print(sep)
    for r in results:
        tag = ""
        if r["threshold"] == best_f1_row["threshold"]:
            tag += " ◀ best F1"
        if r["threshold"] == best_recall_row["threshold"] and r["threshold"] != best_f1_row["threshold"]:
            tag += " ◀ best Recall"
        print(
            f"{r['threshold']:>9.1f} | {r['preds']:>5} | {r['tp']:>4} | {r['fp']:>4} | {r['fn']:>4} | "
            f"{r['precision']:>9.3f} | {r['recall']:>6.3f} | {r['f1']:>6.3f}{tag}"
        )

    print(sep)
    print(f"\nBest F1     : threshold={best_f1_row['threshold']}  "
          f"P={best_f1_row['precision']:.3f}  R={best_f1_row['recall']:.3f}  F1={best_f1_row['f1']:.3f}")
    print(f"Best Recall : threshold={best_recall_row['threshold']}  "
          f"P={best_recall_row['precision']:.3f}  R={best_recall_row['recall']:.3f}  F1={best_recall_row['f1']:.3f}")


if __name__ == "__main__":
    main()
