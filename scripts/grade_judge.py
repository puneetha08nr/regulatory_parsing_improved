#!/usr/bin/env python3
"""
Grade the Claude judge against the frozen human answer key (Step 4).

Reports the single honest number (agreement), a 3x3 confusion matrix,
precision/recall/F1 on the "addressed" (FA+PA) class, Cohen's kappa, and a
disagreement list to read.

Usage:
    python3 scripts/grade_judge.py \
        --judged data/10_judge/answer_key.judged.json \
        --out    data/10_judge/grade_report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"
LABELS = [FA, PA, NA]
POS = {FA, PA}


def kappa(rows):
    n = len(rows)
    if not n:
        return None
    obs = sum(1 for r in rows if r["human_label"] == r["ai_label"]) / n
    h = Counter(r["human_label"] for r in rows)
    a = Counter(r["ai_label"] for r in rows)
    exp = sum((h[l] / n) * (a[l] / n) for l in LABELS)
    return (obs - exp) / (1 - exp) if (1 - exp) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default="data/10_judge/answer_key.judged.json")
    ap.add_argument("--out", default="data/10_judge/grade_report.json")
    args = ap.parse_args()

    data = json.load(open(args.judged, encoding="utf-8"))
    rows = data["results"] if isinstance(data, dict) else data
    rows = [r for r in rows if r.get("human_label") and r.get("ai_label")]
    n = len(rows)
    if not n:
        raise SystemExit("no gradable rows")

    agree = sum(1 for r in rows if r["human_label"] == r["ai_label"])
    # 3x3 confusion: conf[human][ai]
    conf = {h: Counter() for h in LABELS}
    for r in rows:
        conf[r["human_label"]][r["ai_label"]] += 1

    # binary addressed/not for precision-recall on positives
    tp = sum(1 for r in rows if r["human_label"] in POS and r["ai_label"] in POS)
    fp = sum(1 for r in rows if r["human_label"] not in POS and r["ai_label"] in POS)
    fn = sum(1 for r in rows if r["human_label"] in POS and r["ai_label"] not in POS)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    disagree = [r for r in rows if r["human_label"] != r["ai_label"]]

    print(f"Graded {n} pairs against the frozen human answer key\n")
    print(f"  AGREEMENT       : {agree}/{n} = {agree/n:.1%}")
    print(f"  Cohen's kappa   : {kappa(rows):.3f}")
    print(f"\n  Positive class (Addressed = Fully or Partially):")
    print(f"    Precision {prec:.2f}  Recall {rec:.2f}  F1 {f1:.2f}   (TP={tp} FP={fp} FN={fn})")
    print(f"\n  Confusion (rows = human, cols = AI):")
    print(f"    {'human\\AI':22s} {'Fully':>10s} {'Partial':>10s} {'Not':>10s}")
    for h in LABELS:
        print(f"    {h:22s} {conf[h][FA]:>10d} {conf[h][PA]:>10d} {conf[h][NA]:>10d}")
    print(f"\n  Disagreements: {len(disagree)}  (read these — some are AI errors, some are label errors)")
    for r in disagree[:15]:
        print(f"    {r['control_id']:8s} human={r['human_label'][:9]:9s} ai={r['ai_label'][:9]:9s} "
              f":: {r.get('ai_reason','')[:80]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "n": n, "agreement": agree / n, "kappa": kappa(rows),
        "positive_class": {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
        "confusion": {h: dict(conf[h]) for h in LABELS},
        "disagreements": disagree,
    }, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
