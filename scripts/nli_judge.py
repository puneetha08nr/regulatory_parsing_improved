#!/usr/bin/env python3
"""
Local NLI per-atom compliance judge (CPU, offline).

For each (control, passage) pair: score entailment(passage -> atom_claim) with a
small NLI cross-encoder; an atom is covered if P(entailment) >= threshold; the
FA/PA/NA label is computed mechanically from atom coverage. Atomless (title)
controls fall back to a single whole-obligation entailment check.

The threshold is calibrated on the TRAIN split (pair-level agreement with the
corrected v2 labels); results are reported on the held-out TEST split and overall.

Run:  .venv/bin/python scripts/nli_judge.py
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    mdl.eval()
    ent_idx = [i for i, l in mdl.config.id2label.items() if l.lower() == "entailment"][0]
    return tok, mdl, ent_idx


@torch.no_grad()
def entail_scores(tok, mdl, ent_idx, prem_hyp, batch=16):
    """prem_hyp: list of (premise, hypothesis) -> list of P(entailment)."""
    out = []
    for i in range(0, len(prem_hyp), batch):
        chunk = prem_hyp[i:i + batch]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk],
                  return_tensors="pt", truncation=True, max_length=512, padding=True)
        probs = torch.softmax(mdl(**enc).logits, -1)[:, ent_idx]
        out.extend(probs.tolist())
    return out


def label_from(atom_flags):
    k = sum(atom_flags)
    n = len(atom_flags)
    if n == 0:
        return None
    return FA if k == n else (PA if k > 0 else NA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/12_atoms/pairs_eval.json")
    ap.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    ap.add_argument("--out", default="data/10_judge/answer_key.judged_nli.json")
    args = ap.parse_args()

    pairs = json.load(open(args.pairs, encoding="utf-8"))
    tok, mdl, ent_idx = load_model(args.model)
    print(f"model: {args.model}  (entailment idx {ent_idx})")

    # score every atom + atomless fallback
    atom_ph, atom_ref = [], []          # (premise,hyp) and back-ref (pair_idx, atom_key)
    pairless = []                        # atomless pairs -> single check
    for pi, p in enumerate(pairs):
        if p["atoms"]:
            for a in p["atoms"]:
                atom_ph.append((p["passage"], a["claim"]))
                atom_ref.append((pi, a["key"]))
        else:
            hyp = (p["obligation"] or "").strip() or "This passage addresses the control."
            pairless.append((pi, len(atom_ph)))
            atom_ph.append((p["passage"], f"The entity addresses: {hyp}"))
            atom_ref.append((pi, "_whole"))

    print(f"scoring {len(atom_ph)} entailment pairs on CPU ...")
    scores = entail_scores(tok, mdl, ent_idx, atom_ph)

    # per-pair atom scores
    pair_atom = {pi: [] for pi in range(len(pairs))}
    pair_whole = {}
    for (pi, key), sc in zip(atom_ref, scores):
        if key == "_whole":
            pair_whole[pi] = sc
        else:
            pair_atom[pi].append(sc)

    def predict(thr):
        preds = {}
        for pi, p in enumerate(pairs):
            if p["atoms"]:
                flags = [1 if s >= thr else 0 for s in pair_atom[pi]]
                preds[pi] = label_from(flags)
            else:
                preds[pi] = PA if pair_whole.get(pi, 0) >= thr else NA
        return preds

    # calibrate threshold on TRAIN pairs (pair-level agreement with v2)
    train = [pi for pi, p in enumerate(pairs) if p["split"] == "train"]
    test = [pi for pi, p in enumerate(pairs) if p["split"] == "test"]
    best_thr, best_acc = 0.5, -1
    for t in [x / 100 for x in range(30, 91, 5)]:
        pr = predict(t)
        acc = sum(1 for pi in train if pr[pi] == pairs[pi]["v2_label"]) / len(train)
        if acc > best_acc:
            best_acc, best_thr = acc, t
    print(f"calibrated threshold: {best_thr}  (train pair-agreement {best_acc:.1%})")

    preds = predict(best_thr)

    def report(name, idxs):
        agree = sum(1 for pi in idxs if preds[pi] == pairs[pi]["v2_label"])
        POS = {FA, PA}
        tp = sum(1 for pi in idxs if pairs[pi]["v2_label"] in POS and preds[pi] in POS)
        fp = sum(1 for pi in idxs if pairs[pi]["v2_label"] not in POS and preds[pi] in POS)
        fn = sum(1 for pi in idxs if pairs[pi]["v2_label"] in POS and preds[pi] not in POS)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"  {name:12s} n={len(idxs):3d}  agree={agree/len(idxs):.1%}  "
              f"posF1={f1:.2f} (P{prec:.2f}/R{rec:.2f})")

    print("\nvs corrected v2 labels:")
    report("TEST (held)", test)
    report("train", train)
    report("overall", list(range(len(pairs))))

    # write judged file (human_label = v2) for grade_judge.py
    results = [{
        "control_id": p["control_id"], "passage_id": p["passage_id"],
        "human_label": p["v2_label"], "ai_label": preds[pi],
        "ai_reason": f"NLI atom coverage @thr={best_thr}",
    } for pi, p in enumerate(pairs)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "threshold": best_thr, "n": len(results), "results": results},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n-> {args.out}   (grade: .venv/bin/python scripts/grade_judge.py --judged {args.out})")


if __name__ == "__main__":
    main()
