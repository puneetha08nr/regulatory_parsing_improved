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
  # Single policy (uses pre-computed mappings):
  python3 scripts/calibrate_threshold.py
  python3 scripts/calibrate_threshold.py \
      --mappings single_policy_e2e/output/mappings.json \
      --golden   single_policy_e2e/output/golden_filtered.json \
      --min-threshold 0.0 --max-threshold 10.0 --step 0.5

  # Per-policy calibration (runs pipeline for all policies, saves thresholds):
  python3 scripts/calibrate_threshold.py --per-policy
  python3 scripts/calibrate_threshold.py --per-policy \
      --policy-dir data/02_processed/policies \
      --golden     data/07_golden_mapping/golden_mapping_dataset_clean.json \
      --output     data/02_processed/policy_thresholds.json
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


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


def print_threshold_table(results: list) -> None:
    best_f1_row     = max(results, key=lambda r: r["f1"])
    best_recall_row = max(results, key=lambda r: (r["recall"], r["f1"]))
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


def build_thresholds(min_t: float, max_t: float, step: float) -> list:
    thresholds = []
    t = min_t
    while t <= max_t + 1e-9:
        thresholds.append(round(t, 2))
        t += step
    return thresholds


def run_pipeline_for_policy(passages: list, doc_id: str, controls_path: str,
                             routing_path: str, reranker_model: str) -> list:
    """Run ComplianceMappingPipeline for one policy, return list of mapping dicts."""
    from compliance_mapping_pipeline import ComplianceMappingPipeline
    pipeline = ComplianceMappingPipeline(
        obligation_classifier="rule",
        use_reranker=True,
        reranker_model=reranker_model,
    )
    pipeline.load_ia_controls(controls_path)
    pipeline.load_policy_passages_from_list(passages)
    if Path(routing_path).exists():
        pipeline.load_family_routing(routing_path)
    pipeline.create_mappings(
        filter_obligations_only=True,
        use_retrieval=True,
        top_k_retrieve=int(os.environ.get("TOP_K_RETRIEVE", "50")),
        top_k_per_doc=int(os.environ.get("TOP_K_PER_DOC", "10")),
        top_k_per_control=int(os.environ.get("TOP_K_PER_CONTROL", "25")),
        threshold_full=0.0,
        threshold_partial=0.0,
    )
    return [asdict(m) for m in pipeline.mappings if (m.target_policy_id or "").startswith(doc_id)]


def per_policy_calibrate(args) -> None:
    """Run pipeline for all policies, sweep thresholds, save best per-policy thresholds."""
    DEFAULT_THRESHOLD = 3.0

    golden_path = Path(args.golden)
    if not golden_path.exists():
        fallback = ROOT / "data/07_golden_mapping/golden_mapping_dataset.json"
        print(f"Clean golden not found; using {fallback.name}")
        golden_path = fallback
    golden_all = json.load(open(golden_path, encoding="utf-8"))
    POSITIVE = {"Fully Addressed", "Partially Addressed"}

    policy_dir = Path(args.policy_dir)
    policy_files = sorted(policy_dir.glob("*_corrected.json"))
    policy_files = [f for f in policy_files if "all_policies" not in f.name]

    controls_path = str(ROOT / "data/02_processed/uae_ia_controls_clean.json")
    routing_path  = str(ROOT / "data/02_processed/family_routing.jsonl")
    reranker_model = args.reranker or os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
    thresholds = build_thresholds(args.min_threshold, args.max_threshold, args.step)

    thresholds_map = {}  # doc_id -> best threshold
    summary_rows = []

    for policy_file in policy_files:
        passages = json.load(open(policy_file, encoding="utf-8"))
        if not isinstance(passages, list) or not passages:
            continue
        first_id = passages[0].get("id", "")
        doc_id = first_id.rsplit("_passage_", 1)[0] if "_passage_" in first_id else first_id
        if not doc_id:
            continue

        golden_rows = [r for r in golden_all if (r.get("policy_passage_id") or "").startswith(doc_id)]
        if not golden_rows:
            print(f"  {policy_file.name:<55} no golden rows — skip")
            continue

        golden_pos, golden_neg = set(), set()
        for r in golden_rows:
            cid = (r.get("corrected_control_id") or r.get("control_id") or "").strip()
            pid = (r.get("policy_passage_id") or "").strip()
            if not cid or not pid:
                continue
            if r.get("compliance_status") in POSITIVE:
                golden_pos.add((cid, pid))
            elif r.get("compliance_status") == "Not Addressed":
                golden_neg.add((cid, pid))

        if not golden_pos:
            print(f"  {policy_file.name:<55} no positive golden pairs — skip")
            continue

        print(f"\n[{policy_file.name}]  doc_id={doc_id}")
        print(f"  Running pipeline... (golden_pos={len(golden_pos)}, golden_neg={len(golden_neg)})")
        try:
            mappings = run_pipeline_for_policy(passages, doc_id, controls_path, routing_path, reranker_model)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        scores = [m.get("entailment_score", m.get("score", 0.0)) for m in mappings]
        if scores:
            print(f"  Mappings: {len(mappings)}  score range: {min(scores):.3f}–{max(scores):.3f}")

        results = [evaluate_threshold(mappings, golden_pos, golden_neg, t) for t in thresholds]
        best = max(results, key=lambda r: (r["f1"], r["recall"]))
        print(f"  Best F1={best['f1']:.3f} at threshold={best['threshold']}  "
              f"P={best['precision']:.3f}  R={best['recall']:.3f}  TP={best['tp']} FP={best['fp']} FN={best['fn']}")

        thresholds_map[doc_id] = best["threshold"]
        summary_rows.append({
            "doc_id": doc_id,
            "policy_file": policy_file.name,
            "best_threshold": best["threshold"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
        })

    # Print summary
    print(f"\n{'Policy':<55} {'Threshold':>9} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 85)
    for row in summary_rows:
        short = row["doc_id"].replace("clientname-IS-POL-00-", "")[:55]
        print(f"{short:<55} {row['best_threshold']:>9.1f} {row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}")

    # Save thresholds
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "default_threshold": DEFAULT_THRESHOLD,
        "thresholds": thresholds_map,
        "summary": summary_rows,
    }
    json.dump(result, open(output_path, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved {len(thresholds_map)} policy thresholds → {output_path}")
    print(f"Default fallback threshold: {DEFAULT_THRESHOLD}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-policy",     action="store_true",
                    help="Run pipeline for all policies and calibrate per-policy thresholds.")
    # Single-policy mode args
    ap.add_argument("--mappings",       default=str(ROOT / "single_policy_e2e/output/mappings.json"))
    ap.add_argument("--golden",         default=str(ROOT / "single_policy_e2e/output/golden_filtered.json"))
    # Per-policy mode args
    ap.add_argument("--policy-dir",     default=str(ROOT / "data/02_processed/policies"))
    ap.add_argument("--reranker",       default=None, help="Reranker model path or HF name")
    ap.add_argument("--output",         default=str(ROOT / "data/02_processed/policy_thresholds.json"))
    # Shared
    ap.add_argument("--min-threshold",  type=float, default=1.0)
    ap.add_argument("--max-threshold",  type=float, default=10.0)
    ap.add_argument("--step",           type=float, default=0.5)
    args = ap.parse_args()

    if args.per_policy:
        per_policy_calibrate(args)
        return

    # Single-policy mode (original behaviour)
    mappings = load_mappings(args.mappings)
    golden_pos, golden_neg = load_golden(args.golden)

    print(f"Mappings loaded  : {len(mappings)}")
    print(f"Score field      : entailment_score")
    scores = [m.get("entailment_score", m.get("score", 0.0)) for m in mappings]
    print(f"Score range      : {min(scores):.3f} – {max(scores):.3f}")
    print(f"Golden positives : {len(golden_pos)}")
    print(f"Golden negatives : {len(golden_neg)}")
    print()

    thresholds = build_thresholds(args.min_threshold, args.max_threshold, args.step)
    results = [evaluate_threshold(mappings, golden_pos, golden_neg, t) for t in thresholds]
    print_threshold_table(results)


if __name__ == "__main__":
    main()
