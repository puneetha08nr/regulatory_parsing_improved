"""
Evaluate pipeline mappings against golden set (filtered to one policy).
Can be used standalone: python3 evaluate.py --mappings output/mappings.json --golden output/golden_filtered.json
"""
import argparse
import json
from pathlib import Path

# Reuse main repo's loaders if run from repo
try:
    from scripts.evaluate_pipeline import (
        load_golden,
        load_pipeline,
        load_retrieval_log,
        recall_at_k,
        f1,
    )
except ImportError:
    # Standalone: minimal impl
    def load_golden(path):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        positives = set()
        negatives = set()
        meta = {}
        for r in rows:
            cid = (r.get("corrected_control_id") or r.get("control_id") or "").strip()
            pid = r.get("policy_passage_id", "").strip()
            if not cid or not pid:
                continue
            pair = (cid, pid)
            meta[pair] = r
            if r.get("compliance_status") in ("Fully Addressed", "Partially Addressed"):
                positives.add(pair)
            elif r.get("compliance_status") == "Not Addressed":
                negatives.add(pair)
        return positives, negatives, meta

    def load_pipeline(path):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        predicted = set()
        pipeline_meta = {}
        for r in rows:
            cid = r.get("source_control_id", "").strip()
            pid = r.get("target_policy_id", "").strip()
            if cid and pid:
                pipeline_meta[(cid, pid)] = r
                if r.get("status") in ("Fully Addressed", "Partially Addressed"):
                    predicted.add((cid, pid))
        return predicted, pipeline_meta

    def load_retrieval_log(path):
        if not path or not Path(path).exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def recall_at_k(retrieval_log, golden_pos, ks=(5, 10, 20, 50)):
        from collections import defaultdict
        gold_by_ctrl = defaultdict(set)
        for cid, pid in golden_pos:
            gold_by_ctrl[cid].add(pid)
        results = {}
        misses_by_k = {k: [] for k in ks}
        for k in ks:
            hits = total = 0
            for cid, gold_pids in gold_by_ctrl.items():
                if cid not in retrieval_log:
                    if retrieval_log:
                        total += 1
                    continue
                total += 1
                if gold_pids & set(retrieval_log[cid][:k]):
                    hits += 1
            results[k] = round(hits / total, 3) if total else 0.0
        return results, misses_by_k

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_and_save(
    mappings_path: str,
    golden_path: str,
    retrieval_log_path: str = "",
    output_path: str = "",
) -> dict:
    """Load mappings and golden, compute metrics, optionally save to output_path. Returns dict with precision, recall, f1, counts."""
    golden_pos, golden_neg, meta = load_golden(golden_path)
    predicted, pipeline_meta = load_pipeline(mappings_path)
    retrieval_log = load_retrieval_log(retrieval_log_path) if retrieval_log_path else {}

    tp = predicted & golden_pos
    fp = predicted & golden_neg
    fn = golden_pos - predicted

    precision = len(tp) / len(predicted) if predicted else 0.0
    recall = len(tp) / len(golden_pos) if golden_pos else 0.0
    f1_score = f1(precision, recall)

    result = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_score, 4),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "golden_positives": len(golden_pos),
        "golden_negatives": len(golden_neg),
        "predicted_positives": len(predicted),
    }

    # Recall@K
    if retrieval_log and golden_pos:
        rk, _ = recall_at_k(retrieval_log, golden_pos, ks=(5, 10, 20, 50))
        result["recall_at_k"] = rk

    # Console
    print("\n  ── Evaluation (this policy) ──")
    print(f"  Golden positives : {len(golden_pos)}  |  Golden negatives : {len(golden_neg)}")
    print(f"  Predicted positives : {len(predicted)}")
    print(f"  TP : {result['tp']}  |  FP : {result['fp']}  |  FN : {result['fn']}")
    print(f"  Precision : {result['precision']:.3f}  |  Recall : {result['recall']:.3f}  |  F1 : {result['f1']:.3f}")
    if result.get("recall_at_k"):
        print(f"  Recall@5/10/20/50 : {result['recall_at_k']}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description="Evaluate mappings vs golden (single-policy filtered).")
    ap.add_argument("--mappings", default="output/mappings.json", help="Pipeline mappings JSON")
    ap.add_argument("--golden", default="output/golden_filtered.json", help="Golden JSON (filtered to one policy)")
    ap.add_argument("--retrieval-log", default="", help="Optional retrieval_log.json for Recall@K")
    ap.add_argument("--output", default="output/evaluation.json", help="Write metrics here")
    args = ap.parse_args()
    evaluate_and_save(
        mappings_path=args.mappings,
        golden_path=args.golden,
        retrieval_log_path=args.retrieval_log or "",
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
