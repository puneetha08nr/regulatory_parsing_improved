#!/usr/bin/env python3
"""
Batch evaluation: run the full compliance mapping pipeline across all 13 policy documents
and evaluate each against its golden subset.

Steps per policy:
  1. ObligationFilter pre-filter
  2. ComplianceMappingPipeline (BM25 + Dense + Reranker)
  3. Evaluate vs golden_mapping_dataset_clean.json
  4. Append result to data/08_evaluation/results_tracker.json

Output:
  data/08_evaluation/batch_results.json     — per-policy metrics
  data/08_evaluation/results_tracker.json   — cumulative run history

Usage:
  python3 scripts/batch_eval.py
  python3 scripts/batch_eval.py --reranker models/compliance-reranker-v3
  python3 scripts/batch_eval.py --policy "Asset Management*"   (glob filter)
  python3 scripts/batch_eval.py --skip-obligation-filter
  python3 scripts/batch_eval.py --dry-run   (list matched policies, no pipeline)
"""

import argparse
import fnmatch
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "data/08_evaluation"


# ── Policy → golden doc_id mapping ───────────────────────────────────────────

def build_policy_index(policy_dir: Path) -> list[dict]:
    """
    Return list of {file, doc_id, policy_name} for every valid policy file.
    Duplicate doc_ids are deduplicated — latest version (by filename sort) wins.
    """
    seen_doc_ids: dict[str, dict] = {}
    for f in sorted(policy_dir.glob("*_corrected.json")):
        if "all_policies" in f.name:
            continue
        try:
            passages = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(passages, list) or not passages:
            continue
        first_id = passages[0].get("id", "")
        doc_id = first_id.rsplit("_passage_", 1)[0] if "_passage_" in first_id else first_id
        if not doc_id:
            continue
        # Keep the last file that maps to this doc_id (alphabetically latest = newest version)
        seen_doc_ids[doc_id] = {
            "file": f,
            "doc_id": doc_id,
            "passages": passages,
        }

    return list(seen_doc_ids.values())


def match_golden(golden: list, doc_id: str) -> list:
    """Return golden rows whose policy_passage_id starts with doc_id."""
    return [r for r in golden if (r.get("policy_passage_id") or "").startswith(doc_id)]


# ── Evaluation helpers ────────────────────────────────────────────────────────

POSITIVE_STATUSES = {"Fully Addressed", "Partially Addressed"}


def compute_metrics(mappings: list, golden_rows: list) -> dict:
    """Compute TP/FP/FN/P/R/F1 against golden rows for this policy."""
    golden_pos = set()
    golden_neg = set()
    for r in golden_rows:
        cid = (r.get("corrected_control_id") or r.get("control_id") or "").strip()
        pid = r.get("policy_passage_id", "").strip()
        if not cid or not pid:
            continue
        if r.get("compliance_status") in POSITIVE_STATUSES:
            golden_pos.add((cid, pid))
        elif r.get("compliance_status") == "Not Addressed":
            golden_neg.add((cid, pid))

    predicted = set()
    for m in mappings:
        cid = m.get("source_control_id", "").strip()
        pid = m.get("target_policy_id", "").strip()
        status = m.get("status") or m.get("final_status") or ""
        if cid and pid and status in POSITIVE_STATUSES:
            predicted.add((cid, pid))

    tp = predicted & golden_pos
    fp = predicted & golden_neg
    fn = golden_pos - predicted

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall    = len(tp) / len(golden_pos) if golden_pos else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "golden_pos": len(golden_pos),
        "golden_neg": len(golden_neg),
        "predicted":  len(predicted),
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# ── Main batch runner ─────────────────────────────────────────────────────────

def run_policy(entry: dict, golden: list, args) -> dict:
    """Run full pipeline for one policy and return metrics dict."""
    from compliance_mapping_pipeline import ComplianceMappingPipeline

    policy_file = entry["file"]
    doc_id      = entry["doc_id"]
    passages    = entry["passages"]

    golden_rows = match_golden(golden, doc_id)
    if not golden_rows:
        return {"doc_id": doc_id, "skipped": "no golden rows"}

    # ── ObligationFilter ──────────────────────────────────────────────────────
    if not args.skip_obligation_filter:
        from obligation_filter import ObligationFilter
        ob = ObligationFilter(
            model_path=args.obligation_model,
            threshold=args.obligation_threshold,
        )
        passages_filtered, removed = ob.filter_passages(passages, text_key="text")
        n_removed = len(removed)
    else:
        passages_filtered = passages
        n_removed = 0

    # ── Pipeline ──────────────────────────────────────────────────────────────
    reranker_model = args.reranker or os.environ.get(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2"
    )
    pipeline = ComplianceMappingPipeline(
        obligation_classifier="rule",
        use_reranker=True,
        reranker_model=reranker_model,
    )
    pipeline.load_ia_controls(str(ROOT / "data/02_processed/uae_ia_controls_clean.json"))
    pipeline.load_policy_passages_from_list(passages_filtered)

    # Family routing
    routing_path = ROOT / "data/02_processed/family_routing.jsonl"
    if routing_path.exists() and not args.all_controls:
        pipeline.load_family_routing(str(routing_path))

    pipeline.create_mappings(
        filter_obligations_only=True,
        use_retrieval=True,
        top_k_retrieve=int(os.environ.get("TOP_K_RETRIEVE", "50")),
        top_k_per_doc=int(os.environ.get("TOP_K_PER_DOC", "10")),
        top_k_per_control=int(os.environ.get("TOP_K_PER_CONTROL", "25")),
        threshold_full=float(os.environ.get("THRESHOLD_FULL", "0.30")),
        threshold_partial=float(os.environ.get("THRESHOLD_PARTIAL", "0.10")),
    )

    mappings_this_doc = [
        asdict(m) for m in pipeline.mappings
        if (m.target_policy_id or "").startswith(doc_id)
    ]

    metrics = compute_metrics(mappings_this_doc, golden_rows)
    metrics.update({
        "doc_id":          doc_id,
        "policy_file":     policy_file.name,
        "passages_raw":    len(passages),
        "passages_indexed": len(passages_filtered),
        "passages_removed": n_removed,
        "mappings_total":  len(mappings_this_doc),
        "reranker_model":  reranker_model,
    })
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-dir",  default=str(ROOT / "data/02_processed/policies"))
    ap.add_argument("--golden",      default=str(ROOT / "data/07_golden_mapping/golden_mapping_dataset_clean.json"),
                    help="Golden dataset (clean/deduped). Falls back to raw if not found.")
    ap.add_argument("--reranker",    default=None,  help="Reranker model path or HF name")
    ap.add_argument("--policy",      default="*",   help="Glob filter on policy filename, e.g. 'Asset*'")
    ap.add_argument("--skip-obligation-filter", action="store_true")
    ap.add_argument("--obligation-model",   default=None)
    ap.add_argument("--obligation-threshold", type=float, default=0.5)
    ap.add_argument("--all-controls", action="store_true",
                    help="Bypass family routing, evaluate all 188 controls per policy")
    ap.add_argument("--dry-run", action="store_true",
                    help="List matched policies and golden coverage without running pipeline")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load golden (prefer clean version)
    golden_path = Path(args.golden)
    if not golden_path.exists():
        fallback = ROOT / "data/07_golden_mapping/golden_mapping_dataset.json"
        print(f"  ⚠ Clean golden not found at {golden_path}; falling back to {fallback.name}")
        print(f"    Run: python3 scripts/fix_dataset.py --golden-only")
        golden_path = fallback
    golden = json.load(open(golden_path, encoding="utf-8"))
    print(f"Loaded golden: {len(golden)} rows from {golden_path.name}")

    # Build policy index
    policy_index = build_policy_index(Path(args.policy_dir))
    # Apply filename glob filter
    policy_index = [e for e in policy_index if fnmatch.fnmatch(e["file"].name, args.policy)]
    # Filter to policies that have golden rows
    policy_index = [e for e in policy_index if match_golden(golden, e["doc_id"])]
    print(f"Policies matched: {len(policy_index)}\n")

    if args.dry_run:
        print(f"  {'File':<55} {'DocID':<60} {'Golden':>6}")
        print("  " + "-" * 125)
        for e in policy_index:
            g = match_golden(golden, e["doc_id"])
            pos = sum(1 for r in g if r.get("compliance_status") in POSITIVE_STATUSES)
            neg = sum(1 for r in g if r.get("compliance_status") == "Not Addressed")
            print(f"  {e['file'].name:<55} {e['doc_id']:<60} {len(g):>4} rows (FA/PA={pos} NA={neg})")
        return

    # ── Run pipeline per policy ───────────────────────────────────────────────
    all_results = []
    for i, entry in enumerate(policy_index, 1):
        print(f"\n[{i}/{len(policy_index)}] {entry['file'].name}")
        t0 = time.time()
        try:
            result = run_policy(entry, golden, args)
        except Exception as e:
            result = {"doc_id": entry["doc_id"], "policy_file": entry["file"].name, "error": str(e)}
        result["elapsed_s"] = round(time.time() - t0, 1)
        all_results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        elif "skipped" in result:
            print(f"  Skipped: {result['skipped']}")
        else:
            print(f"  P={result['precision']:.3f}  R={result['recall']:.3f}  "
                  f"F1={result['f1']:.3f}  TP={result['tp']} FP={result['fp']} FN={result['fn']}")

    # ── Save batch results ────────────────────────────────────────────────────
    batch_path = OUT_DIR / "batch_results.json"
    json.dump(all_results, open(batch_path, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved batch results → {batch_path}")

    # ── Aggregate table ───────────────────────────────────────────────────────
    good = [r for r in all_results if "error" not in r and "skipped" not in r]
    print(f"\n{'Policy File':<50} {'Controls':>8} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 98)
    for r in good:
        print(f"{r['policy_file'][:50]:<50} {r['golden_pos']:>8} "
              f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
    if good:
        avg_p  = sum(r["precision"] for r in good) / len(good)
        avg_r  = sum(r["recall"]    for r in good) / len(good)
        avg_f1 = sum(r["f1"]        for r in good) / len(good)
        print("-" * 98)
        print(f"{'AVERAGE':<50} {'':>8} {'':>4} {'':>4} {'':>4} "
              f"{avg_p:>6.3f} {avg_r:>6.3f} {avg_f1:>6.3f}")

    # ── Append to results tracker ─────────────────────────────────────────────
    tracker_path = OUT_DIR / "results_tracker.json"
    tracker = json.load(open(tracker_path)) if tracker_path.exists() else []
    tracker.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_type": "batch",
        "reranker": args.reranker,
        "policies_run": len(good),
        "avg_precision": round(avg_p, 4) if good else 0,
        "avg_recall":    round(avg_r, 4) if good else 0,
        "avg_f1":        round(avg_f1, 4) if good else 0,
        "results": all_results,
    })
    json.dump(tracker, open(tracker_path, "w", encoding="utf-8"), indent=2)
    print(f"Appended to results tracker → {tracker_path}")


if __name__ == "__main__":
    main()
