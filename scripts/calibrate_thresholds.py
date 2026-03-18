"""
calibrate_thresholds.py — Find optimal reranker score thresholds for FA/PA/NA
classification using the golden set as ground truth.

Usage:
    python3 scripts/calibrate_thresholds.py
    python3 scripts/calibrate_thresholds.py --mappings path/to/mappings.json
    python3 scripts/calibrate_thresholds.py --mappings single_policy_e2e/output/mappings.json

Outputs:
    data/06_compliance_mappings/calibration_report.json
    data/06_compliance_mappings/mappings_calibrated.json
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median, stdev

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_FA = "Fully Addressed"
LABEL_PA = "Partially Addressed"
LABEL_NA = "Not Addressed"
POSITIVE_LABELS = {LABEL_FA, LABEL_PA}


# ── Step 1: Join reranker scores to golden labels ─────────────────────────────

def load_and_join(mappings_path: str, golden_path: str) -> list[dict]:
    """Join mappings (reranker scores) to golden labels.

    Join key: (source_control_id, target_policy_id) in mappings
              → (original_control_id, policy_passage_id) in golden.

    We use original_control_id (not corrected_control_id) because the pipeline's
    source_control_id corresponds to what the pipeline originally assigned — which
    the annotator then corrected. Calibration needs to know what score the reranker
    produced for each (pipeline-assigned-control, passage) pair regardless of
    whether the control was later corrected.
    """
    with open(mappings_path, encoding="utf-8") as f:
        mappings = json.load(f)
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    # Build golden index: (original_control_id, passage_id) → record
    golden_by_orig = {}
    for g in golden:
        orig = (g.get("original_control_id") or "").strip()
        pid  = (g.get("policy_passage_id") or "").strip()
        if orig and pid:
            golden_by_orig[(orig, pid)] = g

    # Also build index by corrected_control_id for supplementary lookup
    golden_by_corr = {}
    for g in golden:
        corr = (g.get("corrected_control_id") or "").strip()
        pid  = (g.get("policy_passage_id") or "").strip()
        if corr and pid:
            golden_by_corr.setdefault((corr, pid), g)

    joined = []
    skipped = 0
    for m in mappings:
        ctrl = (m.get("source_control_id") or "").strip()
        pid  = (m.get("target_policy_id") or "").strip()
        try:
            score = float(m.get("entailment_score", 0))
        except (TypeError, ValueError):
            score = 0.0

        g = golden_by_orig.get((ctrl, pid)) or golden_by_corr.get((ctrl, pid))
        if g is None:
            skipped += 1
            continue

        joined.append({
            "control_id":      ctrl,
            "passage_id":      pid,
            "reranker_score":  score,
            "pipeline_status": m.get("status", ""),
            "golden_label":    g.get("compliance_status", LABEL_NA),
            "corrected_ctrl":  (g.get("corrected_control_id") or ctrl),
            "confidence":      g.get("confidence"),
            "is_hard_negative": g.get("is_hard_negative"),
        })

    print(f"  Mappings loaded      : {len(mappings)}")
    print(f"  Joined to golden     : {len(joined)}")
    print(f"  No golden match      : {skipped}")
    return joined


# ── Step 2: Score distribution analysis ───────────────────────────────────────

def print_score_distributions(joined: list[dict]) -> None:
    print("\n── Score distributions by golden label ──────────────────────────")
    for label in (LABEL_FA, LABEL_PA, LABEL_NA):
        scores = [r["reranker_score"] for r in joined if r["golden_label"] == label]
        if not scores:
            print(f"  {label:<20} n=0  (no data)")
            continue
        mn, mx = min(scores), max(scores)
        avg = mean(scores)
        med = median(scores)
        sd  = stdev(scores) if len(scores) > 1 else 0.0
        print(f"  {label:<20} n={len(scores):3d}  "
              f"min={mn:6.3f}  max={mx:6.3f}  "
              f"mean={avg:6.3f}  median={med:6.3f}  std={sd:5.3f}")

    # Print overlap warning
    fa_scores = [r["reranker_score"] for r in joined if r["golden_label"] == LABEL_FA]
    na_scores = [r["reranker_score"] for r in joined if r["golden_label"] == LABEL_NA]
    if fa_scores and na_scores:
        fa_mean = mean(fa_scores)
        na_mean = mean(na_scores)
        if abs(fa_mean - na_mean) < 1.0:
            print("\n  ⚠  FA and NA score distributions overlap significantly.")
            print("     The reranker alone may not cleanly separate them.")
            print("     LLM judge is recommended as a second stage.")


# ── Step 3: Threshold sweep ────────────────────────────────────────────────────

def _classify(score: float, threshold_full: float, threshold_partial: float) -> str:
    if score >= threshold_full:
        return LABEL_FA
    if score >= threshold_partial:
        return LABEL_PA
    return LABEL_NA


def sweep_thresholds(joined: list[dict], step: float = 0.1) -> dict:
    """Grid search over threshold_full and threshold_partial to maximise F1."""
    scores = [r["reranker_score"] for r in joined]
    score_min = min(scores)
    score_max = max(scores)

    # Candidate values: evenly spaced between min and max
    candidates = []
    v = max(0.0, round(score_min - step, 2))
    while v <= score_max + step:
        candidates.append(round(v, 2))
        v += step
    candidates = sorted(set(candidates))

    best_f1     = {"score": -1, "tf": None, "tp": None, "stats": {}}
    best_recall = {"score": -1, "tf": None, "tp": None, "stats": {}}

    for tf in candidates:
        for tp in candidates:
            if tp > tf:
                continue
            tp_count = fp_count = fn_count = 0
            for r in joined:
                pred  = _classify(r["reranker_score"], tf, tp)
                truth = r["golden_label"]
                pred_pos  = pred  in POSITIVE_LABELS
                truth_pos = truth in POSITIVE_LABELS
                if pred_pos and truth_pos:
                    tp_count += 1
                elif pred_pos and not truth_pos:
                    fp_count += 1
                elif not pred_pos and truth_pos:
                    fn_count += 1

            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall    = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            stats = {"precision": precision, "recall": recall, "f1": f1,
                     "tp": tp_count, "fp": fp_count, "fn": fn_count}

            if f1 > best_f1["score"]:
                best_f1 = {"score": f1, "tf": tf, "tp": tp, "stats": stats}
            if recall > best_recall["score"] or (
                    recall == best_recall["score"] and f1 > best_recall["stats"].get("f1", 0)):
                best_recall = {"score": recall, "tf": tf, "tp": tp, "stats": stats}

    return {"best_f1": best_f1, "best_recall": best_recall}


# ── Step 4: Calibrated relabelling ────────────────────────────────────────────

def relabel_mappings(mappings_path: str, threshold_full: float, threshold_partial: float,
                     output_path: str) -> dict:
    """Re-label all mappings with calibrated thresholds and save."""
    with open(mappings_path, encoding="utf-8") as f:
        mappings = json.load(f)

    before = {"Fully Addressed": 0, "Partially Addressed": 0, "Not Addressed": 0}
    after  = {"Fully Addressed": 0, "Partially Addressed": 0, "Not Addressed": 0}

    relabelled = []
    for m in mappings:
        old_status = m.get("status", LABEL_NA)
        before[old_status] = before.get(old_status, 0) + 1
        try:
            score = float(m.get("entailment_score", 0))
        except (TypeError, ValueError):
            score = 0.0
        new_status = _classify(score, threshold_full, threshold_partial)
        after[new_status] = after.get(new_status, 0) + 1
        relabelled.append({**m, "status": new_status,
                           "original_status": old_status,
                           "calibrated": new_status != old_status})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(relabelled, f, indent=2, ensure_ascii=False)

    return {"before": before, "after": after}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Calibrate reranker thresholds against golden set")
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--mappings",
                        default=str(root / "data/06_compliance_mappings/mappings.json"),
                        help="Path to pipeline mappings JSON (with entailment_score)")
    parser.add_argument("--golden",
                        default=str(root / "data/07_golden_mapping/golden_mapping_dataset.json"),
                        help="Path to golden_mapping_dataset.json")
    parser.add_argument("--output-report",
                        default=str(root / "data/06_compliance_mappings/calibration_report.json"),
                        help="Where to save calibration_report.json")
    parser.add_argument("--output-mappings",
                        default=str(root / "data/06_compliance_mappings/mappings_calibrated.json"),
                        help="Where to save relabelled mappings")
    parser.add_argument("--step", type=float, default=0.1,
                        help="Threshold sweep step size (default 0.1)")
    parser.add_argument("--mode", choices=("f1", "recall"), default="f1",
                        help="Which optimisation target to use for recommended thresholds")
    args = parser.parse_args()

    # Auto-derive output paths alongside the input mappings file
    mappings_path = Path(args.mappings)
    default_report = str(root / "data/06_compliance_mappings/calibration_report.json")
    default_cal    = str(root / "data/06_compliance_mappings/mappings_calibrated.json")
    if args.output_report == default_report:
        args.output_report = str(mappings_path.parent / "calibration_report.json")
    if args.output_mappings == default_cal:
        args.output_mappings = str(mappings_path.parent / "mappings_calibrated.json")

    print("=" * 60)
    print("Reranker Threshold Calibration")
    print("=" * 60)
    print(f"Mappings : {args.mappings}")
    print(f"Golden   : {args.golden}")

    # ── Step 1: Join ──────────────────────────────────────────────────────────
    print("\n── Step 1: Join reranker scores to golden labels ────────────────")
    joined = load_and_join(args.mappings, args.golden)
    if len(joined) < 5:
        print(f"\nERROR: Only {len(joined)} joined records — not enough for calibration.")
        print("  Try providing a larger mappings file, e.g. data/06_compliance_mappings/mappings.json")
        sys.exit(1)

    label_counts = {}
    for r in joined:
        label_counts[r["golden_label"]] = label_counts.get(r["golden_label"], 0) + 1
    print(f"  Label counts: {label_counts}")

    positives = sum(v for k, v in label_counts.items() if k in POSITIVE_LABELS)
    negatives = label_counts.get(LABEL_NA, 0)
    total     = len(joined)
    print(f"  Positive rate: {positives}/{total} = {100*positives/total:.1f}%")
    print(f"  Negative rate: {negatives}/{total} = {100*negatives/total:.1f}%")

    # ── Step 2: Distributions ─────────────────────────────────────────────────
    print_score_distributions(joined)

    # ── Step 3: Sweep ─────────────────────────────────────────────────────────
    print(f"\n── Step 3: Threshold sweep (step={args.step}) ───────────────────")
    results = sweep_thresholds(joined, step=args.step)
    bf  = results["best_f1"]
    br  = results["best_recall"]

    print(f"\n  Best F1     : threshold_full={bf['tf']}  threshold_partial={bf['tp']}")
    s = bf["stats"]
    print(f"    Precision={s['precision']:.3f}  Recall={s['recall']:.3f}  F1={s['f1']:.3f}  "
          f"TP={s['tp']}  FP={s['fp']}  FN={s['fn']}")

    print(f"\n  Best Recall : threshold_full={br['tf']}  threshold_partial={br['tp']}")
    s = br["stats"]
    print(f"    Precision={s['precision']:.3f}  Recall={s['recall']:.3f}  F1={s['f1']:.3f}  "
          f"TP={s['tp']}  FP={s['fp']}  FN={s['fn']}")

    # Pick recommended based on --mode
    rec = bf if args.mode == "f1" else br
    threshold_full    = rec["tf"]
    threshold_partial = rec["tp"]

    print(f"\n── Recommended thresholds ({args.mode}-optimised) ────────────────")
    print(f"  THRESHOLD_FULL    = {threshold_full}")
    print(f"  THRESHOLD_PARTIAL = {threshold_partial}")
    s = rec["stats"]
    print(f"  Expected on golden: Precision={s['precision']:.3f}  "
          f"Recall={s['recall']:.3f}  F1={s['f1']:.3f}")

    # ── Step 4: Apply to mappings ─────────────────────────────────────────────
    print(f"\n── Step 4: Relabelling {args.mappings} ──────────────────────────")
    change = relabel_mappings(args.mappings, threshold_full, threshold_partial,
                              args.output_mappings)
    print("  Before relabelling:", change["before"])
    print("  After  relabelling:", change["after"])
    print(f"  Saved → {args.output_mappings}")

    # ── Step 5: Save calibration report ───────────────────────────────────────
    report = {
        "mappings_source":    args.mappings,
        "golden_source":      args.golden,
        "joined_count":       len(joined),
        "label_counts":       label_counts,
        "score_distributions": {
            label: {
                "n": len(sc := [r["reranker_score"] for r in joined if r["golden_label"] == label]),
                "min":    round(min(sc), 4) if sc else None,
                "max":    round(max(sc), 4) if sc else None,
                "mean":   round(mean(sc), 4) if sc else None,
                "median": round(median(sc), 4) if sc else None,
                "std":    round(stdev(sc), 4) if len(sc) > 1 else None,
            }
            for label in (LABEL_FA, LABEL_PA, LABEL_NA)
        },
        "best_f1": {
            "threshold_full":    bf["tf"],
            "threshold_partial": bf["tp"],
            **bf["stats"],
        },
        "best_recall": {
            "threshold_full":    br["tf"],
            "threshold_partial": br["tp"],
            **br["stats"],
        },
        "recommended": {
            "mode":              args.mode,
            "threshold_full":    threshold_full,
            "threshold_partial": threshold_partial,
            **rec["stats"],
        },
        "relabelling_before": change["before"],
        "relabelling_after":  change["after"],
    }

    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Calibration report → {args.output_report}")

    print("\n" + "=" * 60)
    print("Done. To use calibrated thresholds in llm_judge.py, run:")
    print(f"  python3 scripts/llm_judge.py \\")
    print(f"    --mappings {args.output_mappings} \\")
    print(f"    --calibration {args.output_report} \\")
    print(f"    --dry-run")
    print("=" * 60)


if __name__ == "__main__":
    main()
