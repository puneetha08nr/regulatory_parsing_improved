#!/usr/bin/env python3
"""
Evaluate pipeline mappings against the human-annotated golden dataset.

Compares data/06_compliance_mappings/mappings.json (pipeline predictions)
against data/07_golden_mapping/golden_mapping_dataset.json (human truth).

Metrics reported
────────────────
Standard IR metrics (pair-level):
  Precision / Recall / F1

RegNLP RePASs-adapted metrics (per mapping):
  Entailment score  – NLI score stored by pipeline (does policy entail control?)
  Obligation coverage – fraction of sub-controls mentioned in the matched passage
  Answer stability  – how often same control gets same status across runs
                      (approximated here as confidence score from annotator)

Breakdown:
  By mismatch reason  (keyword-overlap, wrong-topic, scope)
  Per-policy          (TP / FP / FN / Precision per document)
  Still-wrong list    (FP detail)
  Missed-match list   (FN detail)

Recall@K (retrieval quality):
  For each control that has a human-verified positive passage,
  check whether that passage appeared in the top-K retrieved candidates
  (before reranking/NLI). Requires retrieval_log.json saved by the pipeline.
  K values evaluated: 5, 10, 20, 50.

Usage:
    python3 scripts/evaluate_pipeline.py
    python3 scripts/evaluate_pipeline.py \\
        --mappings      data/06_compliance_mappings/mappings.json \\
        --golden        data/07_golden_mapping/golden_mapping_dataset.json \\
        --retrieval-log data/06_compliance_mappings/retrieval_log.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


POSITIVE_STATUSES = {"Fully Addressed", "Partially Addressed"}
NEGATIVE_STATUSES  = {"Not Addressed"}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_golden(path: str):
    """Load golden set. Use corrected_control_id when present (human-approved control)."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    positives = set()   # (control_id, passage_id) human says MATCH
    negatives  = set()  # (control_id, passage_id) human says NOT A MATCH
    meta = {}           # (control_id, passage_id) -> row metadata
    for r in rows:
        # Human truth: use corrected control when annotator fixed the pipeline's wrong control
        cid = (r.get("corrected_control_id") or r.get("control_id") or "").strip()
        pid = r.get("policy_passage_id", "").strip()
        if not cid or not pid:
            continue
        status = r.get("compliance_status", "")
        pair = (cid, pid)
        meta[pair] = r
        if status in POSITIVE_STATUSES:
            positives.add(pair)
        elif status in NEGATIVE_STATUSES:
            negatives.add(pair)
    return positives, negatives, meta


def load_pipeline(path: str):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    predicted = set()         # (control_id, passage_id) pipeline says MATCH
    pipeline_meta = {}        # (control_id, passage_id) -> row
    for r in rows:
        cid = r.get("source_control_id", "").strip()
        pid = r.get("target_policy_id", "").strip()
        status = r.get("status", "")
        if cid and pid:
            pipeline_meta[(cid, pid)] = r
            if status in POSITIVE_STATUSES:
                predicted.add((cid, pid))
    return predicted, pipeline_meta


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


# ── main ─────────────────────────────────────────────────────────────────────

def load_retrieval_log(path: str) -> dict:
    """Load retrieval_log.json → {control_id: [passage_id, ...] in rank order}."""
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieval_log: dict, golden_pos: set, ks=(5, 10, 20, 50)) -> dict:
    """Recall@K: for each control with ≥1 human-verified positive passage,
    check if ANY of its positive passages appears in the top-K retrieved list.

    Returns dict {k: recall_float} and per-control miss details.
    """
    # Group golden positives by control
    gold_by_ctrl: dict = defaultdict(set)
    for cid, pid in golden_pos:
        gold_by_ctrl[cid].add(pid)

    results = {}
    misses_by_k: dict = {k: [] for k in ks}

    for k in ks:
        hits = 0
        total = 0
        for cid, gold_pids in gold_by_ctrl.items():
            if cid not in retrieval_log:
                # Control not in log → pipeline never ran retrieval for it
                # Count as miss only if log is non-empty (i.e. pipeline did run)
                if retrieval_log:
                    total += 1
                    misses_by_k[k].append({
                        "control_id": cid,
                        "reason": "control not in retrieval log",
                        "gold_passages": list(gold_pids),
                    })
                continue
            top_k_retrieved = set(retrieval_log[cid][:k])
            total += 1
            if gold_pids & top_k_retrieved:
                hits += 1
            else:
                misses_by_k[k].append({
                    "control_id": cid,
                    "reason": f"gold passage not in top-{k}",
                    "gold_passages": list(gold_pids),
                    "top_retrieved": retrieval_log[cid][:5],
                })
        results[k] = round(hits / total, 3) if total else 0.0

    return results, misses_by_k


def _obligation_coverage(sub_controls: list, passage_text: str) -> float:
    """RePASs: fraction of sub-controls whose key noun/verb appears in the passage.

    Sub-controls look like 'M1.1.1.a: The entity shall determine interested parties...'
    We extract the main verb phrase after 'shall' and check presence in passage text.
    """
    if not sub_controls or not passage_text:
        return 0.0
    passage_lower = passage_text.lower()
    hits = 0
    for sc in sub_controls:
        # Extract key words after 'shall' (first 5 content words)
        sc_text = sc.lower()
        idx = sc_text.find("shall")
        if idx == -1:
            idx = 0
        keywords = [w for w in sc_text[idx:].split() if len(w) > 4 and w.isalpha()][:5]
        if any(kw in passage_lower for kw in keywords):
            hits += 1
    return round(hits / len(sub_controls), 3)


def _repas_score(entailment_score, obligation_coverage, confidence,
                 entailment_w=0.5, coverage_w=0.35, stability_w=0.15) -> float:
    """Composite RePASs-adapted score for one (control, passage) mapping.

    Components (adapted from RegNLP paper):
      entailment_score  (0–1) – NLI/reranker score from pipeline
      obligation_coverage (0–1) – fraction of sub-controls mentioned in passage
      stability         (0–1) – annotator confidence normalised to [0,1]
    """
    stability = ((confidence or 3) - 1) / 4  # 1–5 → 0–1
    score = (entailment_w * entailment_score
             + coverage_w  * obligation_coverage
             + stability_w * stability)
    return round(score, 3)


def evaluate(mappings_path: str, golden_path: str, retrieval_log_path: str = ""):
    golden_pos, golden_neg, meta = load_golden(golden_path)
    predicted, pipeline_meta = load_pipeline(mappings_path)
    retrieval_log = load_retrieval_log(retrieval_log_path)

    all_golden  = golden_pos | golden_neg

    # Pairs that exist in both sets
    tp = predicted & golden_pos           # pipeline correct, human agrees MATCH
    fp = predicted & golden_neg           # pipeline says match, human says NOT
    fn = golden_pos - predicted           # human says match, pipeline missed it

    precision = len(tp) / len(predicted) if predicted else 0
    recall    = len(tp) / len(golden_pos) if golden_pos else 0
    f1_score  = f1(precision, recall)

    print("=" * 64)
    print("  PIPELINE EVALUATION vs GOLDEN DATASET")
    print("=" * 64)
    print(f"\n  Golden positives (human-verified matches) : {len(golden_pos)}")
    print(f"  Golden negatives (human-verified non-match): {len(golden_neg)}")
    print(f"  Pipeline predicted positives               : {len(predicted)}")
    print()
    print(f"  True  Positives (TP) – correct matches kept : {len(tp)}")
    print(f"  False Positives (FP) – wrong matches surfaced: {len(fp)}")
    print(f"  False Negatives (FN) – correct matches missed: {len(fn)}")
    print()
    print(f"  Precision : {precision:.3f}  ({pct(len(tp), len(predicted))} of pipeline output is correct)")
    print(f"  Recall    : {recall:.3f}  ({pct(len(tp), len(golden_pos))} of human matches found)")
    print(f"  F1        : {f1_score:.3f}")

    # ── False Positives breakdown by mismatch reason ──────────────────────
    if fp:
        fp_reasons = Counter(meta[p].get("mismatch_reason", "unknown") for p in fp if p in meta)
        print(f"\n  ── False Positives by mismatch reason ({len(fp)} total) ──")
        for reason, cnt in fp_reasons.most_common():
            label = {
                "kw": "Keyword overlap",
                "topic": "Wrong topic",
                "scope": "Boilerplate/out of scope",
                "ok": "Marked correct (annotation conflict)",
                "unknown": "Unknown",
            }.get(reason, reason)
            print(f"    {label:35s}: {cnt}")

    # ── Still-wrong pairs (FP detail) ────────────────────────────────────
    if fp:
        print(f"\n  ── Still Wrong (pipeline still surfaces these) ──")
        print(f"  {'Control':<12} {'Reason':<12} {'Policy / Section'}")
        print(f"  {'-'*12} {'-'*12} {'-'*40}")
        for pair in sorted(fp)[:30]:
            cid, pid = pair
            r = meta.get(pair, {})
            reason  = r.get("mismatch_reason", "?")
            policy  = r.get("policy_name", "")[:28]
            section = r.get("policy_section", "")[:28]
            print(f"  {cid:<12} {reason:<12} {policy} | {section}")
        if len(fp) > 30:
            print(f"  ... and {len(fp) - 30} more")

    # ── Missed pairs (FN detail) ──────────────────────────────────────────
    if fn:
        print(f"\n  ── Missed Matches (pipeline did not surface these human-verified pairs) ──")
        print(f"  {'Control':<12} {'Policy / Section'}")
        print(f"  {'-'*12} {'-'*40}")
        for pair in sorted(fn)[:20]:
            cid, pid = pair
            r = meta.get(pair, {})
            policy  = r.get("policy_name", "")[:30]
            section = r.get("policy_section", "")[:30]
            print(f"  {cid:<12} {policy} | {section}")
        if len(fn) > 20:
            print(f"  ... and {len(fn) - 20} more")

    # ── Recall@K (retrieval quality) ─────────────────────────────────────
    KS = (5, 10, 20, 50)
    if retrieval_log:
        rk_scores, rk_misses = recall_at_k(retrieval_log, golden_pos, ks=KS)
        print(f"\n  ── Recall@K (retrieval quality — is the right passage in top-K?) ──")
        print(f"  {'K':>4}  {'Recall':>8}  {'Target':>8}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*8}")
        targets = {5: 0.60, 10: 0.70, 20: 0.80, 50: 0.90}
        for k in KS:
            r = rk_scores[k]
            t = targets[k]
            flag = "✅" if r >= t else "⚠️ "
            print(f"  {k:>4}  {r:>8.3f}  {t:>8.2f}  {flag}")
        # Show misses at K=20
        misses_20 = rk_misses.get(20, [])
        if misses_20:
            print(f"\n  Controls where correct passage was NOT in top-20 retrieved:")
            for m in misses_20[:10]:
                print(f"    {m['control_id']:<12}  {m['reason']}")
                for gp in m.get("gold_passages", [])[:2]:
                    print(f"              gold: {gp[-60:]}")
            if len(misses_20) > 10:
                print(f"    ... and {len(misses_20) - 10} more")
    else:
        rk_scores = {}
        rk_misses = {}
        print(f"\n  ── Recall@K ──")
        print(f"  (Not available yet — run quick_start_compliance.py first to generate")
        print(f"   data/06_compliance_mappings/retrieval_log.json)")

    # ── RePASs-adapted scores on True Positives ───────────────────────────
    # Load controls for sub-control coverage calculation
    controls_path = Path("data/02_processed/uae_ia_controls_structured.json")
    control_map = {}
    if controls_path.exists():
        with open(controls_path, encoding="utf-8") as f:
            for ctrl in json.load(f):
                c = ctrl.get("control", {})
                cid = c.get("id", "")
                if cid:
                    control_map[cid] = c

    repas_scores = []
    low_repas_alerts = []
    for pair in tp:
        cid, pid = pair
        pm = pipeline_meta.get(pair, {})
        gm = meta.get(pair, {})
        entailment = float(pm.get("entailment_score") or 0.5)
        passage_text = pm.get("evidence_text", "") or gm.get("policy_text_snippet", "")
        sub_controls = control_map.get(cid, {}).get("sub_controls", [])
        coverage = _obligation_coverage(sub_controls, passage_text)
        confidence = gm.get("confidence") or 3
        score = _repas_score(entailment, coverage, confidence)
        repas_scores.append(score)
        if score < 0.5:
            low_repas_alerts.append({
                "control_id": cid,
                "policy_passage_id": pid,
                "repas_score": score,
                "entailment": entailment,
                "coverage": coverage,
                "confidence": confidence,
                "policy_name": gm.get("policy_name", ""),
                "policy_section": gm.get("policy_section", ""),
                "mapping_status": gm.get("compliance_status", ""),
            })

    avg_repas = round(sum(repas_scores) / len(repas_scores), 3) if repas_scores else 0

    print(f"\n  ── RePASs-adapted scores (on {len(tp)} True Positives) ──")
    print(f"  Average RePASs score : {avg_repas}  (target ≥ 0.70)")
    print(f"  Components:")
    print(f"    Entailment (NLI/reranker score)  weight 50%")
    print(f"    Obligation coverage (sub-controls) weight 35%")
    print(f"    Answer stability (annotator conf)  weight 15%")
    if low_repas_alerts:
        print(f"\n  ⚠️  Weak mappings (RePASs < 0.50) — review these {len(low_repas_alerts)}:")
        print(f"  {'Control':<12} {'Score':>6} {'Entail':>7} {'Cover':>6} {'Conf':>5}  Section")
        print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*6} {'-'*5}  {'-'*30}")
        for a in sorted(low_repas_alerts, key=lambda x: x["repas_score"])[:15]:
            print(f"  {a['control_id']:<12} {a['repas_score']:>6.3f} "
                  f"{a['entailment']:>7.3f} {a['coverage']:>6.3f} {a['confidence']:>5}  "
                  f"{a['policy_section'][:30]}")

    # ── Per-policy breakdown ──────────────────────────────────────────────
    print(f"\n  ── Per-Policy Precision ──")
    policy_tp = defaultdict(int)
    policy_fp = defaultdict(int)
    policy_fn = defaultdict(int)
    for pair in tp:
        policy_tp[meta.get(pair, {}).get("policy_name", "?")] += 1
    for pair in fp:
        policy_fp[meta.get(pair, {}).get("policy_name", "?")] += 1
    for pair in fn:
        policy_fn[meta.get(pair, {}).get("policy_name", "?")] += 1

    all_policies = sorted(set(list(policy_tp) + list(policy_fp) + list(policy_fn)))
    print(f"  {'Policy':<45} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6}")
    print(f"  {'-'*45} {'-'*4} {'-'*4} {'-'*4} {'-'*6}")
    for pol in all_policies:
        t = policy_tp[pol]
        fp_ = policy_fp[pol]
        fn_ = policy_fn[pol]
        prec = f"{100*t/(t+fp_):.0f}%" if (t + fp_) else "n/a"
        print(f"  {pol[:45]:<45} {t:>4} {fp_:>4} {fn_:>4} {prec:>6}")

    # ── Summary JSON ─────────────────────────────────────────────────────
    out = {
        "golden_positives": len(golden_pos),
        "golden_negatives": len(golden_neg),
        "pipeline_predicted": len(predicted),
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_score, 4),
        "recall_at_k": {
            f"R@{k}": {"score": rk_scores.get(k), "target": {5: 0.60, 10: 0.70, 20: 0.80, 50: 0.90}[k]}
            for k in KS
        } if rk_scores else "retrieval_log.json not found — re-run pipeline first",
        "recall_at_k_misses_at_20": rk_misses.get(20, []) if rk_scores else [],
        "repas": {
            "avg_score": avg_repas,
            "target": 0.70,
            "components": {
                "entailment_weight": 0.50,
                "obligation_coverage_weight": 0.35,
                "answer_stability_weight": 0.15,
            },
            "weak_mappings": low_repas_alerts,
        },
        "still_wrong_pairs": [
            {
                "control_id": cid,
                "policy_passage_id": pid,
                "policy_name": meta.get((cid, pid), {}).get("policy_name"),
                "policy_section": meta.get((cid, pid), {}).get("policy_section"),
                "mismatch_reason": meta.get((cid, pid), {}).get("mismatch_reason"),
            }
            for cid, pid in sorted(fp)
        ],
        "missed_pairs": [
            {
                "control_id": cid,
                "policy_passage_id": pid,
                "policy_name": meta.get((cid, pid), {}).get("policy_name"),
                "policy_section": meta.get((cid, pid), {}).get("policy_section"),
            }
            for cid, pid in sorted(fn)
        ],
    }
    out_path = Path(mappings_path).parent / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report saved → {out_path}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mappings",      default="data/06_compliance_mappings/mappings.json")
    ap.add_argument("--golden",        default="data/07_golden_mapping/golden_mapping_dataset.json")
    ap.add_argument("--retrieval-log", default="data/06_compliance_mappings/retrieval_log.json",
                    dest="retrieval_log",
                    help="Retrieval log saved by pipeline (for Recall@K). Auto-detected if present.")
    args = ap.parse_args()

    if not Path(args.mappings).exists():
        print(f"ERROR: mappings file not found: {args.mappings}")
        return
    if not Path(args.golden).exists():
        print(f"ERROR: golden dataset not found: {args.golden}")
        return

    retrieval_log = args.retrieval_log if Path(args.retrieval_log).exists() else ""
    evaluate(args.mappings, args.golden, retrieval_log_path=retrieval_log)


if __name__ == "__main__":
    main()
