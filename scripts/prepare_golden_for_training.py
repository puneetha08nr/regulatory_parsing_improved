#!/usr/bin/env python3
"""
Prepare golden mapping dataset for training an NLI or Reranker model.

Reads:
  - Golden JSON (from create_golden_set_tasks.py --mode export)
  - Synthetic JSON (from scripts/generate_synthetic_pairs.py)  [optional]
  - Controls JSON (UAE IA structured)
  - Policy passages (dir of *_for_mapping.json or single JSON)

Outputs:
  - train/dev splits in NLI format (premise, hypothesis, label) or
    reranker format (query, passage, label/score).

Usage:
  # Real data only
  python3 scripts/prepare_golden_for_training.py \\
    --golden data/07_golden_mapping/golden_mapping_dataset.json \\
    --controls data/02_processed/uae_ia_controls_structured.json \\
    --policies data/02_processed/policies \\
    --output data/07_golden_mapping/training_data \\
    --format reranker

  # With synthetic data (recommended after generate_synthetic_pairs.py)
  python3 scripts/prepare_golden_for_training.py \\
    --golden data/07_golden_mapping/golden_mapping_dataset.json \\
    --synthetic data/07_golden_mapping/synthetic_pairs.json \\
    --real-weight 3 \\
    --format reranker
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow imports from the project root (flexible_policy_extractor, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Status -> NLI label (3-class)
STATUS_TO_NLI = {
    "Fully Addressed": "entailment",
    "Fully addressed": "entailment",
    "Partially Addressed": "neutral",
    "Partially addressed": "neutral",
    "Not Addressed": "contradiction",
    "Not addressed": "contradiction",
}

# Status -> reranker score (for regression) or keep as label
STATUS_TO_SCORE = {
    "Fully Addressed": 1.0,
    "Fully addressed": 1.0,
    "Partially Addressed": 0.5,
    "Partially addressed": 0.5,
    "Not Addressed": 0.0,
    "Not addressed": 0.0,
}

# Synthetic score downgrade — synthetic pairs count as slightly weaker signal
# than human-verified ones. This prevents overfitting to LLM-generated text.
SYNTHETIC_SCORE_MAP = {
    "Fully Addressed": 0.85,
    "Fully addressed": 0.85,
    "Partially Addressed": 0.55,
    "Partially addressed": 0.55,
    "Not Addressed": 0.0,
    "Not addressed": 0.0,
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_controls(path: str) -> Dict[str, Dict]:
    """control_id -> control row (with control.description, etc.)."""
    data = load_json(path)
    items = data if isinstance(data, list) else [data]
    out = {}
    for item in items:
        c = item.get("control", {})
        cid = c.get("id", "")
        if cid:
            out[cid] = item
    return out


def build_control_text(ctrl: Dict, use_sub_controls: bool = True, max_sub: int = 3) -> str:
    """Obligation text for hypothesis / query."""
    c = ctrl.get("control", {})
    desc = (c.get("description") or "").strip()
    if use_sub_controls:
        subs = c.get("sub_controls", [])[:max_sub]
        if subs:
            desc += "\n" + "\n".join(subs)
    return desc or c.get("name", "")


def load_policy_passages(path: str) -> Dict[str, Dict]:
    """policy_passage_id -> { id, name, text, section }."""
    p = Path(path)
    out = {}
    if p.is_file():
        data = load_json(path)
        items = data if isinstance(data, list) else [data]
        for item in items:
            pid = item.get("id", "")
            if pid:
                out[pid] = item
        return out
    if p.is_dir():
        # Load all .json files in the directory regardless of naming convention
        # (handles both *_corrected.json and *_for_mapping.json)
        for f in sorted(p.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    pid = item.get("id", "")
                    if pid:
                        out[pid] = item
            except Exception:
                continue
        return out
    return {}


def _corrected_positive_rows_nli(g: Dict, control_map: Dict, policy_map: Dict) -> List[Dict]:
    """When annotator entered a corrected control ID, generate a positive NLI row for it.

    Example: pipeline guessed M1.1.1 for a Training passage. Annotator set
    edit_control_id=M5.2.1. We emit (passage, M5.2.1 obligation) = entailment.
    The original wrong pair (passage, M1.1.1) stays as contradiction in the main loop.
    """
    corrected_cid = (g.get("corrected_control_id") or "").strip()
    original_cid = (g.get("original_control_id") or g.get("control_id") or "").strip()
    if not corrected_cid or corrected_cid == original_cid:
        return []
    pid = g.get("policy_passage_id")
    ctrl = control_map.get(corrected_cid)
    policy = policy_map.get(pid)
    if not ctrl or not policy:
        return []
    premise = (policy.get("text") or "").strip()
    hypothesis = build_control_text(ctrl, use_sub_controls=True)
    if not premise or not hypothesis:
        return []
    return [{
        "premise": premise[:2000],
        "hypothesis": hypothesis[:512],
        "label": "entailment",
        "control_id": corrected_cid,
        "policy_passage_id": pid,
        "source": "corrected_annotation",
    }]


def _corrected_positive_rows_reranker(g: Dict, control_map: Dict, policy_map: Dict) -> List[Dict]:
    """Reranker version of the above — positive pair for the corrected control."""
    corrected_cid = (g.get("corrected_control_id") or "").strip()
    original_cid = (g.get("original_control_id") or g.get("control_id") or "").strip()
    if not corrected_cid or corrected_cid == original_cid:
        return []
    pid = g.get("policy_passage_id")
    ctrl = control_map.get(corrected_cid)
    policy = policy_map.get(pid)
    if not ctrl or not policy:
        return []
    query = build_control_text(ctrl, use_sub_controls=True)
    passage = (policy.get("text") or "").strip()
    if not query or not passage:
        return []
    return [{
        "query": query[:512],
        "passage": passage[:2000],
        "label": "Fully Addressed",
        "score": 1.0,
        "control_id": corrected_cid,
        "policy_passage_id": pid,
        "is_hard_negative": False,
        "mismatch_reason": None,
        "source": "corrected_annotation",
    }]


def prepare_nli(golden: List[Dict], control_map: Dict, policy_map: Dict) -> List[Dict]:
    """Build rows: premise, hypothesis, label (entailment/neutral/contradiction).

    Multi-control passages annotated as "Partially Addressed":
      "Partially Addressed" maps to "neutral" regardless of how many controls
      a passage covers. This is correct — the passage is relevant (not contradiction)
      but doesn't fully entail the control on its own. No adjustment needed for NLI.

    When an annotator corrected the control ID (edit_control_id in Label Studio),
    the export stores it as corrected_control_id. We automatically add a positive
    (entailment) row for that corrected control in addition to the main row.
    """
    # Count controls per passage (for metadata only — NLI label stays "neutral")
    passage_control_counts: Dict[str, int] = {}
    for g in golden:
        status = (g.get("compliance_status") or "").strip()
        pid = g.get("policy_passage_id")
        if pid and status in {"Fully Addressed", "Fully addressed",
                              "Partially Addressed", "Partially addressed"}:
            passage_control_counts[pid] = passage_control_counts.get(pid, 0) + 1

    rows = []
    for g in golden:
        status = (g.get("compliance_status") or "").strip()
        nli_label = STATUS_TO_NLI.get(status)
        if nli_label is None:
            continue
        cid = g.get("control_id")
        pid = g.get("policy_passage_id")
        if not cid or not pid:
            continue
        ctrl = control_map.get(cid)
        policy = policy_map.get(pid)
        premise = (policy.get("text") or g.get("policy_text_snippet") or "").strip() if policy else (g.get("policy_text_snippet") or "").strip()
        hypothesis = build_control_text(ctrl, use_sub_controls=True) if ctrl else (g.get("control_text_snippet") or "").strip()
        if not premise or not hypothesis:
            continue
        rows.append({
            "premise":   premise[:2000],
            "hypothesis": hypothesis[:512],
            "label":     nli_label,
            "control_id": cid,
            "policy_passage_id": pid,
            "n_controls_for_passage": passage_control_counts.get(pid, 1),
        })
        # Auto-generate positive row for the corrected control
        rows.extend(_corrected_positive_rows_nli(g, control_map, policy_map))
    return rows


def prepare_reranker(golden: List[Dict], control_map: Dict, policy_map: Dict) -> List[Dict]:
    """Build rows: query (control), passage, label (Fully/Partially/Not) or score.

    Multi-control passages (one passage annotated for several controls):
      A passage annotated as "Partially Addressed" for N controls gets score=0.5
      per pair. We also compute an adjusted score:
        - If the passage is annotated against ONLY ONE control → score stays as-is
          (0.5 means it genuinely only partially satisfies that control)
        - If annotated against MULTIPLE controls → each pair's score is boosted
          toward 0.7, because the passage is genuinely relevant but split coverage
          is why it's partial — not because it's a weak match.
      This prevents the model from penalising valid multi-topic passages.

    Hard negatives (is_hard_negative=True) are duplicated so the model sees them
    more often during training — they are the most informative negatives (confirmed
    keyword-overlap or topic-mismatch false matches, not just low-scoring pairs).
    """
    # Pre-compute: how many controls each passage is mapped to (positives only)
    passage_control_counts: Dict[str, int] = {}
    for g in golden:
        status = (g.get("compliance_status") or "").strip()
        pid = g.get("policy_passage_id")
        if pid and status in {"Fully Addressed", "Fully addressed",
                              "Partially Addressed", "Partially addressed"}:
            passage_control_counts[pid] = passage_control_counts.get(pid, 0) + 1

    rows = []
    for g in golden:
        status = (g.get("compliance_status") or "").strip()
        if status not in STATUS_TO_SCORE:
            continue
        cid = g.get("control_id")
        pid = g.get("policy_passage_id")
        if not cid or not pid:
            continue
        ctrl = control_map.get(cid)
        policy = policy_map.get(pid)
        query = build_control_text(ctrl, use_sub_controls=True) if ctrl else (g.get("control_text_snippet") or "").strip()
        passage = (policy.get("text") or g.get("policy_text_snippet") or "").strip() if policy else (g.get("policy_text_snippet") or "").strip()
        if not query or not passage:
            continue
        is_hard_negative = bool(g.get("is_hard_negative"))
        mismatch_reason = g.get("mismatch_reason")

        # Adjust score for multi-control passages annotated as "Partially Addressed":
        # the passage is genuinely relevant to several controls; score it higher
        # (0.7 vs 0.5) so the model doesn't learn to rank it low.
        base_score = STATUS_TO_SCORE[status]
        n_controls_for_passage = passage_control_counts.get(pid, 1)
        if "partially" in status.lower() and n_controls_for_passage > 1:
            adjusted_score = 0.70   # multi-control partial → boosted soft positive
        else:
            adjusted_score = base_score

        row = {
            "query":             query[:512],
            "passage":           passage[:2000],
            "label":             status,
            "score":             adjusted_score,
            "control_id":        cid,
            "policy_passage_id": pid,
            "is_hard_negative":  is_hard_negative,
            "mismatch_reason":   mismatch_reason,
            "n_controls_for_passage": n_controls_for_passage,
        }
        rows.append(row)
        # Duplicate hard negatives so the model sees them twice during training
        if is_hard_negative:
            rows.append(dict(row))
        # Auto-generate positive row for the corrected control
        rows.extend(_corrected_positive_rows_reranker(g, control_map, policy_map))
    return rows


def prepare_synthetic_reranker(synthetic: List[Dict]) -> List[Dict]:
    """Convert synthetic_pairs.json records into reranker training rows.

    Synthetic pairs use a lower score than real annotations to reflect the
    slightly weaker signal quality of LLM-generated text.  The dev split
    intentionally excludes synthetic rows so evaluation stays honest.
    """
    rows = []
    for g in synthetic:
        status = (g.get("compliance_status") or "").strip()
        score = SYNTHETIC_SCORE_MAP.get(status)
        if score is None:
            continue
        query = (g.get("control_text_snippet") or "").strip()
        passage = (g.get("policy_text_snippet") or "").strip()
        if not query or not passage or len(passage.split()) < 10:
            continue
        rows.append({
            "query":             query[:512],
            "passage":           passage[:2000],
            "label":             status,
            "score":             score,
            "control_id":        g.get("control_id", ""),
            "policy_passage_id": g.get("policy_passage_id", ""),
            "is_hard_negative":  False,
            "mismatch_reason":   None,
            "is_synthetic":      True,
            "source":            g.get("synthetic_source", "synthetic"),
        })
    return rows


def prepare_synthetic_nli(synthetic: List[Dict]) -> List[Dict]:
    """NLI version of synthetic rows."""
    rows = []
    for g in synthetic:
        status = (g.get("compliance_status") or "").strip()
        nli_label = STATUS_TO_NLI.get(status)
        if nli_label is None:
            continue
        premise = (g.get("policy_text_snippet") or "").strip()
        hypothesis = (g.get("control_text_snippet") or "").strip()
        if not premise or not hypothesis or len(premise.split()) < 10:
            continue
        rows.append({
            "premise":           premise[:2000],
            "hypothesis":        hypothesis[:512],
            "label":             nli_label,
            "control_id":        g.get("control_id", ""),
            "policy_passage_id": g.get("policy_passage_id", ""),
            "is_synthetic":      True,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Prepare golden data for NLI or Reranker training")
    ap.add_argument("--golden",      required=True,
                    help="Golden mapping JSON (from create_golden_set_tasks export)")
    ap.add_argument("--synthetic",   default=None,
                    help="Synthetic pairs JSON (from scripts/generate_synthetic_pairs.py). "
                         "When provided, synthetic rows are added to train split only.")
    ap.add_argument("--real-weight", type=int, default=3,
                    help="Duplicate real human-annotated rows this many times so they "
                         "outweigh synthetic rows during training (default: 3).")
    ap.add_argument("--controls",    default="data/02_processed/uae_ia_controls_structured.json",
                    help="UAE IA controls JSON")
    ap.add_argument("--policies",    default="data/02_processed/policies",
                    help="Policy passages: dir with *_for_mapping.json or single JSON")
    ap.add_argument("--output",      default="data/07_golden_mapping/training_data",
                    help="Output dir for train/dev files")
    ap.add_argument("--format",      choices=["nli", "reranker"], default="reranker",
                    help="Output format (default: reranker)")
    ap.add_argument("--dev-ratio",   type=float, default=0.15,
                    help="Fraction for dev set from REAL data only (0.15 = 15%%)")
    ap.add_argument("--seed",        type=int, default=42,
                    help="Random seed for split")
    args = ap.parse_args()

    golden_raw = load_json(args.golden)
    golden_list = golden_raw if isinstance(golden_raw, list) else [golden_raw]
    control_map = load_controls(args.controls)
    policy_map = load_policy_passages(args.policies)

    print(f"Golden rows  : {len(golden_list)}")
    print(f"Controls     : {len(control_map)}  |  Passages: {len(policy_map)}")

    # ── Real rows ─────────────────────────────────────────────────────────────
    if args.format == "nli":
        real_rows = prepare_nli(golden_list, control_map, policy_map)
    else:
        real_rows = prepare_reranker(golden_list, control_map, policy_map)

    if not real_rows:
        print("No rows produced from real data. Check control_id/policy_passage_id fields.")
        return 1

    # ── Split real rows into train/dev (dev stays real-only for honest eval) ──
    random.seed(args.seed)
    random.shuffle(real_rows)
    n_dev = max(1, int(len(real_rows) * args.dev_ratio))
    dev_rows = real_rows[:n_dev]
    train_real = real_rows[n_dev:]

    # Apply real-weight duplication to train split so real pairs dominate
    train_rows = train_real * args.real_weight

    print(f"\nReal data    : {len(real_rows)} rows  "
          f"(train={len(train_real)}, dev={len(dev_rows)})")
    print(f"Real-weight  : ×{args.real_weight}  → {len(train_rows)} weighted train rows")

    # ── Synthetic rows (train only) ───────────────────────────────────────────
    if args.synthetic:
        syn_raw = load_json(args.synthetic)
        syn_list = syn_raw if isinstance(syn_raw, list) else [syn_raw]
        if args.format == "nli":
            syn_rows = prepare_synthetic_nli(syn_list)
        else:
            syn_rows = prepare_synthetic_reranker(syn_list)

        train_rows = train_rows + syn_rows
        random.shuffle(train_rows)

        from collections import Counter as _Counter
        syn_status = _Counter(r.get("label") or r.get("score") for r in syn_rows)
        print(f"Synthetic    : {len(syn_list)} records → {len(syn_rows)} train rows")
        print(f"  Status dist: {dict(syn_status)}")
        print(f"Train total  : {len(train_rows)} rows  (real×{args.real_weight} + synthetic)")
    else:
        random.shuffle(train_rows)
        print(f"Train total  : {len(train_rows)} rows  (real only, no synthetic)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.json"
    dev_path   = out_dir / "dev.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_rows, f, indent=2, ensure_ascii=False)
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_rows, f, indent=2, ensure_ascii=False)

    print(f"\nTrain: {len(train_rows)} → {train_path}")
    print(f"Dev  : {len(dev_rows)} → {dev_path}  (real data only)")

    if args.format == "reranker":
        from collections import Counter
        sc = Counter(r["score"] for r in train_rows)
        print(f"Score distribution (train): {dict(sorted(sc.items()))}")
    else:
        from collections import Counter
        print("Label counts (train):", Counter(r["label"] for r in train_rows))

    return 0


if __name__ == "__main__":
    exit(main())
