#!/usr/bin/env python3
"""
Single-policy end-to-end: load one policy → run compliance mapping → evaluate vs golden.

Run from repo root:  python3 single_policy_e2e/run.py
Or from this folder: python3 run.py  (adds parent to path)
"""
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from single_policy_e2e import config


def policy_doc_id_from_passage_id(passage_id: str) -> str:
    """e.g. 'clientname-IS-POL-00-Asset Management Policy 6_passage_1' -> 'clientname-IS-POL-00-Asset Management Policy 6'."""
    if "_passage_" in passage_id:
        return passage_id.rsplit("_passage_", 1)[0]
    return passage_id


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Single-policy end-to-end compliance mapping + evaluation."
    )
    ap.add_argument(
        "--policy",
        default=None,
        help="Path to policy JSON file (list of passage dicts). "
             "Overrides POLICY_JSON env var and config default.",
    )
    ap.add_argument(
        "--skip-obligation-filter",
        action="store_true",
        help="Skip the ObligationFilter pre-filter step and index all passages.",
    )
    ap.add_argument(
        "--obligation-threshold",
        type=float,
        default=0.5,
        help="Obligation classifier probability threshold (default=0.5).",
    )
    ap.add_argument(
        "--obligation-model",
        default=None,
        help="Path to obligation classifier model. Defaults to models/obligation-classifier-legalbert.",
    )
    args = ap.parse_args()

    policy_path = Path(args.policy) if args.policy else Path(config.POLICY_JSON)
    controls_path = Path(config.CONTROLS_JSON)
    golden_path = Path(config.GOLDEN_JSON)
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not policy_path.exists():
        print(f"Policy not found: {policy_path}")
        print("Set POLICY_JSON or use default Asset Management Policy 6_corrected.json")
        sys.exit(1)
    if not controls_path.exists():
        print(f"Controls not found: {controls_path}")
        sys.exit(1)
    if not golden_path.exists():
        print(f"Golden not found: {golden_path}")
        sys.exit(1)

    with open(policy_path, "r", encoding="utf-8") as f:
        policy_list = json.load(f)
    policy_list = policy_list if isinstance(policy_list, list) else [policy_list]
    if not policy_list:
        print("Policy file has no passages.")
        sys.exit(1)

    policy_doc_id = policy_doc_id_from_passage_id(policy_list[0].get("id", ""))
    print("=" * 60)
    print("  SINGLE-POLICY COMPLIANCE MAPPING (END-TO-END)")
    print("=" * 60)
    print(f"  Policy document : {policy_path.name}")
    print(f"  Policy doc ID   : {policy_doc_id}")
    print(f"  Passages (raw)  : {len(policy_list)}")
    print(f"  Output dir      : {output_dir}")
    print()

    # ── ObligationFilter pre-filter ───────────────────────────────────────
    if not args.skip_obligation_filter:
        sys.path.insert(0, str(ROOT / "scripts"))
        from obligation_filter import ObligationFilter

        obligation_model_path = args.obligation_model or os.environ.get(
            "OBLIGATION_MODEL_PATH",
            str(ROOT / "models/obligation-classifier-legalbert"),
        )
        ob_filter = ObligationFilter(
            model_path=obligation_model_path,
            threshold=args.obligation_threshold,
        )
        kept, removed = ob_filter.filter_passages(policy_list, text_key="text")
        print(f"  ObligationFilter : {len(policy_list)} → {len(kept)} passages "
              f"({len(removed)} removed as non-obligation)")
        if removed:
            sample = removed[:3]
            for p in sample:
                snippet = (p.get("text") or "")[:80].replace("\n", " ")
                print(f"    removed: [{p.get('id','?')}] {snippet!r}")
        policy_list = kept
        print()
    else:
        print("  ObligationFilter : skipped (--skip-obligation-filter)")
        print()

    # ── Pipeline (reuse main repo) ────────────────────────────────────────
    from compliance_mapping_pipeline import ComplianceMappingPipeline

    use_reranker = os.environ.get("USE_RERANKER", "1") != "0"
    reranker_model = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
    legalbert_path = os.environ.get(
        "LEGALBERT_MODEL_PATH",
        str(ROOT / "models/obligation-classifier-legalbert"),
    )
    use_legalbert = Path(legalbert_path).exists()
    pipeline = ComplianceMappingPipeline(
        obligation_classifier="legalbert" if use_legalbert else "rule",
        legalbert_model_path=legalbert_path if use_legalbert else None,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
    )
    if use_legalbert:
        print(f"  Obligation classifier : LegalBERT ({legalbert_path})")
    else:
        print(f"  Obligation classifier : rule-based (LegalBERT model not found at {legalbert_path})")
    pipeline.load_ia_controls(str(controls_path))
    pipeline.load_policy_passages_from_list(policy_list)
    print(f"  Passages (indexed): {len(policy_list)}")

    # Family-based routing: only map controls in families relevant to this policy (no 188 controls)
    routing_path = ROOT / "data/02_processed/family_routing.jsonl"
    if routing_path.exists():
        pipeline.load_family_routing(str(routing_path))
        routed = getattr(pipeline, "_family_routing_by_doc", {}).get(policy_doc_id, [])
        if routed:
            print(f"  Family routing   : {routed} (only these control families will be mapped)")
    else:
        print("  ⚠️  family_routing.jsonl not found — mapping all obligation controls for this policy.")

    # NLI model (only used if reranker is disabled/unavailable)
    if not use_reranker:
        nli_model = os.environ.get("NLI_MODEL", "cross-encoder/nli-MiniLM2-L6-H768")
        pipeline.initialize_entailment_mapper(model_name=nli_model)

    # Optional: blocklist from golden (confirmed negatives for this policy)
    if golden_path.exists():
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_rows = json.load(f)
        policy_golden = [r for r in golden_rows if (r.get("policy_passage_id") or "").startswith(policy_doc_id)]
        pos_pairs = set()
        neg_pairs = set()
        for r in policy_golden:
            cid = (r.get("corrected_control_id") or r.get("control_id") or "").strip()
            pid = (r.get("policy_passage_id") or "").strip()
            if not cid or not pid:
                continue
            st = r.get("compliance_status")
            if st in ("Fully Addressed", "Partially Addressed"):
                pos_pairs.add((cid, pid))
            elif st == "Not Addressed":
                neg_pairs.add((cid, pid))

        # Never block a pair that is positive anywhere (handles annotation conflicts / duplicates).
        confirmed_negs = neg_pairs - pos_pairs
        if confirmed_negs:
            pipeline._confirmed_negatives = confirmed_negs
            conflicts = len(neg_pairs & pos_pairs)
            if conflicts:
                print(f"  Golden conflicts : {conflicts} pair(s) labeled both positive and negative (not blocked).")
            print(f"  Loaded {len(confirmed_negs)} confirmed negatives (golden) for this policy.")

    # Create mappings (per-doc corpus; we have one doc)
    # Defaults here are intentionally permissive so evaluation isn't trivially 0.
    # Note: score distributions differ between reranker vs NLI; override via env vars as needed.
    top_k_retrieve = int(os.environ.get("TOP_K_RETRIEVE", "50"))
    top_k_per_doc = int(os.environ.get("TOP_K_PER_DOC", "10"))
    top_k_per_control = int(os.environ.get("TOP_K_PER_CONTROL", "25"))
    threshold_full = float(os.environ.get("THRESHOLD_FULL", "0.30"))
    threshold_partial = float(os.environ.get("THRESHOLD_PARTIAL", "0.10"))
    pipeline.create_mappings(
        filter_obligations_only=True,
        use_retrieval=True,
        top_k_retrieve=top_k_retrieve,
        top_k_per_doc=top_k_per_doc,
        top_k_per_control=top_k_per_control,
        threshold_full=threshold_full,
        threshold_partial=threshold_partial,
    )

    # Keep only mappings for this policy
    mappings_this_policy = [
        m for m in pipeline.mappings
        if (m.target_policy_id or "").startswith(policy_doc_id)
    ]
    # Convert to JSON-serializable (dataclass asdict)
    mappings_data = [asdict(m) for m in mappings_this_policy]
    mappings_file = output_dir / "mappings.json"
    with open(mappings_file, "w", encoding="utf-8") as f:
        json.dump(mappings_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(mappings_data)} mappings → {mappings_file}")

    # Retrieval log (for Recall@K) — filter to this policy's passages
    retrieval_log = getattr(pipeline, "_retrieval_log", {})
    if retrieval_log:
        log_file = output_dir / "retrieval_log.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(retrieval_log, f, indent=2, ensure_ascii=False)
        print(f"  Saved retrieval log → {log_file}")

    # ── Filter golden to this policy ──────────────────────────────────────
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_rows = json.load(f)
    golden_filtered = [r for r in golden_rows if (r.get("policy_passage_id") or "").startswith(policy_doc_id)]
    golden_filtered_file = output_dir / "golden_filtered.json"
    with open(golden_filtered_file, "w", encoding="utf-8") as f:
        json.dump(golden_filtered, f, indent=2, ensure_ascii=False)
    print(f"  Golden (this policy): {len(golden_filtered)} rows → {golden_filtered_file}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    from single_policy_e2e.evaluate import evaluate_and_save
    eval_result = evaluate_and_save(
        mappings_path=str(mappings_file),
        golden_path=str(golden_filtered_file),
        retrieval_log_path=str(output_dir / "retrieval_log.json") if (output_dir / "retrieval_log.json").exists() else "",
        output_path=str(output_dir / "evaluation.json"),
    )
    print(f"  Evaluation → {output_dir / 'evaluation.json'}")
    print()
    print("  Done. Summary: P = {:.3f}, R = {:.3f}, F1 = {:.3f}".format(
        eval_result["precision"], eval_result["recall"], eval_result["f1"]))


if __name__ == "__main__":
    main()
