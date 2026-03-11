"""
Quick Start Script for Compliance Mapping
Run this to get started with compliance mapping
"""

from compliance_mapping_pipeline import ComplianceMappingPipeline
from pathlib import Path
import json
import os

def main():
    print("=" * 60)
    print("UAE IA Regulation Compliance Mapping - Quick Start")
    print("=" * 60)

    policies_dir = Path("data/02_processed/policies")
    backup_dir = Path("data/02_processed/backup")
    combined_path = policies_dir / "all_policies_for_mapping.json"
    template_path = Path("data/policies/internal_policies.json")

    # Prefer extracted policies (policies/ then backup/); fall back to combined or manual template
    policy_path = None
    policy_list = None
    policy_source_label = None
    from flexible_policy_extractor import load_all_policies_from_dir
    policy_list = load_all_policies_from_dir(str(policies_dir))
    if policy_list:
        policy_source_label = "data/02_processed/policies/*_for_mapping.json"
    else:
        policy_list = load_all_policies_from_dir(str(backup_dir))
        if policy_list:
            policy_source_label = "data/02_processed/backup/*_for_mapping.json"
    if policy_list:
        pass  # use policy_list
    elif combined_path.exists():
        policy_path = combined_path
        policy_list = None
    elif template_path.exists():
        policy_path = template_path
        policy_list = None
    else:
        print("\n⚠️  Policy file not found!")
        print("\n   Expected:")
        print("   - data/02_processed/policies/*_for_mapping.json (one per doc, from run_policy_extraction_and_label_studio.py)")
        print("   - data/02_processed/backup/*_for_mapping.json (if policies moved to backup)")
        print("   - data/02_processed/policies/all_policies_for_mapping.json (legacy combined)")
        print("   - data/policies/internal_policies.json (manual)")
        print("\n📋 To extract policies: put DOCX/PDF in data/01_raw/policies/ then run:")
        print("   python3 run_policy_extraction_and_label_studio.py")
        return
    
    # Optional: use LegalBERT obligation classifier (RegNLP-style)
    # Set LEGALBERT_MODEL_PATH to a local path (e.g. from RePASs train_model.py) or leave unset for rule-based.
    legalbert_path = os.environ.get("LEGALBERT_MODEL_PATH")
    # Check candidate paths in priority order (fine-tuned preferred)
    _legalbert_candidates = [
        Path("obligation-classifier-legalbert"),           # repo root (fine-tuned)
        Path("models/obligation-classifier-legalbert"),    # models/ subfolder
        Path("models/obligation-classifier-legalbert-finetuned"),
    ]
    if not legalbert_path:
        for _candidate in _legalbert_candidates:
            if _candidate.exists() and (_candidate / "config.json").exists():
                legalbert_path = str(_candidate.resolve())
                break
    use_legalbert = bool(legalbert_path)
    if use_legalbert:
        print("   Using LegalBERT obligation classifier:", legalbert_path)
    use_reranker = os.environ.get("USE_RERANKER", "1") != "0"
    use_graph    = os.environ.get("USE_GRAPH", "0") == "1"

    # ── Model selection ────────────────────────────────────────────────────────
    # Priority order for reranker:
    #   1. RERANKER_MODEL env var  (explicit override)
    #   2. models/compliance-reranker  (fine-tuned, produced by finetune_reranker.py)
    #   3. BAAI/bge-reranker-base  (best base model, GPU)
    #   4. cross-encoder/ms-marco-MiniLM-L-2-v2  (CPU fallback, 22 MB)
    #
    # Override any of these with env vars:
    #   RERANKER_MODEL=models/compliance-reranker python3 quick_start_compliance.py
    #   CPU_MODE=0 python3 quick_start_compliance.py   (to disable CPU swap)
    import torch as _torch
    _cpu_mode = not _torch.cuda.is_available() and os.environ.get("CPU_MODE", "1") != "0"
    if _cpu_mode:
        _torch.set_num_threads(8)
        _default_reranker = "cross-encoder/ms-marco-MiniLM-L-2-v2"
        _default_nli      = "cross-encoder/nli-MiniLM2-L6-H768"
        print("\n   CPU-only mode: using lightweight models for speed")
        print(f"      Torch threads : 8")
        print(f"      Reranker      : {_default_reranker}  (~22 MB)")
        print(f"      NLI           : {_default_nli}  (~85 MB)")
        print("      (Set CPU_MODE=0 to use full models, or set RERANKER_MODEL / NLI_MODEL env vars)")
    else:
        _default_reranker = "BAAI/bge-reranker-base"
        _default_nli      = None  # pipeline picks best available

    # Auto-detect fine-tuned reranker if not explicitly overridden
    _finetuned_reranker = Path("models/compliance-reranker")
    if os.environ.get("RERANKER_MODEL"):
        reranker_model = os.environ["RERANKER_MODEL"]
        print(f"\n   Reranker (env override) : {reranker_model}")
    elif (_finetuned_reranker / "config.json").exists():
        reranker_model = str(_finetuned_reranker.resolve())
        print(f"\n   Reranker (fine-tuned)   : {reranker_model}")
        print(f"      Detected models/compliance-reranker — using domain-tuned model.")
    else:
        reranker_model = _default_reranker
        print(f"\n   Reranker (base model)   : {reranker_model}")
        print(f"      No fine-tuned model found at models/compliance-reranker.")
        print(f"      Run scripts/finetune_reranker.py to train a domain-tuned version.")

    nli_model = os.environ.get("NLI_MODEL", _default_nli)

    # Auto-detect fine-tuned LLM judge
    _finetuned_judge = Path("models/compliance-llm-judge")
    _use_finetuned_judge = (_finetuned_judge / "config.json").exists()
    if _use_finetuned_judge:
        print(f"   LLM Judge (fine-tuned)  : {_finetuned_judge.resolve()}")
        print(f"      Domain-tuned judge detected — will be used in post-processing step.")
    else:
        print(f"   LLM Judge               : Ollama (base model)")
        print(f"      No fine-tuned judge at models/compliance-llm-judge.")
        print(f"      Run scripts/finetune_llm_compliance.py to train one.")

    pipeline = ComplianceMappingPipeline(
        obligation_classifier="legalbert" if use_legalbert else "rule",
        legalbert_model_path=legalbert_path or None,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        use_graph=use_graph,
    )
    # Store NLI model choice so initialize_entailment_mapper can use it
    pipeline._cpu_nli_model = nli_model
    
    # Step 1: Load IA Controls
    print("\n[Step 1/6] Loading UAE IA Controls...")
    controls_candidates = [
        Path("data/02_processed/uae_ia_controls_structured.json"),
        Path("data/02_processed/uae_ia_controls_corrected.json"),
        Path("data/02_processed/uae_ia_controls_from_label_studio.json"),
    ]
    controls_path = None
    for p in controls_candidates:
        if p.exists():
            controls_path = p
            break
    if not controls_path:
        print("❌ No controls file found. Tried:")
        for p in controls_candidates:
            print(f"   - {p}")
        print("   Run improved_control_extractor.py first, or add uae_ia_controls_corrected.json")
        return
    print(f"   Using: {controls_path}")
    pipeline.load_ia_controls(str(controls_path))
    
    # Step 2: Load Policies
    print("\n[Step 2/6] Loading Internal Policies...")
    if policy_list is not None:
        print(f"   Using: {len(policy_list)} passages from {policy_source_label}")
        pipeline.load_policy_passages_from_list(policy_list)
    else:
        print(f"   Using: {policy_path}")
        pipeline.load_policy_passages(str(policy_path))
    
    if len(pipeline.policy_passages) == 0:
        print("⚠️  No policy passages loaded. Check your policy file format.")
        return
    
    # Step 3: Initialize Entailment Mapper
    print("\n[Step 3/6] Initializing Entailment Mapper...")
    print("   (This will download the NLI model if not already present)")
    try:
        _nli_kwarg = {}
        _nli_model = getattr(pipeline, "_cpu_nli_model", None)
        if _nli_model:
            _nli_kwarg["model_name"] = _nli_model
        pipeline.initialize_entailment_mapper(**_nli_kwarg)
    except Exception as e:
        print(f"❌ Error initializing entailment mapper: {e}")
        print("   Install: pip install transformers torch")
        return
    
    # Step 4: Create Mappings
    print("\n[Step 4/6] Creating Compliance Mappings...")
    print("   First: building retrieval index (BM25 + Dense) per document — can take several minutes.")
    print("   Then: NLI over retrieved passages per control. Progress is printed every 10 controls.")

    # Load confirmed negatives (pair-level blocklist) and not-applicable passages
    # (passage-level blocklist) from previously reviewed golden data.
    _golden_path = Path("data/07_golden_mapping/golden_mapping_dataset.json")
    _na_path = Path("data/07_golden_mapping/not_applicable_passages.json")
    if _golden_path.exists():
        try:
            pipeline.load_confirmed_negatives(str(_golden_path))
        except Exception as _e:
            print(f"   ⚠️  Could not load confirmed negatives: {_e}. Continuing without pair blocklist.")
    else:
        print("   (No golden dataset found yet — pair blocklist not applied.)")

    if _na_path.exists():
        try:
            pipeline.load_not_applicable_passages(str(_na_path))
        except Exception as _e:
            print(f"   ⚠️  Could not load not-applicable passages: {_e}.")
    elif _golden_path.exists():
        # Derive not-applicable passages directly from golden data as fallback
        try:
            pipeline.load_not_applicable_passages(str(_golden_path))
        except Exception as _e:
            print(f"   ⚠️  Could not derive not-applicable passages: {_e}.")
    else:
        print("   (No not-applicable passage list found — boilerplate sections may appear in output.)")

    # Retrieval + reranking parameters
    # GPU defaults are high enough for good Recall@K without being slow.
    # CPU defaults are conservative to keep runtime reasonable.
    #
    # top_k_retrieve : BM25+Dense candidates per control per doc (first stage)
    # top_k_rerank   : how many of those pass to the cross-encoder  
    # top_k_per_doc  : cross-encoder keeps this many per doc
    # top_k_per_control : final cap across all docs per control
    #
    # Rule of thumb: top_k_retrieve >= top_k_rerank >= top_k_per_doc
    # Recall@K improves with higher top_k_retrieve; cost scales linearly.
    if _cpu_mode:
        _top_k_retrieve   = int(os.environ.get("TOP_K_RETRIEVE",  "30"))
        _top_k_rerank     = int(os.environ.get("TOP_K_RERANK",    "30"))
        _top_k_per_doc    = int(os.environ.get("TOP_K_PER_DOC",    "3"))
        _top_k_per_control= int(os.environ.get("TOP_K",            "8"))
        print(f"   CPU mode  top_k_retrieve={_top_k_retrieve}  top_k_rerank={_top_k_rerank}"
              f"  top_k_per_doc={_top_k_per_doc}  top_k_per_control={_top_k_per_control}")
    else:
        # GPU: retrieve more candidates → much higher Recall@K
        _top_k_retrieve   = int(os.environ.get("TOP_K_RETRIEVE",  "50"))
        _top_k_rerank     = int(os.environ.get("TOP_K_RERANK",   "100"))
        _top_k_per_doc    = int(os.environ.get("TOP_K_PER_DOC",    "5"))
        _top_k_per_control= int(os.environ.get("TOP_K",           "10"))
        print(f"   GPU mode  top_k_retrieve={_top_k_retrieve}  top_k_rerank={_top_k_rerank}"
              f"  top_k_per_doc={_top_k_per_doc}  top_k_per_control={_top_k_per_control}")

    # Reranker score thresholds — lowered from 0.60/0.35 because BGE-reranker-base
    # (untuned) produces scores in the 0.3–0.6 range for genuine matches.
    # Raise back to 0.60/0.35 once the reranker has been fine-tuned on golden data.
    _threshold_full    = float(os.environ.get("THRESHOLD_FULL",    "0.45"))
    _threshold_partial = float(os.environ.get("THRESHOLD_PARTIAL", "0.25"))

    try:
        pipeline.create_mappings(
            filter_obligations_only=True,
            top_k_retrieve=_top_k_retrieve,
            top_k_rerank=_top_k_rerank,
            top_k_per_doc=_top_k_per_doc,
            top_k_per_control=_top_k_per_control,
            threshold_full=_threshold_full,
            threshold_partial=_threshold_partial,
        )
    except Exception as e:
        print(f"❌ Error creating mappings: {e}")
        return
    
    # Step 5: Save Mappings (per-policy JSONs + combined + CSV)
    print("\n[Step 5/6] Saving Mappings...")
    output_dir = Path("data/06_compliance_mappings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline.save_mappings(
        str(output_dir / "mappings.csv"),
        format="csv"
    )
    pipeline.save_mappings_per_policy(
        str(output_dir / "by_policy"),
        also_combined_path=str(output_dir / "mappings.json"),
    )

    # Passage-centric mapping: cross-encoder scored directly (passage as query vs all controls).
    # This is the reverse of the control-centric pass above — no BM25 retrieval,
    # so scores reflect genuine semantic coverage rather than keyword-overlap artefacts.
    print("\n[Step 5b/6] Passage-centric mapping (Cross-Encoder: passage → controls)...")
    pipeline.create_passage_to_control_mappings(
        str(output_dir / "mappings_by_passage.json"),
        filter_obligations_only=True,
        threshold_full=0.45,   # lowered from 0.60 — BGE-reranker-base (untuned) scores
        threshold_partial=0.25,  # cluster 0.3–0.6 for correct pairs; fine-tune later
        batch_size=64,
    )

    # Save retrieval log for Recall@K evaluation
    import json as _json
    retrieval_log = getattr(pipeline, "_retrieval_log", {})
    if retrieval_log:
        retrieval_log_path = output_dir / "retrieval_log.json"
        with open(retrieval_log_path, "w", encoding="utf-8") as _f:
            _json.dump(retrieval_log, _f, indent=2, ensure_ascii=False)
        print(f"   - retrieval_log.json: Retrieved passage IDs per control (for Recall@K)")
    
    # Step 6: Generate Report
    print("\n[Step 6/6] Generating Compliance Report...")
    pipeline.generate_compliance_report(
        str(output_dir / "compliance_report.json")
    )
    
    print("\n" + "=" * 60)
    print("COMPLIANCE MAPPING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"   - mappings.csv             : All mappings (CSV)")
    print(f"   - mappings.json            : All mappings (combined JSON)")
    print(f"   - mappings_by_passage.json : Passage-centric view")
    print(f"   - by_policy/*.json         : One JSON per policy document")
    print(f"   - compliance_report.json   : Summary report")
    print(f"   - retrieval_log.json       : Per-control top-K passage IDs (for R@K)")

    # ── LLM Judge post-processing step ────────────────────────────────────────
    print("\n" + "-" * 60)
    print("LLM Judge Post-Processing (recommended)")
    print("-" * 60)
    if _use_finetuned_judge:
        _judge_cmd = (
            f"python3 scripts/llm_judge.py \\\n"
            f"    --mappings {output_dir}/mappings.json \\\n"
            f"    --use-finetuned \\\n"
            f"    --finetuned-model {_finetuned_judge.resolve()} \\\n"
            f"    --evaluate"
        )
        print(f"Fine-tuned judge detected. Run:")
    else:
        _judge_cmd = (
            f"python3 scripts/llm_judge.py \\\n"
            f"    --mappings {output_dir}/mappings.json \\\n"
            f"    --model llama3.2:1b \\\n"
            f"    --prompt-style cot \\\n"
            f"    --evaluate"
        )
        print(f"Ollama judge (base model). Run:")
    print(f"\n  {_judge_cmd}\n")
    print(f"This filters noisy predictions and re-evaluates against the golden set.")

    # ── Training pipeline ──────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Controls-Aware Training Pipeline (to improve model quality)")
    print("-" * 60)
    print("  Step 1 — Generate synthetic training pairs (~2-3 hrs, run once):")
    print("    python3 scripts/generate_synthetic_pairs.py --model llama3.2:1b")
    print()
    print("  Step 2 — Build unified training set:")
    print("    python3 scripts/prepare_golden_for_training.py \\")
    print("        --golden data/07_golden_mapping/golden_mapping_dataset.json \\")
    print("        --synthetic data/07_golden_mapping/synthetic_pairs.json \\")
    print("        --real-weight 3 --format reranker")
    print()
    print("  Step 3A — Fine-tune cross-encoder (~2 hrs CPU):")
    print("    python3 scripts/finetune_reranker.py \\")
    print("        --loss pairwise --lora --lora-r 16 --epochs 3")
    print()
    print("  Step 3B — Fine-tune LLM judge (~4 hrs CPU):")
    print("    python3 scripts/finetune_llm_compliance.py \\")
    print("        --synthetic data/07_golden_mapping/synthetic_pairs.json")
    print()
    print("  Step 4 — Re-run this script — fine-tuned models are auto-detected.")

    print("\n" + "-" * 60)
    print("Annotation (to grow the golden dataset):")
    print("   python3 create_golden_set_tasks.py \\")
    print("       --candidates data/06_compliance_mappings/mappings.json")
    print("   → Import into Label Studio, annotate, export, then re-run Step 2 above.")


if __name__ == "__main__":
    main()
