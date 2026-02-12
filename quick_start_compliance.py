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
    combined_path = policies_dir / "all_policies_for_mapping.json"
    template_path = Path("data/policies/internal_policies.json")

    # Resolve policy source: single file or individual *_for_mapping.json in policies dir
    policy_path = None
    policy_list = None
    if combined_path.exists():
        policy_path = combined_path
    elif template_path.exists():
        policy_path = template_path
    else:
        from flexible_policy_extractor import load_all_policies_from_dir
        policy_list = load_all_policies_from_dir(str(policies_dir))
        if not policy_list:
            print("\n⚠️  Policy file not found!")
            print("\n   Expected:")
            print("   - data/02_processed/policies/*_for_mapping.json (one per doc, from run_policy_extraction_and_label_studio.py)")
            print("   - data/02_processed/policies/all_policies_for_mapping.json (legacy combined)")
            print("   - data/policies/internal_policies.json (manual)")
            print("\n📋 To extract policies: put DOCX/PDF in data/01_raw/policies/ then run:")
            print("   python3 run_policy_extraction_and_label_studio.py")
            return
    
    # Optional: use LegalBERT obligation classifier (RegNLP-style)
    # Set LEGALBERT_MODEL_PATH to a local path (e.g. from RePASs train_model.py) or leave unset for rule-based.
    legalbert_path = os.environ.get("LEGALBERT_MODEL_PATH")
    if not legalbert_path and Path("models/obligation-classifier-legalbert").exists():
        legalbert_path = str(Path("models/obligation-classifier-legalbert").resolve())
    use_legalbert = bool(legalbert_path)
    if use_legalbert:
        print("   Using LegalBERT obligation classifier:", legalbert_path)
    pipeline = ComplianceMappingPipeline(
        obligation_classifier="legalbert" if use_legalbert else "rule",
        legalbert_model_path=legalbert_path or None,
    )
    
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
        print(f"   Using: {len(policy_list)} passages from data/02_processed/policies/*_for_mapping.json")
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
        pipeline.initialize_entailment_mapper()
    except Exception as e:
        print(f"❌ Error initializing entailment mapper: {e}")
        print("   Install: pip install transformers torch")
        return
    
    # Step 4: Create Mappings
    print("\n[Step 4/6] Creating Compliance Mappings...")
    print("   This may take several minutes depending on the number of controls and policies...")
    
    try:
        pipeline.create_mappings(
            filter_obligations_only=True,
            top_k_per_control=5
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
    
    # Step 6: Generate Report
    print("\n[Step 6/6] Generating Compliance Report...")
    pipeline.generate_compliance_report(
        str(output_dir / "compliance_report.json")
    )
    
    print("\n" + "=" * 60)
    print("✅ COMPLIANCE MAPPING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"   - mappings.csv: All mappings (CSV)")
    print(f"   - mappings.json: All mappings (combined JSON)")
    print(f"   - by_policy/*.json: One JSON per policy document")
    print(f"   - compliance_report.json: Summary report")
    print("\n📋 Next steps:")
    print("   1. Review mappings.csv to see compliance status")
    print("   2. Check compliance_report.json for overall statistics")
    print("   3. Update policies for 'Not Addressed' controls")
    print("   4. Re-run to verify improvements")


if __name__ == "__main__":
    main()
