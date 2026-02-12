#!/usr/bin/env python3
"""
Quick Start Script for RegRAG-Xref Pipeline
Processes all policy documents through the RegRAG-Xref pipeline
"""

import sys
from pathlib import Path
from regrag_xref_pipeline import RegRAGXrefPipeline, create_mapping_output


def find_policy_files(policy_dir: str = "data/02_processed/policies") -> list:
    """Find all policy JSON files for mapping"""
    policy_path = Path(policy_dir)
    
    if not policy_path.exists():
        print(f"❌ Policy directory not found: {policy_dir}")
        return []
    
    # Find files ending with _for_mapping.json
    policy_files = list(policy_path.glob("*_for_mapping.json"))
    
    if not policy_files:
        # Fallback: find any JSON files
        policy_files = list(policy_path.glob("*.json"))
        print(f"⚠️  No '_for_mapping.json' files found. Using all JSON files.")
    
    return sorted(policy_files)


def find_uae_ia_controls() -> str:
    """Find UAE IA controls JSON file"""
    possible_paths = [
        "data/02_processed/uae_ia_controls_structured.json",
        "data/04_label_studio/imports/uae_ia_controls copy.json",
        "data/02_processed/uae_ia_controls.json"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    return None


def main():
    """Main function"""
    print("="*70)
    print("RegRAG-Xref Pipeline - Quick Start")
    print("="*70)
    print()
    
    # Find policy files
    policy_files = find_policy_files()
    
    if not policy_files:
        print("❌ No policy files found!")
        print("\nPlease ensure policy files exist in:")
        print("  data/02_processed/policies/*_for_mapping.json")
        print("\nOr run flexible_policy_extractor.py first to extract policies.")
        sys.exit(1)
    
    print(f"✓ Found {len(policy_files)} policy file(s)")
    print()
    
    # Find UAE IA controls
    uae_ia_path = find_uae_ia_controls()
    if uae_ia_path:
        print(f"✓ Found UAE IA controls: {uae_ia_path}")
    else:
        print("⚠️  UAE IA controls not found. Mapping will use keyword-based approach only.")
        print("   For better accuracy, provide: data/02_processed/uae_ia_controls_structured.json")
    print()
    
    # Initialize pipeline
    pipeline = RegRAGXrefPipeline()
    
    # Process each policy
    results_summary = []
    
    for i, policy_file in enumerate(policy_files, 1):
        print(f"\n[{i}/{len(policy_files)}] Processing: {policy_file.name}")
        print("-" * 70)
        
        try:
            # Process policy
            results = pipeline.process_policy_document(str(policy_file))
            
            # Create mapping
            document_id = results['document_id']
            structured_json_path = results['stage3_structured_json']
            mapping_output_path = pipeline.output_dir / "stage4_mappings" / f"{document_id}_mapping"
            
            print(f"\nCreating final mapping for {document_id}...")
            create_mapping_output(
                structured_json_path,
                str(mapping_output_path),
                uae_ia_path
            )
            
            results_summary.append({
                'policy': policy_file.name,
                'document_id': document_id,
                'stage2_txt': results['stage2_standardized_txt'],
                'stage3_json': results['stage3_structured_json'],
                'stage4_mapping': str(mapping_output_path)
            })
            
            print(f"✓ Completed: {document_id}")
            
        except Exception as e:
            print(f"❌ Error processing {policy_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70)
    print(f"\nProcessed {len(results_summary)} policy document(s)\n")
    
    for result in results_summary:
        print(f"📄 {result['document_id']}")
        print(f"   Stage 2 (TXT):  {result['stage2_txt']}")
        print(f"   Stage 3 (JSON): {result['stage3_json']}")
        print(f"   Stage 4 (Map):  {result['stage4_mapping']}.csv/json")
        print()
    
    print(f"\nAll outputs saved to: {pipeline.output_dir}/")
    print("\nNext steps:")
    print("1. Review Stage 2 TXT files for section segmentation")
    print("2. Review Stage 3 JSON files for structured format")
    print("3. Review Stage 4 mappings for compliance links")
    print("4. Use compliance_mapping_pipeline.py for NLI-based validation")


if __name__ == "__main__":
    main()
