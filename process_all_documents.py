#!/usr/bin/env python3
"""
Process all documents from multiple directories
Handles both policies and adgm directories
"""

import sys
from pathlib import Path
from automated_document_parser import batch_parse_documents


def process_all_documents(
    base_dir: str = "data/01_raw",
    output_dir: str = "data/02_processed/parsed",
    method: str = "unstructured"
):
    """
    Process all .docx files from multiple source directories
    
    Args:
        base_dir: Base directory containing subdirectories
        output_dir: Output directory for parsed files
        method: "llamaparse" or "unstructured"
    """
    base_path = Path(base_dir)
    
    # Directories to check
    directories = [
        base_path / "policies",
        base_path / "adgm",
        base_path  # Also check base directory itself
    ]
    
    total_processed = 0
    
    for directory in directories:
        if not directory.exists():
            print(f"⚠️  Directory does not exist: {directory}")
            continue
        
        docx_files = list(directory.glob("*.docx"))
        if len(docx_files) == 0:
            print(f"ℹ️  No .docx files in: {directory}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing directory: {directory}")
        print(f"Found {len(docx_files)} .docx files")
        print(f"{'='*70}\n")
        
        try:
            batch_parse_documents(
                input_dir=str(directory),
                output_dir=output_dir,
                method=method
            )
            total_processed += len(docx_files)
        except Exception as e:
            print(f"❌ Error processing {directory}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total documents processed: {total_processed}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import os
    
    method = sys.argv[1] if len(sys.argv) > 1 else "unstructured"
    base_dir = sys.argv[2] if len(sys.argv) > 2 else "data/01_raw"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/02_processed/parsed"
    
    print("="*70)
    print("Process All Documents")
    print("="*70)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Method: {method}")
    print("="*70)
    
    process_all_documents(base_dir, output_dir, method)
