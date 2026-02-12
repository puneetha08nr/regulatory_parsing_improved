#!/usr/bin/env python3
"""
Policy extraction + Label Studio task generation (per-policy only).

1. Run flexible policy extractor on data/01_raw/policies → one JSON per document.
2. Generate Label Studio validation tasks per policy → one task JSON per document.

Usage:
  python run_policy_extraction_and_label_studio.py
  python run_policy_extraction_and_label_studio.py --input-dir data/01_raw/policies --no-tasks

Label Studio review (manual):
  - Use labeling config: data/03_label_studio_input/validate_policy_extraction.xml
  - Import one or more task files from: data/03_label_studio_input/policy_validation_tasks/<doc_stem>_validation_tasks.json
  - After review, export and run:
    python validate_extraction_label_studio.py --mode export --input <export.json> --type policies --output data/02_processed/policies/<doc>_corrected.json
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run policy extraction and generate Label Studio validation tasks (one JSON per policy)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/01_raw/policies",
        help="Directory with policy PDF/DOCX files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/02_processed/policies",
        help="Output directory for extracted policy JSON (one *_for_mapping.json per doc)",
    )
    parser.add_argument(
        "--no-tasks",
        action="store_true",
        help="Only run extraction; do not generate Label Studio tasks",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Max Label Studio tasks per policy (0 = all passages)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="data/03_label_studio_input/policy_validation_tasks",
        help="Output directory for per-policy task JSON files",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pdfplumber",
        choices=["pdfplumber", "docling", "unstructured"],
        help="PDF extractor backend: docling or unstructured preserve nested structure (sections, tables)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    policies_dir = base / args.output_dir
    tasks_dir = base / args.tasks_dir

    # 1) Run flexible policy extractor (writes one *_for_mapping.json per doc)
    print("=" * 60)
    print("Step 1: Policy extraction (one JSON per document)")
    print("=" * 60)
    cmd_extract = [
        sys.executable,
        str(base / "flexible_policy_extractor.py"),
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.output_dir,
        "--backend",
        args.backend,
    ]
    code = subprocess.run(cmd_extract, cwd=str(base)).returncode
    if code != 0:
        print("Policy extraction failed.")
        return code

    if args.no_tasks:
        print("Skipping Label Studio task generation (--no-tasks).")
        return 0

    # 2) Generate one task file per policy document
    policy_files = sorted((base / args.output_dir).glob("*_for_mapping.json"))
    if not policy_files:
        print("No *_for_mapping.json files found; skipping task generation.")
        return 0

    print()
    print("=" * 60)
    print("Step 2: Generate Label Studio validation tasks (one file per policy)")
    print("=" * 60)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for policy_path in policy_files:
        stem = policy_path.stem.replace("_for_mapping", "")
        task_path = tasks_dir / f"{stem}_validation_tasks.json"
        cmd = [
            sys.executable,
            str(base / "validate_extraction_label_studio.py"),
            "--mode",
            "policies",
            "--input",
            str(policy_path),
            "--output",
            str(task_path),
        ]
        if args.max_tasks > 0:
            cmd.extend(["--max-tasks", str(args.max_tasks)])
        code = subprocess.run(cmd, cwd=str(base)).returncode
        if code != 0:
            return code
        print(f"  ✓ {policy_path.name} → {task_path.name}")

    print()
    print("Next: In Label Studio")
    print("  1. Use labeling config:")
    print(f"     {base / 'data/03_label_studio_input/validate_policy_extraction.xml'}")
    print("  2. Import task file(s) from:")
    print(f"     {tasks_dir}/")
    print("     (one JSON per policy document)")
    print("  3. After review, export and run:")
    print("     python validate_extraction_label_studio.py --mode export --input <export.json> --type policies --output data/02_processed/policies/<doc>_corrected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
