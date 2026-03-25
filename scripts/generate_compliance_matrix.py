#!/usr/bin/env python3
"""
Generate a compliance matrix from batch_results.json or the full golden dataset.

Rows    = policy documents
Columns = UAE IA control families (T1–T9, M1–M6)
Cell    = count of FA+PA mappings per (policy, family)

Output:
  data/08_evaluation/compliance_matrix.json
  data/08_evaluation/compliance_matrix.csv

Usage:
  # From batch pipeline results
  python3 scripts/generate_compliance_matrix.py

  # From golden dataset directly (ground-truth matrix)
  python3 scripts/generate_compliance_matrix.py --source golden

  # Custom output
  python3 scripts/generate_compliance_matrix.py \
      --batch-results data/08_evaluation/batch_results.json \
      --output-dir    data/08_evaluation
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/08_evaluation"

FAMILIES = ["M1", "M2", "M3", "M4", "M5", "M6",
            "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]

POSITIVE_STATUSES = {"Fully Addressed", "Partially Addressed"}


def family_from_control_id(control_id: str) -> str:
    """'T4.2.1' → 'T4', 'M1.3.2' → 'M1'."""
    parts = control_id.strip().split(".")
    return parts[0] if parts else "??"


def build_matrix_from_batch(batch_results: list) -> tuple[dict, set]:
    """
    Build matrix from pipeline batch_results.json.
    Each result has mappings under 'mappings_per_family' if available,
    otherwise we approximate using TP counts and golden family breakdown.
    Returns (matrix, policy_names).
    """
    # batch_eval.py saves per-policy results; we need per-family breakdown.
    # If not present, use the golden fallback.
    matrix = defaultdict(lambda: defaultdict(int))
    policies = set()

    for r in batch_results:
        if "error" in r or "skipped" in r:
            continue
        doc_id = r.get("doc_id", "")
        policy_name = doc_id.split("clientname-IS-POL-00-")[-1] if "clientname-IS-POL-00-" in doc_id else doc_id
        policies.add(policy_name)

        # Use family breakdown if stored, else use golden_pos count as approximation
        family_breakdown = r.get("family_breakdown", {})
        if family_breakdown:
            for fam, count in family_breakdown.items():
                matrix[policy_name][fam] += count
        else:
            # Fallback: mark TP as distributed across families (unknown exact breakdown)
            matrix[policy_name]["??"] += r.get("tp", 0)

    return dict(matrix), policies


def build_matrix_from_golden(golden: list) -> tuple[dict, set]:
    """
    Build ground-truth matrix from golden dataset.
    Cell = count of FA+PA records per (policy_name, family).
    """
    matrix = defaultdict(lambda: defaultdict(int))
    policies = set()

    for r in golden:
        status = r.get("compliance_status", "")
        if status not in POSITIVE_STATUSES:
            continue
        policy = r.get("policy_name", "unknown")
        cid = (r.get("corrected_control_id") or r.get("control_id") or "").strip()
        family = family_from_control_id(cid)
        if policy and family:
            matrix[policy][family] += 1
            policies.add(policy)

    return dict(matrix), policies


def write_csv(matrix: dict, policies: list, families: list, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Policy"] + families + ["Total"])
        for policy in policies:
            row = [policy]
            total = 0
            for fam in families:
                count = matrix.get(policy, {}).get(fam, 0)
                row.append(count)
                total += count
            row.append(total)
            writer.writerow(row)


def print_matrix(matrix: dict, policies: list, families: list) -> None:
    col_w = 5
    header = f"{'Policy':<50}" + "".join(f"{f:>{col_w}}" for f in families) + f"{'Total':>{col_w}}"
    print(header)
    print("-" * len(header))
    for policy in policies:
        total = sum(matrix.get(policy, {}).get(f, 0) for f in families)
        row = f"{policy[:50]:<50}"
        row += "".join(
            f"{'✓' if matrix.get(policy, {}).get(f, 0) > 0 else '-':>{col_w}}"
            for f in families
        )
        row += f"{total:>{col_w}}"
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["batch", "golden"], default="golden",
                    help="'batch' = use pipeline results; 'golden' = use ground-truth annotations")
    ap.add_argument("--batch-results", default=str(OUT_DIR / "batch_results.json"))
    ap.add_argument("--golden",        default=str(ROOT / "data/07_golden_mapping/golden_mapping_dataset_clean.json"))
    ap.add_argument("--output-dir",    default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "batch":
        batch_path = Path(args.batch_results)
        if not batch_path.exists():
            print(f"Batch results not found: {batch_path}")
            print("Run: python3 scripts/batch_eval.py  first")
            return
        batch = json.load(open(batch_path, encoding="utf-8"))
        matrix, policies = build_matrix_from_batch(batch)
        source_label = "pipeline"
    else:
        golden_path = Path(args.golden)
        if not golden_path.exists():
            fallback = ROOT / "data/07_golden_mapping/golden_mapping_dataset.json"
            print(f"Clean golden not found; using raw: {fallback.name}")
            golden_path = fallback
        golden = json.load(open(golden_path, encoding="utf-8"))
        matrix, policies = build_matrix_from_golden(golden)
        source_label = "golden"

    policies_sorted = sorted(policies)
    active_families = [f for f in FAMILIES if any(matrix.get(p, {}).get(f, 0) > 0 for p in policies_sorted)]

    print(f"\nCompliance Matrix (source={source_label})")
    print(f"  Policies : {len(policies_sorted)}")
    print(f"  Families with coverage: {active_families}\n")
    print_matrix(matrix, policies_sorted, active_families)

    # Serialise
    matrix_data = {
        "source": source_label,
        "families": FAMILIES,
        "active_families": active_families,
        "policies": policies_sorted,
        "matrix": {
            policy: {fam: matrix.get(policy, {}).get(fam, 0) for fam in FAMILIES}
            for policy in policies_sorted
        },
    }

    json_path = out_dir / "compliance_matrix.json"
    csv_path  = out_dir / "compliance_matrix.csv"

    json.dump(matrix_data, open(json_path, "w", encoding="utf-8"), indent=2)
    write_csv(matrix, policies_sorted, FAMILIES, csv_path)

    print(f"\nSaved → {json_path}")
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
