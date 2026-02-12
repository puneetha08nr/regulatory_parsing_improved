#!/usr/bin/env python3
"""
Split mappings.json into one JSON file per policy document.

mappings.json contains results for all policy docs combined. This script
groups by policy (using target_policy_id: doc id is the part before _passage_N)
and writes data/06_compliance_mappings/by_policy/<policy_safe_name>.json.

Usage:
  python3 split_mappings_by_policy.py \
    --mappings data/06_compliance_mappings/mappings.json \
    --output-dir data/06_compliance_mappings/by_policy
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def policy_doc_id_from_target(target_policy_id: str) -> str:
    """Extract policy document id from target_policy_id.
    e.g. 'clientname-IS-POL-00-Logging and Monitoring Policy 2_passage_11' -> same without _passage_11.
    """
    if not target_policy_id:
        return "_unknown_"
    return re.sub(r"_passage_\d+$", "", target_policy_id).strip()


def safe_filename(policy_doc_id: str) -> str:
    """Turn policy doc id into a safe filename (no spaces/special chars)."""
    s = policy_doc_id.strip()
    # Replace path-unsafe and common separators with underscore
    s = re.sub(r'[\s\\/:*?"<>|]+', "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed_policy"


def split_by_policy(mappings_path: str, output_dir: str) -> Dict[str, int]:
    """Group mappings by policy doc and write one JSON per policy. Returns policy_doc_id -> count."""
    mappings = load_json(mappings_path)
    if not isinstance(mappings, list):
        mappings = [mappings]

    by_policy: Dict[str, List[Dict]] = {}
    for m in mappings:
        pid = m.get("target_policy_id") or m.get("policy_passage_id") or ""
        doc_id = policy_doc_id_from_target(pid)
        by_policy.setdefault(doc_id, []).append(m)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for doc_id, rows in sorted(by_policy.items()):
        name = safe_filename(doc_id)
        path = out / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        counts[doc_id] = len(rows)

    # Write a small index so you know which file is which
    index = [
        {"policy_doc_id": doc_id, "file": f"{safe_filename(doc_id)}.json", "count": n}
        for doc_id, n in sorted(counts.items(), key=lambda x: -x[1])
    ]
    with open(out / "_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return counts


def main():
    p = argparse.ArgumentParser(description="Split mappings.json into one JSON per policy document")
    p.add_argument(
        "--mappings",
        default="data/06_compliance_mappings/mappings.json",
        help="Path to combined mappings.json",
    )
    p.add_argument(
        "--output-dir",
        default="data/06_compliance_mappings/by_policy",
        help="Directory to write per-policy JSON files",
    )
    args = p.parse_args()

    counts = split_by_policy(args.mappings, args.output_dir)
    total = sum(counts.values())
    print(f"Split {total} mappings into {len(counts)} policy files under {args.output_dir}/")
    for doc_id, n in sorted(counts.items(), key=lambda x: -x[1]):
        name = safe_filename(doc_id)
        print(f"  {n:5}  {name}.json  ({doc_id})")
    print(f"Index: {args.output_dir}/_index.json")


if __name__ == "__main__":
    main()
