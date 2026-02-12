#!/usr/bin/env python3
"""
Generate Label Studio tasks for assigning UAE IA and ISO 27001 controls to parsed
policy passages. One task = one policy passage; annotators assign control IDs.

Usage (single combined file):
  python3 generate_policy_controls_tasks.py \\
    --policies data/02_processed/policies/all_policies_for_mapping.json \\
    --output data/03_label_studio_input/policy_controls_tasks.json

Usage (one JSON per policy document):
  python3 generate_policy_controls_tasks.py \\
    --policies data/02_processed/policies/all_policies_for_mapping.json \\
    --per-policy \\
    --output-dir data/03_label_studio_input/policy_controls_by_policy

Then in Label Studio: use config data/03_label_studio_input/annotate_policy_controls.xml
and import the desired tasks JSON (combined or one file per policy).
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def _policy_doc_id(item: dict) -> str:
    """Policy document id (without _passage_N)."""
    pid = item.get("id", "")
    return re.sub(r"_passage_\d+$", "", pid).strip() or pid


def _safe_filename(name: str) -> str:
    """Safe filename from policy name."""
    return re.sub(r'[\s\\/:*?"<>|]+', "_", name.strip()) or "unnamed_policy"


def build_tasks(policy_data: list, max_tasks: int = 0) -> list:
    """Build Label Studio tasks from policy passage list."""
    tasks = []
    for item in policy_data:
        policy_passage_id = item.get("id", "")
        policy_name = item.get("name", "")
        policy_text = item.get("text", "")
        section = item.get("section", "")
        meta = item.get("metadata") or {}
        doc_name = meta.get("policy_name", policy_name)

        task = {
            "data": {
                "policy_passage_id": policy_passage_id,
                "policy_name": doc_name,
                "section": section,
                "policy_text": (policy_text or "")[:8000],
            },
            "meta": {"policy_passage_id": policy_passage_id, "policy_name": doc_name},
        }
        tasks.append(task)
        if max_tasks and len(tasks) >= max_tasks:
            break
    return tasks


def main():
    p = argparse.ArgumentParser(
        description="Generate Label Studio tasks: policy passages → assign UAE IA + ISO 27001 controls"
    )
    p.add_argument(
        "--policies",
        default="data/02_processed/policies/all_policies_for_mapping.json",
        help="Path to parsed policy passages JSON (id, name, text, section per item)",
    )
    p.add_argument(
        "--output",
        default="data/03_label_studio_input/policy_controls_tasks.json",
        help="Output Label Studio tasks JSON (used when not --per-policy)",
    )
    p.add_argument(
        "--per-policy",
        action="store_true",
        help="Write one tasks JSON per policy document (separate policies)",
    )
    p.add_argument(
        "--output-dir",
        default="data/03_label_studio_input/policy_controls_by_policy",
        help="Directory for per-policy task files (used with --per-policy)",
    )
    p.add_argument("--max-tasks", type=int, default=0, help="Cap number of tasks per file (0 = all)")
    args = p.parse_args()

    policies_path = Path(args.policies)
    if policies_path.is_file():
        with open(policies_path, "r", encoding="utf-8") as f:
            policy_data = json.load(f)
    elif policies_path.is_dir():
        from flexible_policy_extractor import load_all_policies_from_dir
        policy_data = load_all_policies_from_dir(str(policies_path))
    else:
        from flexible_policy_extractor import load_all_policies_from_dir
        policy_data = load_all_policies_from_dir(str(policies_path.parent))
    if not isinstance(policy_data, list):
        policy_data = [policy_data]

    if args.per_policy:
        # Group by policy document
        by_doc = defaultdict(list)
        for item in policy_data:
            doc_id = _policy_doc_id(item)
            meta = item.get("metadata") or {}
            doc_name = meta.get("policy_name", doc_id) or doc_id
            by_doc[(doc_id, doc_name)].append(item)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for (doc_id, doc_name), items in sorted(by_doc.items(), key=lambda x: x[0][1]):
            tasks = build_tasks(items, args.max_tasks or 0)
            safe = _safe_filename(doc_name)
            path = out_dir / f"{safe}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)
            total += len(tasks)
            print(f"  {path.name}: {len(tasks)} tasks")
        print(f"Generated {len(by_doc)} policy files ({total} tasks) → {out_dir}/")
    else:
        tasks = build_tasks(policy_data, args.max_tasks)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        print(f"Generated {len(tasks)} tasks → {args.output}")

    print("Label Studio: use config data/03_label_studio_input/annotate_policy_controls.xml")
    if args.per_policy:
        print("Import any of the JSON files from", args.output_dir)
    else:
        print("Import file:", args.output)


if __name__ == "__main__":
    main()
