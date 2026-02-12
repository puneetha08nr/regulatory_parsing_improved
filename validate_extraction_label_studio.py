#!/usr/bin/env python3
from __future__ import annotations

"""
Create Label Studio tasks for validating and correcting extracted content.

Two modes:
1. Validate UAE IA controls extraction (from improved_control_extractor.py)
2. Validate policy passages extraction (from flexible_policy_extractor.py)

Usage:
  # Generate tasks for control validation
  python validate_extraction_label_studio.py --mode controls --input data/02_processed/uae_ia_controls_structured.json

  # Generate tasks for one policy document
  python validate_extraction_label_studio.py --mode policies --input data/02_processed/policies/<doc>_for_mapping.json --output data/03_label_studio_input/policy_validation_tasks/<doc>_validation_tasks.json

  # Export corrected content (full list, original order: pass --source)
  python validate_extraction_label_studio.py --mode export --input <label_studio_export.json> --type controls --source data/02_processed/uae_ia_controls_structured_v2.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str) -> List[Dict]:
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def format_list(items: List[str]) -> str:
    """Format list as text (one per line)"""
    if not items:
        return ""
    return "\n".join(str(item) for item in items if item)


def generate_control_validation_tasks(controls_path: str, output_path: str, max_tasks: int = 0) -> int:
    """Generate Label Studio tasks for validating extracted controls"""
    controls = load_json(controls_path)
    
    tasks = []
    for ctrl in controls:
        fam = ctrl.get("control_family", {})
        subfam = ctrl.get("control_subfamily", {})
        c = ctrl.get("control", {})
        
        # Format sub-controls
        sub_controls_text = format_list(c.get("sub_controls", []))
        
        # Format lists
        external_factors_text = format_list(c.get("external_factors", []))
        internal_factors_text = format_list(c.get("internal_factors", []))
        guidance_points_text = format_list(c.get("guidance_points", []))
        
        task = {
            "data": {
                "control_id": c.get("id", ""),
                "control_family": f"{fam.get('number', '')} - {fam.get('name', '')}",
                "control_subfamily": f"{subfam.get('number', '')} - {subfam.get('name', '')}",
                "control_name": c.get("name", ""),
                "control_description": c.get("description", ""),
                "sub_controls": sub_controls_text,
                "implementation_guidelines": c.get("implementation_guidelines", ""),
                "external_factors": external_factors_text,
                "internal_factors": internal_factors_text,
                "guidance_points": guidance_points_text,
            },
            "meta": {
                "control_id": c.get("id", ""),
                "source_file": controls_path
            }
        }
        tasks.append(task)
        
        if max_tasks and len(tasks) >= max_tasks:
            break
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    return len(tasks)


def generate_policy_validation_tasks(policies_path: str, output_path: str, max_tasks: int = 0) -> int:
    """Generate Label Studio tasks for validating extracted policy passages"""
    policies = load_json(policies_path)
    
    tasks = []
    for policy in policies:
        metadata = policy.get("metadata", {})
        
        task = {
            "data": {
                "policy_id": metadata.get("policy_id", policy.get("id", "")),
                "policy_name": metadata.get("policy_name", policy.get("name", "")),
                "passage_id": policy.get("id", ""),
                "section": policy.get("section", ""),
                "heading": metadata.get("heading", policy.get("name", "")),
                "passage_text": policy.get("text", ""),
            },
            "meta": {
                "passage_id": policy.get("id", ""),
                "source_file": policies_path
            }
        }
        tasks.append(task)
        
        if max_tasks and len(tasks) >= max_tasks:
            break
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    return len(tasks)


def _parse_corrected_from_result(result: List[Dict], data: Dict) -> Dict[str, Any]:
    """Parse Label Studio result into a flat corrected dict (description, sub_controls, etc.)."""
    corrected = {
        "description": data.get("control_description", ""),
        "sub_controls": [],
        "implementation_guidelines": data.get("implementation_guidelines", ""),
        "external_factors": [],
        "internal_factors": [],
        "guidance_points": [],
        "validation_status": None,
        "correction_notes": None,
    }
    for r in result:
        from_name = r.get("from_name", "")
        val = r.get("value") or {}
        if "control_description" in from_name:
            text = val.get("text") or []
            corrected["description"] = "\n".join(str(t) for t in (text if isinstance(text, list) else [text])).strip()
        elif "sub_controls" in from_name:
            text = val.get("text") or []
            if isinstance(text, list):
                corrected["sub_controls"] = [t.strip() for t in text if (t and str(t).strip())]
            else:
                corrected["sub_controls"] = [t.strip() for t in str(text).split("\n") if t.strip()]
        elif "implementation_guidelines" in from_name:
            text = val.get("text") or []
            corrected["implementation_guidelines"] = "\n".join(str(t) for t in (text if isinstance(text, list) else [text])).strip()
        elif "external_factors" in from_name:
            text = val.get("text") or []
            if isinstance(text, list):
                corrected["external_factors"] = [t.strip() for t in text if (t and str(t).strip())]
            else:
                corrected["external_factors"] = [t.strip() for t in str(text).split("\n") if t.strip()]
        elif "internal_factors" in from_name:
            text = val.get("text") or []
            if isinstance(text, list):
                corrected["internal_factors"] = [t.strip() for t in text if (t and str(t).strip())]
            else:
                corrected["internal_factors"] = [t.strip() for t in str(text).split("\n") if t.strip()]
        elif "guidance_points" in from_name:
            text = val.get("text") or []
            if isinstance(text, list):
                corrected["guidance_points"] = [t.strip() for t in text if (t and str(t).strip())]
            else:
                corrected["guidance_points"] = [t.strip() for t in str(text).split("\n") if t.strip()]
        elif "validation_status" in from_name:
            choices = val.get("choices") or val.get("selected_labels") or []
            if choices:
                corrected["validation_status"] = choices[0]
        elif "correction_notes" in from_name:
            text = val.get("text") or []
            corrected["correction_notes"] = "\n".join(str(t) for t in (text if isinstance(text, list) else [text])).strip() or None
    return corrected


def export_corrected_controls(
    label_studio_export_path: str,
    output_path: str,
    source_controls_path: Optional[str] = None,
) -> int:
    """Export corrected controls from Label Studio annotations.

    If source_controls_path is provided: merges annotations into the source controls list,
    preserving full count, original order, and nested schema (control_family, control_subfamily, control).
    Unannotated controls are kept as-is. Output is one JSON array matching source structure.

    If source_controls_path is not provided: outputs only annotated tasks as a flat list (legacy).
    """
    with open(label_studio_export_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else [raw]

    # Build map: control_id -> corrected fields (from annotations)
    corrected_by_id: Dict[str, Dict[str, Any]] = {}
    for item in items:
        anns = item.get("annotations", [])
        if not anns:
            continue
        data = item.get("data", {})
        control_id = data.get("control_id") or (data.get("meta") or {}).get("control_id")
        if not control_id:
            continue
        result = anns[0].get("result", [])
        corrected_by_id[control_id] = _parse_corrected_from_result(result, data)

    if source_controls_path:
        # Merge with source: preserve order and full list, same nested schema
        source_controls = load_json(source_controls_path)
        if not isinstance(source_controls, list):
            source_controls = [source_controls]
        merged = []
        for ctrl in source_controls:
            fam = ctrl.get("control_family") or {}
            subfam = ctrl.get("control_subfamily") or {}
            c = ctrl.get("control") or {}
            cid = c.get("id", "")
            if not cid:
                merged.append(ctrl)
                continue
            if cid in corrected_by_id:
                cor = corrected_by_id[cid]
                merged.append({
                    "control_family": fam,
                    "control_subfamily": subfam,
                    "control": {
                        "id": c.get("id"),
                        "name": c.get("name"),
                        "description": cor.get("description", c.get("description", "")),
                        "sub_controls": cor.get("sub_controls") if cor.get("sub_controls") is not None else c.get("sub_controls", []),
                        "implementation_guidelines": cor.get("implementation_guidelines") if cor.get("implementation_guidelines") is not None else c.get("implementation_guidelines", ""),
                        "external_factors": cor.get("external_factors") if cor.get("external_factors") is not None else c.get("external_factors", []),
                        "internal_factors": cor.get("internal_factors") if cor.get("internal_factors") is not None else c.get("internal_factors", []),
                        "guidance_points": cor.get("guidance_points") if cor.get("guidance_points") is not None else c.get("guidance_points", []),
                    },
                    "applicablility": ctrl.get("applicablility", ctrl.get("applicability", [])),
                    "breadcrumb": ctrl.get("breadcrumb", ""),
                    "validation_status": corrected_by_id[cid].get("validation_status"),
                    "correction_notes": corrected_by_id[cid].get("correction_notes"),
                })
            else:
                merged.append(ctrl)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        return len(merged)
    else:
        # Legacy: only annotated tasks, flat list
        corrected_controls = []
        for item in items:
            anns = item.get("annotations", [])
            if not anns:
                continue
            data = item.get("data", {})
            result = anns[0].get("result", [])
            cor = _parse_corrected_from_result(result, data)
            corrected_controls.append({
                "control_id": data.get("control_id"),
                "control_name": data.get("control_name"),
                "control_family": data.get("control_family"),
                "control_subfamily": data.get("control_subfamily"),
                "description": cor["description"],
                "sub_controls": cor["sub_controls"],
                "implementation_guidelines": cor["implementation_guidelines"],
                "external_factors": cor["external_factors"],
                "internal_factors": cor["internal_factors"],
                "guidance_points": cor["guidance_points"],
                "validation_status": cor["validation_status"],
                "correction_notes": cor["correction_notes"],
            })
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(corrected_controls, f, indent=2, ensure_ascii=False)
        return len(corrected_controls)


def export_corrected_policies(label_studio_export_path: str, output_path: str) -> int:
    """Export corrected policy passages from Label Studio annotations"""
    with open(label_studio_export_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    items = raw if isinstance(raw, list) else [raw]
    corrected_policies = []
    
    for item in items:
        anns = item.get("annotations", [])
        if not anns:
            continue
        
        data = item.get("data", {})
        result = anns[0].get("result", [])
        
        # Build output in same format as flexible_policy_extractor (drop-in for all_policies_for_mapping.json)
        passage_id = data.get("passage_id", "")
        policy_id = data.get("policy_id", "")
        policy_name = data.get("policy_name", "")
        section = data.get("section", "")
        heading = data.get("heading", "")
        passage_text = data.get("passage_text", "")
        validation_status = None
        correction_notes = None

        for r in result:
            from_name = r.get("from_name", "")
            val = r.get("value") or {}
            if "passage_text" in from_name:
                text = val.get("text") or []
                if isinstance(text, list):
                    passage_text = "\n".join(str(t) for t in text).strip()
                else:
                    passage_text = str(text).strip()
            elif "validation_status" in from_name:
                choices = val.get("choices") or val.get("selected_labels") or []
                if choices:
                    validation_status = choices[0]
            elif "correction_notes" in from_name:
                text = val.get("text") or []
                if isinstance(text, list):
                    correction_notes = "\n".join(str(t) for t in text).strip() or None
                else:
                    correction_notes = str(text).strip() or None

        corrected_policies.append({
            "id": passage_id,
            "name": f"{policy_name} - {section or heading or passage_id}",
            "text": passage_text,
            "section": section,
            "metadata": {
                "policy_id": policy_id,
                "policy_name": policy_name,
                "heading": heading,
                "validation_status": validation_status,
                "correction_notes": correction_notes,
            },
        })
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(corrected_policies, f, indent=2, ensure_ascii=False)
    
    return len(corrected_policies)


def main():
    parser = argparse.ArgumentParser(
        description="Validate and correct extracted content using Label Studio"
    )
    parser.add_argument(
        "--mode",
        choices=["controls", "policies", "export"],
        required=True,
        help="Mode: generate tasks for controls/policies, or export corrected content"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input: controls/policies JSON (for generate) or Label Studio export (for export)"
    )
    parser.add_argument(
        "--output",
        help="Output path (defaults based on mode)"
    )
    parser.add_argument(
        "--type",
        choices=["controls", "policies"],
        help="Type for export mode (controls or policies)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Limit number of tasks (0 = all)"
    )
    parser.add_argument(
        "--source",
        dest="source_controls",
        help="For export --type controls: path to original controls JSON. Merges annotations into it so output is full list, original order, same schema."
    )

    args = parser.parse_args()
    
    if args.mode == "controls":
        output = args.output or "data/03_label_studio_input/control_validation_tasks.json"
        n = generate_control_validation_tasks(
            controls_path=args.input,
            output_path=output,
            max_tasks=args.max_tasks
        )
        print(f"✓ Generated {n} control validation tasks → {output}")
        print("Next: Import into Label Studio with config:")
        print("  data/03_label_studio_input/validate_control_extraction.xml")
    
    elif args.mode == "policies":
        output = args.output or "data/03_label_studio_input/policy_validation_tasks.json"
        n = generate_policy_validation_tasks(
            policies_path=args.input,
            output_path=output,
            max_tasks=args.max_tasks
        )
        print(f"✓ Generated {n} policy validation tasks → {output}")
        print("Next: Import into Label Studio with config:")
        print("  data/03_label_studio_input/validate_policy_extraction.xml")
    
    elif args.mode == "export":
        if not args.type:
            raise SystemExit("For mode=export, specify --type controls or --type policies")
        
        if args.type == "controls":
            output = args.output or "data/02_processed/uae_ia_controls_corrected.json"
            n = export_corrected_controls(
                label_studio_export_path=args.input,
                output_path=output,
                source_controls_path=getattr(args, "source_controls", None),
            )
            print(f"✓ Exported {n} corrected controls → {output}")
        else:
            output = args.output or "data/02_processed/policies/all_policies_corrected.json"
            n = export_corrected_policies(
                label_studio_export_path=args.input,
                output_path=output
            )
            print(f"✓ Exported {n} corrected policy passages → {output}")


if __name__ == "__main__":
    main()
