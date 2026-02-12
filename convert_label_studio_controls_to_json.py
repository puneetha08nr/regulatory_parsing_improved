#!/usr/bin/env python3
"""
Convert Label Studio export for UAE IA controls to our code's format.

Input: Label Studio export JSON (e.g. data/02_processed/label_Studio_mapperd_uae.json)
       with task data.control_id, data.control_family (string), data.control_subfamily (string),
       data.control_name, data.control_description, data.sub_controls, etc., and
       annotations[0].result with edit_description, edit_control_name, edit_subcontrols,
       edit_guidelines, edit_applicability, validation_status.

Output: JSON in the same shape as uae_ia_controls_corrected.json:
        [{ "control_family": { "number", "name" }, "control_subfamily": { "number", "name" },
          "control": { "id", "name", "description", "sub_controls", "implementation_guidelines",
                       "external_factors", "internal_factors", "guidance_points" },
          "applicablility", "breadcrumb", "validation_status", "correction_notes" }, ...]

Usage:
  python3 convert_label_studio_controls_to_json.py \
    --input data/02_processed/label_Studio_mapperd_uae.json \
    --output data/02_processed/uae_ia_controls_from_label_studio.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _text_from_value(val: Any) -> str:
    t = val.get("text")
    if t is None:
        return ""
    if isinstance(t, list):
        return "\n".join(str(x) for x in t if x).strip()
    return str(t).strip()


def _list_from_value(val: Any) -> List[str]:
    t = val.get("text")
    if t is None:
        return []
    if isinstance(t, list):
        out = []
        for x in t:
            s = str(x).strip()
            if s:
                # One item might be multiline
                out.extend(line.strip() for line in s.split("\n") if line.strip())
        return out
    return [line.strip() for line in str(t).split("\n") if line.strip()]


def _choices_from_value(val: Any) -> Optional[str]:
    ch = val.get("choices") or val.get("selected_labels") or []
    return ch[0] if ch else None


def parse_family(s: str) -> Dict[str, str]:
    """Parse 'M1 - Strategy and Planning' -> { number: 'M1', name: 'Strategy and Planning' }"""
    if not s or not isinstance(s, str):
        return {"number": "", "name": ""}
    s = s.strip()
    m = re.match(r"^([MT]\d+(?:\.\d+)?)\s*[-–]\s*(.*)$", s, re.IGNORECASE)
    if m:
        return {"number": m.group(1).strip(), "name": (m.group(2) or "").strip()}
    if re.match(r"^[MT]\d+", s):
        return {"number": s.split()[0] if s.split() else s, "name": " ".join(s.split()[1:])}
    return {"number": "", "name": s}


def _list_from_data(data: Any) -> List[str]:
    if not data:
        return []
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    return [line.strip() for line in str(data).split("\n") if line.strip()]


def parse_result_to_corrected(result: List[Dict], data: Dict) -> Dict[str, Any]:
    """Extract corrected fields from annotation result. Supports edit_* and control_* from_name."""
    corrected = {
        "description": data.get("control_description", ""),
        "name": data.get("control_name", ""),
        "sub_controls": _list_from_data(data.get("sub_controls")),
        "implementation_guidelines": data.get("implementation_guidelines", ""),
        "external_factors": _list_from_data(data.get("external_factors", "")),
        "internal_factors": _list_from_data(data.get("internal_factors", "")),
        "guidance_points": _list_from_data(data.get("guidance_points", "")),
        "applicability": data.get("edit_applicability", ""),
        "validation_status": None,
        "correction_notes": None,
    }

    for r in result:
        from_name = (r.get("from_name") or "").lower()
        val = r.get("value") or {}
        if "edit_description" in from_name or "control_description" in from_name:
            corrected["description"] = _text_from_value(val)
        elif "edit_control_name" in from_name or "control_name" in from_name:
            corrected["name"] = _text_from_value(val)
        elif "edit_subcontrols" in from_name or "sub_controls" in from_name:
            corrected["sub_controls"] = _list_from_value(val)
        elif "edit_guidelines" in from_name or "implementation_guidelines" in from_name:
            corrected["implementation_guidelines"] = _text_from_value(val)
        elif "edit_applicability" in from_name or "applicability" in from_name:
            corrected["applicability"] = _text_from_value(val)
        elif "edit_external" in from_name or "external_factors" in from_name:
            corrected["external_factors"] = _list_from_value(val)
        elif "edit_internal" in from_name or "internal_factors" in from_name:
            corrected["internal_factors"] = _list_from_value(val)
        elif "edit_guidance" in from_name or "guidance_points" in from_name:
            corrected["guidance_points"] = _list_from_value(val)
        elif "validation_status" in from_name:
            corrected["validation_status"] = _choices_from_value(val)
        elif "correction_notes" in from_name:
            corrected["correction_notes"] = _text_from_value(val) or None
    return corrected


def applicability_to_list(s: str) -> List[str]:
    """'P1, always' -> ['P1', 'always']"""
    if not s:
        return []
    return [x.strip() for x in re.split(r"[,;]", str(s)) if x.strip()]


def convert_export_to_controls(label_studio_path: str, output_path: str) -> int:
    raw = load_json(label_studio_path)
    items = raw if isinstance(raw, list) else [raw]
    out_controls = []
    for item in items:
        anns = item.get("annotations", [])
        data = item.get("data", {})
        control_id = data.get("control_id") or (item.get("meta") or {}).get("control_id")
        if not control_id:
            continue
        if not anns:
            # No annotation: use data only
            fam_str = data.get("control_family", "")
            subfam_str = data.get("control_subfamily", "")
            fam = parse_family(fam_str)
            subfam = parse_family(subfam_str)
            name = data.get("control_name", "")
            desc = data.get("control_description", "")
            subs = data.get("sub_controls", "")
            if isinstance(subs, str):
                sub_list = [x.strip() for x in subs.split("\n") if x.strip()]
            else:
                sub_list = list(subs) if subs else []
            appl = data.get("edit_applicability", "")
            out_controls.append({
                "control_family": fam,
                "control_subfamily": subfam,
                "control": {
                    "id": control_id,
                    "name": name,
                    "description": desc,
                    "sub_controls": sub_list,
                    "implementation_guidelines": data.get("implementation_guidelines", ""),
                    "external_factors": data.get("external_factors") if isinstance(data.get("external_factors"), list) else [],
                    "internal_factors": data.get("internal_factors") if isinstance(data.get("internal_factors"), list) else [],
                    "guidance_points": data.get("guidance_points") if isinstance(data.get("guidance_points"), list) else [],
                },
                "applicablility": applicability_to_list(appl),
                "breadcrumb": f"{fam.get('name', '')} > {subfam.get('name', '')} > {name}".strip(" >"),
                "validation_status": None,
                "correction_notes": None,
            })
            continue
        result = anns[0].get("result", [])
        cor = parse_result_to_corrected(result, data)
        fam_str = data.get("control_family", "")
        subfam_str = data.get("control_subfamily", "")
        fam = parse_family(fam_str)
        subfam = parse_family(subfam_str)
        appl_list = applicability_to_list(cor["applicability"])
        out_controls.append({
            "control_family": fam,
            "control_subfamily": subfam,
            "control": {
                "id": control_id,
                "name": cor["name"],
                "description": cor["description"],
                "sub_controls": cor["sub_controls"],
                "implementation_guidelines": cor["implementation_guidelines"],
                "external_factors": cor["external_factors"],
                "internal_factors": cor["internal_factors"],
                "guidance_points": cor["guidance_points"],
            },
            "applicablility": appl_list,
            "breadcrumb": f"{fam.get('name', '')} > {subfam.get('name', '')} > {cor['name']}".strip(" >"),
            "validation_status": cor["validation_status"],
            "correction_notes": cor["correction_notes"],
        })
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_controls, f, indent=2, ensure_ascii=False)
    return len(out_controls)


def main():
    p = argparse.ArgumentParser(description="Convert Label Studio UAE IA controls export to our JSON format")
    p.add_argument("--input", "-i", default="data/02_processed/label_Studio_mapperd_uae.json", help="Label Studio export JSON")
    p.add_argument("--output", "-o", default="data/02_processed/uae_ia_controls_from_label_studio.json", help="Output path")
    args = p.parse_args()
    n = convert_export_to_controls(args.input, args.output)
    print(f"Converted {n} controls → {args.output}")


if __name__ == "__main__":
    main()
