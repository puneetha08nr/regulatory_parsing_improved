#!/usr/bin/env python3
"""
Generate Label Studio tasks from mappings.json so you can manually annotate
(correct) the pipeline status. One task = one row from mappings.json.

Usage:
  1. Generate tasks:
     python3 annotate_mappings_label_studio.py generate \
       --mappings data/06_compliance_mappings/mappings.json \
       --controls data/02_processed/uae_ia_controls_corrected.json \
       --policies data/02_processed/policies/all_policies_for_mapping.json \
       --output data/03_label_studio_input/annotate_mappings_tasks.json

  2. In Label Studio: use config data/03_label_studio_input/annotate_mappings.xml, import the JSON.

  3. After annotating, export from Label Studio and run:
     python3 annotate_mappings_label_studio.py export \
       --input <label_studio_export.json> \
       --output data/06_compliance_mappings/mappings_annotated.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_control_text(ctrl: Dict) -> str:
    c = ctrl.get("control", {})
    desc = (c.get("description") or "").strip()
    subs = c.get("sub_controls") or []
    if subs:
        desc += "\n\nSub-controls:\n" + "\n".join((s for s in subs[:5] if s))
    return desc or (c.get("name") or "")


def generate_tasks(
    mappings_path: str,
    controls_path: str,
    policies_path: str,
    output_path: str,
    max_tasks: int = 0,
) -> int:
    """Build one Label Studio task per row in mappings.json."""
    mappings = load_json(mappings_path)
    if not isinstance(mappings, list):
        mappings = [mappings]

    controls_raw = load_json(controls_path)
    controls_list = controls_raw if isinstance(controls_raw, list) else [controls_raw]
    control_map = {}
    for ctrl in controls_list:
        c = ctrl.get("control", {})
        control_map[c.get("id", "")] = ctrl

    if Path(policies_path).exists():
        policies_list = load_json(policies_path)
    else:
        from flexible_policy_extractor import load_all_policies_from_dir
        policies_list = load_all_policies_from_dir(str(Path(policies_path).parent))
    if not isinstance(policies_list, list):
        policies_list = [policies_list]
    policy_map = {p["id"]: p for p in policies_list}

    tasks = []
    for m in mappings:
        cid = m.get("source_control_id") or m.get("control_id") or ""
        pid = m.get("target_policy_id") or m.get("policy_passage_id") or ""
        if not cid or not pid:
            continue

        ctrl = control_map.get(cid)
        pp = policy_map.get(pid)

        if ctrl:
            fam = ctrl.get("control_family") or {}
            c = ctrl.get("control") or {}
            control_text = build_control_text(ctrl)
            control_name = (c.get("name") or "").strip()
            control_family = f"{fam.get('number', '')} - {fam.get('name', '')}".strip(" -")
        else:
            control_name = ""
            control_family = ""
            control_text = "(Control not found in controls file — select from dropdown below.)"

        if pp:
            policy_text = (pp.get("text") or "")[:4000]
            policy_section = (pp.get("section") or pp.get("name") or "").strip()
            policy_name = ((pp.get("metadata") or {}).get("policy_name") or pp.get("id") or "").strip()
        else:
            policy_text = "(Policy passage not found in policies file.)"
            policy_section = ""
            policy_name = ""

        pipeline_status = (m.get("status") or "").strip()
        pipeline_evidence = (m.get("evidence_text") or "")[:2000]

        task = {
            "data": {
                "mapping_id": (m.get("mapping_id") or "").strip(),
                "control_id": cid,
                "control_name": control_name,
                "control_family": control_family,
                "control_text": (control_text or "")[:4000],
                "policy_passage_id": pid,
                "policy_name": policy_name,
                "policy_section": policy_section,
                "policy_text": policy_text,
                "pipeline_status": pipeline_status,
                "pipeline_evidence": pipeline_evidence,
            },
            "meta": {"control_id": cid, "policy_passage_id": pid, "mapping_id": m.get("mapping_id", "")},
        }
        tasks.append(task)
        if max_tasks and len(tasks) >= max_tasks:
            break

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    return len(tasks)


def export_annotated(
    label_studio_export_path: str,
    output_path: str,
) -> int:
    """Read Label Studio export and write mappings with corrected_status and notes."""
    raw = load_json(label_studio_export_path)
    items = raw if isinstance(raw, list) else [raw]
    rows = []
    for item in items:
        anns = item.get("annotations", [])
        if not anns:
            continue
        data = item.get("data", {})
        result = anns[0].get("result", [])
        corrected_status = None
        notes = None
        corrected_evidence = None
        uae_ia_control_ids = []
        iso27001_control_ids = []
        adhics_control_ids = []

        def taxonomy_leafs(tax: list) -> list:
            out = []
            if not tax or not isinstance(tax, list):
                return out
            for path in tax:
                if isinstance(path, list) and path:
                    out.append(path[-1])
                elif isinstance(path, str):
                    out.append(path)
            return out

        for r in result:
            from_name = (r.get("from_name") or "")
            val = r.get("value") or {}
            if "corrected_status" in from_name:
                tax = val.get("taxonomy")
                if tax and isinstance(tax, list) and len(tax) > 0:
                    path = tax[0] if isinstance(tax[0], list) else tax
                    corrected_status = path[-1] if isinstance(path, list) and path else None
                else:
                    ch = val.get("choices") or val.get("selected_labels") or []
                    if ch:
                        corrected_status = ch[0] if isinstance(ch[0], str) else str(ch[0])
            if "notes" in from_name:
                t = val.get("text")
                if isinstance(t, list):
                    notes = " ".join(str(x) for x in t).strip() or None
                else:
                    notes = (str(t).strip() if t else None) or None
            if "corrected_evidence" in from_name:
                t = val.get("text")
                if isinstance(t, list):
                    corrected_evidence = " ".join(str(x) for x in t).strip() or None
                else:
                    corrected_evidence = (str(t).strip() if t else None) or None
            if "uae_ia_control_ids" in from_name:
                uae_ia_control_ids = taxonomy_leafs(val.get("taxonomy") or [])
            if "iso27001_control_ids" in from_name:
                iso27001_control_ids = taxonomy_leafs(val.get("taxonomy") or [])
            if "adhics_control_ids" in from_name:
                adhics_control_ids = taxonomy_leafs(val.get("taxonomy") or [])

        rows.append({
            "mapping_id": data.get("mapping_id"),
            "source_control_id": data.get("control_id"),
            "target_policy_id": data.get("policy_passage_id"),
            "pipeline_status": data.get("pipeline_status"),
            "corrected_status": corrected_status,
            "uae_ia_control_ids": uae_ia_control_ids,
            "iso27001_control_ids": iso27001_control_ids,
            "adhics_control_ids": adhics_control_ids,
            "corrected_evidence": corrected_evidence,
            "notes": notes,
            "control_name": data.get("control_name"),
            "policy_name": data.get("policy_name"),
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="Annotate mappings.json in Label Studio")
    p.add_argument("mode", choices=["generate", "export"], help="generate tasks or export annotated")
    p.add_argument("--mappings", default="data/06_compliance_mappings/mappings.json")
    p.add_argument("--controls", default="data/02_processed/uae_ia_controls_corrected.json")
    p.add_argument("--policies", default="data/02_processed/policies/all_policies_for_mapping.json")
    p.add_argument("--output", default="data/03_label_studio_input/annotate_mappings_tasks.json")
    p.add_argument("--input", dest="input_export", help="Label Studio export JSON (for mode=export)")
    p.add_argument("--max-tasks", type=int, default=0, help="Cap number of tasks (0 = all)")
    args = p.parse_args()

    if args.mode == "generate":
        n = generate_tasks(
            mappings_path=args.mappings,
            controls_path=args.controls,
            policies_path=args.policies,
            output_path=args.output,
            max_tasks=args.max_tasks or 0,
        )
        print(f"Generated {n} tasks → {args.output}")
        print("Next: In Label Studio use config data/03_label_studio_input/annotate_mappings.xml")
        print("      then Import → Upload Files and select the JSON above (do not import raw mappings.json).")
    else:
        if not args.input_export:
            raise SystemExit("For mode=export pass --input <label_studio_export.json>")
        out = args.output
        if out == "data/03_label_studio_input/annotate_mappings_tasks.json":
            out = "data/06_compliance_mappings/mappings_annotated.json"
        n = export_annotated(label_studio_export_path=args.input_export, output_path=out)
        print(f"Exported {n} annotated rows → {out}")


if __name__ == "__main__":
    main()
