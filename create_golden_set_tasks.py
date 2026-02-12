#!/usr/bin/env python3
"""
Create Label Studio tasks for compliance mapping golden set.

Each task = one (UAE IA control, policy passage) pair.
Annotators assign: Compliance status (Fully/Partially/Not Addressed) + Confidence (1–5).

Usage:
  1. Generate tasks:    python create_golden_set_tasks.py --mode generate
  2. Import into Label Studio (UI or SDK)
  3. Annotate in Label Studio
  4. Export annotations: python create_golden_set_tasks.py --mode export --input <label_studio_export.json>
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# Paths
DEFAULT_CONTROLS = "data/02_processed/uae_ia_controls_structured.json"
DEFAULT_POLICIES = "data/02_processed/policies/all_policies_for_mapping.json"
DEFAULT_OUTPUT_TASKS = "data/03_label_studio_input/golden_set_mapping_tasks.json"
DEFAULT_OUTPUT_GOLDEN = "data/07_golden_mapping/golden_mapping_dataset.json"


def load_controls(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_policies(path: str) -> List[Dict]:
    p = Path(path)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    # Fallback: load from individual *_for_mapping.json in same directory
    if p.parent.is_dir():
        from flexible_policy_extractor import load_all_policies_from_dir
        return load_all_policies_from_dir(str(p.parent))
    return []


def build_control_text(ctrl: Dict) -> str:
    c = ctrl.get("control", {})
    desc = c.get("description", "")
    subs = c.get("sub_controls", [])
    if subs:
        desc += "\n\nSub-controls:\n" + "\n".join(subs[:5])
    return desc


def snippet(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    return (text[:max_len] + "...") if len(text) > max_len else text


def load_candidate_pairs(candidates_path: str) -> List[tuple]:
    """Load (control_id, policy_passage_id) from mappings CSV or JSON."""
    path = Path(candidates_path)
    if not path.exists():
        return []
    rows = []
    if path.suffix.lower() == ".csv":
        import csv
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                cid = row.get("source_control_id") or row.get("control_id") or ""
                pid = row.get("target_policy_id") or row.get("policy_passage_id") or ""
                if cid and pid:
                    rows.append((cid, pid))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in (data if isinstance(data, list) else [data]):
            cid = item.get("source_control_id") or item.get("control_id") or ""
            pid = item.get("target_policy_id") or item.get("policy_passage_id") or ""
            if cid and pid:
                rows.append((cid, pid))
    return rows


def generate_tasks(
    controls_path: str = DEFAULT_CONTROLS,
    policies_path: str = DEFAULT_POLICIES,
    output_path: str = DEFAULT_OUTPUT_TASKS,
    max_tasks: int = 0,
    pairs_per_control: int = 5,
    policy_passage_ids: List[str] = None,
    candidates_path: str = None,
) -> int:
    """
    Build Label Studio tasks: one task per (control, policy_passage) pair.

    - If candidates_path is set (e.g. mappings.csv): only create tasks for those
      (control_id, passage_id) pairs. Best for “golden set from pipeline candidates”.
    - Else: for each control, pair with the first pairs_per_control passages
      (or all if pairs_per_control is 0). max_tasks caps total tasks.
    """
    controls = load_controls(controls_path)
    policies = load_policies(policies_path)
    control_map = {}
    for ctrl in controls:
        c = ctrl.get("control", {})
        control_map[c.get("id", "")] = ctrl
    policy_map = {p["id"]: p for p in policies}

    pairs_to_build: List[tuple] = []
    if candidates_path:
        pairs_to_build = load_candidate_pairs(candidates_path)
        # Deduplicate and limit
        seen = set()
        unique = []
        for cid, pid in pairs_to_build:
            if (cid, pid) in seen:
                continue
            if cid in control_map and pid in policy_map:
                seen.add((cid, pid))
                unique.append((cid, pid))
            if max_tasks and len(unique) >= max_tasks:
                break
        pairs_to_build = unique
    else:
        passages = policies
        if policy_passage_ids:
            passages = [p for p in policies if p["id"] in policy_passage_ids]
        for ctrl in controls:
            c = ctrl.get("control", {})
            cid = c.get("id", "")
            for i, pp in enumerate(passages):
                if pairs_per_control and i >= pairs_per_control:
                    break
                pairs_to_build.append((cid, pp["id"]))
            if max_tasks and len(pairs_to_build) >= max_tasks:
                break
        if max_tasks:
            pairs_to_build = pairs_to_build[: max_tasks]

    tasks = []
    for cid, pid in pairs_to_build:
        ctrl = control_map.get(cid)
        pp = policy_map.get(pid)
        if not ctrl or not pp:
            continue
        fam = ctrl.get("control_family", {})
        c = ctrl.get("control", {})
        control_text = build_control_text(ctrl)
        sub_snippet = snippet(
            "\n".join(c.get("sub_controls", [])[:3]) or control_text[:150]
        )
        policy_text = pp.get("text", "")
        policy_section = pp.get("section", pp.get("name", ""))
        policy_name = pp.get("metadata", {}).get("policy_name", pp.get("id", ""))

        task = {
            "data": {
                "control_id": cid,
                "control_name": c.get("name", ""),
                "control_family": f"{fam.get('number','')} - {fam.get('name','')}",
                "control_text": control_text[:4000],
                "sub_controls_snippet": sub_snippet[:500],
                "policy_passage_id": pid,
                "policy_name": policy_name,
                "policy_section": policy_section,
                "policy_text": policy_text[:4000],
            },
            "meta": {"control_id": cid, "policy_passage_id": pid},
        }
        tasks.append(task)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    return len(tasks)


def export_golden_set(
    label_studio_export_path: str,
    output_path: str = DEFAULT_OUTPUT_GOLDEN,
) -> int:
    """
    Read Label Studio export JSON and convert to golden-set format:
    [{ control_id, policy_passage_id, compliance_status, confidence, ... }]
    """
    with open(label_studio_export_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw if isinstance(raw, list) else [raw]
    rows = []
    for item in items:
        anns = item.get("annotations", [])
        if not anns:
            continue
        data = item.get("data", {})
        result = anns[0].get("result", [])
        status = None
        confidence = None
        notes = None
        for r in result:
            from_name = (r.get("from_name") or "")
            val = r.get("value") or {}
            if "compliance_status" in from_name:
                ch = val.get("choices") or val.get("selected_labels") or []
                if ch:
                    status = ch[0] if isinstance(ch[0], str) else str(ch[0])
            if "confidence" in from_name:
                ch = val.get("choices") or val.get("selected_labels") or []
                if ch:
                    v = ch[0] if isinstance(ch[0], str) else str(ch[0])
                    confidence = int(v) if str(v).isdigit() else v
            if "evidence_or_notes" in from_name:
                t = val.get("text")
                if isinstance(t, list):
                    notes = " ".join(str(x) for x in t).strip() or None
                else:
                    notes = (str(t).strip() if t else None) or None

        rows.append({
            "control_id": data.get("control_id"),
            "control_name": data.get("control_name"),
            "policy_passage_id": data.get("policy_passage_id"),
            "policy_name": data.get("policy_name"),
            "policy_section": data.get("policy_section"),
            "compliance_status": status,
            "confidence": confidence,
            "evidence_or_notes": notes,
            "control_text_snippet": snippet(data.get("control_text", ""), 200),
            "policy_text_snippet": snippet(data.get("policy_text", ""), 200),
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="Golden set for compliance mapping (Label Studio)")
    p.add_argument("--mode", choices=["generate", "export"], default="generate")
    p.add_argument("--controls", default=DEFAULT_CONTROLS)
    p.add_argument("--policies", default=DEFAULT_POLICIES)
    p.add_argument("--output-tasks", default=DEFAULT_OUTPUT_TASKS)
    p.add_argument("--output-golden", default=DEFAULT_OUTPUT_GOLDEN)
    p.add_argument("--max-tasks", type=int, default=0, help="Cap total tasks (0 = no cap)")
    p.add_argument("--pairs-per-control", type=int, default=5, help="Max policy passages per control")
    p.add_argument("--candidates", help="Use only (control, passage) pairs from this file (e.g. mappings.csv)")
    p.add_argument("--input", dest="input_export", help="Label Studio export JSON (for mode=export)")
    args = p.parse_args()

    if args.mode == "generate":
        n = generate_tasks(
            controls_path=args.controls,
            policies_path=args.policies,
            output_path=args.output_tasks,
            max_tasks=args.max_tasks,
            pairs_per_control=args.pairs_per_control,
            candidates_path=args.candidates,
        )
        print(f"Generated {n} Label Studio tasks → {args.output_tasks}")
        print("Next: Import this file into Label Studio and use label config:")
        print("  data/03_label_studio_input/compliance_mapping_golden_set.xml")
    else:
        if not args.input_export:
            raise SystemExit("For mode=export pass --input <label_studio_export.json>")
        n = export_golden_set(
            label_studio_export_path=args.input_export,
            output_path=args.output_golden,
        )
        print(f"Exported {n} golden mappings → {args.output_golden}")


if __name__ == "__main__":
    main()
