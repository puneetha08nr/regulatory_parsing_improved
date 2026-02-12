#!/usr/bin/env python3
"""
Label Studio support for policy mapping extraction verification.

1. Generate tasks from policy mapping JSON (from policy_mapping_extractor.py).
2. Import tasks + config (validate_policy_mapping.xml) into Label Studio.
3. After review, export from Label Studio and run export mode to get corrected mapping JSON.

Usage:
  # Generate tasks (from data/02_processed/policies_mapping/*_mapping.json)
  python policy_mapping_label_studio.py generate --input data/02_processed/policies_mapping --output data/03_label_studio_input/policy_mapping_tasks.json

  # Export corrected mapping from Label Studio export file
  python policy_mapping_label_studio.py export --input <label_studio_export.json> --output-dir data/02_processed/policies_mapping_corrected
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _format_obligations(obligations: List[Dict]) -> str:
    if not obligations:
        return ""
    return "\n".join(
        (o.get("text") or "").strip() for o in obligations if (o.get("text") or "").strip()
    )


def _parse_obligations(text: str) -> List[Dict]:
    if not (text or text.strip()):
        return []
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return [{"text": ln, "type": "shall", "section_id": ""} for ln in lines]


def _format_roles(roles: List[Dict]) -> str:
    if not roles:
        return ""
    return "\n".join(
        f"{r.get('role', '')}: {r.get('responsibility', '')}" for r in roles
    )


def _parse_roles(text: str) -> List[Dict]:
    if not (text or text.strip()):
        return []
    roles = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s*[:–\-]\s*", line, maxsplit=1)
        if len(parts) == 2:
            roles.append({"role": parts[0].strip(), "responsibility": parts[1].strip(), "source_section": ""})
        else:
            roles.append({"role": "", "responsibility": line, "source_section": ""})
    return roles


def generate_tasks(mapping_dir: str, output_path: str, max_tasks: int = 0) -> int:
    """Generate Label Studio tasks from *_mapping.json files."""
    mapping_path = Path(mapping_dir)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Directory not found: {mapping_dir}")
    files = list(mapping_path.glob("*_mapping.json"))
    if not files:
        raise FileNotFoundError(f"No *_mapping.json files in {mapping_dir}")

    tasks = []
    for f in sorted(files):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        meta = data.get("document_metadata") or {}
        doc_id = meta.get("document_id", f.stem.replace("_mapping", ""))
        doc_title = meta.get("title", doc_id)
        owner = meta.get("owner", "")
        effective_date = meta.get("effective_date", "")
        version = meta.get("version", "")
        scope = data.get("scope", "")
        exceptions = data.get("exceptions", "")
        roles = data.get("roles_and_responsibilities") or []

        # One document-level task per mapping file
        tasks.append({
            "name": f"Document: {doc_title}",
            "data": {
                "task_type": "document",
                "document_id": doc_id,
                "document_title": doc_title,
                "owner": owner,
                "effective_date": effective_date,
                "version": version,
                "scope": scope,
                "exceptions": exceptions,
                "roles": _format_roles(roles),
                "section_id": "",
                "section_title": "",
                "content": "",
                "obligations": "",
            },
            "meta": {"document_id": doc_id, "source_file": str(f)},
        })
        if max_tasks and len(tasks) >= max_tasks:
            break

        # One task per hierarchical passage
        passages = data.get("hierarchical_passages") or []
        for idx, p in enumerate(passages):
            sid = p.get("section_id", "")
            stitle = p.get("section_title", "")
            task_name = f"{sid} {stitle}".strip() or f"Passage {idx + 1}"
            tasks.append({
                "name": task_name,
                "data": {
                    "task_type": "passage",
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "section_id": sid,
                    "section_title": stitle,
                    "content": p.get("content", ""),
                    "obligations": _format_obligations(p.get("obligations") or []),
                    "owner": "",
                    "effective_date": "",
                    "version": "",
                    "scope": "",
                    "exceptions": "",
                    "roles": "",
                    "passage_index": idx + 1,
                    "total_passages": len(passages),
                },
                "meta": {"document_id": doc_id, "passage_index": idx, "source_file": str(f)},
            })
            if max_tasks and len(tasks) >= max_tasks:
                break
        if max_tasks and len(tasks) >= max_tasks:
            break

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(tasks, out, indent=2, ensure_ascii=False)
    return len(tasks)


def _get_result_value(result: List[Dict], from_name_substr: str) -> str:
    for r in result:
        if from_name_substr in (r.get("from_name") or ""):
            val = r.get("value") or {}
            text = val.get("text") or []
            return "\n".join(text).strip() if isinstance(text, list) else str(text).strip()
    return ""


def export_corrected_mappings(export_path: str, output_dir: str) -> List[str]:
    """Read Label Studio export and write corrected *_mapping.json per document."""
    with open(export_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else [raw]
    by_doc: Dict[str, Dict[str, Any]] = {}

    for item in items:
        anns = item.get("annotations", [])
        if not anns:
            continue
        data = item.get("data", {})
        result = anns[0].get("result", [])
        doc_id = data.get("document_id", "unknown")
        task_type = data.get("task_type", "passage")

        if doc_id not in by_doc:
            by_doc[doc_id] = {
                "document_metadata": {
                    "document_id": doc_id,
                    "title": data.get("document_title", ""),
                    "owner": "",
                    "effective_date": "",
                    "version": "",
                    "classification": "",
                },
                "hierarchical_passages": [],
                "roles_and_responsibilities": [],
                "scope": "",
                "exceptions": "",
                "source_path": "",
            }
        doc_state = by_doc[doc_id]

        if task_type == "document":
            doc_state["document_metadata"]["title"] = data.get("document_title") or doc_state["document_metadata"]["title"]
            doc_state["document_metadata"]["owner"] = data.get("owner", "")
            doc_state["document_metadata"]["effective_date"] = data.get("effective_date", "")
            doc_state["document_metadata"]["version"] = data.get("version", "")
            doc_state["scope"] = _get_result_value(result, "scope") or data.get("scope", "")
            doc_state["exceptions"] = _get_result_value(result, "exceptions") or data.get("exceptions", "")
            roles_raw = _get_result_value(result, "roles") or data.get("roles", "")
            doc_state["roles_and_responsibilities"] = _parse_roles(roles_raw)
        else:
            content = _get_result_value(result, "content") or data.get("content", "")
            obligations_raw = _get_result_value(result, "obligations") or data.get("obligations", "")
            section_id = data.get("section_id", "")
            section_title = data.get("section_title", "")
            existing = [
                x for x in doc_state["hierarchical_passages"]
                if x.get("section_id") == section_id and x.get("section_title") == section_title
            ]
            passage_data = {
                "section_id": section_id,
                "section_title": section_title,
                "content": content,
                "obligations": _parse_obligations(obligations_raw),
            }
            if existing:
                existing[0].update(passage_data)
            else:
                doc_state["hierarchical_passages"].append(passage_data)

    # Sort passages by section_id for stable order
    for state in by_doc.values():
        state["hierarchical_passages"].sort(key=lambda p: (p.get("section_id") or "", p.get("section_title") or ""))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written = []
    for doc_id, state in by_doc.items():
        safe_id = re.sub(r"[^\w\-]", "_", doc_id)[:80]
        path = out_path / f"{safe_id}_mapping_corrected.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        written.append(str(path))
    return written


def main():
    parser = argparse.ArgumentParser(description="Label Studio support for policy mapping verification")
    parser.add_argument("mode", choices=["generate", "export"], help="generate tasks or export corrected JSON")
    parser.add_argument("--input", required=True, help="Input: mapping dir (generate) or Label Studio export JSON (export)")
    parser.add_argument("--output", help="Output: task JSON path (generate)")
    parser.add_argument("--output-dir", help="Output directory for corrected mappings (export)")
    parser.add_argument("--max-tasks", type=int, default=0, help="Cap number of tasks (generate)")
    args = parser.parse_args()

    if args.mode == "generate":
        out = args.output or "data/03_label_studio_input/policy_mapping_tasks.json"
        n = generate_tasks(args.input, out, args.max_tasks)
        print(f"Generated {n} Label Studio tasks -> {out}")
        print("Next:")
        print("  1. Create project in Label Studio")
        print("  2. Use labeling config: data/03_label_studio_input/validate_policy_mapping.xml")
        print("  3. Import tasks from the JSON above")
        print("  4. After review, export and run: python policy_mapping_label_studio.py export --input <export.json> --output-dir data/02_processed/policies_mapping_corrected")
    else:
        out_dir = args.output_dir or "data/02_processed/policies_mapping_corrected"
        paths = export_corrected_mappings(args.input, out_dir)
        print(f"Exported {len(paths)} corrected mapping file(s) -> {out_dir}")
        for p in paths:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
