#!/usr/bin/env python3
"""
Build annotate_policy_controls.xml with searchable dropdowns (Taxonomy) for
UAE IA, ISO 27001, and ADHICS control IDs. Run this after updating control ID lists.

Usage:
  python3 build_policy_controls_label_studio_xml.py
  # Writes data/03_label_studio_input/annotate_policy_controls.xml
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


def load_control_ids(path: Path) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return sorted(out)


def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def main():
    base = Path(__file__).resolve().parent
    input_dir = base / "data" / "03_label_studio_input"
    uae_path = input_dir / "uae_ia_control_ids.txt"
    iso_path = input_dir / "iso27001_control_ids.txt"
    adhics_path = input_dir / "adhics_control_ids.txt"
    out_path = input_dir / "annotate_policy_controls.xml"

    uae_ids = load_control_ids(uae_path)
    iso_ids = load_control_ids(iso_path)
    adhics_ids = load_control_ids(adhics_path) if adhics_path.exists() else []

    # Root
    view = ET.Element("View")

    # Hidden target for toName
    hidden = ET.SubElement(view, "View", style="display: none;")
    ET.SubElement(hidden, "Text", name="task", value=" ")

    ET.SubElement(
        view,
        "Header",
        value="Assign UAE IA, ISO 27001, and ADHICS controls to this policy passage",
        size="4",
    )

    # Policy passage block
    ET.SubElement(view, "Header", value="Policy passage", size="5")
    policy_block = ET.SubElement(
        view, "View", style="background: #f0f8ff; padding: 8px; margin-bottom: 8px;"
    )
    ET.SubElement(
        policy_block, "Text", name="policy_doc", value="Document: $policy_name", style="font-weight: bold;"
    )
    ET.SubElement(policy_block, "Text", name="section_label", value="Section: $section")
    ET.SubElement(
        policy_block,
        "Text",
        name="passage_id_label",
        value="Passage ID: $policy_passage_id",
        style="font-size: 0.9em; color: #666;",
    )
    ET.SubElement(policy_block, "Header", value="Passage text", size="6")
    ET.SubElement(
        policy_block, "Text", name="policy_text", value="$policy_text", style="white-space: pre-wrap;"
    )

    # UAE IA control IDs – Taxonomy (dropdown with search)
    ET.SubElement(
        view,
        "Header",
        value="UAE IA control IDs (select all that apply; type to search)",
        size="5",
    )
    uae_tax = ET.SubElement(
        view,
        "Taxonomy",
        name="uae_ia_control_ids",
        toName="task",
        leafsOnly="true",
        placeholder="Search or select UAE IA controls...",
    )
    uae_root = ET.SubElement(uae_tax, "Choice", value="UAE IA")
    for cid in uae_ids:
        ET.SubElement(uae_root, "Choice", value=escape_xml(cid))

    # ISO 27001 control IDs – Taxonomy (dropdown with search)
    ET.SubElement(
        view,
        "Header",
        value="ISO 27001:2013 Annex A control IDs (select all that apply; type to search)",
        size="5",
    )
    iso_tax = ET.SubElement(
        view,
        "Taxonomy",
        name="iso27001_control_ids",
        toName="task",
        leafsOnly="true",
        placeholder="Search or select ISO 27001 controls...",
    )
    iso_root = ET.SubElement(iso_tax, "Choice", value="ISO 27001")
    for cid in iso_ids:
        ET.SubElement(iso_root, "Choice", value=escape_xml(cid))

    # ADHICS control IDs – Taxonomy (dropdown with search)
    if adhics_ids:
        ET.SubElement(
            view,
            "Header",
            value="ADHICS control IDs (select all that apply; type to search)",
            size="5",
        )
        adhics_tax = ET.SubElement(
            view,
            "Taxonomy",
            name="adhics_control_ids",
            toName="task",
            leafsOnly="true",
            placeholder="Search or select ADHICS controls...",
        )
        adhics_root = ET.SubElement(adhics_tax, "Choice", value="ADHICS")
        for cid in adhics_ids:
            ET.SubElement(adhics_root, "Choice", value=escape_xml(cid))

    # Notes
    ET.SubElement(view, "Header", value="Notes (optional)", size="5")
    ET.SubElement(
        view,
        "TextArea",
        name="notes",
        toName="task",
        placeholder="Any clarification or scope note for this assignment.",
        rows="2",
        editable="true",
    )

    # Pretty-print
    rough = ET.tostring(view, encoding="unicode", method="xml")
    dom = minidom.parseString(rough)
    pretty = dom.toprettyxml(indent="  ", encoding=None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pretty)

    msg = f"Wrote {out_path} (UAE IA: {len(uae_ids)}, ISO 27001: {len(iso_ids)}"
    if adhics_ids:
        msg += f", ADHICS: {len(adhics_ids)}"
    msg += " options)"
    print(msg)


if __name__ == "__main__":
    main()
