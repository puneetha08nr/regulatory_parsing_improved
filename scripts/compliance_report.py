#!/usr/bin/env python3
"""
Compliance report (Step 5) from the frozen human answer key.

Rolls (control, passage) judgments up to a per-control compliance status — the
question an auditor actually asks: "is this control addressed by the policy set?"

Control-level status (best evidence wins, because multiple partial passages can
jointly satisfy a control):
  Met            - at least one passage Fully Addresses the control
  Partially Met  - some coverage, but no single passage is Full
  Gap            - assessed, but no passage covers it  (a real compliance gap)
  Not Assessed   - the control was never annotated     (NOT the same as a gap)

The Not-Assessed distinction is the point: a responsible report never claims a
control is unmet if it was never checked.

Usage:
    python3 scripts/compliance_report.py \
        --answer-key data/07_golden_mapping/answer_key.json \
        --controls-raw data/04_label_studio/imports/uae_ia_controls_raw.json \
        --out-json data/11_report/compliance_report.json \
        --out-md   data/11_report/compliance_report.md
"""

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"


def all_control_ids(raw_path, structured_path):
    """Every real UAE IA control id -> best name (structured preferred, raw fallback)."""
    ids = {}
    for c in json.load(open(raw_path, encoding="utf-8")):
        num = c.get("control_number") or re.sub(r"^UAE_IA_CTRL_", "", c.get("control_id", ""))
        if num and re.match(r"^[MT]\d", num):
            nm = c.get("control_name", "")
            ids[num] = "" if nm == "Extracted Control" else nm
    # prefer real names from the structured file where available
    try:
        for c in json.load(open(structured_path, encoding="utf-8")):
            ctl = c["control"]
            if ctl.get("name"):
                ids[ctl["id"]] = ctl["name"]
    except Exception:
        pass
    return ids


def family_of(cid):
    m = re.match(r"^([MT]\d+)", cid)
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-key", default="data/07_golden_mapping/answer_key.json")
    ap.add_argument("--controls-raw", default="data/04_label_studio/imports/uae_ia_controls_raw.json")
    ap.add_argument("--controls", default="data/02_processed/uae_ia_controls_structured.json")
    ap.add_argument("--out-json", default="data/11_report/compliance_report.json")
    ap.add_argument("--out-md", default="data/11_report/compliance_report.md")
    args = ap.parse_args()

    key = json.load(open(args.answer_key, encoding="utf-8"))
    pairs = key["pairs"] if isinstance(key, dict) else key
    universe = all_control_ids(args.controls_raw, args.controls)

    # gather evidence per control
    by_control = defaultdict(list)
    for p in pairs:
        by_control[p["control_id"]].append(p)

    rollup = {}
    for cid, evs in by_control.items():
        labels = [e["label"] for e in evs]
        if FA in labels:
            status = "Met"
        elif PA in labels:
            status = "Partially Met"
        else:
            status = "Gap"
        evidence = [
            {"passage_id": e["passage_id"], "policy": e["policy_name"],
             "section": e.get("policy_section", ""), "label": e["label"],
             "quote": html.unescape(e.get("evidence") or "")[:300]}
            for e in evs if e["label"] in (FA, PA)
        ]
        rollup[cid] = {
            "control_id": cid, "control_name": universe.get(cid, ""),
            "family": family_of(cid), "status": status,
            "n_passages_assessed": len(evs), "evidence": evidence,
        }

    assessed = set(rollup)
    not_assessed = sorted(set(universe) - assessed)

    from collections import Counter
    status_counts = Counter(r["status"] for r in rollup.values())
    fam_break = defaultdict(lambda: Counter())
    for r in rollup.values():
        fam_break[r["family"]][r["status"]] += 1

    report = {
        "meta": {
            "source": args.answer_key,
            "total_controls_in_regulation": len(universe),
            "controls_assessed": len(assessed),
            "controls_not_assessed": len(not_assessed),
            "coverage_pct": round(100 * len(assessed) / max(len(universe), 1), 1),
        },
        "summary": dict(status_counts),
        "controls": sorted(rollup.values(), key=lambda r: r["control_id"]),
        "gaps": sorted([r["control_id"] for r in rollup.values() if r["status"] == "Gap"]),
        "not_assessed": not_assessed,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ── markdown ─────────────────────────────────────────────────────────────
    m = report["meta"]
    L = []
    L.append("# UAE IA Compliance Assessment — Human Answer Key\n")
    L.append("> Source: frozen human annotations (`answer_key.json`). "
             "This reflects what has been **assessed so far**, not the full regulation.\n")
    L.append("## Coverage caveat\n")
    L.append(f"- Controls in regulation: **{m['total_controls_in_regulation']}**")
    L.append(f"- Controls assessed: **{m['controls_assessed']}** ({m['coverage_pct']}%)")
    L.append(f"- **Not yet assessed: {m['controls_not_assessed']}** — these are *unknown*, "
             "not failures. They must be assessed before any completeness claim.\n")
    L.append("## Summary of assessed controls\n")
    L.append(f"| Status | Count |\n|---|---|")
    for s in ("Met", "Partially Met", "Gap"):
        L.append(f"| {s} | {report['summary'].get(s, 0)} |")
    L.append("")
    L.append("## Compliance gaps (assessed, no coverage found)\n")
    if report["gaps"]:
        L.append("These controls were checked and **no policy passage covers them** — remediate:\n")
        for cid in report["gaps"]:
            L.append(f"- **{cid}** — {rollup[cid]['control_name']} ({rollup[cid]['n_passages_assessed']} passage(s) checked)")
    else:
        L.append("_None among assessed controls._")
    L.append("")
    L.append("## Met / Partially Met — with evidence\n")
    for r in report["controls"]:
        if r["status"] == "Gap":
            continue
        L.append(f"### {r['control_id']} — {r['control_name']}  · **{r['status']}**")
        for e in r["evidence"]:
            L.append(f"- _{e['label']}_ — {e['policy']} / {e['section']}  \n  `{e['passage_id']}`")
            if e["quote"].strip():
                L.append(f"  > {e['quote'].strip()}")
        L.append("")
    L.append("## Not assessed (per family)\n")
    fam_missing = defaultdict(list)
    for cid in not_assessed:
        fam_missing[family_of(cid)].append(cid)
    for fam in sorted(fam_missing):
        L.append(f"- **{fam}**: {len(fam_missing[fam])} controls unassessed")
    Path(args.out_md).write_text("\n".join(L), encoding="utf-8")

    # console
    print("STEP 5 - COMPLIANCE REPORT (from human answer key)\n")
    print(f"  Regulation controls : {m['total_controls_in_regulation']}")
    print(f"  Assessed            : {m['controls_assessed']} ({m['coverage_pct']}%)")
    print(f"  Not assessed        : {m['controls_not_assessed']}  (unknown, not failures)")
    print(f"\n  Of assessed controls:")
    print(f"    Met           : {status_counts.get('Met',0)}")
    print(f"    Partially Met : {status_counts.get('Partially Met',0)}")
    print(f"    Gap           : {status_counts.get('Gap',0)}")
    print(f"\n  -> {args.out_json}")
    print(f"  -> {args.out_md}")


if __name__ == "__main__":
    main()
