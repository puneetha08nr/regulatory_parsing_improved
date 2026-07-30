#!/usr/bin/env python3
"""
Reconciled compliance report (Steps 3-4-5 merged).

Rolls both the HUMAN answer key and the AI judge up to per-control status, shows
them side by side, and flags every disagreement for human adjudication. The
point: the AI judge finds substantial coverage the human key marked as gaps, so
the "corrected" compliance picture is materially better — but each change is
listed as evidence, not silently applied.

Usage:
    python3 scripts/reconcile_report.py
"""

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"


def rollup(pairs, label_key):
    """control_id -> (status, evidence_pairs) from a list of pair dicts."""
    by_c = defaultdict(list)
    for p in pairs:
        by_c[p["control_id"]].append(p)
    out = {}
    for cid, evs in by_c.items():
        labs = [e[label_key] for e in evs]
        status = "Met" if FA in labs else ("Partially Met" if PA in labs else "Gap")
        out[cid] = status
    return out


def all_names(raw_path, structured_path):
    ids = {}
    for c in json.load(open(raw_path, encoding="utf-8")):
        num = c.get("control_number") or re.sub(r"^UAE_IA_CTRL_", "", c.get("control_id", ""))
        if num and re.match(r"^[MT]\d", num):
            nm = c.get("control_name", "")
            ids[num] = "" if nm == "Extracted Control" else nm
    try:
        for c in json.load(open(structured_path, encoding="utf-8")):
            ctl = c["control"]
            if ctl.get("name"):
                ids[ctl["id"]] = ctl["name"]
    except Exception:
        pass
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-key", default="data/07_golden_mapping/answer_key.json")
    ap.add_argument("--judged", default="data/10_judge/answer_key.judged.json")
    ap.add_argument("--controls-raw", default="data/04_label_studio/imports/uae_ia_controls_raw.json")
    ap.add_argument("--controls", default="data/02_processed/uae_ia_controls_structured.json")
    ap.add_argument("--out-md", default="data/11_report/reconciled_report.md")
    ap.add_argument("--out-json", default="data/11_report/reconciled_report.json")
    args = ap.parse_args()

    names = all_names(args.controls_raw, args.controls)
    key = json.load(open(args.answer_key, encoding="utf-8"))["pairs"]
    judged = json.load(open(args.judged, encoding="utf-8"))["results"]

    human_pairs = [{"control_id": p["control_id"], "passage_id": p["passage_id"], "lab": p["label"]} for p in key]
    ai_pairs = [{"control_id": r["control_id"], "passage_id": r["passage_id"], "lab": r["ai_label"]} for r in judged]
    ai_by_pair = {(r["control_id"], r["passage_id"]): r for r in judged}

    human_roll = rollup(human_pairs, "lab")
    ai_roll = rollup(ai_pairs, "lab")

    controls = sorted(set(human_roll) | set(ai_roll))
    rows = []
    for cid in controls:
        h, a = human_roll.get(cid, "-"), ai_roll.get(cid, "-")
        rows.append({"control_id": cid, "name": names.get(cid, ""), "human": h, "ai": a, "changed": h != a})

    # pair-level disagreements for adjudication
    adjud = []
    for p in key:
        r = ai_by_pair.get((p["control_id"], p["passage_id"]))
        if r and r["ai_label"] != p["label"]:
            adjud.append({
                "control_id": p["control_id"], "passage_id": p["passage_id"],
                "human": p["label"], "ai": r["ai_label"],
                "ai_evidence": html.unescape(r.get("ai_evidence", ""))[:300],
            })

    h_counts = Counter(r["human"] for r in rows)
    a_counts = Counter(r["ai"] for r in rows)
    upgraded = [r for r in rows if r["human"] == "Gap" and r["ai"] in ("Met", "Partially Met")]
    downgraded = [r for r in rows if r["human"] in ("Met", "Partially Met") and r["ai"] == "Gap"]

    report = {
        "controls_assessed": len(controls),
        "human_summary": dict(h_counts),
        "ai_summary": dict(a_counts),
        "controls_changed": sum(1 for r in rows if r["changed"]),
        "gap_to_covered": [r["control_id"] for r in upgraded],
        "covered_to_gap": [r["control_id"] for r in downgraded],
        "pair_disagreements": len(adjud),
        "rows": rows,
        "adjudication": adjud,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    L = []
    L.append("# Reconciled Compliance Assessment — Human key vs Claude judge\n")
    L.append("> Both the human answer key and the Claude judge were rolled up to per-control "
             "status. Where they differ, the judge's evidence is shown for adjudication. "
             "The judge finds substantial coverage the human key recorded as gaps.\n")
    L.append("## Summary (assessed controls)\n")
    L.append("| Status | Human key | Claude judge |")
    L.append("|---|---|---|")
    for s in ("Met", "Partially Met", "Gap"):
        L.append(f"| {s} | {h_counts.get(s,0)} | {a_counts.get(s,0)} |")
    L.append("")
    L.append(f"- Controls where the verdict changed: **{report['controls_changed']}** of {len(controls)}")
    L.append(f"- Controls the human marked **Gap** but the judge finds **covered**: **{len(upgraded)}**")
    L.append(f"- Controls the human marked covered but the judge finds **Gap**: **{len(downgraded)}** "
             "(check these — likely mis-pooled passages or human over-labels)")
    L.append(f"- Pair-level disagreements to adjudicate: **{len(adjud)}**\n")

    L.append("## Controls the human key under-counted (Gap -> Covered)\n")
    L.append("These were checked and marked a gap, but a policy passage does address them:\n")
    for r in upgraded:
        L.append(f"- **{r['control_id']}** {r['name']} — human: Gap - judge: **{r['ai']}**")
    L.append("")
    if downgraded:
        L.append("## Controls to double-check (human covered -> judge Gap)\n")
        for r in downgraded:
            L.append(f"- **{r['control_id']}** {r['name']} — human: {r['human']} - judge: **Gap**")
        L.append("")

    L.append("## Adjudication list (pair-level disagreements)\n")
    L.append("Resolve each: if the judge is right, the answer key was wrong and should be corrected.\n")
    L.append("| Control | Passage | Human | Judge | Evidence (judge) |")
    L.append("|---|---|---|---|---|")
    for a in adjud:
        pid = a["passage_id"].split("_passage_")[-1]
        pol = a["passage_id"].replace("clientname-IS-POL-00-", "").split("_passage_")[0][:34]
        ev = a["ai_evidence"].replace("|", "/").replace("\n", " ")[:110]
        L.append(f"| {a['control_id']} | {pol} p{pid} | {a['human'][:4]} | **{a['ai'][:9]}** | {ev} |")
    Path(args.out_md).write_text("\n".join(L), encoding="utf-8")

    print("RECONCILED REPORT (human key vs Claude judge)\n")
    print(f"  Controls assessed : {len(controls)}")
    print(f"  Human key  -> Met {h_counts.get('Met',0)}  Partially {h_counts.get('Partially Met',0)}  Gap {h_counts.get('Gap',0)}")
    print(f"  Claude judge -> Met {a_counts.get('Met',0)}  Partially {a_counts.get('Partially Met',0)}  Gap {a_counts.get('Gap',0)}")
    print(f"\n  Controls moved Gap -> Covered by the judge : {len(upgraded)}")
    print(f"  Controls moved Covered -> Gap (recheck)    : {len(downgraded)}")
    print(f"  Pair-level disagreements to adjudicate     : {len(adjud)}")
    print(f"\n  -> {args.out_md}")
    print(f"  -> {args.out_json}")


if __name__ == "__main__":
    main()
