#!/usr/bin/env python3
"""
Adjudication worksheet + proposed corrected answer key (v2).

For every human-vs-judge disagreement it emits a ranked worksheet (clearest
human errors first), with a `decision` column PRE-FILLED with the suggested
correction so a human only edits the rows they disagree with. It then writes a
proposed answer_key_v2.json from those decisions, so you can see the corrected
compliance picture immediately.

Workflow:
    python3 scripts/adjudicate.py                 # build worksheet + proposed v2
    # ... open data/10_judge/adjudication.csv, edit `decision` where you disagree
    python3 scripts/adjudicate.py --apply         # regenerate v2 from your edits

Confidence tiers:
    HIGH   - judge cited specific atom coverage; near-verbatim human under-label
    MEDIUM - section-title match, or a mis-pooled passage (human positive -> NA)
    LOW    - "establish a policy" meta-controls / deferrals: genuinely need a human
"""

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"
ADJ = "data/10_judge/adjudication.csv"


def tier(human, ai, covered, ev):
    ev = (ev or "").lower()
    if "over-label" in ev or "deferred" in ev or "fragment" in ev or "does not establish" in ev:
        return "LOW"
    # reverse direction: human said positive, judge NA -> mis-pool, human should confirm
    if human in (FA, PA) and ai == NA:
        return "MEDIUM"
    if covered:                      # judge named specific atoms
        return "HIGH"
    return "MEDIUM"


def suggested(human, ai, t):
    # prefill: apply HIGH + MEDIUM corrections; leave LOW at the human label (needs review)
    return ai if t in ("HIGH", "MEDIUM") else human


def build(key, judged):
    ai_by = {(r["control_id"], r["passage_id"]): r for r in judged}
    rows = []
    for p in key:
        r = ai_by.get((p["control_id"], p["passage_id"]))
        if not r or r["ai_label"] == p["label"]:
            continue
        covered = r.get("ai_covered_atoms") or []
        t = tier(p["label"], r["ai_label"], covered, r.get("ai_evidence", ""))
        rows.append({
            "confidence": t,
            "control_id": p["control_id"],
            "passage_id": p["passage_id"],
            "policy": p.get("policy_name", ""),
            "human_label": p["label"],
            "judge_label": r["ai_label"],
            "judge_atoms": ",".join(covered),
            "judge_evidence": html.unescape(r.get("ai_evidence", ""))[:400],
            "decision": suggested(p["label"], r["ai_label"], t),
        })
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    rows.sort(key=lambda x: (order[x["confidence"]], x["control_id"]))
    return rows


NORM = {"fa": FA, "fully": FA, "pa": PA, "partial": PA, "partially": PA,
        "na": NA, "not": NA, "none": NA, "": ""}


def norm(s):
    s = (s or "").strip()
    return NORM.get(s.lower(), s)


def write_v2(key, decisions):
    """decisions: {(cid,pid): label}. Produce v2 pairs with decisions applied."""
    changed = 0
    out = []
    for p in key:
        k = (p["control_id"], p["passage_id"])
        new = decisions.get(k)
        q = dict(p)
        if new and new != p["label"]:
            q["label"] = new
            q["original_human_label"] = p["label"]
            q["corrected_by"] = "adjudication"
            changed += 1
        out.append(q)
    return out, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-key", default="data/07_golden_mapping/answer_key.json")
    ap.add_argument("--judged", default="data/10_judge/answer_key.judged.json")
    ap.add_argument("--worksheet", default=ADJ)
    ap.add_argument("--out", default="data/07_golden_mapping/answer_key_v2.json")
    ap.add_argument("--apply", action="store_true", help="regenerate v2 from an edited worksheet")
    args = ap.parse_args()

    key = json.load(open(args.answer_key, encoding="utf-8"))["pairs"]
    judged = json.load(open(args.judged, encoding="utf-8"))["results"]

    if args.apply:
        rows = list(csv.DictReader(open(args.worksheet, encoding="utf-8")))
        decisions = {(r["control_id"], r["passage_id"]): norm(r["decision"]) for r in rows if r.get("decision")}
        src = "edited worksheet"
    else:
        rows = build(key, judged)
        cols = ["confidence", "control_id", "passage_id", "policy", "human_label",
                "judge_label", "judge_atoms", "judge_evidence", "decision"]
        Path(args.worksheet).parent.mkdir(parents=True, exist_ok=True)
        with open(args.worksheet, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        decisions = {(r["control_id"], r["passage_id"]): r["decision"] for r in rows}
        src = "pre-filled suggestions"
        by_tier = Counter(r["confidence"] for r in rows)
        print(f"Worksheet: {len(rows)} disagreements -> {args.worksheet}")
        print(f"  HIGH (clear human under-label): {by_tier['HIGH']}")
        print(f"  MEDIUM (title-match / mis-pool): {by_tier['MEDIUM']}")
        print(f"  LOW (meta-controls, need human): {by_tier['LOW']}")

    v2, changed = write_v2(key, decisions)
    dist = Counter(p["label"] for p in v2)
    payload = {
        "meta": {
            "derived_from": args.answer_key,
            "judged": args.judged,
            "decisions_source": src,
            "n_pairs": len(v2),
            "n_corrected": changed,
            "label_distribution": dict(dist),
            "status": "PROPOSED - pending human sign-off on adjudication.csv",
        },
        "pairs": v2,
    }
    json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nProposed corrected key: {changed} labels changed -> {args.out}")
    print(f"  new distribution: Fully={dist.get(FA,0)}  Partially={dist.get(PA,0)}  Not={dist.get(NA,0)}")
    if not args.apply:
        print("\nNext: review data/10_judge/adjudication.csv (edit `decision` where you disagree),")
        print("      then: python3 scripts/adjudicate.py --apply")


if __name__ == "__main__":
    main()
