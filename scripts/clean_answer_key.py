#!/usr/bin/env python3
"""
Clean + freeze the human answer key (Steps 1-2).

Reads the raw golden annotations, and produces a trustworthy, deduplicated,
frozen answer key:

  - keeps only rows whose control id is a REAL UAE IA control (full 251-id set,
    not the incomplete structured subset -- T6.2.2 etc. are real);
  - sets aside "Not Applicable" rows as passage-level negatives (map to no control);
  - deduplicates (control, passage) pairs;
  - resolves conflicting labels conservatively (ties break toward LOWER coverage,
    because over-claiming compliance is the dangerous error);
  - writes a frozen answer_key.json with a content hash, plus a cleaning report.

Non-destructive: the raw golden file is never modified.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"
SEVERITY = {NA: 0, PA: 1, FA: 2}   # higher = stronger compliance claim
NA_MARKERS = {"not applicable", "n/a", "na", "none"}


def real_control_ids(raw_path):
    ids = set()
    def walk(x):
        if isinstance(x, dict):
            for k in ("control_id", "id"):
                v = x.get(k)
                if isinstance(v, str):
                    m = re.sub(r"^UAE_IA_CTRL_", "", v)
                    if re.match(r"^[MT]\d", m):
                        ids.add(m)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(json.load(open(raw_path, encoding="utf-8")))
    return ids


def eff_cid(r):
    return (r.get("corrected_control_id") or r.get("control_id") or "").strip()


def resolve(rows):
    """Pick one label for a set of duplicate rows. Returns (label, meta)."""
    labels = [r.get("compliance_status") for r in rows if r.get("compliance_status")]
    confs = defaultdict(float)
    counts = Counter()
    for r in rows:
        lab = r.get("compliance_status")
        if lab not in SEVERITY:
            continue
        counts[lab] += 1
        c = r.get("confidence")
        confs[lab] += float(c) if isinstance(c, (int, float)) else 0.0
    if not counts:
        return None, {}
    # order by: most votes, then most confidence, then LOWER severity (conservative)
    best = sorted(counts, key=lambda l: (counts[l], confs[l], -SEVERITY[l]), reverse=True)[0]
    conflicting = len(set(labels)) > 1
    # best evidence: longest non-empty note/snippet
    ev = ""
    for r in rows:
        cand = (r.get("evidence_or_notes") or r.get("policy_text_snippet") or "").strip()
        if len(cand) > len(ev):
            ev = cand
    meta = {
        "n_sources": len(rows),
        "conflicting": conflicting,
        "votes": dict(counts),
        "evidence": ev[:500],
        "policy_section": next((r.get("policy_section") for r in rows if r.get("policy_section")), ""),
    }
    return best, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/07_golden_mapping/golden_mapping_dataset.json")
    ap.add_argument("--controls-raw", default="data/04_label_studio/imports/uae_ia_controls_raw.json")
    ap.add_argument("--out", default="data/07_golden_mapping/answer_key.json")
    ap.add_argument("--na-out", default="data/07_golden_mapping/na_passages.json")
    args = ap.parse_args()

    valid = real_control_ids(args.controls_raw)
    golden = json.load(open(args.golden, encoding="utf-8"))

    na_rows, mapping_rows, dropped = [], [], []
    for r in golden:
        cid = eff_cid(r)
        if cid.lower() in NA_MARKERS:
            na_rows.append(r)
        elif cid in valid:
            mapping_rows.append(r)
        else:
            dropped.append(cid)

    # dedup + resolve
    groups = defaultdict(list)
    for r in mapping_rows:
        groups[(eff_cid(r), r.get("policy_passage_id"))].append(r)

    key = []
    conflicts = 0
    for (cid, pid), rows in sorted(groups.items()):
        label, meta = resolve(rows)
        if not label:
            continue
        if meta["conflicting"]:
            conflicts += 1
        key.append({
            "control_id": cid,
            "passage_id": pid,
            "policy_name": next((r.get("policy_name") for r in rows if r.get("policy_name")), ""),
            "policy_section": meta["policy_section"],
            "label": label,
            "conflicting": meta["conflicting"],
            "n_sources": meta["n_sources"],
            "votes": meta["votes"],
            "evidence": meta["evidence"],
        })

    # na passages (deduped)
    na_pass = {}
    for r in na_rows:
        pid = r.get("policy_passage_id")
        if pid:
            na_pass[pid] = {"passage_id": pid, "policy_name": r.get("policy_name", ""),
                            "policy_section": r.get("policy_section", "")}
    na_list = sorted(na_pass.values(), key=lambda x: x["passage_id"])

    dist = Counter(k["label"] for k in key)
    content = json.dumps(key, sort_keys=True, ensure_ascii=False).encode()
    digest = hashlib.sha256(content).hexdigest()[:16]

    payload = {
        "meta": {
            "frozen": True,
            "content_sha256_16": digest,
            "rubric": "docs/GOLDEN_RUBRIC.md",
            "source_golden": args.golden,
            "n_pairs": len(key),
            "label_distribution": dict(dist),
            "n_conflicting_resolved": conflicts,
            "n_policies": len({k["policy_name"] for k in key}),
            "n_controls_covered": len({k["control_id"] for k in key}),
        },
        "pairs": key,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(na_list, open(args.na_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # report
    print("STEP 1 - CLEAN")
    print(f"  raw rows                : {len(golden)}")
    print(f"  -> Not Applicable set aside: {len(na_rows)}  (deduped {len(na_list)} passages -> {args.na_out})")
    print(f"  -> dropped (unreal ids) : {len(dropped)}  {Counter(dropped).most_common(3)}")
    print(f"  -> real mapping rows    : {len(mapping_rows)}")
    print(f"  distinct (control,passage) pairs: {len(key)}  (removed {len(mapping_rows)-len(key)} duplicate copies)")
    print(f"  conflicting labels resolved     : {conflicts}")
    print("\nSTEP 2 - LOCK")
    print(f"  frozen answer key -> {args.out}")
    print(f"  content hash      : {digest}")
    print(f"  pairs             : {len(key)}")
    print(f"  labels            : Fully={dist.get(FA,0)}  Partially={dist.get(PA,0)}  Not={dist.get(NA,0)}")
    print(f"  policies covered  : {payload['meta']['n_policies']}   controls covered: {payload['meta']['n_controls_covered']}")


if __name__ == "__main__":
    main()
