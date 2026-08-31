#!/usr/bin/env python3
"""
Compile the annotated CSV into the binary gold matrix used by the judge pipeline.

Reads:  data/14_healthcare_gold/annotation_task.csv  (annotator-filled)
Writes:
  data/14_healthcare_gold/gold_matrix.json    — {incident_id: {atom_id: 0/1/0.5}}
  data/14_healthcare_gold/gold_pairs.json     — flat list of (incident_id, atom_id, label) dicts
  data/14_healthcare_gold/summary.json        — statistics

Label mapping:  Y -> 1   P -> 0.5   N -> 0

Usage:
    python3 scripts/compile_healthcare_gold_matrix.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ANNOTATION_CSV = "data/14_healthcare_gold/annotation_task.csv"
OUT_DIR        = Path("data/14_healthcare_gold")

LABEL_MAP = {"Y": 1, "P": 0.5, "N": 0}


def main():
    rows = list(csv.DictReader(open(ANNOTATION_CSV, encoding="utf-8")))

    # Validate
    invalid = [r for r in rows if r["label"].strip() not in LABEL_MAP]
    if invalid:
        print(f"WARNING: {len(invalid)} rows with invalid labels — treated as N")
        for r in invalid[:5]:
            print(f"  {r['incident_id']} {r['atom_full_id']} label='{r['label']}'")

    # Build matrix: {incident_id: {atom_full_id: numeric_label}}
    matrix = defaultdict(dict)
    pairs  = []
    for r in rows:
        inc  = r["incident_id"]
        atom = r["atom_full_id"]
        lbl  = LABEL_MAP.get(r["label"].strip(), 0)
        matrix[inc][atom] = lbl
        pairs.append({
            "incident_id":   inc,
            "attack_type":   r["attack_type"],
            "attack_category": r["attack_category"],
            "atom_full_id":  atom,
            "standard":      r["standard"],
            "control_id":    r["control_id"],
            "atom_key":      r["atom_key"],
            "atom_claim":    r["atom_claim"],
            "label":         lbl,
            "label_str":     r["label"].strip() or "N",
        })

    # Summary stats
    from collections import Counter
    label_counts = Counter(p["label_str"] for p in pairs)
    n_records = len(matrix)
    n_atoms   = len({p["atom_full_id"] for p in pairs})
    positives = [p for p in pairs if p["label"] > 0]

    # Per-standard positive rate
    std_pos = Counter()
    std_tot = Counter()
    for p in pairs:
        std_tot[p["standard"]] += 1
        if p["label"] > 0:
            std_pos[p["standard"]] += 1

    # Per-attack-category positive rate
    cat_pos = Counter()
    cat_tot = Counter()
    for p in pairs:
        cat_tot[p["attack_category"]] += 1
        if p["label"] > 0:
            cat_pos[p["attack_category"]] += 1

    summary = {
        "n_records":    n_records,
        "n_atoms":      n_atoms,
        "n_pairs":      len(pairs),
        "label_counts": dict(label_counts),
        "positive_rate": round(len(positives) / len(pairs), 4),
        "by_standard": {
            s: {"positives": std_pos[s], "total": std_tot[s],
                "rate": round(std_pos[s]/std_tot[s], 4)}
            for s in std_tot
        },
        "by_attack_category": {
            c: {"positives": cat_pos[c], "total": cat_tot[c],
                "rate": round(cat_pos[c]/cat_tot[c], 4)}
            for c in cat_tot
        },
    }

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(dict(matrix), open(OUT_DIR / "gold_matrix.json", "w"), indent=2)
    json.dump(pairs,        open(OUT_DIR / "gold_pairs.json",  "w"), indent=2)
    json.dump(summary,      open(OUT_DIR / "summary.json",     "w"), indent=2)

    # Print report
    print(f"Records:        {n_records}")
    print(f"Atoms:          {n_atoms}")
    print(f"Total pairs:    {len(pairs)}")
    print(f"Labels:         Y={label_counts.get('Y',0)}  P={label_counts.get('P',0)}  N={label_counts.get('N',0)}")
    print(f"Positive rate:  {summary['positive_rate']:.1%}  ({len(positives)} positives)")
    print()
    print("By standard:")
    for s, v in summary["by_standard"].items():
        print(f"  {s:12s}  {v['positives']:4d}/{v['total']} = {v['rate']:.1%}")
    print()
    print("By attack category:")
    for c, v in sorted(summary["by_attack_category"].items()):
        print(f"  {c:12s}  {v['positives']:3d}/{v['total']} = {v['rate']:.1%}")
    print()
    print(f"Wrote: {OUT_DIR}/gold_matrix.json")
    print(f"Wrote: {OUT_DIR}/gold_pairs.json")
    print(f"Wrote: {OUT_DIR}/summary.json")
    print()
    print("Next: python3 scripts/nli_judge_healthcare.py")


if __name__ == "__main__":
    main()
