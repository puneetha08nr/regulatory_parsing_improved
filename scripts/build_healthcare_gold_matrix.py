#!/usr/bin/env python3
"""
Build the annotation CSV for the Work 3 healthcare gold matrix.

Output: one row per (incident_record, atom) pair.
  - All 50 × 76 = 3,800 pairs are included.
  - 'label_hint' is pre-filled from the record's controls_evidenced field:
      Y  = atom listed in controls_evidenced  (likely addressed)
      N  = atom not listed                    (likely not addressed)
  - 'label' is left blank for the annotator to fill in (Y / P / N).

Annotator workflow:
  1. Open the CSV in a spreadsheet (Google Sheets / LibreOffice Calc).
  2. Work through rows where label_hint = Y first — confirm or correct each to Y / P / N.
  3. For label_hint = N rows: scan the observed_behaviour column; mark any as Y or P
     where the incident text clearly provides evidence.  Leave others as N.
  4. Save the filled CSV.
  5. Run scripts/compile_healthcare_gold_matrix.py to produce the binary matrix and kappa stats.

Usage:
    python3 scripts/build_healthcare_gold_matrix.py
    python3 scripts/build_healthcare_gold_matrix.py --only-hints  # Y-hint rows only (250 rows)

Output: data/14_healthcare_gold/annotation_task.csv
"""

import argparse
import csv
import json
from pathlib import Path


RECORDS_PATH  = "data/13_healthcare/incidents/records.json"
HIPAA_PATH    = "data/13_healthcare/hipaa_312_atoms.json"
ISO_PATH      = "data/13_healthcare/iso27002_912_atoms.json"
OUT_PATH      = "data/14_healthcare_gold/annotation_task.csv"


def load_atoms():
    """Return ordered list of atom dicts with full_id, standard, control_id, claim."""
    atoms = []
    for path, standard in [(HIPAA_PATH, "HIPAA"), (ISO_PATH, "ISO 27002")]:
        for block in json.load(open(path)):
            cid  = block["control"]["id"]
            name = block["control"]["name"]
            for atom in block["atoms"]:
                atoms.append({
                    "full_id":    f"{cid}.{atom['key']}",
                    "standard":   standard,
                    "control_id": cid,
                    "control_name": name,
                    "atom_key":   atom["key"],
                    "claim":      atom["claim"],
                })
    return atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-hints", action="store_true",
                    help="Output only rows where label_hint=Y (250 rows instead of 3,800)")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    records = json.load(open(RECORDS_PATH))
    atoms   = load_atoms()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "pair_id",
        "incident_id",
        "attack_type",
        "attack_category",
        "outcome_summary",
        "observed_behaviour",
        "atom_full_id",
        "standard",
        "control_id",
        "control_name",
        "atom_key",
        "atom_claim",
        "label_hint",
        "label",          # annotator fills: Y / P / N
        "evidence_note",  # annotator fills: brief quote or reason
    ]

    rows = []
    for rec in records:
        evidenced = set(rec.get("controls_evidenced", []))
        # Truncate observed_behaviour to ~500 chars for spreadsheet readability
        obs = rec["observed_behaviour"]
        obs_short = obs[:500] + ("…" if len(obs) > 500 else "")
        outcome_short = rec["outcome"][:200] + ("…" if len(rec["outcome"]) > 200 else "")

        for atom in atoms:
            aid = atom["full_id"]
            hint = "Y" if aid in evidenced else "N"

            if args.only_hints and hint != "Y":
                continue

            rows.append({
                "pair_id":          f"{rec['incident_id']}__{aid}",
                "incident_id":      rec["incident_id"],
                "attack_type":      rec["attack_type"],
                "attack_category":  rec["attack_category"],
                "outcome_summary":  outcome_short,
                "observed_behaviour": obs_short,
                "atom_full_id":     aid,
                "standard":         atom["standard"],
                "control_id":       atom["control_id"],
                "control_name":     atom["control_name"],
                "atom_key":         atom["atom_key"],
                "atom_claim":       atom["claim"],
                "label_hint":       hint,
                "label":            "",
                "evidence_note":    "",
            })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    y_count = sum(1 for r in rows if r["label_hint"] == "Y")
    n_count = sum(1 for r in rows if r["label_hint"] == "N")
    print(f"Atoms:       {len(atoms)} (HIPAA {sum(1 for a in atoms if a['standard']=='HIPAA')} + ISO {sum(1 for a in atoms if a['standard']!='HIPAA')})")
    print(f"Records:     {len(records)}")
    print(f"Total rows:  {len(rows)}")
    print(f"  label_hint=Y: {y_count}  (review these first)")
    print(f"  label_hint=N: {n_count}  (scan for missed positives)")
    print(f"Wrote: {args.out}")
    print()
    print("Next steps:")
    print("  1. Open data/14_healthcare_gold/annotation_task.csv in a spreadsheet")
    print("  2. Fill 'label' column (Y / P / N) and 'evidence_note' for each Y-hint row")
    print("  3. Spot-check a sample of N-hint rows for missed positives")
    print("  4. python3 scripts/compile_healthcare_gold_matrix.py")


if __name__ == "__main__":
    main()
