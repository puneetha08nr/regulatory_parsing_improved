#!/usr/bin/env python3
"""
Fix dataset quality issues before retraining.

Fix 1 — Deduplicate conflicting (control, passage) labels in golden_mapping_dataset.json:
  Same (control_id, policy_passage_id) pair → keep majority label (tie → most positive).
  Saves: data/07_golden_mapping/golden_mapping_dataset_clean.json

Fix 2 — Cap NA examples per passage in training data:
  Max 2 NA records per unique policy_passage_id.
  Kills passage memorisation without removing any FA/PA examples.
  Saves: data/07_golden_mapping/training_data/train_clean.json

Usage:
  python3 scripts/fix_dataset.py
  python3 scripts/fix_dataset.py --golden-in  data/07_golden_mapping/golden_mapping_dataset.json
                                  --train-in   data/07_golden_mapping/training_data/train.json
                                  --na-cap     2
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Label precedence for tie-breaking: FA > PA > NA
LABEL_RANK = {"Fully Addressed": 2, "Partially Addressed": 1, "Not Addressed": 0}


def dedup_golden(records: list) -> list:
    """
    For each (control_id, policy_passage_id) pair keep one record.
    Majority vote wins; ties broken by most-positive label.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        key = (r["control_id"], r["policy_passage_id"])
        groups[key].append(r)

    out = []
    n_conflicts = 0
    for key, recs in groups.items():
        if len(recs) == 1:
            out.append(recs[0])
            continue

        label_counts = Counter(r["compliance_status"] for r in recs)
        majority_count = max(label_counts.values())
        candidates = [lbl for lbl, cnt in label_counts.items() if cnt == majority_count]

        if len(candidates) > 1 or len(label_counts) > 1:
            n_conflicts += 1

        # Pick most-positive among tied majority labels
        winner = max(candidates, key=lambda l: LABEL_RANK.get(l, 0))

        # Keep the first record with that label as the base
        chosen = next(r for r in recs if r["compliance_status"] == winner)
        out.append(chosen)

    print(f"  Golden: {len(records)} records → {len(out)} after dedup "
          f"({len(records) - len(out)} duplicates removed, {n_conflicts} conflicting pairs resolved)")
    return out


def cap_na_per_passage(records: list, max_na: int = 2) -> list:
    """
    Keep all FA and PA records untouched.
    For NA records, keep at most `max_na` per unique policy_passage_id.
    """
    na_seen: dict[str, int] = defaultdict(int)
    out = []
    n_dropped = 0

    for r in records:
        if r["label"] != "Not Addressed":
            out.append(r)
            continue

        passage_id = r["policy_passage_id"]
        if na_seen[passage_id] < max_na:
            na_seen[passage_id] += 1
            out.append(r)
        else:
            n_dropped += 1

    label_dist = Counter(r["label"] for r in out)
    print(f"  Train: {len(records)} records → {len(out)} after NA cap={max_na} "
          f"({n_dropped} NA rows dropped)")
    print(f"  Label dist after cap: {dict(label_dist)}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Fix dataset quality issues")
    ap.add_argument("--golden-in",  default="data/07_golden_mapping/golden_mapping_dataset.json")
    ap.add_argument("--golden-out", default="data/07_golden_mapping/golden_mapping_dataset_clean.json")
    ap.add_argument("--train-in",   default="data/07_golden_mapping/training_data/train.json")
    ap.add_argument("--train-out",  default="data/07_golden_mapping/training_data/train_clean.json")
    ap.add_argument("--na-cap",     type=int, default=2,
                    help="Max NA records per passage in train (default: 2)")
    args = ap.parse_args()

    # ── Fix 1: Deduplicate golden ──────────────────────────────────────────────
    print(f"Fix 1: Deduplicating {args.golden_in} ...")
    with open(args.golden_in, encoding="utf-8") as f:
        golden = json.load(f)

    golden_clean = dedup_golden(golden)

    Path(args.golden_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.golden_out, "w", encoding="utf-8") as f:
        json.dump(golden_clean, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {args.golden_out}\n")

    # ── Fix 2: Cap NA per passage in train ────────────────────────────────────
    print(f"Fix 2: Capping NA per passage in {args.train_in} (max={args.na_cap}) ...")
    with open(args.train_in, encoding="utf-8") as f:
        train = json.load(f)

    train_clean = cap_na_per_passage(train, max_na=args.na_cap)

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as f:
        json.dump(train_clean, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {args.train_out}")


if __name__ == "__main__":
    main()
