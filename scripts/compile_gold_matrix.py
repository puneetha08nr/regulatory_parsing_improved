#!/usr/bin/env python3
"""
Compile filled annotation CSV(s) into the frozen gold_matrix.json.

- Derives FA/PA/NA per pair from the per-atom `covered_atoms` column
  (see docs/GOLDEN_RUBRIC.md), falling back to `label_manual` for
  atom-less controls or explicit overrides.
- Reports label distribution, precision-pool completeness, and the
  residual-miss (FN) estimate from the neg_sample cells.
- With two CSVs (--csv a.csv --csv b.csv), computes Cohen's kappa on the
  overlapping pairs and writes an adjudication CSV of disagreements.

Usage:
    # single annotator -> freeze
    python3 scripts/compile_gold_matrix.py \
        --csv data/09_gold_matrix/asset_management.pairs.csv \
        --out data/09_gold_matrix/asset_management.gold.json

    # two annotators -> kappa + adjudication
    python3 scripts/compile_gold_matrix.py \
        --csv annot_a.csv --csv annot_b.csv \
        --out data/09_gold_matrix/asset_management.gold.json
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"
_NORM = {
    "fa": FA, "fully": FA, "fully addressed": FA, "full": FA,
    "pa": PA, "partial": PA, "partially": PA, "partially addressed": PA,
    "na": NA, "not": NA, "none": NA, "not addressed": NA, "": "",
}


def norm_label(s):
    return _NORM.get((s or "").strip().lower(), None)


# Confirmed-negative tokens: annotator reviewed the pair, nothing is covered.
# Distinct from a BLANK cell, which means "not yet annotated".
NEG_TOKENS = {"-", "na", "n/a", "none", "x", "0"}


def parse_covered(raw, atom_keys):
    """Return set of covered atom keys. 'ALL' -> every key."""
    raw = (raw or "").strip()
    keys = [k.strip() for k in atom_keys.split(",") if k.strip()]
    if raw.upper() == "ALL":
        return set(keys), keys
    covered = {c.strip() for c in raw.replace(";", ",").split(",") if c.strip()}
    return covered, keys


def derive_label(row):
    """
    Return (label, atom_fraction, issues[]). label is None when the row is
    unlabelled (must be surfaced, never silently dropped).
    """
    issues = []
    n = int(row.get("n_atoms") or 0)
    manual = norm_label(row.get("label_manual"))
    if manual is None and (row.get("label_manual") or "").strip():
        issues.append(f"unrecognized label_manual={row.get('label_manual')!r}")

    if n > 0:
        raw = (row.get("covered_atoms") or "").strip()
        if manual:  # explicit override wins, but flag it
            issues.append("label_manual overrides atom-derived label")
            covered, _ = parse_covered(raw, row.get("atom_keys", ""))
            return manual, (len(covered) / n if n else 0.0), issues
        if raw == "":
            # blank == not yet annotated (distinct from a confirmed NA)
            return None, 0.0, issues
        if raw.lower() in NEG_TOKENS:
            return NA, 0.0, issues  # reviewed, nothing covered
        covered, keys = parse_covered(raw, row.get("atom_keys", ""))
        unknown = covered - set(keys)
        if unknown:
            issues.append(f"covered_atoms has unknown keys {sorted(unknown)}")
            covered &= set(keys)
        k = len(covered)
        if k > 0 and not (row.get("evidence") or "").strip():
            issues.append("covered atoms but no evidence quote")
        label = FA if k == n else (PA if k > 0 else NA)
        return label, k / n, issues

    # n == 0 -> must use label_manual
    if manual:
        return manual, (1.0 if manual == FA else 0.5 if manual == PA else 0.0), issues
    return None, 0.0, issues


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cohens_kappa(pairs_a, pairs_b):
    """3-class Cohen's kappa on overlapping pair_ids. pairs_* : {pair_id: label}."""
    common = [pid for pid in pairs_a if pid in pairs_b]
    if not common:
        return None, 0, []
    labels = [FA, PA, NA]
    n = len(common)
    obs = sum(1 for pid in common if pairs_a[pid] == pairs_b[pid]) / n
    ca = Counter(pairs_a[pid] for pid in common)
    cb = Counter(pairs_b[pid] for pid in common)
    exp = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    kappa = (obs - exp) / (1 - exp) if (1 - exp) else 1.0
    disagree = [pid for pid in common if pairs_a[pid] != pairs_b[pid]]
    return kappa, n, disagree


def compile_one(path):
    rows = load_csv(path)
    records, unlabelled, issues_all = [], [], []
    for r in rows:
        label, frac, issues = derive_label(r)
        rec = {
            "pair_id": r["pair_id"],
            "control_id": r["control_id"],
            "control_family": r.get("control_family", ""),
            "passage_id": r["passage_id"],
            "label": label,
            "atom_fraction": round(frac, 3),
            "covered_atoms": r.get("covered_atoms", ""),
            "evidence": r.get("evidence", ""),
            "source": r.get("source", ""),
            "confidence": r.get("confidence", ""),
            "note": r.get("note", ""),
            "is_fn_check": "neg_sample" in r.get("source", ""),
            "in_precision_pool": ("prediction" in r.get("source", "")),
        }
        records.append(rec)
        if label is None:
            unlabelled.append(r["pair_id"])
        if issues:
            issues_all.append((r["pair_id"], issues))
    return records, unlabelled, issues_all


def main():
    ap = argparse.ArgumentParser(description="Compile golden-matrix CSV -> gold_matrix.json")
    ap.add_argument("--csv", action="append", required=True, help="filled annotation CSV (repeat for 2 annotators)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unlabelled", action="store_true",
                    help="freeze even if some pairs are unlabelled (default: refuse)")
    args = ap.parse_args()

    per_annotator = [compile_one(p) for p in args.csv]

    # ── dual-annotation: kappa + adjudication ────────────────────────────────
    kappa_report = None
    if len(args.csv) == 2:
        a = {r["pair_id"]: r["label"] for r in per_annotator[0][0] if r["label"]}
        b = {r["pair_id"]: r["label"] for r in per_annotator[1][0] if r["label"]}
        kappa, n_common, disagree = cohens_kappa(a, b)
        kappa_report = {"kappa": kappa, "n_common": n_common, "n_disagree": len(disagree)}
        adj = Path(args.out).with_suffix(".adjudicate.csv")
        with open(adj, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "annotator_a", "annotator_b"])
            for pid in disagree:
                w.writerow([pid, a.get(pid, ""), b.get(pid, "")])
        print(f"Cohen's kappa (3-class): {kappa:.3f} over {n_common} shared pairs")
        print(f"  disagreements: {len(disagree)} -> {adj}")
        print("  Resolve disagreements, then compile the reconciled single CSV.\n")

    # primary annotator (first CSV) is the frozen set
    records, unlabelled, issues_all = per_annotator[0]

    dist = Counter(r["label"] for r in records if r["label"])
    n_pred = sum(1 for r in records if r["in_precision_pool"])
    n_pred_unlab = sum(1 for r in records if r["in_precision_pool"] and r["label"] is None)
    fn_checks = [r for r in records if r["is_fn_check"]]
    fn_pos = sum(1 for r in fn_checks if r["label"] in (FA, PA))

    print(f"Pairs           : {len(records)}")
    print(f"Labelled        : {sum(dist.values())}   Unlabelled: {len(unlabelled)}")
    print(f"  Fully Addressed    : {dist.get(FA, 0)}")
    print(f"  Partially Addressed: {dist.get(PA, 0)}")
    print(f"  Not Addressed      : {dist.get(NA, 0)}")
    print(f"Precision pool (predictions): {n_pred} in pool, {n_pred_unlab} still unlabelled")
    if fn_checks:
        rate = fn_pos / len(fn_checks)
        print(f"FN estimate     : {fn_pos}/{len(fn_checks)} sampled unpooled cells were positive "
              f"(~{rate:.1%} residual miss rate)")
    if issues_all:
        print(f"\nData-quality flags ({len(issues_all)}):")
        for pid, iss in issues_all[:20]:
            print(f"  {pid}: {'; '.join(iss)}")
        if len(issues_all) > 20:
            print(f"  ... and {len(issues_all) - 20} more")

    if unlabelled and not args.allow_unlabelled:
        print(f"\nREFUSING to freeze: {len(unlabelled)} unlabelled pairs. "
              f"Fill them or pass --allow-unlabelled.", file=sys.stderr)
        print("  first few:", ", ".join(unlabelled[:5]), file=sys.stderr)
        sys.exit(1)

    frozen = [r for r in records if r["label"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "n_pairs": len(frozen),
            "label_distribution": dict(dist),
            "precision_pool_size": n_pred,
            "fn_check": {"sampled": len(fn_checks), "positive": fn_pos},
            "kappa": kappa_report,
            "source_csvs": args.csv,
            "rubric": "docs/GOLDEN_RUBRIC.md",
        },
        "pairs": frozen,
    }
    json.dump(payload, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nFroze {len(frozen)} labelled pairs -> {out}")


if __name__ == "__main__":
    main()
