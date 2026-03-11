#!/usr/bin/env python3
"""
Deduplicate + Enrich UAE IA Controls
=====================================
The corrected controls file has 963 rows but only 191 unique control IDs.
Each control appears 4-12 times across different source files with varying
field completeness. This script:

  1. Groups all rows by control ID
  2. For every text field, takes the LONGEST non-empty value across all
     duplicate rows (richest-wins merge strategy)
  3. Cross-references two additional source files to fill remaining gaps:
       - data/02_processed/uae_ia_controls_structured_v2.json  (has descriptions)
       - data/04_label_studio/imports/uae_ia_controls_raw.json (has control_statement)
  4. Builds a composite "full_text" field that concatenates all available
     text for controls — used by generate_synthetic_pairs.py as the prompt
  5. Flags controls that still lack useful text after all enrichment

Output: data/02_processed/uae_ia_controls_clean.json
  - Exactly 191 records (one per unique control ID)
  - Each record has the richest available text from all sources
  - family_balanced flag for use by generate_synthetic_pairs.py

Usage:
  python3 scripts/deduplicate_controls.py
  python3 scripts/deduplicate_controls.py --verbose   # show per-control merge details
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def richest(*values) -> str:
    """Return the longest non-empty string among the provided values."""
    candidates = [str(v).strip() for v in values if v and str(v).strip()]
    return max(candidates, key=len) if candidates else ""


def richest_list(*lists) -> list:
    """Return the longest non-empty list among the provided lists."""
    candidates = [lst for lst in lists if lst and len(lst) > 0]
    return max(candidates, key=len) if candidates else []


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Source loaders ────────────────────────────────────────────────────────────

def load_corrected(path: str) -> dict:
    """Load corrected file → {control_id: [rows]}."""
    data = load_json(path)
    groups = defaultdict(list)
    for row in data:
        ctrl = row.get("control", {})
        cid = ctrl.get("id", "").strip()
        if cid:
            groups[cid].append(row)
    return dict(groups)


def load_v2(path: str) -> dict:
    """Load v2 file → {control_id: row}  (first occurrence wins)."""
    data = load_json(path)
    index = {}
    for row in data:
        ctrl = row.get("control", {})
        cid = ctrl.get("id", "").strip()
        if cid and cid not in index:
            index[cid] = row
    return index


def load_raw(path: str) -> dict:
    """Load raw LabelStudio controls → {control_number: row}."""
    data = load_json(path)
    index = {}
    for row in data:
        num = row.get("control_number", "").strip()
        if num and num not in index:
            index[num] = row
    return index


# ── Merge logic ───────────────────────────────────────────────────────────────

def merge_control(cid: str, rows: list, v2_idx: dict, raw_idx: dict,
                  verbose: bool = False) -> dict:
    """Merge all duplicate rows + cross-reference sources into one rich record."""

    # ── Step 1: richest-wins merge within corrected rows ──────────────────────
    base = rows[0]
    ctrl_merged = {
        "id":   cid,
        "name": richest(*(r["control"].get("name", "") for r in rows)),
        "description": richest(*(r["control"].get("description", "") for r in rows)),
        "implementation_guidelines": richest(
            *(r["control"].get("implementation_guidelines", "") for r in rows)
        ),
        "sub_controls": richest_list(
            *(r["control"].get("sub_controls", []) for r in rows)
        ),
        "guidance_points": richest_list(
            *(r["control"].get("guidance_points", []) for r in rows)
        ),
        "internal_factors": richest_list(
            *(r["control"].get("internal_factors", []) for r in rows)
        ),
        "external_factors": richest_list(
            *(r["control"].get("external_factors", []) for r in rows)
        ),
    }

    # Track enrichment sources used
    sources_used = ["corrected"]
    gaps_filled = []

    # ── Step 2: fill gaps from v2 ─────────────────────────────────────────────
    v2_row = v2_idx.get(cid)
    if v2_row:
        v2_ctrl = v2_row.get("control", {})
        before_desc = ctrl_merged["description"]
        before_impl = ctrl_merged["implementation_guidelines"]
        before_subs = ctrl_merged["sub_controls"]

        ctrl_merged["description"] = richest(
            ctrl_merged["description"], v2_ctrl.get("description", "")
        )
        ctrl_merged["implementation_guidelines"] = richest(
            ctrl_merged["implementation_guidelines"],
            v2_ctrl.get("implementation_guidelines", "")
        )
        ctrl_merged["sub_controls"] = richest_list(
            ctrl_merged["sub_controls"], v2_ctrl.get("sub_controls", [])
        )
        ctrl_merged["internal_factors"] = richest_list(
            ctrl_merged["internal_factors"], v2_ctrl.get("internal_factors", [])
        )

        if len(ctrl_merged["description"]) > len(before_desc):
            gaps_filled.append("description←v2")
        if len(ctrl_merged["implementation_guidelines"]) > len(before_impl):
            gaps_filled.append("impl←v2")
        if len(ctrl_merged["sub_controls"]) > len(before_subs):
            gaps_filled.append("sub_controls←v2")
        if gaps_filled:
            sources_used.append("v2")

    # ── Step 3: fill gaps from raw (control_statement) ────────────────────────
    raw_row = raw_idx.get(cid)
    if raw_row:
        stmt = raw_row.get("control_statement", "") or ""
        before_desc = ctrl_merged["description"]
        # Use control_statement as description if description is still empty
        ctrl_merged["description"] = richest(ctrl_merged["description"], stmt)
        if len(ctrl_merged["description"]) > len(before_desc):
            gaps_filled.append("description←raw_stmt")
            sources_used.append("raw")

        # Also grab sub_controls from raw if we have none
        if not ctrl_merged["sub_controls"]:
            raw_subs = raw_row.get("sub_controls", [])
            if raw_subs:
                ctrl_merged["sub_controls"] = raw_subs
                gaps_filled.append("sub_controls←raw")
                if "raw" not in sources_used:
                    sources_used.append("raw")

    # ── Step 4: build composite full_text for prompting ───────────────────────
    parts = []
    if ctrl_merged["description"]:
        parts.append(ctrl_merged["description"])
    if ctrl_merged["implementation_guidelines"]:
        parts.append("Implementation: " + ctrl_merged["implementation_guidelines"])
    for sc in ctrl_merged["sub_controls"][:5]:   # cap at 5 sub-controls
        sc_text = sc if isinstance(sc, str) else str(sc)
        if sc_text.strip():
            parts.append(sc_text.strip())
    ctrl_merged["full_text"] = "\n".join(parts).strip()

    # ── Step 5: assess data quality ───────────────────────────────────────────
    has_text = len(ctrl_merged["full_text"]) > 20
    text_len = len(ctrl_merged["full_text"])

    if verbose and (gaps_filled or not has_text):
        status = "OK" if has_text else "EMPTY"
        print(f"  [{status}] {cid:12s}  sources={sources_used}  "
              f"gaps_filled={gaps_filled}  text_len={text_len}")

    # ── Build final record ────────────────────────────────────────────────────
    # Pick best validation_status (Complete > Incomplete > None)
    statuses = [r.get("validation_status", "") for r in rows]
    validation_status = (
        "Complete" if "Complete" in statuses else
        "Incomplete" if "Incomplete" in statuses else
        statuses[0] if statuses else None
    )

    # Pick best applicability
    applicability = richest_list(
        *(r.get("applicablility", []) or [] for r in rows)
    )

    return {
        "control_family":    base.get("control_family", {}),
        "control_subfamily": base.get("control_subfamily", {}),
        "control":           ctrl_merged,
        "applicability":     applicability,
        "breadcrumb":        richest(*(r.get("breadcrumb", "") for r in rows)),
        "validation_status": validation_status,
        "enrichment_sources": sources_used,
        "has_useful_text":   has_text,
        "full_text_length":  text_len,
        "source_row_count":  len(rows),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Deduplicate and enrich UAE IA controls")
    ap.add_argument("--corrected", default="data/02_processed/uae_ia_controls_corrected.json")
    ap.add_argument("--v2",        default="data/02_processed/uae_ia_controls_structured_v2.json")
    ap.add_argument("--raw",       default="data/04_label_studio/imports/uae_ia_controls_raw.json")
    ap.add_argument("--output",    default="data/02_processed/uae_ia_controls_clean.json")
    ap.add_argument("--verbose",   action="store_true",
                    help="Print per-control merge details for changed/empty controls")
    args = ap.parse_args()

    print("Loading source files ...")
    corrected_groups = load_corrected(args.corrected)
    v2_idx           = load_v2(args.v2)
    raw_idx          = load_raw(args.raw)

    print(f"  corrected.json : {sum(len(v) for v in corrected_groups.values())} rows, "
          f"{len(corrected_groups)} unique IDs")
    print(f"  v2.json        : {len(v2_idx)} unique IDs")
    print(f"  raw.json       : {len(raw_idx)} entries")
    print()

    if args.verbose:
        print("Per-control merge details:")

    results = []
    stats = {
        "total":             0,
        "has_description":   0,
        "has_impl":          0,
        "has_sub_controls":  0,
        "has_full_text":     0,
        "still_empty":       0,
        "enriched_from_v2":  0,
        "enriched_from_raw": 0,
    }

    for cid, rows in sorted(corrected_groups.items()):
        merged = merge_control(cid, rows, v2_idx, raw_idx, verbose=args.verbose)
        results.append(merged)

        stats["total"] += 1
        ctrl = merged["control"]
        if len(ctrl.get("description", "") or "") > 10:
            stats["has_description"] += 1
        if len(ctrl.get("implementation_guidelines", "") or "") > 10:
            stats["has_impl"] += 1
        if ctrl.get("sub_controls"):
            stats["has_sub_controls"] += 1
        if merged["has_useful_text"]:
            stats["has_full_text"] += 1
        else:
            stats["still_empty"] += 1
        if "v2" in merged["enrichment_sources"]:
            stats["enriched_from_v2"] += 1
        if "raw" in merged["enrichment_sources"]:
            stats["enriched_from_raw"] += 1

    # ── Sort by family then ID for consistent ordering ────────────────────────
    def sort_key(r):
        cid = r["control"]["id"]
        # Sort T-series before M-series, then numerically
        prefix = cid[0]
        rest = cid[1:]
        return (prefix, rest)
    results.sort(key=sort_key)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Family distribution ───────────────────────────────────────────────────
    family_counts = Counter(
        r["control"]["id"].split(".")[0]
        for r in results
    )

    print()
    print("=" * 60)
    print(f"Deduplication complete → {out_path}")
    print("=" * 60)
    print(f"  Input rows          : {sum(len(v) for v in corrected_groups.values())}")
    print(f"  Output records      : {stats['total']}  (one per unique control ID)")
    print()
    print(f"  Field coverage after enrichment:")
    print(f"    description       : {stats['has_description']}/{stats['total']} "
          f"({100*stats['has_description']//stats['total']}%)")
    print(f"    impl_guidelines   : {stats['has_impl']}/{stats['total']} "
          f"({100*stats['has_impl']//stats['total']}%)")
    print(f"    sub_controls      : {stats['has_sub_controls']}/{stats['total']} "
          f"({100*stats['has_sub_controls']//stats['total']}%)")
    print(f"    full_text (any)   : {stats['has_full_text']}/{stats['total']} "
          f"({100*stats['has_full_text']//stats['total']}%)")
    print()
    print(f"  Enrichment applied  :")
    print(f"    from v2           : {stats['enriched_from_v2']} controls")
    print(f"    from raw          : {stats['enriched_from_raw']} controls")
    print(f"  Still no text       : {stats['still_empty']} controls")
    print()
    print(f"  Family distribution :")
    for fam, cnt in sorted(family_counts.items()):
        bar = "█" * (cnt // 2)
        print(f"    {fam:5s}: {cnt:3d}  {bar}")

    # Print still-empty controls
    empty = [r for r in results if not r["has_useful_text"]]
    if empty:
        print()
        print(f"  Controls with no useful text after enrichment ({len(empty)}):")
        for r in empty:
            print(f"    {r['control']['id']:12s}  {r['control']['name']}")

    print()
    print(f"Next step — update generate_synthetic_pairs.py to use this file:")
    print(f"  python3 scripts/generate_synthetic_pairs.py \\")
    print(f"      --controls {out_path}")


if __name__ == "__main__":
    main()
