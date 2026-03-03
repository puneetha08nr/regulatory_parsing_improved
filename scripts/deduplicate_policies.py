"""
Deduplicate policy files in data/02_processed/policies/.

Problem: the same policy document exists under multiple filenames
(e.g. different version suffixes, corrected variants). This causes:
  1. Passage IDs that differ between pipeline output and golden data
     (the annotation was done against one version, pipeline loads another)
  2. Artificially low Recall@K (evaluation can't match IDs across versions)
  3. Duplicate passages inflating the retrieval index

Strategy:
  - Group files by their "canonical" policy name (strip version suffixes,
    numbers, corrected suffix)
  - Within each group, keep ONE file — prefer the version whose passage IDs
    appear in the golden dataset; fall back to the most recently modified
  - Move duplicates to data/02_processed/policies_archive/
  - Print a clear mapping of what was kept vs archived

Usage:
  python3 scripts/deduplicate_policies.py [--dry-run] [--golden PATH]

Options:
  --dry-run   Print what would happen without moving files
  --golden    Path to golden_mapping_dataset.json (to prefer the version
              whose passage IDs appear in the golden set)
              Default: data/07_golden_mapping/golden_mapping_dataset.json
  --policies  Directory of policy files
              Default: data/02_processed/policies
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


# ── Normalise a filename to a canonical policy name ────────────────────────
_STRIP_RE = re.compile(
    r"""
    [_\s]*                           # leading separator
    (?:
        v\d+(?:\.\d+)*               # version: v1, v1.2, v1.2.3
      | \d+(?:\.\d+)+                # bare version: 1.2, 2.0
      | [_\s]+\d+                    # trailing number: _2, _6
      | corrected                    # corrected suffix
      | _corrected                   # with underscore
      | \.json                       # extension
    )
    [_\s]*                           # trailing separator
    """,
    re.VERBOSE | re.IGNORECASE,
)

_WORD_ABBREV = {
    "ISSecurityIncidentMgmnt":   "IS Security Incident Management",
    "InformationRiskManagement": "Information Risk Management",
    "InformationRiskManagementPolicy": "Information Risk Management Policy",
    "InformationSecurity":       "Information Security",
    "InformationSecurityComPolicy": "Information Security Compliance Policy",
    "AcquisitionDevelopmentMaintenance": "Acquisition Development Maintenance",
    "SecurityAwarenessTraining": "Security Awareness Training",
    "LogginMonitoring":          "Logging Monitoring",
    "NetworkCommSecurity":       "Network Communications Security",
    "PhysicalEnvSecurity":       "Physical Environmental Security",
    "SecurityOperations":        "Security Operations",
}


def canonical_name(filename: str) -> str:
    """Return a normalised canonical name for grouping."""
    stem = Path(filename).stem
    # Expand known abbreviations
    for abbr, full in _WORD_ABBREV.items():
        stem = stem.replace(abbr, full)
    # Strip version numbers, 'corrected', trailing digits
    name = _STRIP_RE.sub(" ", stem)
    # Collapse whitespace, lowercase
    return " ".join(name.split()).lower()


def load_golden_passage_ids(golden_path: Path) -> set:
    """Return the set of passage IDs referenced in the golden dataset."""
    if not golden_path.exists():
        return set()
    with open(golden_path, encoding="utf-8") as f:
        rows = json.load(f)
    return {r.get("policy_passage_id", "") for r in rows if r.get("policy_passage_id")}


def load_passage_ids_from_file(policy_path: Path) -> set:
    """Return passage IDs in a policy *_for_mapping.json or _corrected.json file."""
    try:
        with open(policy_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {item.get("id", "") for item in data if item.get("id")}
        return set()
    except Exception:
        return set()


def check_content_duplicates(files: list, golden_ids: set) -> list:
    """Detect files whose passage IDs are subsets of another file (content-level duplicates)."""
    all_ids = {}
    for f in files:
        all_ids[f] = load_passage_ids_from_file(f)

    problems = []
    file_list = list(all_ids.items())
    for i, (fa, ids_a) in enumerate(file_list):
        for j, (fb, ids_b) in enumerate(file_list):
            if i >= j:
                continue
            if not ids_a or not ids_b:
                continue
            overlap = ids_a & ids_b
            if len(overlap) == len(ids_a) and len(overlap) == len(ids_b):
                problems.append(("exact_duplicate", fa, fb, len(overlap)))
            elif len(overlap) >= 0.9 * min(len(ids_a), len(ids_b)):
                problems.append(("near_duplicate", fa, fb, len(overlap)))
    return problems


def main():
    ap = argparse.ArgumentParser(description="Deduplicate policy files")
    ap.add_argument("--dry-run", action="store_true", help="Print only, do not move files")
    ap.add_argument("--check", action="store_true", help="Audit only — report all issues")
    ap.add_argument(
        "--golden",
        default="data/07_golden_mapping/golden_mapping_dataset.json",
        help="Golden dataset path (to prefer version whose IDs appear in golden)",
    )
    ap.add_argument(
        "--policies",
        default="data/02_processed/policies",
        help="Policy files directory",
    )
    args = ap.parse_args()

    policies_dir = Path(args.policies)
    archive_dir  = policies_dir.parent / "policies_archive"
    golden_path  = Path(args.golden)

    if not policies_dir.exists():
        print(f"❌ Policies directory not found: {policies_dir}")
        return

    golden_ids = load_golden_passage_ids(golden_path)
    print(f"Golden passage IDs loaded: {len(golden_ids)}")

    files = sorted(policies_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]

    # ── 1. Content-level duplicate detection ──────────────────────────────
    print(f"\n── Content-level duplicate check ({'audit' if args.check else 'dedup'}) ──")
    problems = check_content_duplicates(files, golden_ids)
    if problems:
        for kind, fa, fb, cnt in problems:
            print(f"  [{kind}] {fa.name}  ↔  {fb.name}  ({cnt} shared IDs)")
    else:
        print("  No content-level duplicates found.")

    # ── 2. Group by canonical name ──────────────────────────────────────
    print(f"\n── Canonical name groups ({len(files)} files) ──")
    groups: dict = defaultdict(list)
    for f in files:
        key = canonical_name(f.name)
        groups[key].append(f)

    to_keep:    list = []
    to_archive: list = []

    for canon, group in sorted(groups.items()):
        if len(group) == 1:
            to_keep.append(group[0])
            continue

        # Multiple files — check if they're content duplicates or distinct parts
        group_ids = {f: load_passage_ids_from_file(f) for f in group}
        all_ids_union = set().union(*group_ids.values())

        # Check cross-overlap: if any two share >80% IDs they're duplicates
        scored = []
        for f in group:
            ids = group_ids[f]
            golden_overlap = len(ids & golden_ids) if golden_ids else 0
            mtime = f.stat().st_mtime
            scored.append((golden_overlap, mtime, f, ids))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Are they actually distinct parts (non-overlapping IDs)?
        parts = [s[3] for s in scored]
        pairwise_overlaps = []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                ov = len(parts[i] & parts[j])
                pairwise_overlaps.append(ov)
        max_overlap = max(pairwise_overlaps) if pairwise_overlaps else 0
        total_unique = len(all_ids_union)
        total_combined = sum(len(p) for p in parts)
        are_distinct_parts = max_overlap == 0 or (total_unique / total_combined > 0.85)

        print(f"\n  [group '{canon}']  {len(group)} files")
        for overlap_count, mtime, f, ids in scored:
            print(f"    golden={overlap_count:3d}  passages={len(ids):3d}  {f.name}")
        if are_distinct_parts:
            print(f"    → KEEP ALL (distinct parts, max pairwise overlap={max_overlap})")
            to_keep.extend([s[2] for s in scored])
        else:
            winner = scored[0][2]
            losers  = [s[2] for s in scored[1:]]
            to_keep.append(winner)
            to_archive.extend(losers)
            print(f"    → KEEP:    {winner.name}")
            for loser in losers:
                print(f"    → ARCHIVE: {loser.name}")

    # ── 3. Missing files (golden IDs with no policy file) ─────────────────
    all_loaded_ids: set = set()
    for f in files:
        all_loaded_ids |= load_passage_ids_from_file(f)
    missing_golden = golden_ids - all_loaded_ids
    if missing_golden:
        missing_docs = defaultdict(list)
        for pid in sorted(missing_golden):
            doc = "_".join(pid.split("_")[:-1]) if "_passage_" in pid else pid
            missing_docs[doc].append(pid)
        print(f"\n  ⚠️  {len(missing_golden)} golden passage IDs have NO matching policy file:")
        for doc, pids in sorted(missing_docs.items()):
            print(f"    {len(pids):3d} passages  {doc}")
        print("  → Upload the missing policy files to Drive and re-copy.")

    print(f"\nSummary:  keep={len(to_keep)}  archive={len(to_archive)}")

    if args.check or args.dry_run:
        print("(audit/dry-run — no files moved)")
        return
    if not to_archive:
        print("Nothing to archive.")
        return

    archive_dir.mkdir(parents=True, exist_ok=True)
    for f in to_archive:
        shutil.move(str(f), str(archive_dir / f.name))
        print(f"  archived: {f.name}")
    print(f"\n✅ {len(to_archive)} files archived. Re-run the pipeline to rebuild indices.")


if __name__ == "__main__":
    main()
