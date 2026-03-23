#!/usr/bin/env python3
"""
Generate synthetic FA/PA/NA training examples for UAE IA controls.

Input:  data/02_processed/uae_ia_controls_clean.json
Output: data/07_golden_mapping/synthetic_training_data.json

Uses Anthropic API (claude-sonnet-4-20250514) to generate 3 policy
clause examples per control: Fully Addressed, Partially Addressed,
Not Addressed.

Usage:
  python3 scripts/generate_synthetic_data.py
  python3 scripts/generate_synthetic_data.py --family T4
  python3 scripts/generate_synthetic_data.py --resume
  python3 scripts/generate_synthetic_data.py --dry-run
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
CONTROLS_PATH = ROOT / "data/02_processed/uae_ia_controls_clean.json"
PROGRESS_PATH = ROOT / "data/07_golden_mapping/synthetic_progress.json"
OUTPUT_PATH   = ROOT / "data/07_golden_mapping/synthetic_training_data.json"

LABEL_TO_STATUS = {
    "FA": "Fully Addressed",
    "PA": "Partially Addressed",
    "NA": "Not Addressed",
}

# ── Control loading ────────────────────────────────────────────────────────────

def load_controls(family_filter: str = None) -> list[dict]:
    """
    Load and deduplicate controls from uae_ia_controls_clean.json.
    Returns list of dicts with keys: id, name, text.
    Skips controls where both description and sub_controls are empty.
    """
    with open(CONTROLS_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    seen = {}
    for record in raw:
        ctrl = record.get("control", {})
        if isinstance(ctrl, str):
            ctrl = json.loads(ctrl)

        cid = ctrl.get("id", "").strip()
        if not cid or not re.match(r"^[A-Z]\d+\.\d+\.\d+", cid):
            continue

        # Deduplicate: keep first occurrence
        if cid in seen:
            continue

        # Family filter
        if family_filter:
            family = re.match(r"^([A-Z]\d+)", cid)
            if not family or not family.group(1).startswith(family_filter):
                continue

        desc = ctrl.get("description", "").strip()
        subs = ctrl.get("sub_controls") or []
        sub_text = "\n".join(str(s) for s in subs if str(s).strip())

        full_text = desc
        if sub_text:
            full_text = (desc + "\n" + sub_text).strip() if desc else sub_text

        if not full_text:
            continue

        seen[cid] = {
            "id": cid,
            "name": ctrl.get("name", "").strip(),
            "text": full_text,
        }

    return list(seen.values())


# ── Prompt building ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an information security policy expert. "
    "Generate realistic policy clauses in formal IS policy language. "
    "Respond only in valid JSON. No explanation. No markdown fences."
)


def build_user_prompt(control: dict) -> str:
    return (
        f"UAE IA Control:\n"
        f"ID: {control['id']}\n"
        f"Name: {control['name']}\n"
        f"Requirements: {control['text']}\n\n"
        f"Generate 3 realistic policy clause examples:\n\n"
        f"FA (Fully Addressed): explicitly satisfies ALL requirements "
        f"and sub-controls listed above. Must reference the specific "
        f"requirements directly.\n\n"
        f"PA (Partially Addressed): addresses SOME sub-controls but "
        f"misses at least one specific requirement. Must be genuinely "
        f"related but incomplete.\n\n"
        f"NA (Not Addressed): topically related to information security "
        f"but clearly does not address this specific control's requirements.\n\n"
        f'Return ONLY this JSON:\n{{"FA": "policy clause text", "PA": "policy clause text", "NA": "policy clause text"}}'
    )


# ── API call ──────────────────────────────────────────────────────────────────

def call_api(client, control: dict) -> dict | None:
    """
    Call Anthropic API for one control.
    Returns {"FA": ..., "PA": ..., "NA": ...} or None on failure.
    Retries once on JSON parse failure.
    """
    user_prompt = build_user_prompt(control)

    for attempt in range(2):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                temperature=0.7,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = message.content[0].text.strip()

            # Strip markdown fences if present despite instructions
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed = json.loads(raw_text)

            # Validate all three keys present
            if all(k in parsed for k in ("FA", "PA", "NA")):
                return parsed

            print(f"  ✗ Missing keys in response for {control['id']} (attempt {attempt+1}): {list(parsed.keys())}")

        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parse error for {control['id']} (attempt {attempt+1}): {e}")
            if attempt == 0:
                time.sleep(1)
        except Exception as e:
            print(f"  ✗ API error for {control['id']} (attempt {attempt+1}): {type(e).__name__}: {e}")
            return None  # Don't retry on API errors

    return None


# ── Output record builder ──────────────────────────────────────────────────────

def make_records(control: dict, clauses: dict) -> list[dict]:
    """Build 3 golden_mapping_dataset.json-format records from API response."""
    records = []
    snippet = control["text"][:300]
    for label, status in LABEL_TO_STATUS.items():
        clause_text = clauses.get(label, "").strip()
        if not clause_text:
            continue
        records.append({
            "control_id":           control["id"],
            "corrected_control_id": control["id"],
            "control_name":         control["name"],
            "control_text_snippet": snippet,
            "policy_passage_id":    f"synthetic_{control['id']}_{label}",
            "policy_name":          "synthetic",
            "policy_section":       "synthetic",
            "compliance_status":    status,
            "confidence":           4,
            "is_hard_negative":     False,
            "mismatch_reason":      "synthetic",
            "evidence_or_notes":    "synthetically generated",
            "comments":             "auto-generated by generate_synthetic_data.py",
            "policy_text_snippet":  clause_text,
        })
    return records


# ── Progress helpers ───────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic FA/PA/NA training data via Anthropic API")
    ap.add_argument("--family",  default=None,
                    help="Process only this control family (e.g. T4, M2). Default: all families.")
    ap.add_argument("--resume",  action="store_true",
                    help="Skip controls already in synthetic_progress.json.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print first 3 prompts without calling the API.")
    args = ap.parse_args()

    # ── Load controls ─────────────────────────────────────────────────────────
    controls = load_controls(family_filter=args.family)
    print(f"Controls loaded : {len(controls)}"
          + (f"  (family={args.family})" if args.family else ""))

    # ── Load progress ─────────────────────────────────────────────────────────
    progress = load_progress()
    if args.resume and progress:
        before = len(controls)
        controls = [c for c in controls if c["id"] not in progress]
        print(f"Resuming        : {before - len(controls)} already done, {len(controls)} remaining")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n--- DRY RUN: first 3 prompts ---\n")
        for ctrl in controls[:3]:
            print(f"=== {ctrl['id']} — {ctrl['name']} ===")
            print(f"SYSTEM: {SYSTEM_PROMPT}\n")
            print(f"USER:\n{build_user_prompt(ctrl)}\n")
            print("-" * 60)
        return

    # ── Init Anthropic client ─────────────────────────────────────────────────
    try:
        import anthropic
    except ImportError:
        print("Install anthropic SDK: pip install anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # ── Process controls ──────────────────────────────────────────────────────
    all_records: list[dict] = []
    counts = {"FA": 0, "PA": 0, "NA": 0, "skipped": 0}

    total = len(controls)
    for i, ctrl in enumerate(controls, 1):
        print(f"[{i:3d}/{total}] {ctrl['id']}  {ctrl['name'][:50]}", end=" ... ", flush=True)

        clauses = call_api(client, ctrl)

        if clauses is None:
            print("SKIPPED")
            counts["skipped"] += 1
        else:
            records = make_records(ctrl, clauses)
            all_records.extend(records)
            for label in ("FA", "PA", "NA"):
                if clauses.get(label):
                    counts[label] += 1
            progress[ctrl["id"]] = clauses
            print(f"OK  (FA/PA/NA generated)")

        # Save progress every 10 controls
        if i % 10 == 0:
            save_progress(progress)
            print(f"  → Progress saved ({i}/{total})")

        time.sleep(0.5)

    # Final progress save
    save_progress(progress)

    # ── Save output ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Controls processed : {total - counts['skipped']}/{total}")
    print(f"FA generated       : {counts['FA']}")
    print(f"PA generated       : {counts['PA']}")
    print(f"NA generated       : {counts['NA']}")
    print(f"Total new examples : {counts['FA'] + counts['PA'] + counts['NA']}")
    print(f"Skipped (error)    : {counts['skipped']}")
    print(f"Output             : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
