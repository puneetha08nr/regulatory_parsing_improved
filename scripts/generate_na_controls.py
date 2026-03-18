#!/usr/bin/env python3
"""
generate_na_controls.py — Generate confusable "Not Addressed" (NA) control pairs.

For each passage in the golden set that has at least one positive (FA/PA) control,
ask an LLM to identify 3 controls that an NLP system would plausibly but wrongly
match to the passage — confusable neighbours due to vocabulary/topic overlap.

These are NOT random negatives. They are hard negatives that specifically target
the vocabulary overlap failure mode of reranker/NLI systems.

Usage:
    python3 scripts/generate_na_controls.py \\
        --golden data/07_golden_mapping/golden_mapping_dataset.json \\
        --controls data/04_label_studio/imports/uae_ia_controls_raw.json \\
        --output data/07_golden_mapping/na_confusable_pairs.json \\
        --model llama3.2

    # Dry-run (print prompts, no Ollama calls)
    python3 scripts/generate_na_controls.py --dry-run --limit 3

    # Resume from partial output
    python3 scripts/generate_na_controls.py --resume
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYSTEM_PROMPT = """You are a UAE IA compliance expert.
You understand how automated NLP systems make mistakes by matching controls
to passages due to vocabulary overlap rather than actual compliance coverage.
Your task is to identify confusable controls — controls that a keyword or 
embedding system would incorrectly match to a passage, but that the passage
does not actually satisfy."""

USER_PROMPT = """This policy passage is about:
---
{passage_text}
---

This passage is correctly mapped to these controls:
{positive_controls_list}

Below is a list of UAE IA controls that share vocabulary or domain with this passage.
Choose exactly 3 that an automated NLP system would INCORRECTLY match to this passage
due to keyword or topic overlap — but that this passage does NOT actually satisfy.

You MUST only use control IDs from the list below. Do not invent IDs.

Candidate controls to choose from:
{control_menu}

For each chosen wrong control give exactly this format:
CONTROL_ID: [ID from the list above]
CONTROL_NAME: [name from the list above]
WHY_CONFUSED: [which keywords or phrases cause the false match]
WHY_WRONG: [what requirement this passage is missing]

CONTROL_ID: [ID from the list above]
CONTROL_NAME: [name]
WHY_CONFUSED: [keywords]
WHY_WRONG: [missing requirement]

CONTROL_ID: [ID from the list above]
CONTROL_NAME: [name]
WHY_CONFUSED: [keywords]
WHY_WRONG: [missing requirement]"""


def call_ollama(prompt: str, system: str, model: str,
                host: str = "http://localhost:11434",
                timeout: int = 90) -> str:
    import requests
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 400},
    }
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def parse_na_response(response: str) -> list[dict]:
    """Parse 3-block CONTROL_ID/NAME/WHY_CONFUSED/WHY_WRONG response."""
    entries = []
    current: dict = {}
    for line in response.strip().splitlines():
        line = line.strip()
        up = line.upper()
        if up.startswith("CONTROL_ID:"):
            if current.get("control_id"):
                entries.append(current)
            cid = line.split(":", 1)[1].strip().strip("[]").upper()
            current = {"control_id": cid, "control_name": "", "why_confused": "", "why_wrong": ""}
        elif up.startswith("CONTROL_NAME:"):
            current["control_name"] = line.split(":", 1)[1].strip().strip("[]")
        elif up.startswith("WHY_CONFUSED:"):
            current["why_confused"] = line.split(":", 1)[1].strip().strip("[]")
        elif up.startswith("WHY_WRONG:"):
            current["why_wrong"] = line.split(":", 1)[1].strip().strip("[]")
    if current.get("control_id"):
        entries.append(current)
    return entries


def load_controls_index(controls_path: str) -> dict:
    """Build control_id → row from controls JSON.
    Prefers uae_ia_controls_clean.json (M1.1.1 format).
    Falls back to raw JSON and strips the UAE_IA_CTRL_ prefix if present.
    """
    # Always merge clean controls first (canonical IDs)
    root = Path(controls_path).resolve().parent.parent
    clean_path = root / "data/02_processed/uae_ia_controls_clean.json"
    index = {}

    if clean_path.exists():
        with open(clean_path, encoding="utf-8") as f:
            clean = json.load(f)
        for item in clean:
            c = item.get("control", {})
            cid = (c.get("id") or "").strip()
            name = (c.get("name") or "").strip()
            # Strip leading "M1.1.1 - " or "M1.1.1 — " prefix from name
            import re as _re
            name = _re.sub(r"^[A-Z][0-9]+\.[0-9]+(?:\.[0-9]+)?\s*[-–—]\s*", "", name).strip()
            if not name:
                name = cid
            if cid:
                index[cid] = {
                    "control_id": cid,
                    "control_name": name,
                    "control_statement": (c.get("description") or "").strip(),
                    "control_family": item.get("control_family", ""),
                }

    # Supplement with raw JSON (strip UAE_IA_CTRL_ prefix)
    with open(controls_path, encoding="utf-8") as f:
        raw = json.load(f)
    for item in raw:
        cid = (item.get("control_id") or item.get("control_number") or "").strip()
        # Normalise: UAE_IA_CTRL_M1.1.1 → M1.1.1
        if cid.startswith("UAE_IA_CTRL_"):
            cid = cid[len("UAE_IA_CTRL_"):]
        if cid and cid not in index:
            index[cid] = item

    return index


def build_control_menu(controls_index: dict, positive_ids: set,
                       passage_text: str, max_per_family: int = 4) -> str:
    """Build a compact menu of candidate control IDs for the prompt.
    Filters to families that share vocabulary with the passage text,
    excludes known positives, and limits to max_per_family per family.
    """
    passage_upper = passage_text.upper()

    # Family keyword heuristics
    family_keywords = {
        "M1": ["POLICY", "MANAGEMENT", "GOVERNANCE", "LEADERSHIP", "CONTEXT"],
        "M2": ["RISK", "THREAT", "VULNERABILITY", "ASSESSMENT"],
        "M3": ["TRAINING", "AWARENESS", "EDUCATION", "PERSONNEL"],
        "M4": ["AUDIT", "REVIEW", "COMPLIANCE", "MONITORING"],
        "M5": ["ACCESS", "USER", "IDENTITY", "AUTHENTICATION", "PRIVILEGE"],
        "M6": ["INCIDENT", "RESPONSE", "CONTINUITY", "RECOVERY"],
        "T1": ["ASSET", "INVENTORY", "CLASSIFICATION", "LABEL", "MEDIA", "DISPOSAL", "OWNERSHIP"],
        "T2": ["PHYSICAL", "ENVIRONMENTAL", "FACILITY", "BUILDING", "EQUIPMENT"],
        "T3": ["ACCESS CONTROL", "NETWORK", "FIREWALL", "BOUNDARY", "REMOTE"],
        "T4": ["CRYPTOGRAPHY", "ENCRYPTION", "KEY", "CERTIFICATE", "PKI"],
        "T5": ["SYSTEM", "APPLICATION", "SOFTWARE", "DEVELOPMENT", "PATCH", "VULNERABILITY"],
        "T6": ["SUPPLIER", "THIRD.PARTY", "VENDOR", "OUTSOURCE", "CLOUD"],
        "T7": ["INCIDENT", "EVENT", "LOG", "MONITOR", "ALERT", "SIEM"],
        "T8": ["CONTINUITY", "RECOVERY", "BACKUP", "DISASTER", "BCP"],
        "T9": ["COMPLIANCE", "LEGAL", "REGULATION", "AUDIT", "CONTROL"],
    }

    # Score each family by keyword hits
    family_scores = {}
    for family, keywords in family_keywords.items():
        hits = sum(1 for kw in keywords if kw in passage_upper)
        if hits > 0:
            family_scores[family] = hits

    if not family_scores:
        # Fallback: include all families
        family_scores = {f: 1 for f in family_keywords}

    # Collect candidate IDs, excluding positives
    by_family: dict[str, list] = defaultdict(list)
    for cid, row in controls_index.items():
        if cid in positive_ids:
            continue
        if cid.count(".") != 2:  # only leaf controls like M1.1.1, T1.2.3
            continue
        family = cid[:2]
        if family in family_scores:
            name = (row.get("control_name") or "").strip()
            if not name or name == "Extracted Control":
                name = (row.get("control_statement") or "")[:60].strip()
            by_family[family].append((cid, name))

    lines = []
    for family, score in sorted(family_scores.items(), key=lambda x: -x[1]):
        for cid, name in by_family.get(family, [])[:max_per_family]:
            lines.append(f"  {cid} — {name[:70]}")

    return "\n".join(lines) if lines else "  (no candidates)"


def build_passage_groups(golden: list) -> dict:
    """Group golden entries by passage_id.
    Returns {passage_id: {positives: [...], policy_name, policy_section, text}}
    """
    groups = defaultdict(lambda: {
        "positives": [],
        "all_positive_ctrl_ids": set(),
        "policy_name": "",
        "policy_section": "",
        "text": "",
    })
    for g in golden:
        pid = g.get("policy_passage_id", "")
        status = (g.get("compliance_status") or "").strip()
        if not pid:
            continue
        grp = groups[pid]
        if not grp["policy_name"]:
            grp["policy_name"] = g.get("policy_name", "")
            grp["policy_section"] = g.get("policy_section", "")
            grp["text"] = (g.get("policy_text_snippet") or "").strip()
        if status in ("Fully Addressed", "Partially Addressed"):
            ctrl_id = (g.get("corrected_control_id") or g.get("control_id") or "").strip()
            ctrl_name = (g.get("control_name") or "").strip()
            grp["positives"].append({
                "control_id": ctrl_id,
                "control_name": ctrl_name,
                "status": status,
            })
            grp["all_positive_ctrl_ids"].add(ctrl_id)
    return groups


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description="Generate confusable NA control pairs using LLM")
    ap.add_argument("--golden",   default=str(root / "data/07_golden_mapping/golden_mapping_dataset.json"))
    ap.add_argument("--controls", default=str(root / "data/04_label_studio/imports/uae_ia_controls_raw.json"))
    ap.add_argument("--output",   default=str(root / "data/07_golden_mapping/na_confusable_pairs.json"))
    ap.add_argument("--model",    default="auto",
                    help="Ollama model. 'auto' (default) picks best available: "
                         "llama3.2:3b → llama3.2:1b → gpt-oss:20b. "
                         "Specify explicitly, e.g. --model llama3.2:1b")
    ap.add_argument("--host",     default="http://localhost:11434")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Only process first N passages (for testing)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print prompts without calling Ollama")
    ap.add_argument("--resume",   action="store_true",
                    help="Load existing output and skip already-processed passages")
    ap.add_argument("--timeout",  type=int, default=90,
                    help="Ollama timeout per call in seconds (default: 90)")
    args = ap.parse_args()

    print("=" * 60)
    print("Generate Confusable NA Control Pairs")
    print("=" * 60)

    # Load inputs
    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)
    controls_index = load_controls_index(args.controls)
    print(f"Golden entries : {len(golden)}")
    print(f"Controls index : {len(controls_index)}")

    # Group passages
    passage_groups = build_passage_groups(golden)
    passages_with_positives = {
        pid: grp for pid, grp in passage_groups.items()
        if grp["positives"] and grp["text"]
    }
    print(f"Unique passages: {len(passage_groups)}")
    print(f"  With positives (FA/PA): {len(passages_with_positives)}")

    # Apply limit
    passage_items = list(passages_with_positives.items())
    if args.limit:
        passage_items = passage_items[:args.limit]
        print(f"  Limited to first {args.limit} passages (--limit)")

    # Resume: load existing output and skip already-processed passage IDs
    existing_pairs: list = []
    already_done: set = set()
    if args.resume and Path(args.output).exists():
        with open(args.output, encoding="utf-8") as f:
            existing_pairs = json.load(f)
        already_done = {p["policy_passage_id"] for p in existing_pairs}
        print(f"Resuming: {len(existing_pairs)} pairs already saved, "
              f"{len(already_done)} passages already processed")

    if args.dry_run:
        print("\nDry-run mode: printing prompts, no Ollama calls.\n")
    else:
        # Verify Ollama and resolve model
        try:
            import requests
            r = requests.get(f"{args.host}/api/tags", timeout=5)
            r.raise_for_status()
            available = [m["name"] for m in r.json().get("models", [])]
            print(f"\nOllama models available: {available}")
        except Exception as e:
            print(f"\nERROR: Cannot reach Ollama at {args.host}: {e}")
            sys.exit(1)

        # Auto-resolve model
        if args.model == "auto":
            # Prefer larger models for better generation quality
            candidates = ["llama3.2:3b", "llama3.2", "llama3.1:8b", "gpt-oss:20b", "llama3.2:1b"]
            args.model = next((m for m in candidates if m in available), available[0] if available else "llama3.2:1b")
            print(f"Model (auto)   : {args.model}")
        elif args.model not in available:
            # Try prefix match (e.g. "llama3.2" matches "llama3.2:1b")
            match = next((m for m in available if m.startswith(args.model)), None)
            if match:
                print(f"Model '{args.model}' not found exactly; using '{match}'")
                args.model = match
            else:
                print(f"  WARNING: '{args.model}' not available. Available: {available}")
                print(f"  Defaulting to: {available[0]}")
                args.model = available[0]
        else:
            print(f"Model          : {args.model}")

    print()
    results = list(existing_pairs)
    skipped_hallucinated = 0
    skipped_positive_conflict = 0
    t0 = time.time()

    for idx, (passage_id, grp) in enumerate(passage_items):
        if passage_id in already_done:
            continue

        passage_text = grp["text"][:1000]
        positives = grp["positives"]
        positive_ids = grp["all_positive_ctrl_ids"]
        policy_name = grp["policy_name"]
        policy_section = grp["policy_section"]

        positive_list = "\n".join(
            f"  {p['control_id']} — {p['control_name']} ({p['status']})"
            for p in positives
        )
        control_menu = build_control_menu(controls_index, positive_ids, passage_text)

        prompt = USER_PROMPT.format(
            passage_text=passage_text,
            positive_controls_list=positive_list or "  (none recorded)",
            control_menu=control_menu,
        )

        if args.dry_run:
            print(f"\n--- Passage {idx+1}/{len(passage_items)}: {passage_id} ---")
            print(f"Positives: {[p['control_id'] for p in positives]}")
            print(prompt[:1200])
            print("---")
            continue

        try:
            response = call_ollama(
                prompt, system=SYSTEM_PROMPT, model=args.model,
                host=args.host, timeout=args.timeout,
            )
        except Exception as e:
            print(f"  [{idx+1}] ERROR for {passage_id}: {e}", flush=True)
            continue

        parsed = parse_na_response(response)

        added = 0
        for entry in parsed:
            cid = entry["control_id"]
            # Validate: must exist in controls index
            ctrl_row = controls_index.get(cid)
            if not ctrl_row:
                skipped_hallucinated += 1
                continue
            # Dedup: skip if already a positive for this passage
            if cid in positive_ids:
                skipped_positive_conflict += 1
                continue

            ctrl_text = (
                ctrl_row.get("control_statement") or ctrl_row.get("control_description") or ""
            ).strip()

            results.append({
                "control_id":           cid,
                "original_control_id":  cid,
                "corrected_control_id": cid,
                "control_name":         entry["control_name"] or ctrl_row.get("control_name", ""),
                "policy_passage_id":    passage_id,
                "policy_name":          policy_name,
                "policy_section":       policy_section,
                "compliance_status":    "Not Addressed",
                "confidence":           5,
                "mismatch_reason":      "confusable_neighbour",
                "is_hard_negative":     True,
                "is_synthetic_negative": False,
                "is_llm_generated_na":  True,
                "why_confused":         entry["why_confused"],
                "why_wrong":            entry["why_wrong"],
                "control_text_snippet": ctrl_text[:300],
                "policy_text_snippet":  passage_text[:300],
            })
            added += 1

        print(f"  [{idx+1}/{len(passage_items)}] {passage_id[:60]}  "
              f"→ {len(parsed)} parsed, {added} added  "
              f"(hallucinated={skipped_hallucinated}, conflicts={skipped_positive_conflict})",
              flush=True)

        # Save partial every 10 passages
        if (idx + 1) % 10 == 0:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(f"\nDry-run complete. {len(passage_items)} prompts shown.")
        return

    # Final save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("Done.")
    print(f"  Passages processed  : {len(passage_items) - len(already_done)}")
    print(f"  Confusable NA pairs : {len(results)}")
    print(f"  Hallucinated IDs    : {skipped_hallucinated}")
    print(f"  Positive conflicts  : {skipped_positive_conflict}")
    print(f"  Elapsed             : {elapsed/60:.1f} min")
    print(f"  Saved to            : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
