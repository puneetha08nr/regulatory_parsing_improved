#!/usr/bin/env python3
"""
Phase 1 — Synthetic Training Data Generator
============================================
Uses a local Ollama LLM to generate synthetic (control, passage) pairs for all
UAE IA controls. This expands the 100-positive golden dataset to ~2,000 labeled
pairs covering all 263 controls, giving the cross-encoder enough domain signal
to learn UAE IA-specific compliance patterns.

For each control the script generates:
  - 2 Fully Addressed passages  (explicit, complete, specific)
  - 2 Partially Addressed passages  (relevant but incomplete)
  - 3 Not Addressed passages  (plausible policy text, different domain)

Total: 263 controls × 7 passages = ~1,841 pairs

Features:
  - Resume support: saves a checkpoint every --save-every controls so a
    2–3 hr run can be interrupted and restarted without losing work.
  - Validation: generated text is checked for minimum length and coherence.
  - Domain variety for negatives: cycles through unrelated policy domains so
    the model sees diverse false-match patterns.

Usage:
  # Full run (all 263 controls)
  python3 scripts/generate_synthetic_pairs.py

  # Test with first 10 controls
  python3 scripts/generate_synthetic_pairs.py --limit 10

  # Resume interrupted run
  python3 scripts/generate_synthetic_pairs.py --resume

  # Use a different model
  python3 scripts/generate_synthetic_pairs.py --model llama3.2
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Negative domain pool ──────────────────────────────────────────────────────
# Cycled through when generating "Not Addressed" passages, to ensure variety.
NEGATIVE_DOMAINS = [
    "visitor access management and badge issuance procedures",
    "email and internet acceptable use policy",
    "clean desk and clear screen policy",
    "software licensing and procurement",
    "business travel expense reimbursement",
    "employee onboarding and offboarding HR procedures",
    "canteen and facilities management",
    "printer and photocopier usage guidelines",
    "mobile phone usage during working hours",
    "vehicle fleet management and driver safety",
    "stationery and office supplies requisition",
    "document retention and archival procedures",
    "training room booking and AV equipment usage",
    "social media usage guidelines for employees",
    "whistleblowing and ethics hotline procedures",
]

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_WRITER = """You are a senior information security policy writer specialising in UAE \
Information Assurance (UAE IA) Regulation compliance. You write clear, specific, realistic \
internal policy text for government and enterprise organisations in the UAE.

Your policy passages must:
- Be written in formal policy language (not bullet points)
- Name specific mechanisms, responsible parties, and timeframes where applicable
- Be 3–5 sentences long
- Sound like an extract from a real internal policy document
- Output ONLY the passage text — no titles, no labels, no explanations"""

PROMPT_FULLY = """Control requirement (UAE IA {control_number}):
{control_text}

Write a policy passage that FULLY satisfies this control.
The passage must explicitly address ALL of these:
1. The specific mechanism or control measure
2. Who is responsible (role/department)
3. How often or under what trigger it applies
4. Any verification or review process

Output only the passage text (3–5 sentences):"""

PROMPT_PARTIAL = """Control requirement (UAE IA {control_number}):
{control_text}

Write a policy passage that is RELEVANT to this control but INCOMPLETE.
The passage should:
- Cover the general topic of the control
- Mention at least one related procedure or measure
- But OMIT at least one key requirement (e.g. missing the verification step,
  or not specifying frequency, or not naming the responsible party)

Output only the passage text (2–4 sentences):"""

PROMPT_NEGATIVE = """Policy domain: {domain}

Write a realistic internal policy passage about {domain}.
The passage must:
- Be specific to this domain only
- NOT relate to: {control_topic}
- Sound like it comes from a real internal policy document

Output only the passage text (2–4 sentences):"""


# ── Ollama client ─────────────────────────────────────────────────────────────

def call_ollama(prompt: str, system: str, model: str,
                host: str = "http://localhost:11434",
                timeout: int = 180, num_predict: int = 250) -> str:
    import requests
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": num_predict, "top_p": 0.9},
    }
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def generate_with_retry(prompt: str, system: str, model: str, host: str,
                        timeout: int, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            text = call_ollama(prompt, system, model, host, timeout)
            if len(text.split()) >= 20:   # minimum coherence check
                return text
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


# ── Controls loader ───────────────────────────────────────────────────────────

def load_controls(path: str) -> list:
    """Load controls, filtering to those with a non-trivial control_statement."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    controls = []
    seen_ids = set()
    for c in raw:
        cid = c.get("control_id", "")
        stmt = (c.get("control_statement") or "").strip()
        num = c.get("control_number", cid)
        # Skip chapter-level entries (no dot in number = family header like "M1", "T2")
        if not stmt or len(stmt.split()) < 5:
            continue
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        controls.append({
            "control_id":     cid,
            "control_number": num,
            "control_statement": stmt,
            "control_family":  c.get("control_family", ""),
            "section_title":   c.get("section_title", ""),
        })
    return controls


def build_control_text(ctrl: dict, max_chars: int = 600) -> str:
    return ctrl["control_statement"][:max_chars]


# ── Output schema (matches golden_mapping_dataset.json) ──────────────────────

def make_record(ctrl: dict, status: str, passage: str, source: str) -> dict:
    return {
        "control_id":          ctrl["control_number"],
        "control_name":        ctrl.get("control_family", ""),
        "policy_passage_id":   f"synthetic_{ctrl['control_id']}_{status.replace(' ', '_').lower()}",
        "policy_name":         "synthetic",
        "policy_section":      "synthetic",
        "compliance_status":   status,
        "confidence":          3,
        "mismatch_reason":     None,
        "is_hard_negative":    False,
        "is_synthetic":        True,
        "synthetic_source":    source,
        "evidence_or_notes":   "",
        "comments":            "",
        "control_text_snippet": ctrl["control_statement"][:400],
        "policy_text_snippet":  passage[:800],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic (control, passage) training pairs")
    ap.add_argument("--controls",    default="data/04_label_studio/imports/uae_ia_controls_raw.json")
    ap.add_argument("--output",      default="data/07_golden_mapping/synthetic_pairs.json")
    ap.add_argument("--checkpoint",  default="data/07_golden_mapping/synthetic_pairs_checkpoint.json",
                    help="Partial results saved here every --save-every controls")
    ap.add_argument("--model",       default="llama3.2:1b")
    ap.add_argument("--host",        default="http://localhost:11434")
    ap.add_argument("--timeout",     type=int, default=180)
    ap.add_argument("--limit",       type=int, default=None,
                    help="Only process first N controls (for testing)")
    ap.add_argument("--resume",      action="store_true",
                    help="Resume from checkpoint if it exists")
    ap.add_argument("--save-every",  type=int, default=20,
                    help="Save checkpoint every N controls (default: 20)")
    ap.add_argument("--n-fully",     type=int, default=2,
                    help="Fully Addressed passages per control")
    ap.add_argument("--n-partial",   type=int, default=2,
                    help="Partially Addressed passages per control")
    ap.add_argument("--n-negative",  type=int, default=3,
                    help="Not Addressed passages per control")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # ── Load controls ─────────────────────────────────────────────────────────
    print(f"Loading controls from {args.controls} ...")
    controls = load_controls(args.controls)
    print(f"  {len(controls)} controls with non-trivial statements")

    if args.limit:
        controls = controls[:args.limit]
        print(f"  Limited to first {args.limit} controls (--limit)")

    # ── Resume support ────────────────────────────────────────────────────────
    already_done: set = set()
    results: list = []
    ckpt_path = Path(args.checkpoint)
    if args.resume and ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            results = json.load(f)
        already_done = {r["control_id"] for r in results}
        print(f"  Resuming: {len(already_done)} controls already done, "
              f"{len(results)} records loaded from checkpoint")

    remaining = [c for c in controls if c["control_number"] not in already_done]
    print(f"  {len(remaining)} controls to process\n")

    # ── Verify Ollama ─────────────────────────────────────────────────────────
    print(f"Verifying Ollama ({args.model}) ...")
    try:
        import requests
        r = requests.post(f"{args.host}/api/generate",
                          json={"model": args.model, "prompt": "Reply OK",
                                "stream": False, "options": {"num_predict": 5}},
                          timeout=30)
        r.raise_for_status()
        print(f"  Ollama OK\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama — {e}")
        print("  Start it with: ollama serve")
        sys.exit(1)

    neg_domain_cycle = NEGATIVE_DOMAINS.copy()
    random.shuffle(neg_domain_cycle)
    neg_idx = 0

    t0 = time.time()
    errors = 0

    for ctrl_idx, ctrl in enumerate(remaining):
        ctrl_text = build_control_text(ctrl)
        ctrl_num = ctrl["control_number"]
        ctrl_topic = ctrl.get("section_title") or ctrl.get("control_family") or ctrl_num
        ctrl_records = []
        ctrl_errors = 0

        # ── Fully Addressed ───────────────────────────────────────────────────
        for i in range(args.n_fully):
            prompt = PROMPT_FULLY.format(
                control_number=ctrl_num,
                control_text=ctrl_text,
            )
            try:
                text = generate_with_retry(prompt, SYSTEM_WRITER, args.model,
                                           args.host, args.timeout)
                if text:
                    ctrl_records.append(make_record(ctrl, "Fully Addressed", text,
                                                    f"synthetic_fa_{i+1}"))
            except Exception as e:
                ctrl_errors += 1
                errors += 1

        # ── Partially Addressed ───────────────────────────────────────────────
        for i in range(args.n_partial):
            prompt = PROMPT_PARTIAL.format(
                control_number=ctrl_num,
                control_text=ctrl_text,
            )
            try:
                text = generate_with_retry(prompt, SYSTEM_WRITER, args.model,
                                           args.host, args.timeout)
                if text:
                    ctrl_records.append(make_record(ctrl, "Partially Addressed", text,
                                                    f"synthetic_pa_{i+1}"))
            except Exception as e:
                ctrl_errors += 1
                errors += 1

        # ── Not Addressed ─────────────────────────────────────────────────────
        for i in range(args.n_negative):
            domain = neg_domain_cycle[neg_idx % len(neg_domain_cycle)]
            neg_idx += 1
            prompt = PROMPT_NEGATIVE.format(
                domain=domain,
                control_topic=ctrl_topic,
            )
            try:
                text = generate_with_retry(prompt, SYSTEM_WRITER, args.model,
                                           args.host, args.timeout)
                if text:
                    ctrl_records.append(make_record(ctrl, "Not Addressed", text,
                                                    f"synthetic_na_{i+1}"))
            except Exception as e:
                ctrl_errors += 1
                errors += 1

        results.extend(ctrl_records)

        # ── Progress ──────────────────────────────────────────────────────────
        done = ctrl_idx + 1
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(remaining) - done) / rate if rate > 0 else 0
        total_so_far = len(results)
        print(f"  [{done}/{len(remaining)}] {ctrl_num:10s}  "
              f"records={len(ctrl_records)} (err={ctrl_errors})  "
              f"total={total_so_far}  ETA={eta/60:.1f}min",
              flush=True)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if done % args.save_every == 0:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"    checkpoint saved → {ckpt_path}  ({len(results)} records)")

    # ── Final save ────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed_total = time.time() - t0
    from collections import Counter
    status_counts = Counter(r["compliance_status"] for r in results)

    print(f"\n{'='*60}")
    print(f"Synthetic data generation complete in {elapsed_total/60:.1f} min")
    print(f"  Controls processed : {len(remaining)}")
    print(f"  Total records      : {len(results)}")
    print(f"  By status          : {dict(status_counts)}")
    print(f"  Errors             : {errors}")
    print(f"  Output             : {out_path}")
    print(f"\nNext step:")
    print(f"  python3 scripts/prepare_golden_for_training.py \\")
    print(f"    --golden data/07_golden_mapping/golden_mapping_dataset.json \\")
    print(f"    --synthetic {out_path} \\")
    print(f"    --format reranker")


if __name__ == "__main__":
    main()
