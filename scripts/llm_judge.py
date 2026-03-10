#!/usr/bin/env python3
"""
LLM-as-Judge: post-processing filter for compliance mappings using a local Ollama model.

The pipeline (BM25 + Dense + Cross-Encoder) produces ~1,400 noisy "Partially Addressed"
predictions with very low precision (~0.3%).  This script re-evaluates each predicted
(control, passage) pair using a local LLM and keeps only the ones the LLM confirms.

Pipeline:
  quick_start_compliance.py → data/06_compliance_mappings/mappings.json
                                          ↓
  llm_judge.py              → data/06_compliance_mappings/mappings_llm_judged.json
                                          ↓
  (optional) re-run evaluation against golden set

Requirements:
  pip install requests
  ollama pull llama3.2          # 4 GB, good quality (recommended)
  ollama pull mistral            # 4 GB, faster
  ollama pull llama3.1:8b        # 5 GB, better for nuanced compliance text

Usage:
  # Default: filter mappings.json with llama3.2
  python3 scripts/llm_judge.py

  # Use a different model
  python3 scripts/llm_judge.py --model mistral

  # Dry-run: only judge first 20 pairs to test
  python3 scripts/llm_judge.py --limit 20

  # Re-evaluate against golden set after judging
  python3 scripts/llm_judge.py --evaluate
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a compliance auditor specialising in UAE Information Assurance (UAE IA) Regulation.
Your task is to assess whether a policy passage addresses a specific compliance control.

Be strict and precise:
- "Fully Addressed": the passage explicitly and completely covers the control requirement.
- "Partially Addressed": the passage touches on the topic but does not fully satisfy the control.
- "Not Addressed": the passage is about a different topic and does not address the control.

Respond with EXACTLY one of these three labels followed by a brief reason (1 sentence).
Format: LABEL | reason"""

USER_PROMPT_TEMPLATE = """Control ID: {control_id}
Control requirement:
{control_text}

Policy passage:
{passage_text}

Does this passage address the control?"""


# ── Ollama client ─────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str, host: str = "http://localhost:11434",
                timeout: int = 60) -> str:
    import requests
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 80},
    }
    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it's running:\n"
            "  ollama serve\n"
            "And that the model is pulled:\n"
            f"  ollama pull {model}"
        )


def parse_llm_verdict(response: str) -> tuple[str, str]:
    """Extract (label, reason) from LLM response."""
    valid = {"fully addressed", "partially addressed", "not addressed"}
    for line in response.strip().splitlines():
        parts = line.split("|", 1)
        label_raw = parts[0].strip().lower()
        for v in valid:
            if v in label_raw:
                label = v.title()
                reason = parts[1].strip() if len(parts) > 1 else ""
                return label, reason
    # Fallback: scan for any valid label in full response
    low = response.lower()
    if "fully addressed" in low:
        return "Fully Addressed", response[:120]
    if "partially addressed" in low:
        return "Partially Addressed", response[:120]
    return "Not Addressed", response[:120]


# ── Controls index ────────────────────────────────────────────────────────────

def build_control_index(controls_path: str) -> dict:
    """Build control_id → control_text mapping from the controls file."""
    with open(controls_path, encoding="utf-8") as f:
        raw = json.load(f)

    index = {}
    for item in raw:
        # Handle both flat {control_id, control_statement} and
        # Label Studio {data: {article_id, text}} formats
        if "data" in item:
            d = item["data"]
            cid = d.get("article_id") or d.get("control_id", "")
            text = d.get("text", "")
        else:
            cid = item.get("control_id") or item.get("control_number", "")
            text = item.get("control_statement") or item.get("control_text", "")
            # Append sub-controls
            for sc in item.get("sub_controls", []):
                sc_text = sc.get("control_statement") or sc.get("text", "")
                if sc_text:
                    text += f"\n{sc.get('sub_control_id', '')}: {sc_text}"
        if cid:
            index[cid] = text.strip()
    return index


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge for compliance mappings")
    parser.add_argument("--mappings",  default="data/06_compliance_mappings/mappings.json")
    parser.add_argument("--controls",  default="data/04_label_studio/imports/uae_ia_controls_raw.json")
    parser.add_argument("--output",    default="data/06_compliance_mappings/mappings_llm_judged.json")
    parser.add_argument("--model",     default="llama3.2",
                        help="Ollama model name (default: llama3.2). "
                             "Alternatives: mistral, llama3.1:8b")
    parser.add_argument("--host",      default="http://localhost:11434")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Only judge first N mappings (for testing)")
    parser.add_argument("--keep-not-addressed", action="store_true",
                        help="Also include 'Not Addressed' LLM verdicts in output "
                             "(default: drop them)")
    parser.add_argument("--evaluate",  action="store_true",
                        help="Run evaluation against golden set after judging")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading mappings from {args.mappings} ...")
    with open(args.mappings, encoding="utf-8") as f:
        mappings = json.load(f)
    print(f"  {len(mappings)} pipeline predictions loaded")

    # Only judge Fully/Partially Addressed — Not Addressed are already filtered
    to_judge = [m for m in mappings
                if m.get("status", "") in ("Fully Addressed", "Partially Addressed")]
    if args.limit:
        to_judge = to_judge[:args.limit]
        print(f"  Limiting to first {args.limit} predictions (--limit)")

    print(f"  {len(to_judge)} predictions to judge with LLM")

    print(f"\nLoading controls index from {args.controls} ...")
    ctrl_index = build_control_index(args.controls)
    print(f"  {len(ctrl_index)} controls indexed")

    # ── Verify Ollama is up ───────────────────────────────────────────────────
    print(f"\nVerifying Ollama connection (model: {args.model}) ...")
    try:
        test = call_ollama("Reply with OK", model=args.model, host=args.host, timeout=30)
        print(f"  Ollama OK — response: {test[:40]!r}")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # ── Judge each mapping ────────────────────────────────────────────────────
    print(f"\nJudging {len(to_judge)} mappings ...")
    results = []
    verdict_counts = {"Fully Addressed": 0, "Partially Addressed": 0, "Not Addressed": 0}
    t0 = time.time()

    for i, m in enumerate(to_judge):
        ctrl_id = m.get("source_control_id", "")
        passage = m.get("evidence_text", "")[:1200]   # cap to avoid token overflow
        ctrl_text = ctrl_index.get(ctrl_id, f"Control {ctrl_id}")

        prompt = USER_PROMPT_TEMPLATE.format(
            control_id=ctrl_id,
            control_text=ctrl_text[:600],
            passage_text=passage,
        )

        try:
            response = call_ollama(prompt, model=args.model, host=args.host)
            label, reason = parse_llm_verdict(response)
        except Exception as e:
            label, reason = "Not Addressed", f"LLM error: {e}"

        verdict_counts[label] = verdict_counts.get(label, 0) + 1

        judged = dict(m)
        judged["llm_verdict"]       = label
        judged["llm_reason"]        = reason
        judged["llm_model"]         = args.model
        judged["original_status"]   = m.get("status")
        judged["status"]            = label     # update status with LLM verdict

        if label != "Not Addressed" or args.keep_not_addressed:
            results.append(judged)

        # Progress every 10 items
        if (i + 1) % 10 == 0 or i == len(to_judge) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(to_judge) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(to_judge)}]  "
                  f"kept={len(results)}  "
                  f"verdicts={verdict_counts}  "
                  f"ETA={remaining/60:.1f}min", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed_total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"LLM Judge complete in {elapsed_total/60:.1f} min")
    print(f"  Input predictions  : {len(to_judge)}")
    print(f"  LLM verdicts       : {verdict_counts}")
    print(f"  Kept (non-NA)      : {len(results)}")
    print(f"  Reduction          : {len(to_judge)} → {len(results)} "
          f"({100*(1-len(results)/max(len(to_judge),1)):.0f}% noise removed)")
    print(f"  Output             : {args.output}")

    if args.evaluate:
        print("\nRunning evaluation against golden set ...")
        try:
            import subprocess
            result = subprocess.run(
                ["python3", "scripts/evaluate_pipeline.py",
                 "--mappings", args.output],
                capture_output=True, text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
        except FileNotFoundError:
            print("  evaluate_pipeline.py not found — run evaluation manually")


if __name__ == "__main__":
    main()
