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

# ── Prompt templates ──────────────────────────────────────────────────────────
#
# Strategy:
#   1. Role + domain framing  — grounds the model in UAE IA compliance
#   2. Criteria with examples — removes label ambiguity with concrete cases
#   3. Chain-of-Thought (CoT) — forces the model to reason before labelling,
#      which improves accuracy on small (1B) models significantly
#   4. Structured output      — deterministic parsing even with verbose CoT
#
# Prompt variant is selectable via --prompt-style {strict|cot|fewshot}
# default = "cot"  (best accuracy/speed balance for 1B models)

# ── Style: strict (original, fastest) ─────────────────────────────────────────
SYSTEM_PROMPT_STRICT = """You are a compliance auditor specialising in UAE Information Assurance (UAE IA) Regulation.
Your task is to assess whether a policy passage addresses a specific compliance control.

Definitions:
- "Fully Addressed"   : The passage explicitly and completely satisfies the control requirement — \
the control objective is clearly met with no gaps.
- "Partially Addressed": The passage is relevant and touches on the control topic but leaves \
some requirements unmet or only implied.
- "Not Addressed"     : The passage is about a different topic, or only mentions the subject in \
passing without addressing the control requirement.

Rules:
- Generic statements like "we comply with regulations" do NOT count as Fully Addressed.
- A passage that merely names the topic (e.g. "access control") without describing HOW it is \
implemented is at most Partially Addressed.
- If uncertain between Fully and Partially, choose Partially Addressed.

Respond with EXACTLY one of the three labels, then a pipe, then one sentence of reasoning.
Format: LABEL | reasoning"""

USER_PROMPT_STRICT = """Control ID: {control_id}
Control requirement:
{control_text}

Policy passage:
{passage_text}

Verdict:"""

# ── Style: cot (Chain-of-Thought, recommended for 1B models) ──────────────────
SYSTEM_PROMPT_COT = """You are a compliance auditor specialising in UAE Information Assurance (UAE IA) Regulation.

Your job: decide whether a POLICY PASSAGE addresses a COMPLIANCE CONTROL.

Label definitions:
- Fully Addressed    : passage explicitly and completely covers the control — no gaps.
- Partially Addressed: passage is relevant but leaves requirements unmet or only implied.
- Not Addressed      : passage is off-topic or too generic to satisfy the control.

Strict rules:
- Vague statements ("we follow best practices") = Not Addressed.
- Topic mentioned but no HOW/WHAT details = Partially Addressed at best.
- When in doubt between Fully and Partially → choose Partially Addressed.

Think step by step BEFORE giving your verdict:
  Step 1: What does the control require?
  Step 2: What does the passage actually say?
  Step 3: Is there a match? What is missing?
  Verdict: LABEL | one-sentence reason

Always end with the line:
  Verdict: LABEL | reason"""

USER_PROMPT_COT = """=== COMPLIANCE CONTROL ===
ID: {control_id}
Requirement:
{control_text}

=== POLICY PASSAGE ===
{passage_text}

=== YOUR ANALYSIS ===
Step 1: What does the control require?"""

# ── Style: fewshot (few-shot examples, best accuracy, slower) ─────────────────
SYSTEM_PROMPT_FEWSHOT = """You are a compliance auditor specialising in UAE Information Assurance (UAE IA) Regulation.
Assess whether a policy passage addresses a compliance control.

Labels:
- Fully Addressed    : passage explicitly and completely covers the control — no gaps.
- Partially Addressed: passage is relevant but leaves requirements unmet or only implied.
- Not Addressed      : passage is off-topic or too generic to satisfy the control.

Here are examples of correct verdicts:

--- EXAMPLE 1 ---
Control: T1.1.1 - The organization shall maintain an asset inventory including hardware, software and data assets.
Passage: "All information assets are classified and recorded in the asset register maintained by IT. The register is reviewed quarterly."
Verdict: Fully Addressed | The passage explicitly describes maintaining an asset inventory (asset register) with regular review, directly satisfying the control.

--- EXAMPLE 2 ---
Control: T2.2.6 - Physical access to server rooms must be controlled and logged with biometric authentication.
Passage: "Physical security of the organization premises is managed through access cards and CCTV surveillance."
Verdict: Partially Addressed | The passage covers physical access control but does not mention server-room-specific controls or the required biometric authentication and access logging.

--- EXAMPLE 3 ---
Control: M3.2.1 - All staff must complete annual security awareness training covering phishing and social engineering.
Passage: "The organization complies with all applicable laws and regulations relating to information security."
Verdict: Not Addressed | The passage is a generic compliance statement and does not describe any training programme or address the control requirement.

Now assess the following:"""

USER_PROMPT_FEWSHOT = """Control ID: {control_id}
Control requirement:
{control_text}

Policy passage:
{passage_text}

Verdict:"""

# ── Active style registry ──────────────────────────────────────────────────────
PROMPT_STYLES = {
    "strict":  (SYSTEM_PROMPT_STRICT,  USER_PROMPT_STRICT),
    "cot":     (SYSTEM_PROMPT_COT,     USER_PROMPT_COT),
    "fewshot": (SYSTEM_PROMPT_FEWSHOT, USER_PROMPT_FEWSHOT),
}
DEFAULT_PROMPT_STYLE = "cot"


# ── Ollama client ─────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str, system: str,
                host: str = "http://localhost:11434",
                timeout: int = 180, num_predict: int = 200) -> str:
    import requests
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
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
    """Extract (label, reason) from LLM response.

    Handles three formats:
      - Direct:   "Fully Addressed | reason"
      - CoT:      "Verdict: Fully Addressed | reason"
      - Fallback: scans full text for any valid label
    """
    valid = {"fully addressed", "partially addressed", "not addressed"}

    # Prioritise the explicit "Verdict:" line produced by CoT prompts
    verdict_line = None
    for line in response.strip().splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("verdict:"):
            verdict_line = stripped[len("verdict:"):].strip()
            break

    candidates = ([verdict_line] if verdict_line else []) + response.strip().splitlines()

    for line in candidates:
        if not line:
            continue
        parts = line.split("|", 1)
        label_raw = parts[0].strip().lower()
        for v in valid:
            if v in label_raw:
                label = v.title()
                reason = parts[1].strip() if len(parts) > 1 else ""
                return label, reason

    # Last-resort scan of full response
    low = response.lower()
    if "fully addressed" in low:
        return "Fully Addressed", response[:200]
    if "partially addressed" in low:
        return "Partially Addressed", response[:200]
    return "Not Addressed", response[:200]


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
    parser.add_argument("--model",        default="llama3.2:1b",
                        help="Ollama model name (default: llama3.2:1b). "
                             "Alternatives: mistral, llama3.1:8b, llama3.2")
    parser.add_argument("--prompt-style", default=DEFAULT_PROMPT_STYLE,
                        choices=list(PROMPT_STYLES),
                        help=("Prompting strategy: "
                              "'strict' = direct label (fastest), "
                              "'cot' = chain-of-thought reasoning (default, best for 1B), "
                              "'fewshot' = 3 labelled examples (most accurate, slower)"))
    parser.add_argument("--host",         default="http://localhost:11434")
    parser.add_argument("--timeout",      type=int, default=180,
                        help="Seconds to wait per Ollama response (default=180). "
                             "Increase if you see ReadTimeout errors.")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Only judge first N mappings (for testing)")
    parser.add_argument("--keep-not-addressed", action="store_true",
                        help="Also include 'Not Addressed' LLM verdicts in output "
                             "(default: drop them)")
    parser.add_argument("--evaluate",     action="store_true",
                        help="Run evaluation against golden set after judging")
    args = parser.parse_args()

    system_prompt, user_template = PROMPT_STYLES[args.prompt_style]
    # CoT needs more tokens to reason; strict/fewshot only need the label line
    num_predict = 300 if args.prompt_style == "cot" else 100
    print(f"Prompt style : {args.prompt_style}  (num_predict={num_predict})")

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
        test = call_ollama("Reply with OK", model=args.model, system="You are a helpful assistant.",
                           host=args.host, timeout=args.timeout, num_predict=20)
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

        prompt = user_template.format(
            control_id=ctrl_id,
            control_text=ctrl_text[:600],
            passage_text=passage,
        )

        try:
            response = call_ollama(prompt, model=args.model, system=system_prompt,
                                   host=args.host, timeout=args.timeout,
                                   num_predict=num_predict)
            label, reason = parse_llm_verdict(response)
        except Exception as e:
            label, reason = "Not Addressed", f"LLM error: {e}"

        verdict_counts[label] = verdict_counts.get(label, 0) + 1

        judged = dict(m)
        judged["llm_verdict"]       = label
        judged["llm_reason"]        = reason
        judged["llm_model"]         = args.model
        judged["llm_prompt_style"]  = args.prompt_style
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
