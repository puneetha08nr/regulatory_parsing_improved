#!/usr/bin/env python3
"""
LLM-as-Judge: verifies correct control assignment for compliance mappings.

The pipeline assigns policy passages to UAE IA controls.  This script asks a
local LLM a single question per pair: "Did the pipeline assign the CORRECT
control to this passage, or the wrong one?"

Every passage in an annotated policy document maps to some control.  "Not
Addressed" in the output means the pipeline assigned the wrong control —
not that the passage is irrelevant.

Pipeline:
  quick_start_compliance.py → data/06_compliance_mappings/mappings.json
                                          ↓
  llm_judge.py              → data/06_compliance_mappings/mappings_llm_judged.json
                                          ↓
  (optional) re-run evaluation against golden set

Requirements:
  pip install requests
  ollama pull llama3.2          # 4 GB, good quality (recommended)
  ollama pull llama3.2:1b       # 1.3 GB, fast for testing

Usage:
  python3 scripts/llm_judge.py
  python3 scripts/llm_judge.py --mappings data/06_compliance_mappings/mappings.json --limit 5 --dry-run
  python3 scripts/llm_judge.py --model gpt-oss:20b
  python3 scripts/llm_judge.py --use-finetuned --finetuned-model models/compliance-llm-judge
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


# ── Assignment-verification judge (default when not --use-finetuned) ───────────
#
# The judge answers one question:
#   "Did the pipeline assign the CORRECT control to this passage?"
# Score-based routing is removed — every pair gets the same prompt.

SYSTEM_ASSIGNMENT = """You are a UAE IA compliance auditor.
You are checking ONE specific (control, passage) pair.
A passage can satisfy multiple controls — your job is only to verify THIS specific control against THIS passage.
Topic similarity is not enough.
The passage must contain direct evidence for THIS control's specific requirement.
Answer only in the exact format shown."""

USER_ASSIGNMENT = """Control {control_id} — {control_name}
Specific requirement: {control_description}{sub_requirements_block}

Policy passage:
---
{passage_text}
---

Step 1 — State in one phrase what this passage is primarily about:
PASSAGE_ABOUT: [one phrase]

Step 2 — State in one phrase what control {control_id} specifically requires:
CONTROL_REQUIRES: [one phrase]

Step 3 — Does the passage contain DIRECT EVIDENCE that satisfies the control requirement?
Direct evidence means the passage explicitly states it — not implies, not relates to it.
EVIDENCE: YES/NO

Step 4 — If EVIDENCE is NO, is there any partial coverage — passage mentions part of the requirement but not completely?
PARTIAL: YES/NO

ASSIGNMENT:
  CORRECT  — if EVIDENCE = YES
  PARTIAL  — if EVIDENCE = NO but PARTIAL = YES
  WRONG    — if both EVIDENCE = NO and PARTIAL = NO

Format your answer exactly as (no extra lines):
PASSAGE_ABOUT: [phrase]
CONTROL_REQUIRES: [phrase]
EVIDENCE: YES or NO
PARTIAL: YES or NO
ASSIGNMENT: CORRECT or PARTIAL or WRONG
REASON: one sentence"""


def _parse_yn(val: str) -> bool | None:
    v = (val or "").strip().upper()
    if v.startswith("YES"):
        return True
    if v.startswith("NO"):
        return False
    return None


def parse_assignment_verdict(response: str) -> tuple:
    """Parse LLM chain-of-thought response for assignment verification.

    Returns:
        (llm_status, passage_topic, control_req, has_evidence,
         has_partial, judge_assignment, reason)

    llm_status: 'Fully Addressed' | 'Partially Addressed' | 'Not Addressed'
    judge_assignment: 'CORRECT' | 'PARTIAL' | 'WRONG'
    """
    lines = [ln.strip() for ln in (response or "").strip().splitlines()]
    passage_topic   = ""
    control_req     = ""
    has_evidence    = None
    has_partial     = None
    judge_assignment = None
    reason          = ""

    for ln in lines:
        up = ln.upper()
        if up.startswith("PASSAGE_ABOUT:"):
            passage_topic = ln.split(":", 1)[1].strip().strip("[]")
        elif up.startswith("CONTROL_REQUIRES:"):
            control_req = ln.split(":", 1)[1].strip().strip("[]")
        elif up.startswith("EVIDENCE:"):
            has_evidence = _parse_yn(ln.split(":", 1)[1] if ":" in ln else "")
        elif up.startswith("PARTIAL:"):
            has_partial = _parse_yn(ln.split(":", 1)[1] if ":" in ln else "")
        elif up.startswith("ASSIGNMENT:"):
            val = ln.split(":", 1)[1].strip().upper()
            if "CORRECT" in val:
                judge_assignment = "CORRECT"
            elif "PARTIAL" in val:
                judge_assignment = "PARTIAL"
            elif "WRONG" in val:
                judge_assignment = "WRONG"
        elif up.startswith("REASON:"):
            reason = ln.split(":", 1)[1].strip() if ":" in ln else ""

    # Fallback: derive from EVIDENCE/PARTIAL if ASSIGNMENT line missing/unreadable
    if judge_assignment is None:
        if has_evidence is True:
            judge_assignment = "CORRECT"
        elif has_evidence is False and has_partial is True:
            judge_assignment = "PARTIAL"
        else:
            judge_assignment = "WRONG"

    if judge_assignment == "CORRECT":
        llm_status = "Fully Addressed"
    elif judge_assignment == "PARTIAL":
        llm_status = "Partially Addressed"
    else:
        llm_status = "Not Addressed"

    return (llm_status, passage_topic, control_req,
            has_evidence, has_partial, judge_assignment,
            reason or (response or "")[:200])


def _build_sub_requirements_block(info: dict) -> str:
    """Return a formatted sub-requirements block for the prompt (max 3 items).
    Returns an empty string if no sub_controls are available.
    Prefix with a newline so it slots cleanly into the prompt template.
    """
    sub_controls = info.get("sub_controls") or []
    lines = []
    for sc in sub_controls[:3]:
        sc_id   = (sc.get("sub_control_id") or "").strip()
        sc_text = (sc.get("control_statement") or sc.get("text") or "").strip()
        if sc_text:
            lines.append(f"  {sc_id + ': ' if sc_id else ''}{sc_text}")
    if lines:
        return "\nSub-requirements:\n" + "\n".join(lines)
    return ""


def parse_verdict_checklist(response: str, has_checklist: bool) -> tuple:
    """Parse LLM response: (llm_status, q1, q2, q3, reason).
    llm_status is 'Fully Addressed' | 'Partially Addressed' | 'Not Addressed'.
    q1,q2,q3 are bool or None (None when has_checklist is False).
    """
    lines = [ln.strip() for ln in (response or "").strip().splitlines()]
    q1 = q2 = q3 = None
    verdict_raw = None
    reason = ""

    if has_checklist:
        for line in lines:
            if line.upper().startswith("Q1:"):
                q1 = "YES" in line.upper()
            elif line.upper().startswith("Q2:"):
                q2 = "YES" in line.upper()
            elif line.upper().startswith("Q3:"):
                q3 = "YES" in line.upper()
            elif line.upper().startswith("VERDICT:"):
                verdict_raw = line[8:].strip().upper()
            elif line.upper().startswith("REASON:"):
                reason = line[7:].strip()

    if not has_checklist:
        for line in lines:
            if line.upper().startswith("VERDICT:"):
                verdict_raw = line[8:].strip().upper()
            elif line.upper().startswith("REASON:"):
                reason = line[7:].strip()

    # Map VERDICT line to status
    if verdict_raw:
        if "FULL" in verdict_raw:
            status = "Fully Addressed"
        elif "PARTIAL" in verdict_raw:
            status = "Partially Addressed"
        elif "NONE" in verdict_raw:
            status = "Not Addressed"
        else:
            status = None
    else:
        status = None

    # Fallback from Q answers
    if status is None and has_checklist and (q1 is not None or q2 is not None or q3 is not None):
        yes_count = sum(1 for x in (q1, q2, q3) if x is True)
        if yes_count == 3:
            status = "Fully Addressed"
        elif yes_count >= 1:
            status = "Partially Addressed"
        else:
            status = "Not Addressed"

    if status is None:
        status = "Not Addressed"

    return status, q1, q2, q3, reason or response[:200] if response else ""


def build_control_lookup(controls_path: str) -> dict:
    """Build control_id -> {name, description} for score-routed judge.
    Supports: (1) nested control.id/control.name/control.description/implementation_guidelines,
              (2) flat control_id/control_name/control_statement.
    """
    path = Path(controls_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raw = [raw] if isinstance(raw, dict) else []
    index = {}
    for item in raw:
        if "control" in item:
            c = item["control"]
            cid = (c.get("id") or "").strip()
            name = (c.get("name") or cid or "").strip()
            raw_desc = (c.get("description") or "").strip()
            # The clean JSON sometimes concatenates the description with itself.
            # Deduplicate by checking if first half == second half.
            if raw_desc:
                half = len(raw_desc) // 2
                if raw_desc[:half].strip() == raw_desc[half:].strip():
                    raw_desc = raw_desc[:half].strip()
            desc = raw_desc
            if not desc:
                desc = (c.get("implementation_guidelines") or "").strip()
            if not desc:
                desc = name
            sub_controls = c.get("sub_controls") or item.get("sub_controls") or []
        else:
            cid = (item.get("control_id") or item.get("control_number") or "").strip()
            name = (item.get("control_name") or cid or "").strip()
            desc = (item.get("control_statement") or item.get("control_description") or "").strip()
            if not desc:
                desc = name
        sub_controls = item.get("sub_controls") or []
        if cid:
            index[cid] = {
                "name": name or cid,
                "description": desc or "No description available",
                "sub_controls": sub_controls,
            }
    return index


def ollama_resolve_model(host: str = "http://localhost:11434", timeout: int = 10) -> str:
    """Resolve best available Ollama model: llama3.1:8b → llama3.2:3b → llama3.2 → llama3.2:1b."""
    import requests
    candidates = ["llama3.1:8b", "llama3.2:3b", "llama3.2", "llama3.2:1b"]
    for model in candidates:
        try:
            r = requests.post(f"{host}/api/generate", json={"model": model, "prompt": "x", "stream": False}, timeout=timeout)
            if r.status_code == 200:
                return model
        except Exception:
            continue
    return candidates[-1]


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


# ── Fine-tuned HF model client ────────────────────────────────────────────────

_hf_model_cache: dict = {}   # path → (model, tokenizer, is_single_word)

# System prompt matching finetune_llm_compliance.py training format
_FT_SYSTEM = (
    "You are a UAE IA compliance classifier.\n"
    "Classify if the policy passage addresses the control.\n"
    "Answer with exactly one word: FA, PA, or NA.\n"
    "FA = Fully Addressed\n"
    "PA = Partially Addressed\n"
    "NA = Not Addressed"
)

_FT_LABEL_TO_STATUS = {
    "FA": "Fully Addressed",
    "PA": "Partially Addressed",
    "NA": "Not Addressed",
}


def _is_single_word_model(model_path: str) -> bool:
    """Detect if the model was trained with finetune_llm_compliance.py (FA/PA/NA format)."""
    flag = Path(model_path) / "judge_format.json"
    if flag.exists():
        try:
            info = json.loads(flag.read_text())
            return info.get("format") == "single_word"
        except Exception:
            pass
    return False


def _load_hf_model(model_path: str):
    """Load and cache a HuggingFace model + tokenizer."""
    if model_path in _hf_model_cache:
        return _hf_model_cache[model_path]
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        raise RuntimeError(
            "transformers and torch are required for --use-finetuned.\n"
            "Install with: pip install transformers torch"
        )
    print(f"  Loading fine-tuned model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )
    model.eval()
    is_sw = _is_single_word_model(model_path)
    _hf_model_cache[model_path] = (model, tokenizer, is_sw)
    print(f"  Fine-tuned model loaded  (format={'single_word FA/PA/NA' if is_sw else 'legacy'})")
    return _hf_model_cache[model_path]


def call_finetuned(prompt: str, system: str, model_path: str,
                   max_new_tokens: int = 200) -> str:
    """Run inference using a locally fine-tuned HuggingFace model.

    Automatically detects whether the model was trained in single-word mode
    (FA/PA/NA — from finetune_llm_compliance.py) or legacy mode (full verdict).
    """
    import torch
    model, tokenizer, is_single_word = _load_hf_model(model_path)

    if is_single_word:
        # Use the training-time system prompt + compact user message
        messages = [
            {"role": "system", "content": _FT_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        max_new_tokens = 3   # FA/PA/NA + eot = 2-3 tokens
    elif hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
    else:
        # Alpaca fallback for legacy non-chat models
        text = f"### Instruction:\n{system}\n\n### Input:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_finetuned_single_word(response: str) -> tuple[str, str]:
    """Parse FA/PA/NA single-word response from fine-tuned model.

    Maps to full status labels for compatibility with the rest of the pipeline.
    Returns (status_label, reason).
    """
    token = response.strip().upper().split()[0] if response.strip() else "NA"
    token = token.rstrip(".,;:")   # strip trailing punctuation
    if token not in _FT_LABEL_TO_STATUS:
        token = "NA"               # safe default
    return _FT_LABEL_TO_STATUS[token], f"classifier: {token}"


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


# ── Score-routed judge entry point ────────────────────────────────────────────

OLLAMA_TIMEOUT_PER_CALL = 60


def run_score_routed_judge(args) -> None:
    """Score-routed LLM judge: fast_check / full_judge / strict_judge by entailment_score."""
    from datetime import datetime, timezone

    # Resolve model (auto-detect best available when default or 'auto')
    model_used = (args.model or "").strip()
    if model_used == "auto" or not model_used or model_used.startswith("llama3"):
        try:
            model_used = ollama_resolve_model(host=args.host, timeout=10)
            print(f"Model (auto)   : {model_used}")
        except Exception:
            model_used = "llama3.2:1b"
            print(f"Model (fallback): {model_used}")
    else:
        print(f"Model          : {model_used}")

    # Load mappings (from --mappings-dir or --mappings)
    mappings_dir = getattr(args, "mappings_dir", None)
    if mappings_dir:
        dir_path = Path(mappings_dir)
        json_files = sorted(dir_path.glob("*.json"))
        if not json_files:
            print(f"ERROR: No .json files found in {mappings_dir}")
            sys.exit(1)
        print(f"Loading mappings from directory {mappings_dir} ({len(json_files)} files)...")
        mappings = []
        for jf in json_files:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                mappings.extend(data)
            elif isinstance(data, dict) and "mappings" in data:
                mappings.extend(data["mappings"])
            print(f"  {jf.name}: loaded {len(data) if isinstance(data, list) else len(data.get('mappings', []))} entries")
        print(f"  Total combined: {len(mappings)} mappings")
    else:
        print(f"Loading mappings from {args.mappings} ...")
        with open(args.mappings, encoding="utf-8") as f:
            mappings = json.load(f)
        print(f"  {len(mappings)} mappings loaded")

    to_judge = [m for m in mappings if m.get("status", "") in ("Fully Addressed", "Partially Addressed")]
    if args.limit:
        to_judge = to_judge[: args.limit]
        print(f"  Limiting to first {args.limit} mappings (--limit)")
    print(f"  {len(to_judge)} mappings to judge")

    # Score stats (printed for reference only — not used for routing)
    all_scores = [m.get("entailment_score") for m in to_judge if m.get("entailment_score") is not None]
    if all_scores:
        import statistics
        print(f"\nScore stats    : min={min(all_scores):.3f}  max={max(all_scores):.3f}  "
              f"mean={statistics.mean(all_scores):.3f}  median={statistics.median(all_scores):.3f}")
        print("  (scores kept for reference only — routing removed, every pair uses same prompt)")
    print()

    # Control lookup (name + sub_controls/description for prompt requirements)
    print(f"Loading controls from {args.controls} ...")
    control_lookup = build_control_lookup(args.controls)
    # Fallback: merge granular control data from uae_ia_controls_clean.json
    root = Path(__file__).resolve().parent.parent
    fallback_controls = root / "data/02_processed/uae_ia_controls_clean.json"
    if fallback_controls.exists() and str(fallback_controls.resolve()) != str(Path(args.controls).resolve()):
        clean = build_control_lookup(str(fallback_controls))
        for cid, info in clean.items():
            control_lookup.setdefault(cid, info)
        if clean:
            print(f"  Merged fallback controls from {fallback_controls.name} ({len(clean)} entries)")
    print(f"  {len(control_lookup)} controls indexed")

    # Verify Ollama (skip when dry-run)
    if not args.dry_run:
        try:
            call_ollama("Reply OK", model=model_used, system="You are helpful.",
                        host=args.host, timeout=OLLAMA_TIMEOUT_PER_CALL, num_predict=10)
        except RuntimeError:
            print("\nERROR: Ollama not running. Start with: ollama serve")
            sys.exit(1)
    else:
        print("Dry-run: printing prompts only; no Ollama calls, no output file written.\n")

    # Auto-derive output path: place judged file next to the input when not explicitly overridden
    default_output = "data/06_compliance_mappings/mappings_llm_judged.json"
    if args.output == default_output and not getattr(args, "mappings_dir", None):
        input_path = Path(args.mappings)
        args.output = str(input_path.parent / (input_path.stem + "_llm_judged.json"))
    print(f"Output         : {args.output}\n")

    assignment_counts = {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0}
    results = []
    partial_path = Path(args.output).parent / "mappings_llm_judged_partial.json"
    t0 = time.time()

    for i, m in enumerate(to_judge):
        ctrl_id = m.get("source_control_id", "")
        score   = m.get("entailment_score", 0.0)

        info = control_lookup.get(ctrl_id, {})
        control_name        = info.get("name") or ctrl_id
        control_description = (info.get("description") or control_name)[:600]
        sub_block           = _build_sub_requirements_block(info)
        passage_text        = (m.get("evidence_text") or "")[:1200]

        prompt = USER_ASSIGNMENT.format(
            control_id=ctrl_id,
            control_name=control_name,
            control_description=control_description,
            sub_requirements_block=sub_block,
            passage_text=passage_text,
        )

        if args.dry_run:
            print(f"\n--- Mapping {i+1}  control={ctrl_id}  score={score} ---")
            print(prompt)
            (llm_status, passage_topic, control_req,
             has_evidence, has_partial, judge_assignment, llm_reason) = (
                "Not Addressed", "", "", None, None, "WRONG", "(dry-run)"
            )
        else:
            try:
                response = call_ollama(
                    prompt, model=model_used, system=SYSTEM_ASSIGNMENT,
                    host=args.host, timeout=OLLAMA_TIMEOUT_PER_CALL, num_predict=250,
                )
                (llm_status, passage_topic, control_req,
                 has_evidence, has_partial, judge_assignment, llm_reason) = \
                    parse_assignment_verdict(response)
                if getattr(args, "verbose", False):
                    print(f"  [verbose] {ctrl_id} → {judge_assignment}  "
                          f"evidence={has_evidence}  partial={has_partial}  "
                          f"| {response[:200]}", flush=True)
            except Exception as e:
                if "timeout" in str(e).lower() or "Timeout" in str(type(e).__name__):
                    print(f"  Warning: timeout for {ctrl_id} -> "
                          f"{m.get('target_policy_id', '')[:50]}", flush=True)
                (llm_status, passage_topic, control_req,
                 has_evidence, has_partial, judge_assignment, llm_reason) = (
                    "Not Addressed", "", "", None, None, "WRONG", f"Error: {e}"
                )

        assignment_counts[judge_assignment or "WRONG"] = \
            assignment_counts.get(judge_assignment or "WRONG", 0) + 1

        # CORRECT and PARTIAL are kept; WRONG is discarded
        kept = llm_status in ("Fully Addressed", "Partially Addressed")

        out = {
            "mapping_id":         m.get("mapping_id", ""),
            "source_control_id":  ctrl_id,
            "target_policy_id":   m.get("target_policy_id", ""),
            "original_status":    m.get("status", ""),
            "llm_status":         llm_status,
            "final_status":       llm_status,
            "entailment_score":   score,
            "passage_topic":      passage_topic,
            "control_requirement": control_req,
            "has_evidence":       has_evidence,
            "has_partial":        has_partial,
            "judge_assignment":   judge_assignment,
            "llm_reason":         llm_reason,
            "evidence_text":      m.get("evidence_text", ""),
            "kept":               kept,
            "mapping_date":       datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_used":         model_used,
        }
        results.append(out)

        if (i + 1) % 10 == 0 and not args.dry_run:
            Path(partial_path).parent.mkdir(parents=True, exist_ok=True)
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_judge) - i - 1) / rate if rate > 0 else 0
            kept_so_far = sum(1 for r in results if r["kept"])
            print(f"  [{i+1}/{len(to_judge)}]  kept={kept_so_far}  "
                  f"CORRECT={assignment_counts['CORRECT']}  "
                  f"PARTIAL={assignment_counts['PARTIAL']}  "
                  f"WRONG={assignment_counts['WRONG']}  "
                  f"ETA={eta/60:.1f}min", flush=True)

    if not args.dry_run:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        print(f"\nDry-run complete: {len(results)} prompts shown (no output file written).")

    kept_total = sum(1 for r in results if r["kept"])
    correct_count  = assignment_counts.get("CORRECT", 0)
    partial_count  = assignment_counts.get("PARTIAL", 0)
    wrong_count    = assignment_counts.get("WRONG",   0)

    print(f"\n{'='*60}")
    print("LLM Assignment Judge complete")
    print(f"  Total pairs evaluated               : {len(to_judge)}")
    print(f"  CORRECT  (Fully Addressed)   → kept : {correct_count}")
    print(f"  PARTIAL  (Partially Addressed) → kept: {partial_count}")
    print(f"  WRONG    (Not Addressed)  → discard : {wrong_count}")
    print(f"\n  Final kept mappings                 : {kept_total}")
    print(f"  Saved to                            : {args.output}")

    if args.evaluate:
        print("\nRunning evaluation against golden set ...")
        try:
            import subprocess
            result = subprocess.run(
                ["python3", "scripts/evaluate_pipeline.py", "--mappings", args.output],
                capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
            )
            print(result.stdout or result.stderr)
        except FileNotFoundError:
            print("  evaluate_pipeline.py not found — run evaluation manually")


# ── Negative validation ───────────────────────────────────────────────────────

def run_negative_validation(args) -> None:
    """Judge confusable NA pairs and report how many are correctly marked WRONG.

    A confusable NA pair is a (control, passage) pair that an NLP system
    plausibly but wrongly matched.  The judge should mark all of them as WRONG.
    If it marks them CORRECT the judge cannot distinguish confusable controls.
    """
    from datetime import datetime, timezone

    na_path = args.inject_confusable_na
    print(f"\n{'='*60}")
    print("Negative Validation — Confusable NA Pairs")
    print(f"{'='*60}")
    print(f"Loading confusable NA pairs from {na_path} ...")

    with open(na_path, encoding="utf-8") as f:
        na_pairs = json.load(f)

    # Build model name (re-resolve)
    model_used = (args.model or "").strip()
    if model_used == "auto" or not model_used or model_used.startswith("llama3"):
        try:
            model_used = ollama_resolve_model(host=args.host, timeout=10)
        except Exception:
            model_used = "llama3.2:1b"

    # Control lookup
    control_lookup = build_control_lookup(args.controls)
    root = Path(__file__).resolve().parent.parent
    fallback = root / "data/02_processed/uae_ia_controls_clean.json"
    if fallback.exists():
        clean = build_control_lookup(str(fallback))
        for cid, info in clean.items():
            control_lookup.setdefault(cid, info)

    # Apply limit if set
    to_validate = na_pairs
    if args.limit:
        to_validate = to_validate[:args.limit]
    print(f"  {len(to_validate)} confusable NA pairs to validate")

    correct_count = wrong_count = error_count = 0
    false_positives = []  # pairs the judge incorrectly accepted

    for i, m in enumerate(to_validate):
        ctrl_id = (m.get("corrected_control_id") or m.get("control_id") or "").strip()
        passage_text = (m.get("policy_text_snippet") or "")[:1200]

        info = control_lookup.get(ctrl_id, {})
        control_name        = info.get("name") or ctrl_id
        control_description = (info.get("description") or control_name)[:600]
        sub_block           = _build_sub_requirements_block(info)

        prompt = USER_ASSIGNMENT.format(
            control_id=ctrl_id,
            control_name=control_name,
            control_description=control_description,
            sub_requirements_block=sub_block,
            passage_text=passage_text,
        )

        if args.dry_run:
            print(f"\n[NA-{i+1}] control={ctrl_id}  why_confused={m.get('why_confused','')[:60]}")
            print(prompt[:600])
            continue

        try:
            response = call_ollama(
                prompt, model=model_used, system=SYSTEM_ASSIGNMENT,
                host=args.host, timeout=OLLAMA_TIMEOUT_PER_CALL, num_predict=250,
            )
            (llm_status, passage_topic, control_req,
             has_evidence, has_partial, judge_assignment, llm_reason) = \
                parse_assignment_verdict(response)
        except Exception as e:
            error_count += 1
            continue

        if judge_assignment == "WRONG":
            wrong_count += 1
        else:
            correct_count += 1
            false_positives.append({
                "control_id":   ctrl_id,
                "passage_id":   m.get("policy_passage_id", ""),
                "why_confused": m.get("why_confused", ""),
                "why_wrong":    m.get("why_wrong", ""),
                "judge_said":   judge_assignment,
                "judge_reason": llm_reason,
            })

        if getattr(args, "verbose", False):
            marker = "✓ WRONG" if judge_assignment == "WRONG" else "✗ KEPT"
            print(f"  [{i+1}] {ctrl_id} → {judge_assignment}  {marker}  "
                  f"evidence={has_evidence}  | {llm_reason[:80]}", flush=True)

    if args.dry_run:
        return

    total = wrong_count + correct_count + error_count
    na_accuracy = wrong_count / total * 100 if total > 0 else 0

    # Load kept mappings from main judge run (real pairs)
    real_judged_path = args.output
    real_correct = real_wrong = 0
    if Path(real_judged_path).exists():
        with open(real_judged_path, encoding="utf-8") as f:
            real_judged = json.load(f)
        real_correct = sum(1 for r in real_judged if r.get("kept"))
        real_wrong   = sum(1 for r in real_judged if not r.get("kept"))
        real_total   = len(real_judged)
        real_accuracy = real_correct / real_total * 100 if real_total > 0 else 0
    else:
        real_total = real_accuracy = 0

    print(f"\n── Validation Results ───────────────────────────────────────")
    print(f"  Confusable NAs correctly marked WRONG : {wrong_count}/{total}  ({na_accuracy:.1f}%)")
    print(f"  Confusable NAs incorrectly kept        : {correct_count}/{total}")
    if error_count:
        print(f"  Errors (timeout/parse)                 : {error_count}")
    if real_total:
        print(f"\n  Real pairs correctly kept (CORRECT/PARTIAL): {real_correct}/{real_total}  ({real_accuracy:.1f}%)")
        overall = (wrong_count + real_correct) / (total + real_total) * 100
        print(f"  Overall judge accuracy                : {overall:.1f}%")

    if false_positives:
        print(f"\n  False positives (judge cannot distinguish):")
        for fp in false_positives[:10]:
            print(f"    {fp['control_id']}  |  confused_by: {fp['why_confused'][:60]}")
            print(f"      judge said {fp['judge_said']}: {fp['judge_reason'][:80]}")

    # Save validation report
    val_report_path = Path(args.output).parent / "judge_validation_report.json"
    report = {
        "confusable_na_total":   total,
        "correctly_wrong":       wrong_count,
        "incorrectly_kept":      correct_count,
        "errors":                error_count,
        "na_accuracy_pct":       round(na_accuracy, 2),
        "real_pairs_total":      real_total,
        "real_correctly_kept":   real_correct,
        "real_accuracy_pct":     round(real_accuracy, 2) if real_total else None,
        "overall_accuracy_pct":  round((wrong_count + real_correct) / (total + real_total) * 100, 2)
                                  if (total + real_total) > 0 else None,
        "false_positives":       false_positives,
        "model_used":            model_used,
        "na_source":             na_path,
        "generated_at":          datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Validation report → {val_report_path}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge for compliance mappings")
    parser.add_argument("--mappings",  default="data/06_compliance_mappings/mappings.json")
    parser.add_argument("--controls",  default="data/04_label_studio/imports/uae_ia_controls_raw.json")
    parser.add_argument("--output",    default="data/06_compliance_mappings/mappings_llm_judged.json")
    parser.add_argument("--model",        default="llama3.2:1b",
                        help="Ollama model name (default: llama3.2:1b). "
                             "Alternatives: mistral, llama3.1:8b, llama3.2. "
                             "Ignored when --use-finetuned is set.")
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
    # ── Fine-tuned model options ───────────────────────────────────────────────
    parser.add_argument("--use-finetuned", action="store_true",
                        help="Use locally fine-tuned HuggingFace model instead of Ollama. "
                             "Requires --finetuned-model to point to the saved model directory "
                             "(produced by scripts/finetune_llm_compliance.py).")
    parser.add_argument("--finetuned-model", default="models/compliance-llm-judge",
                        help="Path to the fine-tuned HF model directory "
                             "(default: models/compliance-llm-judge).")
    parser.add_argument("--mappings-dir",  default=None,
                        help="Load all *.json files from a directory as input mappings. "
                             "Overrides --mappings when specified. "
                             "Example: --mappings-dir data/06_compliance_mappings/by_policy/")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Only judge first N mappings (for testing)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print prompts without calling Ollama (for debugging)")
    parser.add_argument("--verbose",      action="store_true",
                        help="Log raw LLM responses when parsing fails")
    parser.add_argument("--keep-not-addressed", action="store_true",
                        help="Also include 'Not Addressed' LLM verdicts in output "
                             "(default: drop them)")
    parser.add_argument("--evaluate",     action="store_true",
                        help="Run evaluation against golden set after judging")
    parser.add_argument("--inject-confusable-na", default=None,
                        help="Path to na_confusable_pairs.json (from generate_na_controls.py). "
                             "When set, injects these pairs as expected-WRONG pairs for validation. "
                             "Use with --validate-negatives.")
    parser.add_argument("--validate-negatives", action="store_true",
                        help="After judging real mappings, run judge on confusable NA pairs "
                             "and report how many are correctly marked WRONG. "
                             "Requires --inject-confusable-na.")
    args = parser.parse_args()

    # ── Validation mode: judge real pairs + confusable NA pairs ───────────────
    if not args.use_finetuned:
        run_score_routed_judge(args)
        if getattr(args, "validate_negatives", False) and getattr(args, "inject_confusable_na", None):
            run_negative_validation(args)
        return

    system_prompt, user_template = PROMPT_STYLES[args.prompt_style]
    # CoT needs more tokens to reason; strict/fewshot only need the label line
    num_predict = 300 if args.prompt_style == "cot" else 100

    # ── Routing: fine-tuned HF model vs Ollama ────────────────────────────────
    use_finetuned = args.use_finetuned
    if use_finetuned:
        ft_path = str(Path(args.finetuned_model).resolve())
        if not Path(ft_path).exists():
            print(f"ERROR: Fine-tuned model not found at {ft_path}")
            print("  Run scripts/finetune_llm_compliance.py first, or check the path.")
            sys.exit(1)
        model_label = f"fine-tuned:{ft_path}"
        print(f"Mode         : fine-tuned HF model")
        print(f"Model path   : {ft_path}")
    else:
        model_label = args.model
        print(f"Mode         : Ollama ({args.model})")

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

    # ── Verify backend is available ───────────────────────────────────────────
    if use_finetuned:
        print(f"\nLoading fine-tuned model (first call may take ~30 sec on CPU) ...")
        try:
            test = call_finetuned("Reply with OK", "You are helpful.", ft_path,
                                  max_new_tokens=10)
            print(f"  Fine-tuned model OK — response: {test[:40]!r}")
        except Exception as e:
            print(f"\nERROR loading fine-tuned model: {e}")
            sys.exit(1)
    else:
        print(f"\nVerifying Ollama connection (model: {args.model}) ...")
        try:
            test = call_ollama("Reply with OK", model=args.model,
                               system="You are a helpful assistant.",
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
            if use_finetuned:
                # For single-word models, build a compact user message matching
                # the training-time format (control_id + query[:200] + passage[:300])
                if _is_single_word_model(ft_path):
                    ft_prompt = (
                        f"Control {ctrl_id}: {ctrl_text[:200]}\n\n"
                        f"Passage: {passage[:300]}"
                    )
                else:
                    ft_prompt = prompt
                response = call_finetuned(ft_prompt, system_prompt, ft_path,
                                          max_new_tokens=num_predict)
                if _is_single_word_model(ft_path):
                    label, reason = parse_finetuned_single_word(response)
                else:
                    label, reason = parse_llm_verdict(response)
            else:
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
        judged["llm_model"]         = model_label
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
