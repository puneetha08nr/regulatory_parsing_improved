#!/usr/bin/env python3
"""
Claude compliance judge (Step 3). For each (control, passage) pair, Claude reads
the control's atomic obligations against the passage and returns per-atom
coverage; FA/PA/NA is derived from that, per docs/GOLDEN_RUBRIC.md.

This is the *system* under sparse labels: retrieval + a frontier LLM judge.
The answer key exists to grade it, not to train it.

Usage:
    python3 scripts/llm_judge_anthropic.py \
        --pairs data/07_golden_mapping/answer_key.json \
        --out   data/10_judge/answer_key.judged.json
"""

import argparse
import glob
import html
import json
import re
import sys
import time
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-8"
FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"

SYSTEM = """You are a meticulous UAE Information Assurance (UAE IA) compliance auditor.

You judge whether ONE internal policy passage, ON ITS OWN, satisfies the atomic
obligations of ONE regulatory control. Judge coverage per atom, not by topic.

Rules:
- An atom is COVERED only if the passage states a requirement, mechanism, or
  assignment that would let an auditor tick that obligation off. Same TOPIC is
  NOT coverage. A cross-reference to another document ("see the X Policy") does
  NOT cover the atom.
- You must quote the exact span of the passage that covers each atom. If you
  cannot quote it, the atom is NOT covered.
- Label is mechanical: all atoms covered -> "Fully Addressed"; some but not all
  -> "Partially Addressed"; none -> "Not Addressed". If the control has no
  atoms, judge the single top-level obligation as one atom.
Be conservative: over-claiming compliance is the dangerous error."""

TOOL = {
    "name": "record_judgment",
    "description": "Record the per-atom coverage judgment for this control/passage pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            "covered_atoms": {
                "type": "array", "items": {"type": "string"},
                "description": "Atom keys (e.g. 'a','c') that the passage genuinely satisfies. Empty if none.",
            },
            "evidence": {
                "type": "string",
                "description": "Quoted span(s) from the passage justifying each covered atom. Empty if none covered.",
            },
            "label": {"type": "string", "enum": [FA, PA, NA]},
            "reason": {"type": "string", "description": "One sentence."},
        },
        "required": ["covered_atoms", "evidence", "label", "reason"],
    },
}


def clean(t): return html.unescape(t or "").strip()


def build_control_index(structured_path, raw_path):
    idx = {}
    # richer structured source first
    for c in json.load(open(structured_path, encoding="utf-8")):
        ctl = c["control"]
        idx[ctl["id"]] = {
            "name": ctl.get("name", ""), "desc": clean(ctl.get("description", "")),
            "subs": ctl.get("sub_controls") or [],
        }
    # fill gaps from raw
    for c in json.load(open(raw_path, encoding="utf-8")):
        num = c.get("control_number") or re.sub(r"^UAE_IA_CTRL_", "", c.get("control_id", ""))
        if num and num not in idx:
            idx[num] = {
                "name": c.get("control_name", ""), "desc": clean(c.get("control_statement", "")),
                "subs": c.get("sub_controls") or [],
            }
    return idx


def build_passage_index(policy_glob):
    idx = {}
    for fp in glob.glob(policy_glob):
        try:
            for p in json.load(open(fp, encoding="utf-8")):
                if isinstance(p, dict) and p.get("id"):
                    idx[p["id"]] = clean(p.get("text", ""))
        except Exception:
            continue
    return idx


def atom_key(sub, i):
    head = sub.split(":", 1)[0].strip()
    last = head.split(".")[-1].strip()
    return last if (last and len(last) <= 3 and not last.isdigit()) else str(i + 1)


def format_control(ctl):
    lines = [f"Control obligation: {ctl['desc'] or ctl['name']}"]
    subs = ctl["subs"]
    if subs:
        lines.append("Atoms:")
        for i, s in enumerate(subs):
            k = atom_key(s, i)
            body = s.split(":", 1)[1].strip() if ":" in s else s
            lines.append(f"  {k}: {body}")
    else:
        lines.append("Atoms: (none — judge the single obligation above; use atom key 'a')")
    return "\n".join(lines), (subs or [None])


def derive_label(covered, n_atoms, model_label):
    if n_atoms == 0:
        return model_label
    k = len({c for c in covered})
    return FA if k >= n_atoms else (PA if k > 0 else NA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/07_golden_mapping/answer_key.json")
    ap.add_argument("--controls", default="data/02_processed/uae_ia_controls_structured.json")
    ap.add_argument("--controls-raw", default="data/04_label_studio/imports/uae_ia_controls_raw.json")
    ap.add_argument("--policies", default="data/02_processed/policies/*.json")
    ap.add_argument("--out", default="data/10_judge/answer_key.judged.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    data = json.load(open(args.pairs, encoding="utf-8"))
    pairs = data["pairs"] if isinstance(data, dict) else data
    if args.limit:
        pairs = pairs[: args.limit]

    cidx = build_control_index(args.controls, args.controls_raw)
    pidx = build_passage_index(args.policies)
    client = anthropic.Anthropic()

    results, skipped = [], 0
    t0 = time.time()
    for i, pr in enumerate(pairs, 1):
        cid, pid = pr["control_id"], pr["passage_id"]
        ctl, passage = cidx.get(cid), pidx.get(pid)
        if not ctl or not passage:
            skipped += 1
            continue
        ctl_text, subs = format_control(ctl)
        n_atoms = len([s for s in subs if s is not None])
        user = (f"{ctl_text}\n\nPolicy passage:\n\"\"\"\n{passage[:6000]}\n\"\"\"\n\n"
                "Judge which atoms this passage satisfies and record your judgment.")
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=1024,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=[TOOL], tool_choice={"type": "tool", "name": "record_judgment"},  # type: ignore[arg-type]
                messages=[{"role": "user", "content": user}],
            )
            ju: dict = {}
            for b in msg.content:
                if b.type == "tool_use":
                    ju = dict(b.input)  # type: ignore[arg-type]
                    break
        except Exception as e:
            print(f"  [{i}] API error {cid}/{pid}: {e}", file=sys.stderr)
            skipped += 1
            continue
        covered = [c for c in ju.get("covered_atoms", []) if c]
        label = derive_label(covered, n_atoms, ju.get("label", NA))
        results.append({
            "control_id": cid, "passage_id": pid, "policy_name": pr.get("policy_name", ""),
            "human_label": pr.get("label"),
            "ai_label": label, "ai_covered_atoms": covered,
            "ai_evidence": ju.get("evidence", ""), "ai_reason": ju.get("reason", ""),
            "n_atoms": n_atoms,
        })
        if i % 20 == 0:
            print(f"  judged {i}/{len(pairs)}  ({time.time()-t0:.0f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": MODEL, "n": len(results), "skipped": skipped, "results": results},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nJudged {len(results)} pairs ({skipped} skipped, no text) in {time.time()-t0:.0f}s")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
