#!/usr/bin/env python3
"""
Build the shared per-atom dataset + pair manifest for the local judges.

From the corrected per-atom labels (data/10_judge/answer_key.judged.json, where
ai_covered_atoms are the teacher labels), produce:

  data/12_atoms/atoms_train.jsonl / atoms_test.jsonl
      one row per (passage, atom): {premise, hypothesis, covered 0/1, ...}
      split by PAIR (no leakage), stratified-ish by covered.
      -> NLI threshold calibration + unsloth per-atom fine-tuning.

  data/12_atoms/pairs_eval.json
      all 162 pairs with their control atoms + passage + the v2 (corrected)
      FA/PA/NA label -> pair-level evaluation of either judge.

Atom hypothesis = the atom obligation restated as a declarative claim, so an NLI
model can score entailment(passage -> claim).
"""

import glob
import hashlib
import html
import json
import re
from pathlib import Path

FA, PA, NA = "Fully Addressed", "Partially Addressed", "Not Addressed"


def clean(t): return html.unescape(t or "").strip()


def atom_key(sub, i):
    head = sub.split(":", 1)[0].strip(); last = head.split(".")[-1].strip()
    return last if (last and len(last) <= 3 and not last.isdigit()) else str(i + 1)


def atom_claim(body):
    """Restate an atom obligation as a declarative hypothesis for NLI."""
    b = body.strip().rstrip(":").strip()
    b = re.sub(r"^[Tt]he entity shall\s*", "", b)
    b = re.sub(r"^[Tt]he .*? shall\s*", "", b)
    b = re.sub(r"^shall\s*", "", b)
    if not b:
        return ""
    b = b[0].lower() + b[1:]
    return f"The entity {b}."


def build_control_index(structured, raw):
    idx = {}
    for c in json.load(open(structured, encoding="utf-8")):
        ctl = c["control"]
        idx[ctl["id"]] = {"desc": clean(ctl.get("description", "")), "subs": ctl.get("sub_controls") or []}
    for c in json.load(open(raw, encoding="utf-8")):
        num = c.get("control_number") or re.sub(r"^UAE_IA_CTRL_", "", c.get("control_id", ""))
        if num and num not in idx:
            idx[num] = {"desc": clean(c.get("control_statement", "")), "subs": c.get("sub_controls") or []}
    return idx


def build_passages(glob_pat):
    idx = {}
    for fp in glob.glob(glob_pat):
        try:
            for p in json.load(open(fp, encoding="utf-8")):
                if isinstance(p, dict) and p.get("id"):
                    idx[p["id"]] = clean(p.get("text", ""))
        except Exception:
            pass
    return idx


def atoms_of(ctl):
    """Return [(key, claim, raw_body)] for real atoms (skip pure stems)."""
    out = []
    for i, s in enumerate(ctl["subs"]):
        k = atom_key(s, i)
        body = s.split(":", 1)[1].strip() if ":" in s else s
        claim = atom_claim(body)
        if claim and len(body.strip().rstrip(":")) > 3:   # skip "The entity shall:" stems
            out.append((k, claim, body))
    return out


def in_split(pair_id, frac_test=0.3):
    h = int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 100
    return "test" if h < frac_test * 100 else "train"


def main():
    judged = json.load(open("data/10_judge/answer_key.judged.json", encoding="utf-8"))["results"]
    v2 = {(p["control_id"], p["passage_id"]): p["label"]
          for p in json.load(open("data/07_golden_mapping/answer_key_v2.json", encoding="utf-8"))["pairs"]}
    cidx = build_control_index("data/02_processed/uae_ia_controls_structured.json",
                               "data/04_label_studio/imports/uae_ia_controls_raw.json")
    pidx = build_passages("data/02_processed/policies/*.json")

    atom_rows, pairs = [], []
    for r in judged:
        cid, pid = r["control_id"], r["passage_id"]
        ctl, passage = cidx.get(cid), pidx.get(pid)
        if not ctl or not passage:
            continue
        atoms = atoms_of(ctl)
        covered = set(r.get("ai_covered_atoms") or [])
        pair_id = f"{cid}__{pid}"
        split = in_split(pair_id)
        pairs.append({
            "pair_id": pair_id, "control_id": cid, "passage_id": pid,
            "obligation": ctl["desc"], "passage": passage[:2200],
            "atoms": [{"key": k, "claim": c} for k, c, _ in atoms],
            "v2_label": v2.get((cid, pid), r.get("human_label")),
            "split": split,
        })
        for k, claim, _ in atoms:
            atom_rows.append({
                "pair_id": pair_id, "control_id": cid, "passage_id": pid, "atom_key": k,
                "premise": passage[:2000], "hypothesis": claim,
                "covered": 1 if k in covered else 0, "split": split,
            })

    out = Path("data/12_atoms"); out.mkdir(parents=True, exist_ok=True)
    tr = [a for a in atom_rows if a["split"] == "train"]
    te = [a for a in atom_rows if a["split"] == "test"]
    with open(out / "atoms_train.jsonl", "w", encoding="utf-8") as f:
        for a in tr: f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(out / "atoms_test.jsonl", "w", encoding="utf-8") as f:
        for a in te: f.write(json.dumps(a, ensure_ascii=False) + "\n")
    json.dump(pairs, open(out / "pairs_eval.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    from collections import Counter
    pos = sum(a["covered"] for a in atom_rows)
    print(f"pairs: {len(pairs)}  (atomless: {sum(1 for p in pairs if not p['atoms'])})")
    print(f"atom instances: {len(atom_rows)}  (covered={pos}, not={len(atom_rows)-pos})")
    print(f"  train atoms: {len(tr)}   test atoms: {len(te)}")
    print(f"  pair split : train {sum(1 for p in pairs if p['split']=='train')} / test {sum(1 for p in pairs if p['split']=='test')}")
    print(f"  v2 pair labels: {Counter(p['v2_label'] for p in pairs)}")
    print(f"-> {out}/atoms_train.jsonl, atoms_test.jsonl, pairs_eval.json")


if __name__ == "__main__":
    main()
