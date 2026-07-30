#!/usr/bin/env python3
"""Dump answer-key pairs with control atoms + passage text, for inline judging."""
import glob, html, json, re, sys

def clean(t): return html.unescape(t or "").strip()

def atom_key(sub, i):
    head = sub.split(":", 1)[0].strip(); last = head.split(".")[-1].strip()
    return last if (last and len(last) <= 3 and not last.isdigit()) else str(i + 1)

# control index (structured + raw fallback)
cidx = {}
for c in json.load(open("data/02_processed/uae_ia_controls_structured.json")):
    ctl = c["control"]; cidx[ctl["id"]] = {"desc": clean(ctl.get("description","")), "subs": ctl.get("sub_controls") or []}
for c in json.load(open("data/04_label_studio/imports/uae_ia_controls_raw.json")):
    num = c.get("control_number") or re.sub(r"^UAE_IA_CTRL_","",c.get("control_id",""))
    if num and num not in cidx:
        cidx[num] = {"desc": clean(c.get("control_statement","")), "subs": c.get("sub_controls") or []}

pidx = {}
for fp in glob.glob("data/02_processed/policies/*.json"):
    try:
        for p in json.load(open(fp)):
            if isinstance(p, dict) and p.get("id"): pidx[p["id"]] = clean(p.get("text",""))
    except Exception: pass

pairs = json.load(open("data/07_golden_mapping/answer_key.json"))["pairs"]
out = []
for pr in pairs:
    c = cidx.get(pr["control_id"]); t = pidx.get(pr["passage_id"])
    if not c or not t: continue
    atoms = []
    for i, s in enumerate(c["subs"]):
        k = atom_key(s, i); body = s.split(":",1)[1].strip() if ":" in s else s
        atoms.append(f"{k}: {body}")
    out.append({
        "control_id": pr["control_id"], "passage_id": pr["passage_id"],
        "human_label": pr["label"],
        "obligation": c["desc"], "atoms": atoms,
        "passage": t[:2200],
    })
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print(f"dumped {len(out)} pairs -> {sys.argv[1]}")
