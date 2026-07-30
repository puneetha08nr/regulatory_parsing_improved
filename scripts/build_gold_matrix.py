#!/usr/bin/env python3
"""
Build a *pooled* annotation CSV for one policy — the honest golden-set matrix.

Coverage strategy: POOLED (see docs/GOLDEN_RUBRIC.md). For a chosen policy we
label the union of:
  1. retrieval  — BM25 top-K controls per passage (recall pool)
  2. prediction — every prior pipeline prediction for the policy (precision pool)
  3. golden     — every existing human annotation for the policy (carried, re-reviewed)
  4. neg_sample — a random sample of *unpooled* cells (estimates residual misses)

Each output row is one (control, passage) pair. The control's atomic obligations
(sub_controls) are surfaced so the annotator marks per-atom coverage; FA/PA/NA is
computed from that at compile time (scripts/compile_gold_matrix.py).

Usage:
    python3 scripts/build_gold_matrix.py \
        --policy   "data/02_processed/policies/Asset Management Policy 6_corrected.json" \
        --controls data/02_processed/uae_ia_controls_structured.json \
        --golden   data/07_golden_mapping/golden_mapping_dataset.json \
        --out      data/09_gold_matrix/asset_management.pairs.csv

    # predictions default to common locations; override / add with --predictions
"""

import argparse
import csv
import glob
import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise SystemExit("rank_bm25 required:  pip install rank_bm25") from e


_TOK = re.compile(r"[a-z0-9]+")


def tok(text: str):
    return _TOK.findall((text or "").lower())


def clean(text: str) -> str:
    """Unescape &lt;CLIENT&gt; etc. and trim trailing whitespace per line."""
    return html.unescape(text or "").strip()


# ── control atoms ────────────────────────────────────────────────────────────

def atom_key(sub: str, idx: int) -> str:
    """'M1.1.1.a: The entity ...' -> 'a'. Fallback to 1-based index."""
    head = sub.split(":", 1)[0].strip()
    last = head.split(".")[-1].strip()
    if last and len(last) <= 3 and not last.isdigit():
        return last
    return str(idx + 1)


def load_controls(path: str):
    controls = json.load(open(path, encoding="utf-8"))
    out = {}
    for c in controls:
        ctl = c["control"]
        cid = ctl["id"]
        subs = ctl.get("sub_controls") or []
        atoms = []
        for i, s in enumerate(subs):
            k = atom_key(s, i)
            body = s.split(":", 1)[1].strip() if ":" in s else s.strip()
            atoms.append((k, body))
        out[cid] = {
            "id": cid,
            "family": c.get("control_family", {}).get("number", ""),
            "name": ctl.get("name", ""),
            "desc": clean(ctl.get("description", "")),
            "atoms": atoms,  # list[(key, text)]
            "bm25_text": " ".join(
                [ctl.get("name", ""), ctl.get("description", "")] + subs
            ),
        }
    return out


# ── pool sources ─────────────────────────────────────────────────────────────

DEFAULT_PRED_GLOBS = [
    "data/06_compliance_mappings/mappings.json",
    "data/06_compliance_mappings/by_policy/*.json",
    "single_policy_e2e/output/mappings.json",
]


def load_predictions(globs, passage_ids):
    """Return set[(control_id, passage_id)] for predictions touching this policy."""
    pairs = set()
    seen_files = 0
    for pat in globs:
        for fp in glob.glob(pat):
            try:
                data = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            seen_files += 1
            for r in data:
                if not isinstance(r, dict):
                    continue
                cid = r.get("source_control_id") or r.get("control_id")
                pid = r.get("target_policy_id") or r.get("policy_passage_id")
                if cid and pid in passage_ids:
                    pairs.add((cid, pid))
    return pairs, seen_files


def load_golden(path, passage_ids):
    """Return {(control_id, passage_id): prior_label} for this policy's passages."""
    prior = {}
    if not path or not Path(path).exists():
        return prior
    for r in json.load(open(path, encoding="utf-8")):
        pid = r.get("policy_passage_id")
        if pid not in passage_ids:
            continue
        cid = r.get("corrected_control_id") or r.get("control_id")
        if cid:
            prior[(cid, pid)] = r.get("compliance_status", "")
    return prior


# ── build ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build pooled golden-matrix CSV for one policy")
    ap.add_argument("--policy", required=True, help="policy JSON (list of passages)")
    ap.add_argument("--controls", default="data/02_processed/uae_ia_controls_structured.json")
    ap.add_argument("--golden", default="data/07_golden_mapping/golden_mapping_dataset.json")
    ap.add_argument("--predictions", nargs="*", default=DEFAULT_PRED_GLOBS,
                    help="glob(s) of prior prediction JSON files")
    ap.add_argument("--top-k", type=int, default=12, help="BM25 controls retrieved per passage")
    ap.add_argument("--neg-sample", type=int, default=50, help="random unpooled cells for FN estimate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    passages = json.load(open(args.policy, encoding="utf-8"))
    if not isinstance(passages, list):
        raise SystemExit("policy file must be a list of passages")
    passage_ids = {p["id"] for p in passages}
    pmap = {p["id"]: p for p in passages}
    policy_name = passages[0].get("metadata", {}).get("policy_name", Path(args.policy).stem)

    controls = load_controls(args.controls)
    cids = list(controls)

    # BM25 over control text; query = passage text -> top-K controls per passage
    bm25 = BM25Okapi([tok(controls[c]["bm25_text"]) for c in cids])

    # pair -> set of source tags
    sources = defaultdict(set)

    # 1. retrieval pool
    for p in passages:
        scores = bm25.get_scores(tok(p["text"]))
        ranked = sorted(range(len(cids)), key=lambda i: scores[i], reverse=True)[: args.top_k]
        for i in ranked:
            sources[(cids[i], p["id"])].add("retrieval")

    # 2. prediction pool
    preds, n_pred_files = load_predictions(args.predictions, passage_ids)
    for pair in preds:
        if pair[0] in controls:
            sources[pair].add("prediction")

    # 3. golden pool
    prior = load_golden(args.golden, passage_ids)
    for pair in prior:
        if pair[0] in controls:
            sources[pair].add("golden")

    # 4. neg_sample — random unpooled cells
    pooled = set(sources)
    universe = [(c, pid) for c in cids for pid in passage_ids]
    unpooled = [pr for pr in universe if pr not in pooled]
    rng.shuffle(unpooled)
    for pair in unpooled[: args.neg_sample]:
        sources[pair].add("neg_sample")

    # order: prediction/golden first (must-label), retrieval, neg_sample last
    def sort_key(pr):
        s = sources[pr]
        rank = 0 if ("prediction" in s or "golden" in s) else (1 if "retrieval" in s else 2)
        return (pr[1], rank, pr[0])  # group by passage

    rows = []
    for (cid, pid) in sorted(sources, key=sort_key):
        ctl = controls[cid]
        pas = pmap[pid]
        atom_keys = ",".join(k for k, _ in ctl["atoms"])
        atoms_text = "\n".join(f"{k}: {t}" for k, t in ctl["atoms"]) or "(no sub-controls — judge on description)"
        rows.append({
            "pair_id": f"{cid}__{pid}",
            "source": "+".join(sorted(sources[(cid, pid)])),
            "control_id": cid,
            "control_family": ctl["family"],
            "control_name": ctl["name"],
            "control_desc": ctl["desc"],
            "n_atoms": len(ctl["atoms"]),
            "atom_keys": atom_keys,
            "atoms_text": atoms_text,
            "passage_id": pid,
            "passage_section": pas.get("section", ""),
            "passage_text": clean(pas.get("text", "")),
            # annotator inputs (blank):
            "covered_atoms": "",
            "evidence": "",
            "label_manual": "",
            "confidence": "",
            "note": "",
            # hint:
            "label_prior": prior.get((cid, pid), ""),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pair_id", "source", "control_id", "control_family", "control_name",
            "control_desc", "n_atoms", "atom_keys", "atoms_text",
            "passage_id", "passage_section", "passage_text",
            "covered_atoms", "evidence", "label_manual", "confidence", "note",
            "label_prior"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # summary
    from collections import Counter
    src_counts = Counter()
    for pr in sources:
        for s in sources[pr]:
            src_counts[s] += 1
    print(f"Policy         : {policy_name}")
    print(f"Passages       : {len(passages)}")
    print(f"Prediction files scanned: {n_pred_files}")
    print(f"Pool pairs     : {len(rows)}")
    for s in ("prediction", "golden", "retrieval", "neg_sample"):
        print(f"  {s:11s}: {src_counts.get(s, 0)}")
    print(f"Prior-labelled : {sum(1 for r in rows if r['label_prior'])}")
    print(f"Wrote          : {out}")
    print("\nNext: fill covered_atoms / evidence per docs/GOLDEN_RUBRIC.md, then")
    print(f"      python3 scripts/compile_gold_matrix.py --csv {out}")


if __name__ == "__main__":
    main()
