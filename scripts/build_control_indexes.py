"""
Build per-family BM25 indexes over UAE IA controls.

This prepares Tier-2/3 retrieval for the domain-routed pipeline:
  - One BM25 index per control family (M1–M6, T1–T9)
  - Each index is built over control.full_text for that family
  - Metadata maps BM25 document index -> control_id/control_name/family/subfamily

Usage:
    python3 scripts/build_control_indexes.py \
        --controls data/02_processed/uae_ia_controls_clean.json \
        --output-dir data/02_processed/indexes

Output:
    data/02_processed/indexes/
      M1_bm25.json          # tokenised corpus & BM25 base data
      M1_meta.json          # control metadata for this family
      ...

At query time, you can:
  - Load the *_bm25.json and *_meta.json for routed families
  - Run BM25 scores for a passage/query only within those controls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:  # pragma: no cover - install guidance
    raise SystemExit(
        "rank_bm25 is required for scripts/build_control_indexes.py\n"
        "Install with:\n\n"
        "   pip install rank-bm25\n"
    ) from e


def load_controls(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list at {path}, got {type(data)}")
    return data


def build_family_corpora(controls: List[Dict]) -> Dict[str, Dict]:
    """
    Group controls by family code (e.g., 'M1', 'T2') and prepare:
      - corpus: list[list[str]] tokenised full_text per control
      - meta:   list[dict] with control_id, name, family, subfamily, index
    """
    families: Dict[str, Dict] = {}
    for item in controls:
        fam = item.get("control_family", {}) or {}
        subfam = item.get("control_subfamily", {}) or {}
        c = item.get("control", {}) or {}

        family_code = str(fam.get("number") or "").strip()
        if not family_code:
            continue

        full_text = c.get("full_text") or ""
        if not full_text.strip():
            continue

        fam_bucket = families.setdefault(
            family_code, {"corpus": [], "meta": []}
        )
        idx = len(fam_bucket["corpus"])
        tokens = full_text.lower().split()
        fam_bucket["corpus"].append(tokens)
        fam_bucket["meta"].append(
            {
                "index": idx,
                "control_id": c.get("id", ""),
                "control_name": c.get("name", ""),
                "family": family_code,
                "subfamily": str(subfam.get("number") or "").strip(),
            }
        )

    return families


def serialize_bm25(corpus: List[List[str]]) -> Dict:
    """
    Build BM25Okapi and return a JSON-serialisable snapshot.
    We store:
      - corpus (token lists)
      - idf, avgdl, doc_len, doc_freqs
    So that a small helper can rehydrate BM25 at query time.
    """
    bm25 = BM25Okapi(corpus)
    # BM25Okapi exposes these attributes; we keep them so we can reconstruct.
    return {
        "corpus": corpus,
        "idf": bm25.idf,
        "avgdl": bm25.avgdl,
        "doc_len": bm25.doc_len,
        "doc_freqs": bm25.doc_freqs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build per-family BM25 indexes over UAE IA controls."
    )
    ap.add_argument(
        "--controls",
        type=str,
        default="data/02_processed/uae_ia_controls_clean.json",
        help="Path to clean controls JSON.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="data/02_processed/indexes",
        help="Directory to write per-family BM25 + metadata.",
    )
    args = ap.parse_args()

    controls_path = Path(args.controls)
    output_dir = Path(args.output_dir)

    if not controls_path.exists():
        raise SystemExit(f"Controls file not found: {controls_path}")

    print(f"Loading controls from {controls_path} ...")
    controls = load_controls(controls_path)
    print(f"  Controls: {len(controls)}")

    print("Grouping by family and preparing corpora ...")
    families = build_family_corpora(controls)
    print(f"  Families found: {', '.join(sorted(families.keys()))}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for family_code, data in sorted(families.items()):
        corpus = data["corpus"]
        meta = data["meta"]
        if not corpus:
            continue
        print(f"  Building BM25 for {family_code}: {len(corpus)} controls")
        bm25_data = serialize_bm25(corpus)

        bm25_path = output_dir / f"{family_code}_bm25.json"
        meta_path = output_dir / f"{family_code}_meta.json"

        with bm25_path.open("w", encoding="utf-8") as f:
            json.dump(bm25_data, f, ensure_ascii=False)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Done. Indexes written under {output_dir}")


if __name__ == "__main__":
    main()

