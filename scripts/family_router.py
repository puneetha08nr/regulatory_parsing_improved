"""
Family router for UAE IA policy documents.

Given a policy document (title + headings + text), this script returns
the top 2–3 control families (M1–M6, T1–T9) it most likely belongs to.

Routing strategy (Tier 1 of the pipeline):
1. Fast title-based match against policy_document_types from domain_taxonomy.json.
2. Fallback to embedding similarity between:
   - policy text (title + headings + first N words)
   - each family's description / keywords bundle.

Usage (example):
    python3 scripts/family_router.py \
        --policies data/02_processed/policies/all_policies_for_mapping.json \
        --taxonomy data/02_processed/domain_taxonomy.json \
        --output data/02_processed/family_routing.jsonl

Input format (policies file):
    [
      {
        "id": "access-control-policy",
        "title": "Access Control Policy",
        "headings": ["1. Purpose", "2. Scope", "3. User Access Management"],
        "content": "Full plain-text of the policy document ..."
      },
      ...
    ]

Output format (JSONL, one line per document):
    {
      "document_id": "...",
      "title": "...",
      "routed_families": ["T5", "M4"],
      "confidence": [0.87, 0.61],
      "method": "title" | "embedding"
    }

Dependencies:
    pip install sentence-transformers
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover - import-time guidance
    raise SystemExit(
        "sentence-transformers is required for scripts/family_router.py\n"
        "Install with:\n\n"
        "   pip install sentence-transformers\n"
    ) from e


@dataclass
class FamilyInfo:
    code: str
    name: str
    description: str
    keywords: List[str]
    policy_types: List[str]

    @property
    def text_for_embedding(self) -> str:
        parts = [self.name, self.description]
        if self.policy_types:
            parts.append("Policy types: " + "; ".join(self.policy_types))
        if self.keywords:
            parts.append("Keywords: " + ", ".join(self.keywords))
        return "\n".join(parts)


def load_taxonomy(path: Path) -> Dict[str, FamilyInfo]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, FamilyInfo] = {}
    families = data.get("families", {})
    for code, f in families.items():
        out[code] = FamilyInfo(
            code=code,
            name=f.get("name", code),
            description=f.get("description", ""),
            keywords=f.get("keywords", []) or [],
            policy_types=f.get("policy_document_types", []) or [],
        )
    return out


def build_model(model_name: str) -> SentenceTransformer:
    # Small, CPU-friendly model by default
    return SentenceTransformer(model_name)


def embed_families(
    model: SentenceTransformer, families: Dict[str, FamilyInfo]
) -> Tuple[List[str], np.ndarray]:
    codes = sorted(families.keys())
    texts = [families[c].text_for_embedding for c in codes]
    emb = model.encode(texts, normalize_embeddings=True)
    return codes, np.asarray(emb, dtype=np.float32)


def _normalize_text(s: str | None) -> str:
    return (s or "").strip()


def _first_n_words(text: str, n: int = 500) -> str:
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[:n])


def route_by_title(
    title: str, families: Dict[str, FamilyInfo]
) -> Tuple[List[str], List[float]]:
    """Simple, fast rule-based routing using title contains policy type."""
    title_l = title.lower()
    scores: Dict[str, float] = {}
    for code, fam in families.items():
        for ptype in fam.policy_types:
            p = ptype.lower()
            if p and p in title_l:
                # Score by length of match as a weak proxy for specificity
                scores[code] = max(scores.get(code, 0.0), float(len(p)))
    if not scores:
        return [], []
    # Sort by score descending
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    codes = [c for c, _ in items]
    raw_scores = [s for _, s in items]
    max_s = max(raw_scores) or 1.0
    confidences = [s / max_s for s in raw_scores]
    return codes, confidences


def route_by_embedding(
    text: str,
    model: SentenceTransformer,
    family_codes: List[str],
    family_emb: np.ndarray,
    top_k: int = 3,
) -> Tuple[List[str], List[float]]:
    doc_vec = model.encode([text], normalize_embeddings=True)
    doc_vec = np.asarray(doc_vec, dtype=np.float32)[0]  # (d,)
    sims = family_emb @ doc_vec  # cosine similarity because both are normalised
    idx = np.argsort(-sims)[:top_k]
    codes = [family_codes[i] for i in idx]
    scores = [float(sims[i]) for i in idx]
    # Rescale to [0,1] for easier interpretation
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s > 1e-6:
        conf = [(s - min_s) / (max_s - min_s) for s in scores]
    else:
        conf = [0.0 for _ in scores]
    return codes, conf


def route_document(
    doc: Dict,
    families: Dict[str, FamilyInfo],
    model: SentenceTransformer,
    family_codes: List[str],
    family_emb: np.ndarray,
    max_families: int = 3,
) -> Dict:
    title = _normalize_text(doc.get("title") or doc.get("name") or "")
    headings = doc.get("headings") or []
    content = _normalize_text(doc.get("content") or doc.get("text") or "")

    # 1) Title-based routing
    if title:
        t_codes, t_scores = route_by_title(title, families)
        if t_codes:
            return {
                "document_id": doc.get("id") or doc.get("doc_id") or title,
                "title": title,
                "routed_families": t_codes[:max_families],
                "confidence": t_scores[:max_families],
                "method": "title",
            }

    # 2) Embedding-based routing
    heading_text = " ".join(h for h in headings if isinstance(h, str))
    doc_text_parts = [title, heading_text, _first_n_words(content, 500)]
    doc_text = "\n".join([p for p in doc_text_parts if p])
    if not doc_text.strip():
        # Fallback: cannot route without any text
        return {
            "document_id": doc.get("id") or doc.get("doc_id") or "(unknown)",
            "title": title,
            "routed_families": [],
            "confidence": [],
            "method": "none",
        }

    e_codes, e_scores = route_by_embedding(
        doc_text, model, family_codes, family_emb, top_k=max_families
    )
    return {
        "document_id": doc.get("id") or doc.get("doc_id") or title,
        "title": title,
        "routed_families": e_codes,
        "confidence": e_scores,
        "method": "embedding",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Content-based UAE IA family router")
    ap.add_argument(
        "--policies",
        type=str,
        required=True,
        help="Path to JSON file containing a list of policy docs "
        "(id,title,headings,content).",
    )
    ap.add_argument(
        "--taxonomy",
        type=str,
        default="data/02_processed/domain_taxonomy.json",
        help="Path to domain_taxonomy.json.",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path for routing results.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name for embeddings.",
    )
    ap.add_argument(
        "--max-families",
        type=int,
        default=3,
        help="Maximum number of families to return per document.",
    )
    args = ap.parse_args()

    policies_path = Path(args.policies)
    taxonomy_path = Path(args.taxonomy)
    output_path = Path(args.output)

    if not policies_path.exists():
        raise SystemExit(f"Policies file not found: {policies_path}")
    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy file not found: {taxonomy_path}")

    print(f"Loading taxonomy from {taxonomy_path} ...")
    families = load_taxonomy(taxonomy_path)
    print(f"  Families: {', '.join(sorted(families.keys()))}")

    print(f"Loading embedding model: {args.model} ...")
    model = build_model(args.model)
    fam_codes, fam_emb = embed_families(model, families)

    print(f"Loading policies from {policies_path} ...")
    docs = json.loads(policies_path.read_text(encoding="utf-8"))
    if not isinstance(docs, list):
        raise SystemExit("Expected policies file to contain a JSON list of documents.")
    print(f"  Documents: {len(docs)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    routed = 0
    with output_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            res = route_document(
                doc,
                families=families,
                model=model,
                family_codes=fam_codes,
                family_emb=fam_emb,
                max_families=args.max_families,
            )
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            routed += 1

    print(f"Done. Routed {routed} documents → {output_path}")


if __name__ == "__main__":
    main()

