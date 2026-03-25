#!/usr/bin/env python3
"""
ObligationFilter — pre-filter policy passages to keep only obligation-bearing text.

Uses a fine-tuned LegalBERT sequence classifier trained to distinguish
obligation sentences ("shall", "must", "is required to") from non-obligation
text (TOC entries, headers, definitions, background prose).

Model default: models/obligation-classifier-legalbert
Fallback     : rule-based keyword check (no model needed)

Usage (standalone):
  python3 scripts/obligation_filter.py \
      --policy data/02_processed/policies/Asset\ Management\ Policy\ 6_corrected.json \
      --threshold 0.5 \
      --show-samples 5
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# ── Rule-based fallback ───────────────────────────────────────────────────────

_OBLIGATION_RE = re.compile(
    r"\b(shall|must|is required to|are required to|should|will ensure|"
    r"will maintain|will implement|will establish|will develop|will conduct|"
    r"will review|will not|must not|shall not)\b",
    re.IGNORECASE,
)

_NOISE_RE = re.compile(
    r"^(table of contents|contents|appendix|annex|revision history|"
    r"document control|version|date|author|approved by|references?|"
    r"glossary|acronyms?|definitions?)\b",
    re.IGNORECASE,
)

OBLIGATION_KEYWORDS = {
    "shall", "must", "required", "mandatory", "prohibited",
    "responsible", "accountable", "ensure", "maintain", "implement",
}


def rule_based_is_obligation(text: str) -> bool:
    """True if the text likely contains an obligation sentence."""
    text = text.strip()
    if not text or len(text) < 20:
        return False
    if _NOISE_RE.match(text):
        return False
    if _OBLIGATION_RE.search(text):
        return True
    words = set(text.lower().split())
    return bool(words & OBLIGATION_KEYWORDS)


# ── LegalBERT classifier ──────────────────────────────────────────────────────

class ObligationFilter:
    """
    Filter policy passages using a LegalBERT obligation classifier.

    Falls back to rule-based filtering if the model is not available.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
        self._device = device
        self._use_model = False

        if model_path is None:
            model_path = str(ROOT / "models/obligation-classifier-legalbert")

        if Path(model_path).exists():
            try:
                self._load_model(model_path, device)
                self._use_model = True
            except Exception as e:
                print(f"  [ObligationFilter] Could not load model from {model_path}: {e}")
                print("  [ObligationFilter] Falling back to rule-based filtering.")
        else:
            print(f"  [ObligationFilter] Model not found at {model_path}; using rule-based filter.")

    def _load_model(self, model_path: str, device: Optional[str]):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self._device = _device
        print(f"  [ObligationFilter] Loading LegalBERT from {model_path} on {_device} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.to(_device)
        self._model.eval()
        print("  [ObligationFilter] Model loaded.")

    def is_obligation(self, text: str) -> bool:
        """Return True if the text is obligation-bearing."""
        if self._use_model:
            return self._model_score(text) >= self.threshold
        return rule_based_is_obligation(text)

    def _model_score(self, text: str) -> float:
        """Return obligation probability from the LegalBERT classifier."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Label 1 = obligation (assumed; adjust if model uses different label mapping)
            obligation_prob = probs[0][1].item()
        return obligation_prob

    def filter_passages(self, passages: list, text_key: str = "text") -> tuple:
        """
        Filter a list of passage dicts, keeping only obligation-bearing passages.

        Args:
            passages:  list of dicts, each with a text field (default key: "text")
            text_key:  key for the passage text content

        Returns:
            (kept, removed) — two lists of passage dicts
        """
        kept, removed = [], []
        for p in passages:
            text = p.get(text_key) or p.get("content") or p.get("passage") or ""
            if self.is_obligation(text):
                kept.append(p)
            else:
                removed.append(p)
        return kept, removed


# ── CLI for standalone testing ────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy",    required=True, help="Path to policy JSON file")
    ap.add_argument("--model",     default=None,  help="Path to obligation classifier model")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--show-samples", type=int, default=5, metavar="N",
                    help="Show N sample filtered and kept passages")
    ap.add_argument("--rule-only", action="store_true",
                    help="Force rule-based filtering (ignore model)")
    args = ap.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"Policy not found: {policy_path}")
        return

    passages = json.load(open(policy_path, encoding="utf-8"))
    if not isinstance(passages, list):
        passages = [passages]
    print(f"Loaded {len(passages)} passages from {policy_path.name}")

    if args.rule_only:
        filt = ObligationFilter(model_path="__nonexistent__", threshold=args.threshold)
    else:
        filt = ObligationFilter(model_path=args.model, threshold=args.threshold)

    kept, removed = filt.filter_passages(passages)
    pct_kept = 100 * len(kept) / len(passages) if passages else 0
    print(f"\nResults:")
    print(f"  Total passages : {len(passages)}")
    print(f"  Kept (obligation)  : {len(kept)}  ({pct_kept:.1f}%)")
    print(f"  Removed (non-oblig): {len(removed)}  ({100-pct_kept:.1f}%)")

    if removed and args.show_samples > 0:
        print(f"\nSample REMOVED passages (non-obligation, first {args.show_samples}):")
        for p in removed[:args.show_samples]:
            text = p.get("text") or p.get("content") or ""
            print(f"  [{p.get('id','?')}] {text[:120]!r}")

    if kept and args.show_samples > 0:
        print(f"\nSample KEPT passages (obligation, first {args.show_samples}):")
        for p in kept[:args.show_samples]:
            text = p.get("text") or p.get("content") or ""
            print(f"  [{p.get('id','?')}] {text[:120]!r}")


if __name__ == "__main__":
    main()
