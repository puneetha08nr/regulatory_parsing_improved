#!/usr/bin/env python3
"""
Convert reranker training data → instruction-tuning format for Llama 3.2.

Each training row becomes a chat-formatted example:
  system  : compliance analyst persona
  user    : control + passage (the question)
  assistant: label + one-sentence rationale

Usage:
  python3 scripts/prepare_llm_training_data.py \
    --train data/07_golden_mapping/training_data/train.json \
    --dev   data/07_golden_mapping/training_data/dev.json \
    --output data/07_golden_mapping/llm_training_data

Outputs (in output dir):
  train.jsonl  — one JSON object per line (ShareGPT / Alpaca format)
  dev.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYSTEM_PROMPT = (
    "You are a compliance analyst specialising in the UAE Information Assurance (UAE IA) "
    "regulatory framework. Given a regulatory control and a policy passage, determine "
    "whether the passage addresses the control requirement. "
    "Respond with exactly one of: Fully Addressed, Partially Addressed, Not Addressed. "
    "Follow with a single sentence explaining your reasoning."
)

# Short rationale templates — the model will learn to write its own, but we seed
# the training signal with structured rationales derived from the label and score.
RATIONALE_TEMPLATES = {
    "Fully Addressed": [
        "The passage directly implements the obligation stated in the control.",
        "The passage explicitly covers all requirements described in the control.",
        "The policy text satisfies the control's mandate in full.",
    ],
    "Partially Addressed": [
        "The passage covers some but not all sub-requirements of the control.",
        "The policy addresses the general intent but omits specific obligations.",
        "The passage provides partial coverage; additional controls or sections are needed.",
    ],
    "Not Addressed": [
        "The passage focuses on a different topic and does not address this control.",
        "There is no substantive overlap between the passage content and the control requirement.",
        "The policy passage pertains to a different domain than the control.",
    ],
}


def score_to_label(score: float) -> str:
    if score >= 0.9:
        return "Fully Addressed"
    elif score > 0.05:
        return "Partially Addressed"
    else:
        return "Not Addressed"


def build_user_message(query: str, passage: str) -> str:
    return (
        f"Regulatory Control:\n{query.strip()}\n\n"
        f"Policy Passage:\n{passage.strip()}\n\n"
        "Does this policy passage address the regulatory control?"
    )


def build_assistant_message(label: str, mismatch_reason: str | None = None) -> str:
    rationale = random.choice(RATIONALE_TEMPLATES[label])
    if label == "Not Addressed" and mismatch_reason:
        rationale = f"The passage does not address this control: {mismatch_reason.lower()}."
    return f"{label}\n\n{rationale}"


def to_sharegpt(row: dict) -> dict:
    """Convert one training row to ShareGPT conversation format (used by Unsloth/TRL)."""
    score = float(row.get("score", 0.0))
    label = score_to_label(score)
    mismatch_reason = row.get("mismatch_reason")

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": build_user_message(row["query"], row["passage"])},
            {"from": "gpt", "value": build_assistant_message(label, mismatch_reason)},
        ],
        "metadata": {
            "control_id": row.get("control_id"),
            "policy_passage_id": row.get("policy_passage_id"),
            "score": score,
            "label": label,
            "is_hard_negative": row.get("is_hard_negative", False),
            "source": row.get("source"),
        },
    }


def filter_rows(rows: list, skip_soft_positives: bool = False) -> list:
    """
    Optionally remove Partially Addressed rows — they are the noisiest signal.
    For a first fine-tuning run, keeping only Fully/Not Addressed gives cleaner training.
    """
    if skip_soft_positives:
        return [r for r in rows if float(r.get("score", 0.0)) in (0.0, 1.0)]
    return rows


def write_jsonl(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", default="data/07_golden_mapping/llm_training_data")
    parser.add_argument(
        "--skip-soft-positives",
        action="store_true",
        help="Exclude Partially Addressed rows for a cleaner binary signal",
    )
    args = parser.parse_args()

    out = Path(args.output)

    for split, path in [("train", args.train), ("dev", args.dev)]:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)

        rows = filter_rows(rows, skip_soft_positives=args.skip_soft_positives)
        converted = [to_sharegpt(r) for r in rows]

        label_counts = {}
        for c in converted:
            lbl = c["metadata"]["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        write_jsonl(converted, out / f"{split}.jsonl")
        print(f"{split:5s}: {len(converted)} rows  {label_counts}")
        print(f"       → {out / f'{split}.jsonl'}")

    print("\nExample conversation (first train row):")
    with open(out / "train.jsonl", encoding="utf-8") as f:
        example = json.loads(f.readline())
    for turn in example["conversations"]:
        role = turn["from"].upper()
        preview = turn["value"][:120].replace("\n", " ")
        print(f"  [{role}] {preview}…")


if __name__ == "__main__":
    main()
