# Local Judge Comparison: NLI vs Llama Fine-tune

## Status

| Judge | Script | Status | F1 (v2 labels) |
|-------|--------|--------|----------------|
| NLI cross-encoder (CPU) | `scripts/nli_judge.py` | Done | **0.53** |
| Llama per-atom fine-tune (GPU) | `scripts/finetune_atom_compliance.py` | Needs Colab | TBD |

Both judges score each control atom independently (binary covered / not), then roll up to FA/PA/NA.
The one with higher posF1 on the held-out test set becomes the production judge.

---

## NLI Judge — Results

**Model:** `cross-encoder/nli-deberta-v3-small` (cached offline, CPU)
**Threshold:** 0.80 (calibrated on 102 train pairs vs v2 labels)

| Metric | Overall | Test (held-out) |
|--------|---------|-----------------|
| Pairs | 162 | 60 |
| Agreement | 47.5% | 45.0% |
| Cohen's kappa | 0.050 | — |
| Positive-class F1 | 0.53 | 0.44 |
| Precision | 0.53 | 0.44 |
| Recall | 0.53 | 0.44 |

**Confusion matrix (162 pairs, human=v2 label, AI=NLI prediction):**

```
human \ AI          Fully  Partial     Not
Fully Addressed         1       11       5
Partially Addressed     2       26      31
Not Addressed           0       36      50
```

**Main weakness:** cannot distinguish FA from PA — only 1/17 "Fully Addressed" predicted correctly; tends to call everything Partial when multiple atoms are involved.

---

## Llama Fine-tune — How to Run (Colab/Lightning)

Upload these files to your Colab session:
- `data/12_atoms/atoms_train.jsonl` (386 atom rows, 85 covered)
- `data/12_atoms/atoms_test.jsonl`  (169 atom rows, 34 covered)
- `data/12_atoms/pairs_eval.json`   (162 pairs with v2 labels)
- `scripts/finetune_atom_compliance.py`

```python
# Cell 1 — install (Colab)
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q trl peft accelerate bitsandbytes

# Cell 2 — train + eval
!python scripts/finetune_atom_compliance.py \
    --model  unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
    --epochs 3 \
    --train  data/12_atoms/atoms_train.jsonl \
    --test   data/12_atoms/atoms_test.jsonl \
    --pairs  data/12_atoms/pairs_eval.json \
    --out    data/10_judge/answer_key.judged_llama.json \
    --save_path /content/atom_judge_adapter
```

Script outputs test-set posF1 directly. Then grade formally:

```bash
.venv/bin/python scripts/grade_judge.py --judged data/10_judge/answer_key.judged_llama.json
```

### Inference-only (after training, to re-run on new pairs):

```python
!python scripts/finetune_atom_compliance.py \
    --infer_only \
    --adapter /content/atom_judge_adapter \
    --pairs data/12_atoms/pairs_eval.json \
    --out   data/10_judge/answer_key.judged_llama.json
```

---

## Picking the Winner

Compare test-set posF1:

| Judge | posF1 | Recommendation |
|-------|-------|----------------|
| NLI baseline | 0.44 (test) | Use if Llama ≤ 0.44 |
| Llama fine-tune | TBD | Use if > 0.44 |

Whichever wins, feed its output to the existing pipeline:

```bash
# Roll up to compliance report
.venv/bin/python scripts/compliance_report.py \
    --judged data/10_judge/answer_key.judged_<winner>.json

# Or reconcile against human key
.venv/bin/python scripts/reconcile_report.py \
    --judged data/10_judge/answer_key.judged_<winner>.json
```

---

## Why Per-atom Framing

Previous fine-tunes (holistic FA/PA/NA) failed because:
1. Training key was systematically under-labeled (60/70 disagreements = human missed coverage)
2. Model learned the NA prior, not compliance semantics (PA recall ≈ 0, 86% over-rejection)

Per-atom binary entailment fixes both: each atom is a self-contained yes/no claim, NLI
models are purpose-built for this, and wrong pair-level labels don't corrupt atom-level labels
(a "Not Addressed" pair can still have 0/5 atoms covered — both labels are consistent).

---

## Atom Dataset Stats

| Split | Pairs | Atoms | Covered |
|-------|-------|-------|---------|
| Train | 102 | 386 | 85 (22%) |
| Test | 60 | 169 | 34 (20%) |
| Total | 162 | 555 | 119 (21%) |

v2 pair label distribution: Fully=17, Partially=59, Not=86 (76 positives / 162 = 47%)
