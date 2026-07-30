# Progress Log — Phi-3 Fine-tune + LLM Judge Run

**Date:** 2026-03-20
**Policy tested:** Asset Management Policy 6
**Model:** Phi-3 (LoRA fine-tune) → `models/compliance-llm-judge`

---

## Step 1 — Fine-tuning

**Dataset**
- 1,809 examples (0 skipped)
- Class weights: FA=5.0×, PA=5.0×, NA=1.0×
- Train/dev split: ~1,668 train / 141 dev

**Training config**
- Epochs: 3 | Batch: 4 | Grad accum: 4 | LR: 0.0002 | Warmup: 10/339 steps

**Per-epoch dev results**

| Epoch | avg_loss | FA P | FA R | FA F1 | PA P | PA R | PA F1 | NA P | NA R | NA F1 | Accuracy |
|-------|----------|------|------|-------|------|------|-------|------|------|-------|----------|
| 1     | 0.2347   | 0.448 | 0.765 | 0.565 | 0.000 | 0.000 | 0.000 | 0.937 | 0.874 | 0.904 | 0.830 |
| 2 ✓  | 0.0693   | 0.500 | 0.765 | 0.605 | 0.000 | 0.000 | 0.000 | 0.946 | 0.882 | 0.913 | 0.837 |
| 3     | 0.0156   | 0.667 | 0.471 | 0.552 | 0.000 | 0.000 | 0.000 | 0.906 | 0.966 | 0.935 | 0.872 |

**Best checkpoint: Epoch 2** (FA recall = 0.765)
Epoch 3 overfit — FA recall dropped to 0.471, NA recall inflated to 0.966.

**Predicted vs gold distribution (dev, epoch 2)**
- Predicted: NA=111, FA=26, PA=4
- Gold: NA=119, FA=17, PA=5

**Model saved to:** `models/compliance-llm-judge`

---

## Step 2 — LLM Judge Run (fine-tuned, single-word FA/PA/NA)

**Command**
```bash
python3 scripts/llm_judge.py --mappings single_policy_e2e/output/mappings.json --use-finetuned --finetuned-model models/compliance-llm-judge
```

**Results**
- Input predictions: 37
- Verdicts: FA=20, PA=4, NA=13
- Kept (non-NA): 24
- Noise removed: 37 → 24 (35%)
- Output: `data/06_compliance_mappings/mappings_llm_judged.json`

> Note: fine-tuned mode output does not set `kept` or `judge_assignment` fields.
> All 24 records in the output file are already the kept (non-NA) set.

---

## Step 3 — Evaluation vs Golden (this policy)

**Files**
- Predictions: `data/06_compliance_mappings/mappings_llm_judged.json` (24 records)
- Golden: `single_policy_e2e/output/golden_filtered.json` (10 pos, 18 neg)

**Results**

| | Reranker only | Reranker + fine-tuned judge |
|--|--------------|----------------------------|
| Precision | 0.243 | **0.292** |
| Recall | 0.900 | 0.700 |
| F1 | 0.383 | **0.412** |
| TP | — | 7 |
| FP | — | 6 |
| FN | — | 3 |

The LLM judge improved precision (+0.049) and F1 (+0.029) at the cost of recall (-0.200).
The 3 FNs are true positives that the judge incorrectly labelled NA.

---

## Schema Investigation (debugging)

Two separate judged output files were found — produced by different runs:

| File | Records | Keys present |
|------|---------|-------------|
| `single_policy_e2e/output/mappings_llm_judged.json` | 1 | `llm_status`, `final_status`, `judge_assignment`, `kept`, `model_used` (Ollama run) |
| `data/06_compliance_mappings/mappings_llm_judged.json` | 24 | `llm_verdict`, `llm_reason`, `llm_model` — no `llm_status`/`kept`/`judge_assignment` (fine-tuned run) |

Key finding: the fine-tuned path in `llm_judge.py` does not write `llm_status`, `final_status`, or `kept`
into its output — it only writes `llm_verdict` and `llm_reason`. Evaluation logic using
`r.get("status")` would work correctly here because the fine-tuned path does set `judged["status"] = label`.

---

## Known Weaknesses

- **PA discrimination is the main gap** — PA recall ≈ 0 across all 3 epochs despite 5× oversampling.
  Model consistently predicts FA or NA; rarely predicts PA.
- 3 FNs in evaluation: true positives dropped by the judge (recall loss is the trade-off cost).

---

## Next Steps

1. Run the same pipeline on additional policies to confirm metrics generalise.
2. Use `single_policy_e2e/run.py --policy <path>` for each new policy (CLI arg now supported).
3. If PA recall remains 0 across policies, consider:
   - Increasing PA weight further (e.g. PA=10×)
   - Adding more PA examples to training data
   - Adjusting classification threshold post-training
4. Compare reranker-only vs reranker + fine-tuned judge on 2–3 more policies before concluding.

---

## Multi-policy Run: IS Incident Management Policy (2026-03-20)

**Policy:** `ISSecurityIncidentMgmntPolicy_v2_corrected.json`
**Family routed:** T8 (13 controls mapped)
**Golden:** 6 positives, 18 negatives

### Results

| Stage | Predictions | TP | FP | FN | Precision | Recall | F1 |
|-------|------------|----|----|-----|-----------|--------|----|
| Reranker-only | 64 | 4 | 2 | 2 | 0.062 | 0.667 | 0.114 |
| After fine-tuned judge | 9 | 1 | 0 | 5 | 0.111 | 0.167 | 0.133 |

Recall@5=0.667 | Recall@20=**1.0** — retrieval is working, classification is the problem.

### Observation
The fine-tuned judge removed **86% of predictions** (64→9), dropping 3 true positives in the process.
Retrieval is sound (R@20=1.0) but the judge is over-aggressive on T8 content.
Root cause confirmed below in training data analysis.

---

## Training Data Analysis — Family Distribution

**Dataset:** `data/07_golden_mapping/golden_mapping_dataset.json` (1,146 rows total)

### Family distribution

| Family | Total | FA | PA | NA | Positives (FA+PA) | Positive rate |
|--------|-------|----|----|-----|-------------------|---------------|
| T8 | 188 | 9 | 4 | 175 | 13 | 6.9% |
| No | 150 | 0 | 0 | 150 | 0 | 0% |
| T1 | 147 | 14 | 10 | 123 | 24 | 16.3% |
| T4 | 102 | 7 | 0 | 95 | 7 | 6.9% |
| M2 | 95 | 6 | 3 | 86 | 9 | 9.5% |
| T2 | 94 | 5 | 3 | 86 | 8 | 8.5% |
| M5 | 77 | 2 | 5 | 70 | 7 | 9.1% |
| M3 | 71 | 3 | 3 | 65 | 6 | 8.5% |
| T7 | 63 | 2 | 2 | 59 | 4 | 6.3% |
| T6 | 59 | 0 | 5 | 54 | 5 | 8.5% |
| T5 | 51 | 5 | 2 | 44 | 7 | 13.7% |
| T3 | 35 | 5 | 3 | 27 | 8 | 22.9% |
| M1 | 14 | 2 | 0 | 12 | 2 | 14.3% |
| **Total** | **1,146** | **69** | **40** | **1,037** | **109** | **9.5%** |

### Key findings

**1. "No" family — 150 pure NA rows**
`corrected_control_id` starts with "No" — likely malformed or null IDs.
Zero positives, pure noise inflating the NA count. Needs investigation and cleanup.

**2. T8 is the largest family but heavily NA-skewed (93.1% NA)**
Only 13 positives out of 188 rows. The judge learned T8 ≈ almost always NA.
This directly explains the 86% rejection rate on the Incident Management policy.

**3. PA is nearly invisible across all families**
Only 40 PA examples total across all 1,146 rows (3.5%).
T4, M1 have **zero PA examples** — the model never saw PA for those families.
This is the root cause of PA recall ≈ 0 in fine-tuning.

**4. Overall dataset is 90.5% NA**
Even with 5× oversampling, the model sees ~60% NA in training batches.
The judge's default behaviour is NA — it needs strong evidence to predict FA/PA.

### Actions required before retraining

| Priority | Action |
|----------|--------|
| High | Investigate and fix "No" family rows (malformed control IDs) |
| High | Add more T8 positive examples to training data |
| High | Add PA examples for T4, M1, T7 (currently zero) |
| Medium | Add family-aware thresholds to `config.py` |
| Medium | Retrain judge with balanced family + PA representation |
| Low | Build multi-policy batch runner for aggregate evaluation |
