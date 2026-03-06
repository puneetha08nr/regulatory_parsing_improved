# Run Comparison and Diagnosis: Base Model vs Fine-tuned

## Side-by-side: Run 1 (base) vs Run 2 (fine-tuned + label smoothing)

| Metric | Run 1 (base model) | Run 2 (fine-tuned + smoothed) | Change |
|---|---|---|---|
| Predictions | 386 | 1,448 | +275% |
| True Positives | 8 | 9 | +1 |
| False Positives | 0 | 0 | 0 |
| False Negatives | 81 | 80 | −1 |
| Precision | 2.07% | 0.62% | −1.45 pp |
| Recall | 8.99% | 10.11% | +1.12 pp |
| F1 | 3.37% | 1.17% | −2.2 pp |
| R@5 | **81.5%** | 70.4% | −11.1 pp |
| R@10 | **85.2%** | 74.1% | −11.1 pp |
| R@20 | **85.2%** | 77.8% | −7.4 pp |
| R@50 | **85.2%** | 81.5% | −3.7 pp |
| RePASs | **0.593** | 0.424 | −0.169 |

---

## Diagnosis: Fine-tuning made retrieval worse

The **R@K drop is the critical signal** — R@5 fell from 81.5% to 70.4%. R@K is a retrieval-stage metric, unaffected by classification thresholds. This means the fine-tuned model is ranking correct passages *lower* than the base model did.

This is exactly the **Spearman collapse** we predicted:

- Fine-tuning improved binary accuracy: 49% → 74%
- But destroyed ranking quality: Spearman 0.22 → 0.14
- Even with label smoothing, the ranking damage persists

The other effects follow directly from this:

- **1,448 predictions vs 386** — the 0.18 threshold is too permissive; noise fills in where ranking was lost
- **Precision drops** — more noise predictions, same tiny TP count
- **RePASs drops** — individual mapping quality is worse because reranker scores are less reliable

---

## Missed pairs: unchanged between both runs

Both runs miss the same 80 golden pairs. Fine-tuning did not help find new correct matches — it only reshuffled which ones bubble up.

| Policy | Controls missed | Root cause |
|---|---|---|
| Physical & Environmental Security | T2.1.1, T2.2.2, T2.2.5, T2.2.6, T2.3.1, T2.3.6 | 6 controls — whole policy poorly ranked |
| Asset Management | T1.1.1 – T1.4.2 (10 pairs) | Passage-level specificity lost |
| Security Operations Policy | T3.1.1, T3.2.1, T3.2.2, T3.2.5, T3.3.1 | 5 controls |
| *(retrieval stage)* | **T2.2.6, T1.2.3, T6.2.2** | Never enter BM25/dense retrieval at all |

---

## Actions taken

### 1. Reverted to base model (immediate)

Step 5 of the notebook was hard-reverted to use `BAAI/bge-reranker-base` with standard thresholds while training is fixed:

```python
os.environ['RERANKER_MODEL']    = 'BAAI/bge-reranker-base'
os.environ['THRESHOLD_FULL']    = '0.45'
os.environ['THRESHOLD_PARTIAL'] = '0.25'
```

### 2. Switched to pairwise MarginMSE loss (implemented — Run 3 pending)

The MSELoss regression on soft scores collapses into binary prediction. The fix is **pairwise MarginMSE** which directly trains the model to rank positives above negatives:

```
loss = MSE( score(q, pos) − score(q, neg),  target = 1.0 )
```

Instead of predicting exact score values, the model only needs to satisfy the *ordering* — positive scores higher than negative. This preserves Spearman while keeping BinaryAcc gains.

Changes made in `scripts/finetune_reranker.py`:

- Added `--loss pairwise` (new default) and `--loss mse` (legacy)
- Pairwise mode builds `(query, positive_passage, negative_passage)` triplets grouped by query
- Custom training loop with AdamW + linear warmup, saves best checkpoint per epoch
- Hard negatives appear in more triplets (oversampled) rather than being duplicated as rows

Command to run:

```bash
python3 scripts/finetune_reranker.py \
    --train data/07_golden_mapping/training_data/train.json \
    --dev   data/07_golden_mapping/training_data/dev.json \
    --output models/compliance-reranker \
    --loss pairwise \
    --epochs 5 --batch-size 16
```

**Expected outcome**: Δ Spearman positive (was −0.08 with MSE), R@5 ≥ 81.5%, loss decreasing steadily to ~0.05 by epoch 5.

### 3. Fix 3 missing controls (pending)

`T2.2.6`, `T1.2.3`, `T6.2.2` are never seen in the retrieval log — filtered out before the reranker. Requires:

- Audit the controls source file to confirm they are present and correctly formatted
- Check BM25 index coverage for their control text
- Annotate their relevant policy passages manually as positives for training

---

## What to watch in Run 3 (pairwise fine-tuned)

| Signal | Good | Bad |
|---|---|---|
| Epoch loss trend | Decreasing from ~0.20 → ~0.05 | Plateaus above 0.15 after epoch 2 |
| Δ Spearman | Positive | Negative again |
| R@5 vs baseline | ≥ 81.5% | < 81.5% |
| Predictions count | Similar to Run 1 (~400) | Explodes again (>1000) |
| RePASs | ≥ 0.593 | < 0.593 |
