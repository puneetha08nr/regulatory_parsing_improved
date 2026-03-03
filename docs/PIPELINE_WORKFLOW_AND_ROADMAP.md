# Pipeline Workflow and Future Improvements

This document covers the end-to-end workflow after running the pipeline in Colab,
the iterative annotation loop, and a prioritised roadmap of future improvements.

---

## 1. After Running in Colab — What Happens

### What the pipeline produces

| File | Contents |
|------|----------|
| `data/06_compliance_mappings/mappings.csv` | Flat list of all (control, passage) pairs with score and status |
| `data/06_compliance_mappings/mappings.json` | Same, in JSON format |
| `data/06_compliance_mappings/mappings_by_passage.json` | Passage-centric view: for each passage, which controls it covers (BM25 + Cross-Encoder, passage as query) |
| `data/06_compliance_mappings/by_policy/*.json` | One JSON per policy document, grouped by policy |
| `data/06_compliance_mappings/compliance_report.json` | Summary: control family coverage, status counts |
| `data/06_compliance_mappings/evaluation_report.json` | Precision, Recall, F1, Recall@K, RePASs scores vs golden data |
| `data/06_compliance_mappings/retrieval_log.json` | Retrieved passage IDs per control (used for Recall@K) |

### Pull results locally

After Colab Step 8 (push to GitHub):
```bash
git pull
# data/06_compliance_mappings/ is now updated
```

---

## 2. The Iterative Improvement Loop

Each iteration = one pipeline run → annotate → retrain → re-run.

```
┌─────────────────────────────────────────────────────────┐
│  Pipeline run (Colab GPU)                               │
│  quick_start_compliance.py                              │
│  → mappings.json  (control → passages)                  │
│  → mappings_by_passage.json  (passage → controls)       │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Generate annotation tasks                              │
│  python3 create_golden_set_tasks.py                     │
│    --candidates data/06_compliance_mappings/mappings.json│
│    --output-tasks golden_set_mapping_tasks.json         │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Label Studio annotation                                │
│  - Mark: Fully / Partially / Not Addressed              │
│  - For wrong matches: pick mismatch reason              │
│  - For wrong control: enter correct control ID          │
│  Target: 200-300 pairs per round                        │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Export and merge golden data                           │
│  python3 create_golden_set_tasks.py --mode export       │
│    --input <label_studio_export.json>                   │
│  → data/07_golden_mapping/golden_mapping_dataset.json   │
└───────────────────┬─────────────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
┌─────────────────┐  ┌────────────────────────────────────┐
│  Blocklist      │  │  Training data                     │
│  (auto-applied  │  │  python3 scripts/                  │
│  next run)      │  │    prepare_golden_for_training.py  │
│                 │  │  → reranker / NLI training CSVs    │
│  - pair-level   │  └────────────────┬───────────────────┘
│  - passage-level│                   │
└─────────────────┘                   ▼
                           ┌──────────────────────┐
                           │  Fine-tune reranker  │
                           │  (on Colab GPU)      │
                           │  → models/compliance │
                           │    -reranker/        │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  Re-run pipeline     │
                           │  RERANKER_MODEL=     │
                           │  models/compliance-  │
                           │  reranker/           │
                           └──────────────────────┘
```

### Blocklist (auto-applied every run)

After annotation, two types of exclusions are applied automatically:

| Type | How created | Effect |
|------|-------------|--------|
| **Pair-level blocklist** | Passages marked "Not Addressed" with high confidence + mismatch reason | That specific (control, passage) pair never appears again |
| **Passage-level blocklist** | Boilerplate sections (Scope, Purpose, ToC) marked N/A or never matching | Passage excluded from all control matching |

These live in `data/07_golden_mapping/golden_mapping_dataset.json` and `data/07_golden_mapping/not_applicable_passages.json`.

---

## 3. Key Parameters to Tune

### Retrieval depth (controls Recall@K)

Set via env vars before running `quick_start_compliance.py`:

| Env var | Default (GPU) | What it does |
|---------|---------------|--------------|
| `TOP_K_RETRIEVE` | 50 | BM25+Dense candidates per control per doc (first stage) |
| `TOP_K_RERANK` | 100 | How many pass to the cross-encoder |
| `TOP_K_PER_DOC` | 5 | Cross-encoder keeps this many per doc |
| `TOP_K` | 10 | Final passages kept per control (across all docs) |

Rule: `TOP_K_RETRIEVE` ≥ `TOP_K_RERANK` ≥ `TOP_K_PER_DOC`

Higher `TOP_K_RETRIEVE` → higher Recall@K but more CE calls.
On GPU: 50–100 is fine. On CPU: keep at 30.

### Thresholds (controls Precision vs Recall tradeoff)

In `quick_start_compliance.py` / `compliance_mapping_pipeline.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `threshold_full` | 0.60 | CE score above this → "Fully Addressed" |
| `threshold_partial` | 0.35 | CE score above this → "Partially Addressed" |

Lower `threshold_partial` → more matches found (higher recall, lower precision).
Raise it to 0.45–0.50 once you have a fine-tuned reranker.

---

## 4. Current Metrics (Baseline — Run 1)

| Metric | Value | Target |
|--------|-------|--------|
| Precision | 3% | > 50% |
| Recall | 9% | > 60% |
| F1 | 0.045 | > 0.55 |
| Recall@20 | 9.3% | > 70% |
| Recall@50 | 24.1% | > 85% |
| RePASs (avg) | 0.593 | > 0.70 |

**Root cause of low Recall@K:** BM25 retrieval was only returning 5 candidates per control per doc.
Fixed in Run 2 by setting `TOP_K_RETRIEVE=50, TOP_K_RERANK=100`.

---

## 5. Future Improvements — Prioritised Roadmap

### Priority 1 — Fix retrieval (immediate, already implemented)
- [x] Increase `TOP_K_RETRIEVE` from 5 → 50 on GPU
- [x] Expose `TOP_K_RETRIEVE`, `TOP_K_RERANK`, `TOP_K_PER_DOC` as env vars
- [ ] **Deduplicate policy files** — same document loaded under different names (`_1`, `_2`, `_3` suffixes) inflates FN count. Remove duplicates from `data/02_processed/policies/`.

### Priority 2 — Grow and use golden data
- [ ] Annotate 200–300 more pairs from Run 2 output (with higher top_k, more candidates available)
- [ ] Reach 500+ total annotated pairs → meaningful fine-tuning signal
- [ ] For multi-control passages annotated "Partially Addressed": verify the boosted score (0.70) is appropriate vs keeping 0.50

### Priority 3 — Fine-tune the cross-encoder reranker
Currently using generic `BAAI/bge-reranker-base`. Fine-tuning on domain-specific pairs will give the biggest quality jump.

```bash
python3 scripts/prepare_golden_for_training.py \
  --golden data/07_golden_mapping/golden_mapping_dataset.json \
  --controls data/02_processed/uae_ia_controls_structured.json \
  --policies data/02_processed/policies \
  --output data/07_golden_mapping/training_data \
  --format reranker
```

Then fine-tune on Colab (GPU, ~1 hr with 500+ pairs):
- Base model: `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Positive: (control, passage) pairs marked "Fully Addressed" → score 1.0
- Soft positive: "Partially Addressed" → score 0.7 (multi-control) or 0.5 (single-control partial)
- Hard negative: "Not Addressed" + high confidence + mismatch reason → score 0.0 (duplicated 2×)
- Use with: `RERANKER_MODEL=models/compliance-reranker/`

### Priority 4 — Multi-framework equivalence (XRefRAG-style)
When a passage maps to UAE IA control X, infer mappings to equivalent ISO 27001 / ADHICS controls automatically using an equivalence table.

- Build: `data/02_processed/framework_equivalence.json` mapping UAE IA ↔ ISO ↔ ADHICS
- Add: `derive_cross_framework_mappings()` in pipeline that reads equivalence table and flags derived mappings
- Benefit: annotating one passage for one framework automatically suggests mappings for others

### Priority 5 — Improve BM25 query construction
Current BM25 query = raw control text. Better: extract **obligation keywords** only.

Example:
- Control: "The entity shall implement multi-factor authentication for all privileged accounts"
- Current BM25 query: full text (noisy — "entity", "shall", "all" are stop words)
- Improved query: "multi-factor authentication privileged accounts MFA" (obligation keywords only)

Implementation: add `_build_retrieval_query(control)` in `compliance_mapping_pipeline.py` that strips modal verbs and generic terms, keeps nouns and technical terms.

### Priority 6 — Passage deduplication before retrieval
Policy documents often repeat the same paragraph (e.g., "This policy applies to all employees") across multiple sections. These inflate the passage count and dilute retrieval quality.

Add a deduplication step in `load_policy_passages()`:
- Hash each passage text (after normalisation)
- Keep only the first occurrence per document
- Reduces index size and speeds up BM25/dense retrieval

### Priority 7 — Active learning for annotation prioritisation
Instead of annotating pipeline candidates randomly, prioritise:
1. **Uncertain pairs** — cross-encoder score between 0.35–0.60 (decision boundary)
2. **High-frequency FPs** — controls that repeatedly match wrong passages
3. **Coverage gaps** — controls with 0 TP in golden data (never correctly matched)

Add `scripts/prioritise_annotation.py` that reads `evaluation_report.json` and `mappings.json` and outputs a ranked list of pairs most worth annotating.

### Priority 8 — Evaluation improvements
- [ ] **Per-control-family metrics** — break down Recall by control family (M1, M2, T1, etc.) to identify which families have the worst retrieval
- [ ] **Coverage report** — what % of controls have at least one passage above threshold? Which are uncovered?
- [ ] **Cross-run comparison** — compare evaluation_report.json across pipeline runs to track improvement over time

---

## 6. Quick Reference — Commands

```bash
# Run pipeline locally (CPU)
python3 quick_start_compliance.py

# Run evaluation
python3 scripts/evaluate_pipeline.py

# Generate annotation tasks from pipeline output
python3 create_golden_set_tasks.py \
  --candidates data/06_compliance_mappings/mappings.json \
  --output-tasks data/03_label_studio_input/golden_set_mapping_tasks.json

# Export Label Studio annotations to golden dataset
python3 create_golden_set_tasks.py \
  --mode export \
  --input <label_studio_export.json>

# Prepare training data from golden dataset
python3 scripts/prepare_golden_for_training.py \
  --golden data/07_golden_mapping/golden_mapping_dataset.json \
  --controls data/02_processed/uae_ia_controls_structured.json \
  --policies data/02_processed/policies \
  --output data/07_golden_mapping/training_data \
  --format reranker   # or: nli

# Pull Colab results after Step 8
git pull
```

---

## 7. File Reference

| File | Purpose |
|------|---------|
| `quick_start_compliance.py` | Main pipeline entrypoint |
| `compliance_mapping_pipeline.py` | Core pipeline: retrieval, reranker, NLI, blocklist |
| `create_golden_set_tasks.py` | Generate Label Studio tasks / export golden data |
| `scripts/prepare_golden_for_training.py` | Convert golden data → NLI / reranker training format |
| `scripts/evaluate_pipeline.py` | Precision, Recall, F1, Recall@K, RePASs evaluation |
| `colab_run_pipeline.ipynb` | Google Colab notebook for GPU pipeline run |
| `data/07_golden_mapping/golden_mapping_dataset.json` | Human-verified (control, passage, status) pairs |
| `data/07_golden_mapping/not_applicable_passages.json` | Boilerplate passages excluded from all matching |
| `data/06_compliance_mappings/evaluation_report.json` | Latest evaluation results |
| `docs/WHAT_NEXT_AFTER_GOLDEN_DATA.md` | Detailed strategy: evaluate → train → improve |
| `docs/MAPPING_STRATEGY_AND_REGNLP.md` | RegNLP methodology and Label Studio XML config |
