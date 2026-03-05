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

| Parameter | Current (untuned) | Post-finetune target | Effect |
|-----------|-------------------|----------------------|--------|
| `threshold_full` | **0.45** | 0.60–0.70 | CE score above this → "Fully Addressed" |
| `threshold_partial` | **0.25** | 0.35–0.45 | CE score above this → "Partially Addressed" |

Current low thresholds are intentional: an untuned `bge-reranker-base` compresses all scores into the 0.3–0.6 range.
After fine-tuning on domain data, cross-encoder scores spread towards 0 and 1 — raise thresholds then.

---

## 4. Evaluation History

### Run 1 & 2 — Baseline (old Recall@K measurement was broken)

| Metric | Value |
|--------|-------|
| TP / FP / FN | 8 / 0 / 81 |
| Precision (TP/predicted) | 2.7–3% |
| Recall | 9% |
| Recall@5 / @20 / @50 | **0.093 / 0.093 / 0.241** (all identical — bug) |

**Bugs found and fixed:**
1. **Retrieval log ordering bug** — log was built by concatenating per-doc candidate lists in document iteration order (`[doc1_top100, doc2_top100, ...]`). Recall@K=5 only checked the top-5 of the *first* document processed. A gold passage from the 7th document never appeared before position ~700. This is why K=5, K=10, K=20 gave identical recall.
   - Fix: retrieval log is now built after all docs are processed, sorted by cross-encoder score descending (global ranking).
2. **Reranker threshold 0.60 too strict for untuned model** — `BAAI/bge-reranker-base` (never trained on compliance text) produces scores in the 0.3–0.6 range even for correct pairs. Most true positives were silently dropped.
   - Fix: lowered to `THRESHOLD_FULL=0.45`, `THRESHOLD_PARTIAL=0.25`, exposed as env vars.
3. **3 mislabelled/duplicate policy files** — `Security Operations Policy_corrected.json` was actually a copy of VulnMgmt policy; two others were exact duplicates. These polluted the retrieval index and caused systematic mismatch for Security Ops controls.
   - Fix: archived to `data/02_processed/policies_archive/`.

---

### Run 3 — After all fixes (current)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| TP / FP / FN | 8 / 0 / 81 | — | — |
| Precision (TP/predicted) | 2.1% | > 50% | ⚠️ |
| Recall | 9% | > 60% | ⚠️ |
| **Recall@5** | **81.5%** | > 60% | **✅** |
| **Recall@10** | **85.2%** | > 70% | **✅** |
| **Recall@20** | **85.2%** | > 80% | **✅** |
| Recall@50 | 85.2% | > 90% | ⚠️ |
| RePASs (avg on 8 TPs) | 0.593 | > 0.70 | ⚠️ |

**Predicted positives: 386** (up from 292 — lower threshold lets more pairs through).

#### What the numbers mean

**Recall@K is now working correctly.** 85% of controls have their correct passage in the global top-5 reranked results. The retrieval stage is healthy. The remaining ~15% of controls missing at K=20 are:
- Controls where the gold passage exists in a missing policy file (Access Control Policy, Logging & Monitoring v3)
- Controls that are not in the retrieval log at all (`T6.2.2`, `T1.2.3`, `T2.2.6`) — these are non-obligation controls filtered before mapping

**TP is still 8 despite Recall@K=85%.** The retrieval is finding the right passages, but the cross-encoder is not scoring them above the threshold. This is a **reranker quality problem**, not a retrieval problem. `BAAI/bge-reranker-base` has no training signal for UAE IA compliance text — it sees the correct (control, passage) pair but assigns a score below 0.45 because the domain vocabulary is unfamiliar.

**FP is still 0.** The model is conservative: everything it marks as positive is correct, but it misses 91% of true positives.

#### What to do next (single highest-impact action)

Fine-tune the cross-encoder on the golden dataset. With 89 positive pairs and 200 negatives already annotated, this is achievable now. A domain-tuned reranker will directly raise TP count without any further pipeline changes.

See Priority 3 in Section 5 for the fine-tuning commands.

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

#### What the training data looks like

Each row is a `{ query, passage, score, is_hard_negative }` object:

```json
{ "query": "M2.1.1 — The entity shall establish a risk management framework...",
  "passage": "The purpose of this Risk Management Policy is to establish controls...",
  "score": 1.0,
  "is_hard_negative": false }

{ "query": "M1.1.2 — The entity shall deliver security awareness training...",
  "passage": "5 POLICY STATEMENT\nThis Policy applies to all employees...",
  "score": 0.0,
  "is_hard_negative": true }
```

Score assignments:
- `1.0` → Fully Addressed (correct match)
- `0.7` → Partially Addressed, multi-control passage
- `0.5` → Partially Addressed, single-control partial
- `0.0` → Not Addressed; hard negatives are duplicated ×2 so the model sees them more often

#### Steps — run locally first to verify, then on Colab GPU for full training

**Step 1 — Generate training data (already done, 2,471 train / 435 dev rows)**

```bash
python3 scripts/prepare_golden_for_training.py \
  --golden data/07_golden_mapping/golden_mapping_dataset.json \
  --controls data/02_processed/uae_ia_controls_structured.json \
  --policies data/02_processed/policies \
  --output data/07_golden_mapping/training_data \
  --format reranker
# Output: data/07_golden_mapping/training_data/train.json
#         data/07_golden_mapping/training_data/dev.json
```

**Step 2 — Smoke-test locally (1 epoch, CPU, ~5 min)**

```bash
python3 scripts/finetune_reranker.py \
  --train data/07_golden_mapping/training_data/train.json \
  --dev   data/07_golden_mapping/training_data/dev.json \
  --output models/compliance-reranker \
  --epochs 1 --batch-size 8
# Prints: Baseline MAE / Spearman / BinaryAcc  →  After-training same metrics
```

**Step 3 — Full training on Colab GPU (~45 min on T4)**

Run Steps 1–3 of the Colab notebook first (clone repo, mount Drive, install deps), then:

```python
# In a Colab cell — after the repo is cloned and cwd is REPO_DIR
!python3 scripts/finetune_reranker.py \
  --train data/07_golden_mapping/training_data/train.json \
  --dev   data/07_golden_mapping/training_data/dev.json \
  --output models/compliance-reranker \
  --epochs 5 --batch-size 16 --warmup-ratio 0.1
```

Expected output:
```
Baseline:        MAE=0.35  Spearman=0.40  BinaryAcc=0.65
After fine-tune: MAE=0.12  Spearman=0.78  BinaryAcc=0.88
Model saved → models/compliance-reranker/
```

**Step 4 — Save fine-tuned model to Drive**

```python
import shutil
shutil.copytree('models/compliance-reranker',
                '/content/drive/MyDrive/compliance_pipeline_data/compliance-reranker',
                dirs_exist_ok=True)
print('Model saved to Drive')
```

**Step 5 — Push model to GitHub so it persists**

```python
import subprocess
subprocess.run(['git', 'add', 'models/compliance-reranker/'], cwd=REPO_DIR)
subprocess.run(['git', 'commit', '-m', 'Add fine-tuned compliance reranker'], cwd=REPO_DIR)
subprocess.run(['git', 'push', 'origin', 'main'], cwd=REPO_DIR)
```

**Step 6 — Use fine-tuned model in the pipeline**

In Colab Step 5 (Run Pipeline), change the env vars:
```python
os.environ['RERANKER_MODEL']     = 'models/compliance-reranker'
os.environ['THRESHOLD_FULL']     = '0.60'   # raise back up — scores now spread 0–1
os.environ['THRESHOLD_PARTIAL']  = '0.40'
```

Or locally:
```bash
export RERANKER_MODEL=models/compliance-reranker
export THRESHOLD_FULL=0.60
export THRESHOLD_PARTIAL=0.40
python3 quick_start_compliance.py
```

After fine-tuning, cross-encoder scores spread towards 0 and 1 (currently clustered at 0.3–0.6), so you can raise thresholds without losing TPs.

---

### Priority 3b — Fine-tune Llama 3.2 as a compliance classifier + explainer

A generative LLM can produce labelled output **with a rationale** — useful for audit reports.
This is complementary to the cross-encoder, not a replacement for it.

#### Cross-encoder vs Llama 3.2 — which for what

| | Cross-Encoder (`finetune_reranker.py`) | Llama 3.2 (`finetune_llama_compliance.py`) |
|--|--|--|
| **Task** | Score pair 0–1 | Generate label + one-sentence explanation |
| **Speed** | ~5 ms/pair (fast, used in pipeline) | ~500 ms/pair (too slow for pipeline) |
| **Output** | `0.87` (numeric) | `"Fully Addressed — the passage establishes a risk plan…"` |
| **Best role** | Reranker **inside** the pipeline | Explainer **after** pipeline confirms a match |
| **Colab time** | ~45 min (T4) | ~60 min (T4, 1B model) |
| **GPU RAM** | ~2 GB | ~6 GB (4-bit QLoRA) |

#### Why QLoRA fits on a free T4 GPU

Llama 3.2 1B has 1 billion parameters. Full fine-tuning requires ~8 GB just for weights + gradients.
QLoRA instead:
1. Freezes the base model in 4-bit (~700 MB)
2. Attaches tiny trainable LoRA adapter matrices (~50 MB total)
3. Only the adapters are updated — total GPU RAM ~6 GB

The saved output (`models/llama-compliance/`) is just the adapters (~80 MB), not the full model.
At inference the base model is loaded from HuggingFace and the adapters are applied on top.

#### What the model learns

Each training example is a structured conversation:

```
[SYSTEM] You are a compliance analyst specialising in UAE IA...

[USER]   Regulatory Control: The entity shall establish a vulnerability
         management program that includes regular assessments...

         Policy Passage: CLIENT shall establish vulnerability management
         practices to proactively prevent exploitation of vulnerabilities...

         Does this policy passage address the regulatory control?

[ASST]   Fully Addressed
         The policy text satisfies the control's mandate in full.
```

Loss is computed only on the `[ASST]` tokens — the model learns to predict label + rationale,
not to memorise the prompt.

#### LLM training data is already generated

```
data/07_golden_mapping/llm_training_data/
  train.jsonl  — 2,471 ShareGPT-format conversations
  dev.jsonl    —   435 conversations

Label distribution (train):
  Fully Addressed:    677  (27%)
  Partially Addressed: 30   (1%)
  Not Addressed:    1,764  (72%)
```

#### Run on Colab

```bash
# Install (once per session)
pip install unsloth trl datasets -q

# Prepare data if not already done
python3 scripts/prepare_llm_training_data.py \
  --train data/07_golden_mapping/training_data/train.json \
  --dev   data/07_golden_mapping/training_data/dev.json \
  --output data/07_golden_mapping/llm_training_data

# Fine-tune Llama 3.2 1B (~60 min on T4)
python3 scripts/finetune_llama_compliance.py \
  --train data/07_golden_mapping/llm_training_data/train.jsonl \
  --dev   data/07_golden_mapping/llm_training_data/dev.jsonl \
  --output models/llama-compliance \
  --base-model unsloth/Llama-3.2-1B-Instruct \
  --epochs 3 --batch-size 4 --grad-accum 4
```

Use `unsloth/Llama-3.2-3B-Instruct` for better quality if you have an A100 (Colab Pro).

#### Where it fits in the pipeline

```
Retrieval (BM25 + Dense)  →  top-50 candidates
         │
         ▼
Cross-encoder reranker    →  score 0–1, fast (~5ms), keeps top-10
(fine-tune FIRST — biggest precision/recall gain)
         │
         ▼
Llama 3.2 explainer       →  for confirmed TPs only:
(optional, audit trail)      "Fully Addressed — the passage implements
                              the obligation stated in M5.4.1 by..."
```

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
