# UAE IA Compliance Mapping Pipeline

Automated system that maps internal organisational policy documents to UAE Information Assurance (UAE IA) Regulation controls using a multi-stage NLP pipeline, human-in-the-loop annotation, and an LLM-as-Judge post-processing layer.

---

## What This System Does

Given:
- A set of internal **policy documents** (DOCX/PDF) — e.g. Asset Management Policy, Access Control Policy
- The **UAE IA Regulation controls** catalogue (~263 controls across domains T1–T8, M1–M5)

It automatically answers:
> *"Which passage in which policy document addresses which UAE IA control, and how well?"*

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1 — Document Ingestion                                    │
│  DOCX/PDF → passage extraction → structured JSON per document   │
│  Script: run_policy_extraction_and_label_studio.py              │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  Stage 2 — Retrieval  (per control, per document)               │
│  BM25 keyword search  +  Dense embedding search                 │
│  → top-K candidate passages (default K=50 GPU / 30 CPU)        │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  Stage 3 — Cross-Encoder Reranker                                │
│  Model: BAAI/bge-reranker-base  (270 MB)                        │
│  Scores each (control, passage) pair 0→1                        │
│  CPU fallback: cross-encoder/ms-marco-MiniLM-L-2-v2  (22 MB)   │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  Stage 4 — NLI Entailment Filter                                │
│  Classifies each pair as Fully / Partially / Not Addressed       │
│  Thresholds: Full ≥ 0.45, Partial ≥ 0.25 (base model, untuned) │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  Stage 5 — LLM-as-Judge  (post-processing, optional)            │
│  Local Ollama model re-evaluates each predicted pair             │
│  Prompt styles: strict / chain-of-thought / few-shot            │
│  Script: scripts/llm_judge.py                                   │
│  Removes ~60% of noisy predictions while preserving true matches│
└─────────────────────────┬────────────────────────────────────────┘
                          │
              data/06_compliance_mappings/
              ├── mappings.json              (all predictions)
              ├── mappings_llm_judged.json   (after LLM filter)
              ├── mappings.csv
              ├── by_policy/*.json           (one file per document)
              ├── compliance_report.json
              └── evaluation_report.json
```

---

## Key Scripts

| Script | Purpose |
|---|---|
| `quick_start_compliance.py` | **Main entry point** — runs the full pipeline end-to-end |
| `scripts/llm_judge.py` | LLM-as-Judge post-processing filter via Ollama |
| `scripts/evaluate_pipeline.py` | Evaluates predictions against the golden dataset |
| `scripts/finetune_reranker.py` | Fine-tunes the Cross-Encoder on golden annotations |
| `scripts/prepare_golden_for_training.py` | Converts Label Studio exports → train/dev splits |
| `create_golden_set_tasks.py` | Creates Label Studio annotation tasks from pipeline output |
| `run_policy_extraction_and_label_studio.py` | Ingests raw DOCX/PDF → structured passage JSON |

---

## Human-in-the-Loop Annotation Loop

```
Pipeline predictions
      │
      ▼
create_golden_set_tasks.py  ──►  Label Studio
                                      │
                               Annotator reviews:
                               - Confirms correct pairs
                               - Marks wrong pairs (+ mismatch reason)
                               - Enters correct control if known
                                      │
                                      ▼
                             Export JSON from Label Studio
                                      │
                                      ▼
                   create_golden_set_tasks.py --mode export
                                      │
                          data/07_golden_mapping/
                          golden_mapping_dataset.json
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                 Pair blocklist            Training data
                 (confirmed negatives)     (train/dev split)
                 auto-applied to           for reranker
                 next pipeline run         fine-tuning
```

The golden dataset currently contains **89 verified positive pairs** and **200 verified negative pairs** across 12 policy documents.

---

## Current Performance

Evaluated against the golden dataset (89 human-verified matches):

| Metric | Score | Target | Notes |
|---|---|---|---|
| Recall@5 | **70.4%** | 60% | ✅ Exceeds target |
| Recall@10 | **74.1%** | 70% | ✅ Meets target |
| Recall@20 | **77.8%** | 80% | ⚠️ Near target |
| Recall@50 | **81.5%** | 90% | ⚠️ Below target |
| Precision | 0.6% | — | Low — 1,450 noisy predictions |
| Recall | 10.1% | — | 9 of 89 matches surfaced post-threshold |
| F1 | 1.2% | — | Dominated by precision |
| RePASs | 0.688 | ≥ 0.70 | Near target on confirmed TPs |

**Retrieval is working well** — the correct passage is in the top-5 for 70%+ of controls. The low Precision/Recall/F1 numbers are a threshold and noise problem, not a retrieval problem. The LLM-as-Judge layer directly addresses this.

### Why Precision is Low

The base reranker (untuned) produces score distributions that overlap heavily between true and false positives. With threshold `0.25`, ~1,440 noisy "Partially Addressed" predictions pass through. The LLM judge removes ~60% of these in testing, which should raise precision significantly.

### Known Missing Controls

Three controls never enter the retrieval stage at all (BM25 keyword miss):
- `T2.2.6` — Physical server room biometric access
- `T1.2.3` — Asset classification sub-control
- `T6.2.2` — Third-party security sub-control

These require a controls file audit or query expansion.

---

## Quick Start

### Prerequisites

```bash
# Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# For LLM-as-Judge (optional, local)
ollama pull llama3.2:1b      # 1 GB, fastest
ollama pull llama3.2         # 4 GB, better quality
```

### Run the Pipeline

```bash
# Full pipeline (GPU recommended, CPU works)
python3 quick_start_compliance.py

# LLM-as-Judge post-processing (run after pipeline)
python3 scripts/llm_judge.py --model llama3.2:1b

# Evaluate against golden set
python3 scripts/evaluate_pipeline.py \
    --mappings data/06_compliance_mappings/mappings_llm_judged.json

# Test prompt styles (3 samples each)
python3 scripts/llm_judge.py --model llama3.2:1b --limit 5 --prompt-style strict
python3 scripts/llm_judge.py --model llama3.2:1b --limit 5 --prompt-style cot
python3 scripts/llm_judge.py --model llama3.2:1b --limit 5 --prompt-style fewshot
```

### Environment Variables (all optional)

| Variable | Default | Description |
|---|---|---|
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` (GPU) / `ms-marco-MiniLM-L-2-v2` (CPU) | Cross-encoder model |
| `THRESHOLD_FULL` | `0.45` | Score threshold for "Fully Addressed" |
| `THRESHOLD_PARTIAL` | `0.25` | Score threshold for "Partially Addressed" |
| `TOP_K_RETRIEVE` | `50` (GPU) / `30` (CPU) | BM25+Dense candidates per control |
| `CPU_MODE` | `1` (auto) | Force lightweight models for CPU |
| `USE_RERANKER` | `1` | Set to `0` to disable Cross-Encoder |

---

## LLM-as-Judge Prompt Engineering

Three prompting strategies are available via `--prompt-style`:

| Style | Description | Speed | Accuracy |
|---|---|---|---|
| `strict` | Direct label instruction, strict rules, no examples | Fastest | Good |
| `cot` | **Chain-of-Thought** — model reasons step-by-step before labelling *(default)* | Medium | Best for small (1B) models |
| `fewshot` | 3 labelled UAE IA examples provided in the prompt | Slowest | Most accurate for larger models |

**Why CoT is default for `llama3.2:1b`:** Small models jump to conclusions without reasoning scaffolding. Forcing step-by-step analysis (What does the control require? What does the passage say? Is there a gap?) significantly improves label accuracy at the cost of ~2× more output tokens.

---

## Data Directory Layout

```
data/
├── 01_raw/policies/          Raw DOCX/PDF policy documents
├── 02_processed/policies/    Extracted passage JSON (one file per doc)
├── 02_processed/             UAE IA controls structured JSON
├── 03_label_studio_input/    Label Studio import files
├── 04_label_studio/imports/  UAE IA controls for LLM judge index
├── 06_compliance_mappings/   Pipeline outputs
│   ├── mappings.json         All predictions (1,450 pairs)
│   ├── mappings_llm_judged.json  After LLM filtering
│   ├── mappings.csv
│   ├── by_policy/            One JSON per policy document
│   ├── compliance_report.json
│   ├── evaluation_report.json
│   └── retrieval_log.json    Per-control top-K passage IDs (for R@K)
└── 07_golden_mapping/
    ├── golden_mapping_dataset.json   Human-verified pairs (89+ / 200-)
    └── training_data/
        ├── train.json        Fine-tuning training split
        └── dev.json          Fine-tuning evaluation split
```

---

## What Has Been Tried and Why

### Fine-tuning the Cross-Encoder (Paused)

Fine-tuning `BAAI/bge-reranker-base` on the golden dataset was attempted with:
1. **MSELoss regression** on soft scores → caused Spearman rank correlation collapse (0.22 → 0.14). Binary accuracy improved but ranking order degraded, dropping R@5 from 81.5% to 70.4%.
2. **Pairwise MarginMSE loss** → CUDA OOM on 15 GiB GPU due to dual forward pass per batch.
3. **Binary Pairwise Ranking (BPR) loss** with single combined forward pass + fp16 mixed precision + LoRA adapters → numerically stable, CPU-feasible, but training paused in favour of the LLM-as-Judge approach.

**Current decision:** Use the base model (best R@K), fix precision via LLM-as-Judge. Resume fine-tuning once the golden dataset is larger (currently 89 positives — more annotation needed).

### Graph-Based Mapping (Future)

The control hierarchy (e.g. T2.2 → T2.2.1, T2.2.2 …) and cross-framework equivalences (UAE IA ↔ ISO 27001 ↔ ADHICS) could be represented as a knowledge graph to improve sub-control coverage. Not yet implemented.

---

## Roadmap

- [ ] Fix 3 missing controls (`T2.2.6`, `T1.2.3`, `T6.2.2`) — BM25 query expansion or controls file audit
- [ ] Run full LLM-as-Judge pass on 1,450 predictions and re-evaluate
- [ ] Annotate more golden pairs (target: 200+ positives) via Label Studio
- [ ] Resume BPR + LoRA fine-tuning with larger golden dataset
- [ ] Cross-framework equivalence table (UAE IA ↔ ISO 27001 ↔ ADHICS)
