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

These require a controls file audit or query expansion. **Policy-to-domain routing** (see below) addresses this by scoping retrieval to the relevant control families so the right documents are compared only to the right controls.

---

## Policy-to-Domain Routing (Recommended Approach)

UAE IA controls are hierarchical: each **section** (family) has controls and sub-controls, and **only a subset of controls is applicable to a given policy document**. An Asset Management policy should be matched only against the Asset Management section (e.g. T2); a Logging and Monitoring policy only against the relevant section. Running every policy against all 263 controls dilutes signal and hurts both efficiency and precision.

### Core idea: Policy-to-domain routing

Introduce a **routing layer** that classifies which control domain(s) a policy document belongs to, then run retrieval and ranking **only within that subset** of controls.

```
Policy Document
      ↓
  Domain Router
      ↓
Relevant Control Families (e.g. T2, M4)
      ↓
Retrieval + Reranking within that subset only
      ↓
Mappings
```

This is not only an efficiency gain — it is a **quality gain**. The reranker no longer has to distinguish signal from noise across 263 controls; if scoped to 30–40 controls per document, the ranking task is easier and precision improves naturally.

### Two-level structure to model

UAE IA controls have a natural hierarchy:

- **Family** (e.g. T2 — Asset Management)
  - **Subfamily** (e.g. T2.1 — Asset Inventory)
    - **Control** (e.g. T2.1.1 — Hardware Asset Register)
      - **Sub-control** (e.g. T2.1.1.a — quarterly review)

Routing should happen at **family level first**, then optionally at **subfamily**. Do not route directly to individual controls — that is too granular and will miss documents (e.g. "Access Control Policy") that legitimately touch multiple families (e.g. T3 Access Management and M4 HR Security).

### How domain routing works in practice

**Step 1 — Build a domain taxonomy.**  
For each control family (M1–M6, T1–T9), write a short canonical description of what policy documents belong there (one-time, human-authored). Example: T2 covers asset inventory, asset classification, asset ownership, hardware/software lifecycle. Any policy with those themes routes to T2.

**Step 2 — Policy document classification.**  
When a new policy document arrives, classify it into **one or more** control families. Options, in increasing sophistication:

- **Keyword/title matching** — cheapest; works for obvious cases like "Asset Management Policy" → T2. Fails for generic titles.
- **Embedding similarity** — embed the policy’s title + first 500 words, compare to embedded family descriptions, take top-3 families by cosine similarity. No LLM, works well.
- **LLM classification** — prompt with document summary and family list, return applicable families. Most accurate but slowest; use as fallback when confidence is low (e.g. top-1 similarity below a threshold).

Recommended: **embedding similarity as primary**, LLM as fallback when confidence is low.

**Step 3 — Retrieve only within routed families.**  
BM25 and dense retrieval should be **partitioned by control family** (one index with family as filter, or separate indexes per family). When a document is routed to T2 and M4, retrieval runs only over those ~120 controls, not all 263.

### Multi-domain policy documents

Many policies span more than one family (e.g. "Information Security Policy" → M1, M2, M3). The router must return a **ranked list of families**, not a single label.

- Use **top-K family routing** with a confidence threshold: include any family whose similarity score exceeds the threshold (e.g. 0.4), with a cap of 4–5 families to avoid over-scoping.
- **Always-applicable controls** (e.g. applicability `["P1", "always"]`) should bypass the router and always be included in every document’s candidate set.

### Optional: subfamily refinement

After routing to a family, a second step can narrow to **subfamilies**. For example, "Access Control Policy" routed to T3 is more likely to address T3.1 (User Access Management) and T3.2 (Privileged Access) than T3.5 (Cryptography). Score the policy against each subfamily’s aggregate text (e.g. BM25), keep top-3 subfamilies, and run control-level retrieval only within those. That can reduce the candidate set from ~50 to ~15–20 controls per document.

### What this changes in the pipeline

| Stage | Change |
|-------|--------|
| **Document ingestion** | Add a classification step after extraction. Each document gets `routing_metadata`: assigned families and confidence scores. |
| **Retrieval** | Replace global retrieval with **family-scoped** retrieval (index filtered by family, or per-family indexes). Top-K can drop from 50 to 20–25. |
| **Reranker** | No structural change; runs on fewer, domain-relevant candidates. Precision improves from scoping alone. |
| **NLI / LLM Judge** | Same; both benefit from narrowed candidates. The judge prompt can include family context ("This control is from the Asset Management domain"). |

### Why this fixes the “missing controls” problem

The three controls (T2.2.6, T1.2.3, T6.2.2) that never appear in retrieval are BM25 keyword misses in a **global** pool. With domain-scoped retrieval, T2.2.6 (physical server room biometric access) is only evaluated against documents **routed to T2**, where vocabulary and intent align. The right documents are no longer diluted by unrelated policies.

### Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **Router under-scopes** (misses a family) | Prefer recall over precision in routing: top-3 families, or a low similarity threshold. Include always-applicable controls in every run. |
| **Multi-domain documents** | Explicitly design for multi-domain: top-K families above threshold, cap at 4–5. Tune on a few broad documents. |
| **Taxonomy quality** | One-time human-authored family (and subfamily) descriptions; validate that expected policies route to the right families. |
| **Subfamily over-narrowing** | Add subfamily routing only after family-level routing is stable; keep top-2 or top-3 subfamilies per family. |

### Recommended rollout

1. **Define the taxonomy** — one canonical description per family; list of always-applicable control IDs.
2. **Implement family-level routing only** — e.g. embedding similarity → top-K families + always-applicable. No subfamily yet.
3. **Scope retrieval** — restrict BM25 + dense + reranker to controls in those families. Measure recall/precision vs golden set; verify the three previously missing controls are in scope for the right documents.
4. **Add subfamily routing only if needed** — if family-level scope is still too large (e.g. 40+ controls per doc), add subfamily scoring and narrow to top-2 or top-3 subfamilies per family.
5. **Enrich the judge** — once routing is stable, add family (and optionally subfamily) context to the LLM judge prompt.

This architecture scales: new documents get automatic domain assignment; when adding cross-framework mapping (UAE IA ↔ ISO 27001), the family structure maps cleanly onto ISO clause numbers.

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

- [ ] **Policy-to-domain routing** — Family-level router (embedding similarity → top-K families) + family-scoped retrieval; see [Policy-to-Domain Routing](#policy-to-domain-routing-recommended-approach) above. Addresses missing controls (T2.2.6, T1.2.3, T6.2.2) and improves precision.
- [ ] Fix 3 missing controls (`T2.2.6`, `T1.2.3`, `T6.2.2`) — BM25 query expansion or controls file audit (or superseded by domain routing)
- [ ] Run full LLM-as-Judge pass on 1,450 predictions and re-evaluate
- [ ] Annotate more golden pairs (target: 200+ positives) via Label Studio
- [ ] Resume BPR + LoRA fine-tuning with larger golden dataset
- [ ] Cross-framework equivalence table (UAE IA ↔ ISO 27001 ↔ ADHICS)
