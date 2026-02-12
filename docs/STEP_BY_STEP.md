# Step-by-step implementation (per claude_analysis.md)

Use this checklist to implement the RegNLP roadmap **one step at a time**. When context is lost, say e.g. *“Proceed with Step 3”* and we continue from there.

**Reference:** `claude_analysis.md` (Recommended Implementation Roadmap + Final Evaluation Summary).

---

## Status key

- `[ ]` Not started  
- `[~]` In progress / partial  
- `[x]` Done  

**Current step:** **Step 3** (next: framework_equivalents.json schema/stub).

---

## Phase 1: Foundation (Week 1–2)

| Step | Task | Status | Notes |
|------|------|--------|--------|
| **1** | Keep rule-based ObligationClassifier as default; document where it lives and how to switch later | `[x]` | Done: `compliance_mapping_pipeline.py` + `docs/BUILD_STATUS.md`. |
| **2** | Integrate RegNLP ObligationClassifier (LegalBERT) for UAE IA: add optional model-based classification alongside rule-based | `[x]` | Done: `LegalBertObligationClassifier` in pipeline; `obligation_classifier="rule"\|"legalbert"`, `legalbert_model_path`. Quick start uses `LEGALBERT_MODEL_PATH` or `models/obligation-classifier-legalbert`. |
| **3** | Add `data/02_processed/framework_equivalents.json` (schema only or stub) and document schema for “top 20 controls” | `[ ]` | Schema: control_id → internal_refs, framework_equivalents (iso27001, adhics, …). Fill manually later. |
| **4** | (Optional) Extract obligations from a second framework (e.g. ADHICS or ISO) into same JSON shape as UAE IA | `[ ]` | Defer until multi-framework; or add one small sample. |

---

## Phase 2: Core mapping (Week 3–8)

| Step | Task | Status | Notes |
|------|------|--------|--------|
| **5** | Hybrid retrieval pipeline (BM25 + Dense + RRF) | `[x]` | Done: `PolicyRetrieval` in `compliance_mapping_pipeline.py`, per policy doc. |
| **6** | Multi-passage retrieval (per-doc, top-K, then NLI) | `[x]` | Done: create_mappings with use_retrieval=True, top_k_per_doc, top_k_per_control. |
| **7** | Label Studio project for mapping annotation (single-framework) | `[x]` | Done: annotate_mappings.xml, annotate_mappings_label_studio.py generate/export. |
| **8** | RePASs-inspired metrics: add entailment + obligation_coverage (and optionally contradiction) and composite score | `[ ]` | New module or section in pipeline: compute scores per mapping; optionally flag “Fully Addressed” with low score. |
| **9** | (Later) Multi-framework Label Studio project and map top 50 controls across frameworks | `[ ]` | Depends on Step 3 (framework_equivalents) and Step 4. |

---

## Phase 3: Scaling (Week 9–12)

| Step | Task | Status | Notes |
|------|------|--------|--------|
| **10** | Complete knowledge graph / XRefRAG-style: all controls, internal_refs + framework_equivalents | `[ ]` | Build from framework_equivalents + control lists; use for “map once, satisfy multiple frameworks”. |
| **11** | Automate multi-passage retrieval for new policies (pipeline run from config or CLI) | `[~]` | Already automated; add config file or CLI for paths/limits if needed. |
| **12** | Gap analysis report: controls with no or weak policy mapping | `[ ]` | Report from mappings.json: e.g. Not Addressed or low entailment score. |

---

## Phase 4: Production (Month 4+)

| Step | Task | Status | Notes |
|------|------|--------|--------|
| **13** | Fine-tune ObligationClassifier (or NLI) on UAE IA / ADHICS corpus | `[ ]` | Optional; after Step 2 in use. |
| **14** | Compliance monitoring dashboard (internal tool or export) | `[ ]` | Out of scope for “step by step” code; define later. |
| **15** | Continuous update workflow (new policies → re-run mapping → diff) | `[ ]` | Script or pipeline trigger; define later. |

---

## How to use this doc

1. **Pick the next step** (first `[ ]` or `[~]` in order, or the “Current step” above).  
2. In a new session, say: *“Proceed with Step N”* or *“Implement Step 2”*.  
3. After completing a step, update this file: change `[ ]` to `[x]` and set **Current step** to the next one.

---

## Quick reference: where things live

| What | Where |
|------|--------|
| Rule-based / LegalBERT ObligationClassifier | `compliance_mapping_pipeline.py` → `ObligationClassifier`, `LegalBertObligationClassifier` |
| Hybrid retrieval | `compliance_mapping_pipeline.py` → `PolicyRetrieval` |
| NLI + mapping creation | `compliance_mapping_pipeline.py` → `EntailmentMapper`, `create_mappings` |
| Label Studio (mappings) | `annotate_mappings_label_studio.py`, `data/03_label_studio_input/annotate_mappings.xml` |
| Build status (what’s done vs roadmap) | `docs/BUILD_STATUS.md` |
| Full roadmap & analysis | `claude_analysis.md` |
