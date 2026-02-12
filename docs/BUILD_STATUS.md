# Build status: what’s done vs RegNLP roadmap

This doc maps the **recommended RegNLP roadmap** (from `docs/claude_analysis.md`) to **what is actually implemented** in this repo.

---

## Roadmap vs current implementation

| Component | Roadmap (RegNLP) | What we have | Gap |
|-----------|------------------|--------------|-----|
| **ObligationClassifier** | LegalBERT-based, “Must Have” | **Rule-based** (default) + **optional LegalBERT**. Train with [RegNLP/ObligationClassifier](https://github.com/RegNLP/ObligationClassifier); use `legalbert_model_path` or `LEGALBERT_MODEL_PATH` / `models/obligation-classifier-legalbert`. | Optional LegalBERT integrated; train with RegNLP/ObligationClassifier and point pipeline to saved model. |
| **Hybrid retrieval** | BM25 + Dense + RRF, “Must Have” | **Done.** `PolicyRetrieval` in same pipeline: BM25 (rank_bm25), optional Dense (sentence-transformers), RRF. Per policy doc. | None. |
| **Multi-passage retrieval** | Multi-passage evidence, “Must Have” | **Done.** Retrieval per doc → top-K candidates → NLI; we keep multiple mappings per control (top_k_per_control). | None. |
| **XRefRAG / cross-framework** | Knowledge graph, framework equivalence, “Must Have” | **Not built.** No graph, no `framework_equivalents.json`, single framework (UAE IA) only. | Full gap. |
| **RePASs metrics** | Entailment + contradiction + obligation coverage, “Should Have” | **Partial.** We use **entailment** (NLI) for status (Fully/Partial/Not). No formal RePASs composite, no contradiction or obligation-coverage metrics. | RePASs-style evaluation layer not implemented. |
| **QA generation** | Skip | Not used. | N/A. |

---

## Where classification lives

- **File:** `compliance_mapping_pipeline.py`
- **Classes:**
  - **`ObligationClassifier`** (rule-based): `OBLIGATION_KEYWORDS`, `PERMISSION_KEYWORDS`; `classify_control(control_text)` → `(is_obligation, obligation_text)`.
  - **`LegalBertObligationClassifier`** (optional): Load LegalBERT from local path or HuggingFace id (e.g. RePASs-trained `models/obligation-classifier-legalbert`). Same interface; uses model for `is_obligation`, rule-based extraction for `obligation_text`. Falls back to rule-based if model load fails.
- **Usage:** Pipeline `__init__` accepts `obligation_classifier="rule"|"legalbert"`, `legalbert_model_path`, `legalbert_model_name`. **Train the model** with [RegNLP/ObligationClassifier](https://github.com/RegNLP/ObligationClassifier) (clone repo, run `ObligationClassifier.py`; model saves to `./obligation-classifier-legalbert`). **Quick start:** set `LEGALBERT_MODEL_PATH` or copy that folder to this repo as `models/obligation-classifier-legalbert`.

---

## Roadmap phases vs current state

| Phase | Roadmap | Current state |
|-------|---------|----------------|
| **Phase 1: Foundation** | ObligationClassifier for UAE IA; extract obligations; framework_equivalents for top 20 | Obligation filtering: **rule-based** classifier only. No framework_equivalents, single framework. |
| **Phase 2: Core mapping** | Hybrid retrieval; multi-framework Label Studio; map top 50; RePASs-inspired validation | **Done for single framework:** hybrid retrieval, Label Studio for mappings, NLI status. No multi-framework, no RePASs metrics. |
| **Phase 3: Scaling** | Knowledge graph; automate multi-passage; gap analysis | Not started. |
| **Phase 4: Production** | Fine-tune on UAE; dashboard; continuous update | Not started. |

---

## Summary

- **Classification:** Yes. **Rule-based** (default) and **optional LegalBERT** in `compliance_mapping_pipeline.py`. Train with [RegNLP/ObligationClassifier](https://github.com/RegNLP/ObligationClassifier); use via `legalbert_model_path` or `LEGALBERT_MODEL_PATH`.
- **Aligned with roadmap:** Hybrid retrieval, multi-passage-style flow (retrieval then NLI), per-doc corpus, Label Studio annotation.
- **Not built yet:** RegNLP ObligationClassifier (LegalBERT), XRefRAG / cross-framework graph, RePASs composite metrics, phased rollout as in the roadmap.

To align with the roadmap’s “ObligationClassifier – Must Have” in the intended form, the next step would be to integrate RegNLP’s ObligationClassifier (e.g. from [RegNLP/ObligationClassifier](https://github.com/RegNLP/ObligationClassifier)) and optionally keep the current rule-based classifier as a fallback or for comparison.
