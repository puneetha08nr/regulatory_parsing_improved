# Why This Mapping Strategy? How RegNLP Does It

This document explains **why** the current compliance mapping strategy was chosen, and **how** RegNLP (Regulatory NLP) does mapping and evaluation.

---

## 1. Current Strategy in This Repo (RegNLP-Style, Per-Document Corpus)

| Step | What we do |
|------|------------|
| **Master list** | UAE IA controls (from structured JSON). |
| **Corpus** | **One per policy document.** Passages are grouped by policy doc (id without `_passage_N`); retrieval and NLI run **within each document** separately. |
| **Retrieval** | For each control and **each policy doc**: **BM25 + Dense + RRF** over that doc’s passages → **top_k_retrieve** candidates (e.g. 20). |
| **NLI** | Run **NLI** only on those retrieved passages per doc; keep up to **top_k_per_doc** (e.g. 3) per control per doc. |
| **Merge** | Across docs, sort by entailment score and keep **top_k_per_control** (e.g. 5) per control. |
| **Output** | Status (Fully / Partially / Not Addressed) + evidence; saved per policy document. |

So: **each policy document is handled individually** (own corpus, own retrieval index); retrieval then NLI per doc; then merge.

---

## 2. Why This Strategy Was Chosen

- **RegNLP-aligned:** Retrieval first (BM25 + Dense + RRF), then NLI on candidates only — faster than NLI on full corpus.
- **Obligation-first:** Only obligation controls are mapped (ObligationClassifier). The classifier also supplies the obligation text used as the retrieval query. Default is rule-based; optional LegalBERT (trained with [RegNLP/ObligationClassifier](https://github.com/RegNLP/ObligationClassifier)) improves accuracy and reduces noise — see [HOW_TO_RUN §5](HOW_TO_RUN.md#5-optional-legalbert-obligation-classifier-regnlp) for significance and setup.
- **Structured output:** Mapping = status + evidence (RePASs-style).

---

## 3. How RegNLP Does It

RegNLP does **question → retrieval over corpus → answer**, evaluated with **RePASs**. We recast: control = query, policy passages = corpus, mapping (status + evidence) = answer. Same methodology; our “answer” is compliance mapping.

---

## 4. References

External: [RegNLP GitHub](https://github.com/RegNLP), [ObliQADataset](https://github.com/RegNLP/ObliQADataset), [RePASs](https://github.com/RegNLP/RePASs), [MultiPassage-RegulatoryRAG](https://github.com/RegNLP/MultiPassage-RegulatoryRAG); paper: “RegNLP in Action: Facilitating Compliance Through Automated Information Retrieval and Answer Generation” (arxiv 2409.05677).
