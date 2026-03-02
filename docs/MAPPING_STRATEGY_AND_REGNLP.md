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

## 4. Upgrades (Reranker, Graph, ColBERT)

| Upgrade | Purpose | Status |
|--------|---------|--------|
| **Cross-Encoder reranker** | Replaces NLI with a fast reranker (e.g. BGE-Reranker) on top-50 candidates; ~95% of NLI accuracy, much faster. | **On by default** (`use_reranker=True`). Disable with `USE_RERANKER=0`. |
| **Policy graph** | Same-doc passage neighbors expand retrieval candidates (lightweight graph over passages). | **Optional**: `USE_GRAPH=1`. |
| **ColBERT** | Late-interaction retrieval (per-token vectors) for more precise matching than single-vector Dense. | **Future**: optional backend; requires colbert-ai (or similar); not yet wired into the pipeline. |

---

## 5. Label Studio: What to Import (Golden Set Annotation)

For annotating (control, policy passage) pairs you import **two things** into Label Studio:

| What | Where in Label Studio | File / action |
|------|------------------------|----------------|
| **1. Labeling interface** | Project → **Settings** → **Labeling Interface** | Paste or upload the XML from **`data/03_label_studio_input/compliance_mapping_golden_set.xml`**. |
| **2. Tasks (data to annotate)** | **Import** → Upload file | The JSON file of tasks. Generate it first, then import it. |

**Generate the task JSON** (do this before importing tasks):

```bash
python create_golden_set_tasks.py --mode generate
```

Default output: **`data/03_label_studio_input/golden_set_mapping_tasks.json`**. To use pipeline candidates (control–passage pairs from the pipeline) instead of random sampling:

```bash
python create_golden_set_tasks.py --mode generate --candidates data/06_compliance_mappings/mappings.csv --output-tasks data/03_label_studio_input/golden_set_mapping_tasks.json
```

**Order in Label Studio:** (1) Create project → (2) Set Labeling Interface (XML) → (3) Import the task JSON → (4) Annotate → (5) Export when done, then run `create_golden_set_tasks.py --mode export --input <export.json>` to build the golden dataset.

**XML (copy into Labeling Interface):**

```xml
<View>
  <Header value="UAE IA Control (Regulation Requirement)" size="4"/>
  <View style="background: #f5f5f5; padding: 10px; margin-bottom: 10px;">
    <Text name="control_id" value="Control ID: $control_id" style="font-weight: bold;"/>
    <Text name="control_name" value="Control Name: $control_name"/>
    <Text name="control_family" value="Family: $control_family"/>
    <Header value="Requirement (obligation):" size="5"/>
    <Text name="control_text" value="$control_text" style="white-space: pre-wrap;"/>
    <Text name="sub_controls" value="Sub-controls (excerpt): $sub_controls_snippet" style="font-size: 0.9em; color: #666;"/>
  </View>

  <Header value="Policy Passage (Internal Policy)" size="4"/>
  <View style="background: #e8f4f8; padding: 10px; margin-bottom: 10px;">
    <Text name="policy_id" value="Policy: $policy_name"/>
    <Text name="section" value="Section: $policy_section"/>
    <Header value="Policy text:" size="5"/>
    <Text name="policy_text" value="$policy_text" style="white-space: pre-wrap;"/>
  </View>

  <Header value="Correct control ID (if wrong)" size="5"/>
  <TextArea name="edit_control_id" toName="control_text" placeholder="e.g. M1.1.1 or T4.2.1. Leave blank if the control above is correct." rows="1" editable="true"/>

  <Header value="Your judgment: Does this policy passage address the control?" size="4"/>
  <Choices name="compliance_status" toName="control_text" required="true" showInLine="true">
    <Choice value="Fully Addressed" alias="Full" background="#d4edda"/>
    <Choice value="Partially Addressed" alias="Partial" background="#fff3cd"/>
    <Choice value="Not Addressed" alias="None" background="#f8d7da"/>
  </Choices>

  <Header value="Confidence in your judgment (1–5)" size="5"/>
  <Choices name="confidence" toName="control_text" required="true" showInLine="true">
    <Choice value="1" alias="1 – Low"/>
    <Choice value="2" alias="2"/>
    <Choice value="3" alias="3 – Medium"/>
    <Choice value="4" alias="4"/>
    <Choice value="5" alias="5 – High"/>
  </Choices>

  <Header value="Optional: Evidence or notes" size="5"/>
  <TextArea name="evidence_or_notes" toName="control_text" placeholder="Quote the exact phrase that addresses the control, or explain a gap." rows="3" editable="true"/>

  <Header value="Comments" size="5"/>
  <TextArea name="comments" toName="control_text" placeholder="e.g. Pipeline suggested M1.1.1 but this passage actually addresses M1.1.2." rows="2" editable="true"/>
</View>
```

---

## 6. References

External: [RegNLP GitHub](https://github.com/RegNLP), [ObliQADataset](https://github.com/RegNLP/ObliQADataset), [RePASs](https://github.com/RegNLP/RePASs), [MultiPassage-RegulatoryRAG](https://github.com/RegNLP/MultiPassage-RegulatoryRAG); paper: “RegNLP in Action: Facilitating Compliance Through Automated Information Retrieval and Answer Generation” (arxiv 2409.05677).
