# What Next After Golden Data (RegNLP Strategy)

This doc summarizes **next steps** once you have:
- **Controls** extracted from UAE IA, ISO, ADHIC (structured JSON)
- **Policy passages** extracted per document
- **Golden data**: human-verified policy passage ↔ control mappings (annotated in Label Studio)

It follows the RegNLP strategy and your existing docs (`MAPPING_STRATEGY_AND_REGNLP.md`, `claude_analysis.md`, `GOLDEN_DATA_AND_LLM.md`).

---

## 1. Export and Formalize Golden Mappings

- **Label Studio config** for golden (control, passage) tasks: **`data/03_label_studio_input/compliance_mapping_golden_set.xml`**. It includes:
  - **Correct control ID (if wrong):** annotators can enter the right UAE IA control ID when the pipeline suggested the wrong one; export uses this when present.
  - **Comments:** free text for reviewer notes, wrong-control explanation, or mapping correction.
  - Compliance status (Fully / Partially / Not Addressed), confidence, and evidence/notes.
- **Export** from Label Studio → JSON (per project or combined).
- **Normalize** the export into a single golden set format, e.g.:
  - `passage_id` (or policy + section), `control_ids` (UAE IA, ISO, ADHICS), optional `status` (Fully / Partially / Not Addressed), `notes`, `comments`.
- Store under something like `data/06_compliance_mappings/golden/` or `data/07_golden_mapping/` so it is the **ground truth** for evaluation and tuning.

**Why:** Automated pipeline outputs must be compared to this set; without a formal golden file, you cannot measure precision/recall or RePASs-style metrics.

---

## 2. Evaluate Automated Pipeline Against Golden Data

### How the pipeline gets its input (annotated passages are not the input)

The compliance mapping pipeline **does not** take the Label Studio export (annotated passages) as input. It works like this:

1. **Pipeline input** = the **same** policy passage files you used to create the Label Studio tasks:
   - **Source:** `data/02_processed/policies/*_for_mapping.json` (one JSON per policy doc; each file is a list of passages with `id`, `name`, `text`, `section`, `metadata`).
   - **Loading:** `quick_start_compliance.py` (and the pipeline) loads these via `load_all_policies_from_dir("data/02_processed/policies")` or `load_policy_passages(path)` — see `compliance_mapping_pipeline.py` (`load_policy_passages`, `load_policy_passages_from_list`) and `flexible_policy_extractor.load_all_policies_from_dir`.
2. You **run** the pipeline on that passage list; it produces **predicted** control ↔ passage mappings (retrieval + NLI).
3. **Annotated data** (Label Studio export) is used **only for evaluation**: convert the export to a golden file keyed by `passage_id` (same `id` as in `*_for_mapping.json`), then compare pipeline output to golden — e.g. for each passage id, do predicted control IDs match human-assigned control IDs? That gives you precision/recall.

So: **same passages** (from `*_for_mapping.json`) → pipeline runs → predictions; **golden set** (from Label Studio export) → compare by passage `id` to get metrics. There is currently no script that feeds the Label Studio export *into* the pipeline; the export is for **evaluation** only.

---

- Run your **compliance mapping pipeline** (e.g. `compliance_mapping_pipeline.py`, `regrag_xref_pipeline.py`) on the **same** policy passages that were annotated (i.e. load from `data/02_processed/policies/*_for_mapping.json`).
- **Compare** predicted mappings (control ↔ passage, status) to golden mappings (from Label Studio export, keyed by passage `id`).
- **Metrics to report:**
  - **Per control:** Recall (did we retrieve the right passages?), Precision (are retrieved passages actually correct?).
  - **Per passage:** Which passages got wrong or missing control IDs?
  - **Retrieval:** Recall@K for “given control, is the correct passage in top K?” (aligns with RegNLP retrieval evaluation).
- Use the golden set as **test** (and optionally **dev**) so you can tune retrieval (BM25 + Dense weights, RRF), NLI threshold, and top_k without overfitting.

**Why:** This is the RegNLP-style “evaluate retrieval then answer” applied to mapping: retrieval quality first, then status/evidence quality.

---

## 3. RePASs-Style Compliance Metrics (Optional but Recommended)

- **Entailment:** NLI(premise=policy passage, hypothesis=control obligation) — already in your pipeline; ensure it is logged per mapping.
- **Obligation coverage:** Does the passage address all sub-requirements of the control (if you have structured sub-controls)?
- **Evidence strength / specificity:** Heuristics or a small model: does the passage cite concrete measures (e.g. “12 months”, “MFA”) vs generic (“appropriate measures”)?
- **Composite score:** e.g. weighted combination; **flag** cases where status is “Fully_Addressed” but composite is low → review in Label Studio.

**Why:** From `claude_analysis.md` — RePASs adapted to compliance gives you a consistent way to spot weak mappings and improve quality.

---

## 4. ObligationClassifier (RegNLP)

- **Use** an obligation classifier (rule-based now; optional LegalBERT from RegNLP) so that **only obligation-bearing** controls are mapped.
- **Effect:** Fewer irrelevant controls in the candidate set, less noise, and alignment with RegNLP (map “shall/must” requirements, not descriptive text).

**Why:** Reduces manual review and focuses annotation on what matters for compliance.

---

## 5. Multi-Framework Equivalence (XRefRAG-Style)

- **Build** a small equivalence layer: UAE IA ↔ ISO 27001 ↔ ADHICS (e.g. manual table or semi-automated from existing mappings).
- **When** a policy passage is mapped to UAE IA control X, **infer** mappings to equivalent ISO/ADHICS controls (with a “derived” flag if you want to track source).
- **Use** this to:
  - Propose draft mappings for other frameworks when annotators assign one framework.
  - Report “one policy passage satisfies N controls across M frameworks.”

**Why:** From `claude_analysis.md` — XRefRAG-style cross-framework linking is the main gap; doing it even in a simple form (equivalence table) scales your golden data across frameworks.

---

## 6. Active Learning and Continuous Improvement

- **Prioritize** annotation: passages or controls where the pipeline disagrees with golden data or where RePASs-style score is low.
- **Re-export** golden data periodically; **re-run** evaluation (step 2) to track progress.
- **Optionally** use golden data to fine-tune a ranker or NLI model for your domain (UAE/GCC policy language).

**Why:** RegNLP-style evaluation + golden data is the loop: annotate → export → evaluate → tune → annotate uncertain cases again.

---

## 7. Train / Fine-Tune a Model on Golden Data

Once you have a **golden mapping dataset** (human-verified control ↔ passage with status), you can train or fine-tune a model for your domain (UAE IA / policy language).

### What you have

- **Golden file:** e.g. `data/07_golden_mapping/golden_mapping_dataset.json` from `create_golden_set_tasks.py --mode export --input <label_studio_export.json>`.
- Each row: `control_id`, `policy_passage_id`, `compliance_status` (Fully / Partially / Not Addressed), plus optional snippets.

### What to train

**Current pipeline default:** The pipeline uses a **Cross-Encoder reranker** (e.g. `BAAI/bge-reranker-base`) by default, not NLI. So the model you are actually using for status is the reranker. NLI is only used when `USE_RERANKER=0` or the reranker is unavailable.

| Option | Model type | Input → Output | Use in pipeline |
|--------|------------|----------------|------------------|
| **A. NLI** | 3-class sequence classifier (e.g. DeBERTa, RoBERTa) | (premise=passage, hypothesis=control) → entailment/neutral/contradiction | Drop-in when you turn off the reranker (`USE_RERANKER=0`); replaces `roberta-large-mnli`. |
| **B. Reranker** | **Cross-Encoder** (e.g. BGE-Reranker) | (query=control, passage) → score | **This is what the pipeline uses by default.** Fine-tune to replace or improve `BAAI/bge-reranker-base`. |
| **C. Small LLM** | Causal LM or seq2seq | "Passage: … Control: …" → "Fully / Partially / Not Addressed" | Use as verification or evidence generation. |

**Recommended first step:** **Option B (Reranker / Cross-Encoder)** — that’s the default path. Use **Option A (NLI)** if you prefer the NLI (3-class) path or want a fallback.

### Step 1: Prepare training data from golden

Convert golden JSON + controls + policies into training-ready format (full text):

```bash
python3 scripts/prepare_golden_for_training.py \
  --golden data/07_golden_mapping/golden_mapping_dataset.json \
  --controls data/02_processed/uae_ia_controls_structured.json \
  --policies data/02_processed/policies \
  --output data/07_golden_mapping/training_data \
  --format nli
```

- **`--format nli`** → CSV/JSON with `premise` (passage), `hypothesis` (control obligation), `label` (entailment / neutral / contradiction from Fully / Partially / Not).
- **`--format reranker`** → (query, passage, score or label) for Cross-Encoder training.
- The script enriches golden rows with **full** control and passage text from the controls JSON and policy `*_for_mapping.json` files.

### Step 2: Train the model

- **NLI (A):** Use Hugging Face `transformers` + `datasets`. Base model (e.g. `microsoft/deberta-v3-base`, `roberta-base`) + 3-class head; train on the prepared data. Save to e.g. `models/compliance_nli_finetuned`.
- **Reranker (B):** Use `sentence-transformers` Cross-Encoder training: positive = (control, passage) Fully Addressed; negative = Not Addressed; Partially as soft or separate class.
- **LLM (C):** Instruction format "Passage: … Control: … → Status: …"; fine-tune with LoRA/QLoRA (`trl`, `peft`) on a base LLM.

### Step 3: Use the trained model in the pipeline

- **Reranker (default path):** Pass `reranker_model=path/to/your/reranker` when creating `ComplianceMappingPipeline`, or set env `RERANKER_MODEL`. This is the path used by default.
- **NLI (fallback path):** If you use `USE_RERANKER=0`, the pipeline uses the entailment mapper. Set the NLI model path (e.g. your fine-tuned model instead of `roberta-large-mnli`) in the entailment mapper initialization.

### Data size and quality

- **Minimum:** A few hundred golden pairs; 1k+ is better.
- **How many maximum?** There is no fixed upper limit. In practice:
  - **~500–1k** – Enough to see a gain over the base reranker/NLI.
  - **1k–3k** – Solid for fine-tuning; good target for most setups.
  - **3k–10k** – Can help further; gains tend to taper off unless you have many controls or need very high accuracy.
  - Beyond that, focus on **quality and balance** (Fully / Partially / Not) rather than raw count. Aim for 1k–3k unless you have a clear reason to go larger.
- **Class balance:** If most are "Fully Addressed", oversample or weight Partially/Not.
- **Split:** Hold out 10–20% as dev/test.

### How to get 1k+ golden pairs

Golden data = human-verified (control, passage, status) from Label Studio. Each row is one (control, passage) pair with a label; the model is trained to predict that label from the text pair.

**Why pipeline candidates are better for training**  
The pipeline (retrieval + reranker) already picks *plausible* (control, passage) pairs—ones where the model thought there might be a match. Annotating those gives you:
- **Diverse passages per control** (different controls get different passages),
- **Relevant pairs** (on-distribution for the reranker/NLI you want to train),
- **Useful labels** (Fully / Partially / Not) that teach the model to correct its own ranking/scoring.

Using “random” or “first N” passages per control would often pair irrelevant passages with each control (many “Not Addressed”), and previously the script gave the *same* first N passages to every control—bad for diversity and training. So **prefer (1) pipeline candidates**; use (2) only to add volume with **random** sampling (script now samples different passages per control).

1. **Annotate pipeline candidates (recommended)**  
   Turn pipeline output into Label Studio tasks, then annotate:
   ```bash
   python3 create_golden_set_tasks.py --mode generate \
     --candidates data/06_compliance_mappings/mappings.json \
     --output data/03_label_studio_input/golden_set_mapping_tasks.json
   ```
   Import the JSON into Label Studio, annotate status for each task. Export when done → one golden row per task. To get 1k+, add more policies and re-run the pipeline so `mappings.json` grows, or combine with (2).

2. **Generate more tasks (no candidate file)**  
   The script now **randomly samples** `pairs_per_control` passages **per control** (so each control gets a different set of passages). Use this to grow the task pool:
   ```bash
   python3 create_golden_set_tasks.py --mode generate \
     --controls data/02_processed/uae_ia_controls_structured.json \
     --policies data/02_processed/policies \
     --pairs-per-control 8 \
     --max-tasks 1500 \
     --seed 42
   ```
   Example: ~145 controls × 8 random passages ≈ 1,160 tasks. Many will be “Not Addressed” (random passage often doesn’t address a control); that still helps the model learn the negative class. Then annotate in Label Studio and export.

3. **Get more pipeline candidates (don’t rely only on “more policies”)**  
   The pipeline outputs **at most `num_controls × top_k_per_control`** rows (e.g. 144 × 5 ≈ 725). Adding more policy documents gives the model *more passages to choose from* (better diversity) but **does not increase that total**. To get 1k+ candidate pairs from the pipeline:
   - **Increase `top_k_per_control`** when running the pipeline (e.g. from 5 to 8 or 10 in `quick_start_compliance.py` or `create_mappings(..., top_k_per_control=10)`), then re-run. Example: 144 × 10 ≈ 1,440 candidate pairs.
   - **Adding policy documents** (e.g. 2–5 more) is still useful for *coverage* and *diversity* of passages, but there’s no fixed “how many to add”—add as many as you have that are in scope. Use the new `mappings.json` as `--candidates` in (1).

4. **Batch annotation**  
   Annotate in waves (e.g. 500 from pipeline candidates, then 500 from a random generate run). Merge Label Studio exports into one golden file before `prepare_golden_for_training.py`.

---

## Suggested Order

| Priority | Step | Purpose |
|----------|------|---------|
| 1 | Export & formalize golden mappings | Single source of truth for evaluation |
| 2 | Evaluate pipeline vs golden data | Baseline metrics (recall@K, precision, per-control/per-passage) |
| 3 | RePASs-style metrics | Flag weak “Fully_Addressed” mappings; improve consistency |
| 4 | ObligationClassifier | Reduce noise; focus on obligation controls only |
| 5 | Multi-framework equivalence | Scale mappings across UAE IA, ISO, ADHICS |
| 6 | Active learning loop | Prioritize annotation, re-evaluate, tune |
| 7 | Train on golden data | Fine-tune NLI or reranker (or LLM) for your domain |

---

## References in This Repo

- **Strategy:** `docs/MAPPING_STRATEGY_AND_REGNLP.md`
- **RegNLP evaluation & roadmap:** `docs/claude_analysis.md`
- **Golden data workflow:** `docs/GOLDEN_DATA_AND_LLM.md`
- **Annotation (policy ↔ controls):** `docs/ANNOTATE_POLICY_CONTROLS.md`
- **Pipeline layout:** `PROJECT_STRUCTURE.md` (Phase 5: RePASs, XRefRAG)
- **Training data prep:** `scripts/prepare_golden_for_training.py` (golden → NLI/reranker format)
