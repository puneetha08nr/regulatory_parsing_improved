# How to Run – Compliance Mapping (Simple)

**One main workflow:** run the compliance pipeline, then optionally annotate in Label Studio.

---

## 1. Run the compliance pipeline (one command)

From the **project root** (`regulatory_parsing2/`):

```bash
python3 quick_start_compliance.py
```

**What it uses (auto-detected):**
- **Controls:** `uae_ia_controls_structured.json`, or `uae_ia_controls_corrected.json`, or `uae_ia_controls_from_label_studio.json` (from Label Studio export)
- **Policies:** `data/02_processed/policies/all_policies_for_mapping.json`

**What it does:**
1. Loads UAE IA controls  
2. Loads policy passages (grouped by document)  
3. Builds retrieval (BM25 + Dense) per policy doc  
4. For each control: retrieve + NLI per doc → mappings  
5. Saves: `data/06_compliance_mappings/mappings.csv`, `mappings.json`, `by_policy/*.json`, `compliance_report.json`

**If something is missing:** the script will print what file it expects and how to create it.

**Run in background** (survives closing the terminal; output in a log file):

```bash
nohup python3 quick_start_compliance.py > quick_start_compliance.log 2>&1 &
```

- Log: `quick_start_compliance.log` in the project root. To watch progress: `tail -f quick_start_compliance.log`.
- When the laptop sleeps, the process is suspended and resumes when the laptop wakes. To keep it running across sleep, leave the laptop awake or run it on a server.

---

## 2. If controls or policies are missing

| You need | Command / action |
|----------|-------------------|
| **Controls** (UAE IA) | `python3 improved_control_extractor.py` → writes `data/02_processed/uae_ia_controls_structured.json` (needs PDF in `data/01_raw/regulation/`) |
| **Controls** (ADHICS) | `python3 extract_adhic_controls.py --pdf data/01_raw/regulation/ADHIC.pdf --update-control-ids` → writes `data/02_processed/adhics_controls_structured.json` (PDF only) |
| **Policies** (passages) | Put policy DOCX/PDF in `data/01_raw/policies/`, then run `python3 policy_extractor.py` or your policy extraction script so that `data/02_processed/policies/all_policies_for_mapping.json` exists |

**Convert Label Studio controls export to our format:**
```bash
python3 convert_label_studio_controls_to_json.py \
  --input data/02_processed/label_Studio_mapperd_uae.json \
  --output data/02_processed/uae_ia_controls_from_label_studio.json
```
Then run step 1 again (quick_start will pick the new file if others are missing).

---

## 3. Optional: annotate mappings in Label Studio

After the pipeline has run and you have `data/06_compliance_mappings/mappings.json` (or `by_policy/*.json`):

1. **Generate Label Studio tasks** (one file per policy or combined): see **docs/ANNOTATE_MAPPINGS_STEPS.md**.
2. In **Label Studio:** create project → paste config from `data/03_label_studio_input/annotate_mappings.xml` → import tasks → annotate.
3. **Export** from Label Studio, then run `annotate_mappings_label_studio.py export`.

---

## 4. Files that matter (for this workflow)

| File | Role |
|------|------|
| **quick_start_compliance.py** | Main script to run end-to-end mapping |
| **compliance_mapping_pipeline.py** | Pipeline logic (you don’t run this directly) |
| **data/02_processed/** `uae_ia_controls_*.json` | UAE IA controls (input) |
| **data/02_processed/policies/all_policies_for_mapping.json** | Policy passages (input) |
| **data/06_compliance_mappings/** | All mapping outputs (CSV, JSON, by_policy, report) |
| **annotate_mappings_label_studio.py** | Generate/export Label Studio tasks (optional) |
| **data/03_label_studio_input/annotate_mappings.xml** | Label Studio config (optional) |

---

## 5. Optional: LegalBERT obligation classifier (RegNLP)

By default the pipeline uses a **rule-based** obligation classifier. To use the **RegNLP LegalBERT** classifier, train with the official [RegNLP/ObligationClassifier](https://github.com/RegNLP/ObligationClassifier) repo, then point this pipeline at the saved model.

### Significance of the LegalBERT obligation classifier in this pipeline

The obligation classifier is **the first step** of the compliance mapping flow. For each UAE IA control it decides: *Is this an obligation?* and extracts the **obligation text** used as the query for retrieval.

| Role | What it does |
|------|----------------|
| **Filter** | With `filter_obligations_only=True` (default), **only** controls classified as obligations are sent to retrieval and NLI. Non-obligations (e.g. guidance, recommendations) are skipped. |
| **Query** | The extracted **obligation text** (or full control text) is what we search policy passages for; better obligation extraction → better retrieval and mapping. |
| **RegNLP alignment** | RegNLP uses an obligation classifier so that mapping focuses on “shall/must” style requirements; LegalBERT in this pipeline follows that methodology. |

**Rule-based (default)** uses keyword lists (e.g. *shall*, *must*, *required*). It is fast and works out of the box but can miss nuanced obligations and can false-positive on text that uses those words in a non-obligation way.

**LegalBERT** (trained with RegNLP/ObligationClassifier on legal/regulatory data) typically gives higher accuracy (e.g. ~89% in reported benchmarks), fewer false positives, and better recall on edge cases. That means: fewer controls to map (only real obligations), better evidence quality, and less manual cleanup. Training and pointing the pipeline at the saved model is optional but recommended for production use.

**Where are these?**

| What | Location |
|------|----------|
| **Training repo** | **[RegNLP/ObligationClassifier](https://github.com/RegNLP/ObligationClassifier)** (external). Contains dataset (ADGM-based), LegalBERT fine-tuning script, and saves to `./obligation-classifier-legalbert`. |
| **Training script** | In that repo: `ObligationClassifier.py` (README may say `obligation_classification.py`; use the `.py` file that exists in the repo). |
| **Model after training** | Inside the ObligationClassifier clone: `./obligation-classifier-legalbert` (HuggingFace `save_pretrained` format). |
| **Where this repo looks for the model** | **Option A:** In **this** project: `models/obligation-classifier-legalbert` (i.e. `regulatory_parsing2/models/obligation-classifier-legalbert`). **Option B:** Any path: `export LEGALBERT_MODEL_PATH=/absolute/path/to/obligation-classifier-legalbert`. |

**Next steps (concrete)**

1. **Clone and train** (outside this repo):
   ```bash
   git clone https://github.com/RegNLP/ObligationClassifier.git
   cd ObligationClassifier
   pip install torch transformers scikit-learn
   # Dataset ObligationClassificationDataset.json is already in the repo
   python ObligationClassifier.py
   ```
   This saves the model to `ObligationClassifier/obligation-classifier-legalbert/`.

2. **Use the model in this repo** — choose one:
   - **Copy:** `cp -r obligation-classifier-legalbert /path/to/regulatory_parsing2/models/` (create `models/` if needed), or  
   - **Env:** `export LEGALBERT_MODEL_PATH=/path/to/ObligationClassifier/obligation-classifier-legalbert`

3. **Run this pipeline:** from `regulatory_parsing2/`, run `python3 quick_start_compliance.py`. It will use LegalBERT when the folder exists or `LEGALBERT_MODEL_PATH` is set.

Pipeline API: `ComplianceMappingPipeline(obligation_classifier="legalbert", legalbert_model_path="/path/to/model")`.

---

## 6. Dependencies

```bash
pip install -r requirements.txt
# For retrieval (faster): pip install rank_bm25
# Optional dense retrieval: pip install sentence-transformers
# For NLI: pip install transformers torch
```

---

**TL;DR:** From project root, run **`python3 quick_start_compliance.py`**. Use the paths it prints if something is missing.
