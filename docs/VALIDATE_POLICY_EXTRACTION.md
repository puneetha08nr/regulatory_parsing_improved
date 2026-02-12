# How to validate policy extracted from document files

This flow lets you **review and correct** policy passages extracted from PDF/DOCX using Label Studio. **One JSON per policy document** — no combined file.

**Better structure (fewer truncations):** The default extractor uses `pdfplumber`, which can flatten or truncate nested structure (lists, tables, multi-level sections). To preserve hierarchy and avoid truncation, use **Docling** or **Unstructured** (both work with **PDF and Word (.docx)**):

- **Docling** (recommended, already in `requirements.txt`): layout-aware parsing for PDF and DOCX; preserves sections/tables/lists.
- **Unstructured**: element-level partition for PDF and DOCX. Install with `pip install 'unstructured[pdf]'` and/or `pip install 'unstructured[docx]'` for Word.

Use `--backend docling` or `--backend unstructured` when running extraction (see Step 1).

---

## Files involved

| File | Role |
|------|------|
| **data/01_raw/policies/** | Input: policy DOCX/PDF files. |
| **data/02_processed/policies/*_for_mapping.json** | **Source JSON (per document):** one file per policy doc; each file is a list of passages (id, name, text, section, metadata). |
| **data/03_label_studio_input/validate_policy_extraction.xml** | **Label Studio config (XML):** validation UI (passage text, section, heading, status, notes). |
| **data/03_label_studio_input/policy_validation_tasks/<doc>_validation_tasks.json** | **Label Studio tasks (per document):** one task file per policy; import the one(s) you want to validate. |

---

## Step-by-step

### 1. Extract policy from documents (one JSON per doc)

From project root:

```bash
python3 run_policy_extraction_and_label_studio.py --input-dir data/01_raw/policies
```

To preserve nested structure (sections, tables, lists) and avoid truncation, use Docling or Unstructured:

```bash
python3 run_policy_extraction_and_label_studio.py --input-dir data/01_raw/policies --backend docling
# or:  --backend unstructured   (requires: pip install 'unstructured[pdf]')
```

This:

- Runs **flexible_policy_extractor.py** on `data/01_raw/policies/` and writes **only**:
  - `data/02_processed/policies/<doc>_for_mapping.json` (one file per policy document)
- Then generates **Label Studio tasks per policy** →
  - `data/03_label_studio_input/policy_validation_tasks/<doc>_validation_tasks.json` (one per doc)

To only run extraction (no task generation):

```bash
python3 run_policy_extraction_and_label_studio.py --input-dir data/01_raw/policies --no-tasks
```

To generate tasks for a **single** policy file later:

```bash
python3 validate_extraction_label_studio.py --mode policies \
  --input data/02_processed/policies/<doc>_for_mapping.json \
  --output data/03_label_studio_input/policy_validation_tasks/<doc>_validation_tasks.json
```

Optional: `--max-tasks 50` to limit tasks per file.

---

### 2. Set up Label Studio

1. Create a new project in Label Studio.
2. **Settings** → **Labeling Interface** → paste **`data/03_label_studio_input/validate_policy_extraction.xml`**.
3. **Import** → **Upload Files** → select one or more task files from  
   **`data/03_label_studio_input/policy_validation_tasks/`** (e.g. one policy at a time).

---

### 3. Annotate

For each task: policy ID, policy name, passage ID, section, heading; editable passage text; **Validation status** (Complete / Needs Correction / Missing Content); **Notes / Corrections**.

---

### 4. Export and apply corrections

1. In Label Studio: **Export** → **JSON**.
2. Run (use a per-doc output path if you validated one policy):

```bash
python3 validate_extraction_label_studio.py --mode export \
  --input path/to/label_studio_export.json \
  --type policies \
  --output data/02_processed/policies/<doc>_corrected.json
```

The script writes corrected passages (same schema as `*_for_mapping.json`) with updated `text` and `metadata.validation_status` / `metadata.correction_notes`. Only annotated tasks are in the output.

---

## Summary: which JSON and XML?

- **Source JSON (per document):**  
  `data/02_processed/policies/<doc>_for_mapping.json`
- **Label Studio config (XML):**  
  `data/03_label_studio_input/validate_policy_extraction.xml`
- **Task JSON (per document):**  
  `data/03_label_studio_input/policy_validation_tasks/<doc>_validation_tasks.json`

Other scripts (compliance mapping, annotate mappings, etc.) that need **all** passages load them from the directory via `load_all_policies_from_dir("data/02_processed/policies")` — no combined file is written.
