# Using Policy-Annotated Controls (Label Studio) in Golden Mapping

---

## In simple terms

**What you have:** You already annotated **policy passages** in Label Studio: for each passage, annotators picked **which controls** (e.g. M1.1.1, T2.1.1) that passage addresses. So the export says: “Passage X addresses controls A, B, C.”

**What the mapping pipeline needs:** Golden data in the form of **pairs** with a label: “(Control A, Passage X) → Fully Addressed” or “(Control B, Passage Y) → Not Addressed.” That’s used for training and evaluation.

**The idea:** Your policy annotations already tell us “Passage X addresses Control A, B, C.” So we can **turn that into pairs** without asking annotators again:

- From “Passage X addresses M1.1.1 and M1.1.2” we get two rows:  
  (M1.1.1, Passage X) = Addressed, (M1.1.2, Passage X) = Addressed.

Those rows are in the same format as the golden data from the other Label Studio project (where you label control–passage pairs as Full / Partial / Not). So you can:

1. **Use them as golden data** – Add these rows to your golden file and use for training or evaluation (more data, no extra annotation).
2. **Use them to create tasks** – When generating tasks for the “control–passage pair” project, feed these (control, passage) pairs as candidates; annotators only confirm or change the status instead of finding pairs from scratch.

**Not Applicable (N/A):** Some passages don’t map to any of the controls (e.g. boilerplate, scope text). If annotators can mark a passage as **“Not Applicable”**, we treat it as: **do not create any (control, passage) pairs** for that passage. So N/A = “this passage has no controls defined” → skip it when building golden rows or candidate pairs.

**Not Addressed:** Policy annotations only say “this passage **does** address A, B, C.” They don’t say “this passage does **not** address Z.” So **Not Addressed** is handled in one of two ways: (1) **From the pair-annotation project** – when you label (Control Z, Passage X) as “Not Addressed” there, that’s your source of truth. (2) **Optional closed-world:** For each passage that is *not* N/A, you could treat “every control in scope that was *not* selected = Not Addressed” for that passage; that can create false negatives (annotator might have missed some), so use only if you accept that risk.

---

## 1. Two Annotation Flows (and Their Exports)

| Flow | Task unit | What annotators do | Export shape |
|------|------------|--------------------|--------------|
| **A. Policy → controls** (annotate_policy_controls) | One **policy passage** | For this passage, select which **UAE IA / ISO 27001 / ADHICS** control IDs it addresses (Taxonomy multi-select). | One task per passage; `data.policy_passage_id`, `data.policy_name`, `data.section`, `data.policy_text`; `annotations[].result` with Taxonomy results: `uae_ia_control_ids`, `iso27001_control_ids`, `adhics_control_ids` (each = list of selected paths → leaf = control_id). |
| **B. Control–passage pair** (compliance_mapping_golden_set) | One **(control, passage)** pair | For this pair, choose **Fully / Partially / Not Addressed**, confidence 1–5, optional evidence/comments. | One task per pair; `data.control_id`, `data.policy_passage_id`, etc.; result has `compliance_status`, `confidence`, `evidence_or_notes`, `edit_control_id`, `comments`. |

Flow A is **passage-centric** (passage → list of controls). Flow B is **pair-centric** (pair → status). Golden mapping format we use elsewhere is **pair-centric**: `{ control_id, policy_passage_id, compliance_status, confidence, ... }`.

---

## 2. How Policy-Annotated Controls Can Be Used

### 2.1 Convert policy export → golden-style rows (positive pairs only)

- **Input:** Label Studio export from the **Policy → controls** project (annotate_policy_controls).
- **Logic:**
  - For each task, read `data.policy_passage_id` (and optionally policy_name, section for metadata).
  - **Not Applicable:** If the task has a “Not Applicable” (N/A) field and it is set (e.g. a Choices tag like `not_applicable` selected, or a dedicated column meaning “no controls apply”), **do not emit any rows** for this passage. Skip the task. Optionally record passage_id in a separate “passages_with_no_controls” list for reporting.
  - From `annotations[0].result`, collect Taxonomy regions with `from_name` in `uae_ia_control_ids`, `iso27001_control_ids`, `adhics_control_ids`.
  - If there are **no** selected controls (empty Taxonomy) and N/A is not set, treat according to your convention: either skip (like N/A) or treat as “no positives for this passage.”
  - For each Taxonomy result, `value.taxonomy` is a list of paths (e.g. `[["UAE IA", "M1.1.1"], ["UAE IA", "M1.1.2"]]`). Each path’s **last element** is the selected control ID (leaf).
  - For each selected control_id, emit one row:  
    `{ control_id, policy_passage_id, compliance_status: "Fully Addressed" | "Addressed", policy_name, policy_section, ... }`.
- **Output:** List of golden-style rows in the **same schema** as `create_golden_set_tasks.py --mode export` (control_id, policy_passage_id, compliance_status, etc.) so they can be:
  - Written to a JSON file (e.g. `golden_from_policy_annotations.json`), or
  - **Merged** with golden data from Flow B (e.g. before running `prepare_golden_for_training.py`).

**Design choices:**

- **compliance_status:** Policy annotation doesn’t distinguish Fully vs Partially. Options: (1) Set all to `"Fully Addressed"`, or (2) Introduce a single value `"Addressed"` and document that downstream (e.g. training script) treats it as positive. Recommendation: use `"Fully Addressed"` for simplicity so one golden schema fits both flows.
- **confidence / evidence:** Not collected in Flow A; set to `null` or omit. When merging with Flow B, rows from Flow A are “positive only, no confidence”.
- **Framework:** You can add a field e.g. `source_framework: "uae_ia" | "iso27001" | "adhics"` per row so that filtering (e.g. “only UAE IA for current pipeline”) is easy.

### 2.2 Use to seed or expand golden tasks (Flow B)

- **Option 1 – Seed pairs for annotation:** From policy export, derive all (control_id, passage_id) pairs that were marked as “addressed”. Use this list as **candidates** when generating Flow B tasks: e.g. `create_golden_set_tasks.py --mode generate --candidates <file built from policy export>`. Annotators then only confirm or refine (Full/Partial/None + confidence) instead of discovering pairs from scratch.
- **Option 2 – Add as pre-labelled golden without re-annotation:** Convert policy export to golden rows (2.1) and **merge** with any existing golden JSON. Pairs that appear only in policy export get status “Fully Addressed” (or “Addressed”); pairs that also exist in Flow B export keep the more detailed label (Full/Partial/None + confidence). Dedupe by `(control_id, policy_passage_id)`; decide precedence (e.g. Flow B overwrites when both exist).

### 2.3 Evaluation

- **Pipeline output:** (control, passage, predicted status).
- **Golden from policy export:** For each passage, set of control_ids that humans said the passage addresses (positive set).
- **Metrics:** For a given passage, compare pipeline’s set of predicted (control, passage) with the human set (e.g. precision/recall per passage, or micro-averaged). This uses policy-export data **as ground truth for “which controls does this passage address?”** without needing Flow B for those passages.

### 2.4 Not Applicable and Not Addressed (summary)

| Label / case | Meaning | How to handle when converting to golden |
|--------------|---------|----------------------------------------|
| **Not Applicable** (passage level) | This passage has no controls defined (e.g. boilerplate, out of scope). | **Do not emit any (control, passage) rows** for this passage. Skip the task. Optionally list such passage_ids for reporting. |
| **No controls selected** (empty Taxonomy, N/A not set) | Annotator left all control lists empty. | Treat as N/A and skip, or treat as “no positives” for this passage (no golden rows). |
| **Not Addressed** (pair level) | This specific (control, passage) pair is “does not address.” | **Not present in policy export.** Get it from the **pair-annotation project** (Flow B), where each task is (control, passage) and annotators choose Full / Partial / **Not Addressed**. Optionally, **closed-world**: for each non-N/A passage, treat every control in scope that was *not* selected as “Not Addressed” for that passage (risk: false negatives). |

---

## 3. Export Format Details (Policy → controls)

- **Task data:** `data.policy_passage_id`, `data.policy_name`, `data.section`, `data.policy_text`.
- **Annotations:** `item.annotations[].result[]` — each result has `from_name`, `to_name`, `type`, `value`.
  - **Not Applicable:** In the project’s config, the tag **`passage_scope`** (Choices) has options “Has controls” and “Not Applicable”. In the export, look for `from_name === "passage_scope"` and `value.choices` or `value.selected_labels` containing `"Not Applicable"`. If “Not Applicable” is selected, skip emitting any golden rows for this task.
  - **Taxonomy:** `type === "taxonomy"`, `value.taxonomy` = array of arrays (each inner array = path from root to selected node). With `leafsOnly="true"`, selections are typically one or two levels (e.g. `["UAE IA", "M1.1.1"]`); the **last** element is the control ID.
  - Multiple taxonomies: `uae_ia_control_ids`, `iso27001_control_ids`, `adhics_control_ids` — handle each and tag `source_framework` if needed.

---

## 4. Recommended Integration Points in This Repo

1. **New script (e.g. `scripts/convert_policy_controls_export_to_golden.py`):**
   - Input: Label Studio export JSON from the Policy → controls project (single file or directory of exports).
   - Output: Golden-style JSON (same schema as `create_golden_set_tasks.py --mode export`) with rows derived from Taxonomy selections. Optional: `--framework uae_ia|iso27001|adhics|all`, `--output path`.
2. **Merge step:** Document (or add a small script) to merge:
   - Golden from Flow B (`create_golden_set_tasks.py --mode export`),
   - Golden from policy export (output of the new script),
   - Dedupe by (control_id, policy_passage_id); e.g. Flow B wins on conflict.
3. **Downstream:** `prepare_golden_for_training.py` already consumes golden JSON; it should work unchanged as long as the merged file has `control_id`, `policy_passage_id`, `compliance_status` (and optionally confidence). Rows from policy export will have `compliance_status = "Fully Addressed"` (or "Addressed") and null confidence unless you set a default.

---

## 5. Summary

| Use case | How policy-annotated controls are used |
|----------|----------------------------------------|
| **Generate golden rows** | Convert policy export → (control_id, passage_id, status) rows; status = positive only. **Not Applicable** → no rows for that passage. |
| **Seed Flow B tasks** | Build candidate list (control, passage) from policy export; generate Flow B tasks from that list so annotators confirm/refine. Exclude passages marked N/A. |
| **Merge with Flow B** | Merge golden from policy export with golden from Flow B; dedupe; use merged file for training/eval. |
| **Evaluation** | Use policy export as ground truth “which controls does this passage address?” and compare with pipeline predictions. |
| **Not Addressed** | Get from the pair-annotation project (Flow B). Optionally infer from policy export with closed-world (unselected controls = Not Addressed), with risk of false negatives. |

No code change is required in the **mapping pipeline itself** (retrieval + reranker); the pipeline stays as-is. The reuse is in **data**: turning existing policy annotations into golden mapping rows and optionally merging them with pair-annotated golden data before evaluation and training.
