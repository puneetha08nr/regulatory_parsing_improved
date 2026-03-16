# Editing and Adding UAE IA Controls in Label Studio

The **master file** for UAE IA controls is `data/02_processed/uae_ia_controls_clean.json` (191 controls, one per ID, deduplicated and enriched). You can **edit** existing controls and **add missing controls** using Label Studio, then export back into the same file (or a new file).

---

## 1. Generate Label Studio tasks from the clean file

From the project root:

```bash
python3 validate_extraction_label_studio.py --mode controls \
  --input data/02_processed/uae_ia_controls_clean.json \
  --output data/03_label_studio_input/control_validation_tasks.json
```

This creates one task per control. Default output path is `data/03_label_studio_input/control_validation_tasks.json` if you omit `--output`.

---

## 2. Set up the project in Label Studio

1. Start Label Studio: `label-studio start` (or use your existing instance).
2. Create a new project (or use an existing one for control validation).
3. **Labeling config:** copy the contents of  
   `data/03_label_studio_input/validate_control_extraction.xml`  
   into the project’s **Settings → Labeling Interface** (Code).
4. **Import tasks:**  
   **Import** → upload or paste from  
   `data/03_label_studio_input/control_validation_tasks.json`.

You will see one task per control with:

- Control ID, family, sub-family
- **Editable:** Description, Sub-controls, Implementation guidelines, External/Internal factors, Guidance points
- **Validation:** Complete / Needs Correction / Missing Content
- **Notes**

---

## 3. Edit existing controls

- Open a task and change the text in the editable fields (description, sub-controls, implementation guidelines, etc.).
- Set **Validation Status** and add **Notes** if needed.
- Submit the task. Only **submitted** tasks are included in the export.

---

## 4. Add missing controls

To add controls that are not yet in the clean file:

1. In Label Studio, **Add task** (or import a JSON that includes extra tasks).
2. Each new task must have the same `data` shape as the generated tasks. Example:

```json
{
  "data": {
    "control_id": "T2.2.6",
    "control_family": "T2 - Asset Management",
    "control_subfamily": "T2.2 - Asset Classification",
    "control_name": "Physical server room access controls",
    "control_description": "Physical access to server rooms must be controlled and logged with biometric authentication.",
    "sub_controls": "T2.2.6.a: access log retention\nT2.2.6.b: visitor escort",
    "implementation_guidelines": "",
    "external_factors": "",
    "internal_factors": "",
    "guidance_points": ""
  }
}
```

3. Fill in **control_id**, **control_family**, **control_subfamily**, **control_name**, **control_description**, and **sub_controls** (and other fields if you use them).
4. Annotate the task (edit description/sub-controls as needed, set Validation Status) and **Submit**.

On export, any task whose **control_id** is not in the current clean file is treated as a **new control** and appended to the merged list.

---

## 5. Export from Label Studio and write back to the clean file

1. In Label Studio: **Export** → **JSON** (export only **submitted** annotations).
2. Save the export (e.g. `control_validation_export.json`).
3. Run:

```bash
python3 validate_extraction_label_studio.py --mode export \
  --input path/to/control_validation_export.json \
  --type controls \
  --source data/02_processed/uae_ia_controls_clean.json \
  --output data/02_processed/uae_ia_controls_clean.json
```

- **--source** = current master file (clean JSON). Annotations are merged into this list; order and schema are preserved.
- **--output** = where to write the result. Use the same path as `--source` to update in place, or a different path (e.g. `uae_ia_controls_clean_edited.json`) to keep a backup.

Behaviour:

- **Existing controls** that you annotated: their description, sub_controls, implementation_guidelines, etc. are updated; clean-only fields (`full_text`, `has_useful_text`, `enrichment_sources`, etc.) are preserved or recomputed.
- **Existing controls** that you did not annotate: left unchanged.
- **New controls** (tasks in the export whose `control_id` is not in the source): appended as new records in the same schema, with `enrichment_sources: ["label_studio"]`.

---

## 6. Re-run deduplication (optional)

If you prefer to keep a single source of truth and only add controls in Label Studio (without re-running the full deduplication script), you can skip this. If you later regenerate the clean file from `uae_ia_controls_corrected.json` with `scripts/deduplicate_controls.py`, any controls you added only in Label Studio would need to be re-added or merged separately. For day-to-day “edit and add missing controls” workflow, exporting with `--source` and `--output` to the clean file is enough.

---

## Summary

| Step | Command / action |
|------|-------------------|
| 1. Generate tasks | `validate_extraction_label_studio.py --mode controls --input data/02_processed/uae_ia_controls_clean.json` |
| 2. Label Studio | Use config `validate_control_extraction.xml`, import the generated JSON |
| 3. Edit / add | Edit existing tasks; add new tasks for missing controls (same `data` shape) |
| 4. Export | Export JSON from Label Studio (submitted only) |
| 5. Write back | `validate_extraction_label_studio.py --mode export --input <export.json> --type controls --source data/02_processed/uae_ia_controls_clean.json --output data/02_processed/uae_ia_controls_clean.json` |

The master file for UAE IA controls remains `data/02_processed/uae_ia_controls_clean.json`; the pipeline and `scripts/generate_synthetic_pairs.py` use it by default.
