# Annotate policy passages with UAE IA and ISO 27001 controls (Label Studio)

Use this flow to **assign UAE IA and ISO 27001 control IDs** to each **parsed policy passage**. One task = one policy passage; annotators enter which controls the passage addresses.

---

## Paths (summary)

| What | Path |
|------|------|
| **Code that generates policy controls tasks** | **`generate_policy_controls_tasks.py`** (project root) |
| **Label Studio config (XML)** | `data/03_label_studio_input/annotate_policy_controls.xml` |
| **Tasks JSON (all policies combined)** | `data/03_label_studio_input/policy_controls_tasks.json` |
| **Tasks JSON per policy (separate policies)** | `data/03_label_studio_input/policy_controls_by_policy/<PolicyName>.json` |
| **Source of policy passages** | `data/02_processed/policies/all_policies_for_mapping.json` |
| **UAE IA control IDs** | `data/03_label_studio_input/uae_ia_control_ids.txt` |
| **ISO 27001 control IDs** | `data/03_label_studio_input/iso27001_control_ids.txt` |
| **ADHICS control IDs** | `data/03_label_studio_input/adhics_control_ids.txt` (can be generated from PDF/JSON). |
| **ADHICS structured controls (like UAE IA)** | `data/02_processed/adhics_controls_structured.json` (from `extract_adhic_controls.py`). |

---

## Step 1: Generate tasks (if needed)

Tasks are built from the parsed policy JSON by **`generate_policy_controls_tasks.py`**. Regenerate after adding or changing policy documents.

**Option A – single combined file (all policies):**
```bash
python3 generate_policy_controls_tasks.py \
  --policies data/02_processed/policies/all_policies_for_mapping.json \
  --output data/03_label_studio_input/policy_controls_tasks.json
```

**Option B – one JSON per policy document (separate policies):**
```bash
python3 generate_policy_controls_tasks.py \
  --policies data/02_processed/policies/all_policies_for_mapping.json \
  --per-policy \
  --output-dir data/03_label_studio_input/policy_controls_by_policy
```
This writes e.g. `policy_controls_by_policy/Logging_and_Monitoring_Policy.json`, `policy_controls_by_policy/Information_Security_Incident_Management_Policy.json`, etc. Import the file for the policy you want to annotate.

Optional: `--max-tasks 50` to cap tasks per file (pilot).

---

## Step 2: Set up Label Studio

1. Create a new project (e.g. “Policy → UAE IA & ISO 27001 controls”).
2. **Settings** → **Labeling Interface** → paste the full content of **`data/03_label_studio_input/annotate_policy_controls.xml`**.
3. **Import** → upload **`data/03_label_studio_input/policy_controls_tasks.json`**.

---

## Step 3: Annotate

For each task you will see:

- **Policy passage**: document name, section, and passage text.
- **UAE IA control IDs**: searchable dropdown (Taxonomy); select all that apply. Type to filter the list.
- **ISO 27001 control IDs**: searchable dropdown (Taxonomy); select all that apply. Type to filter the list.
- **ADHICS control IDs**: searchable dropdown (Taxonomy); select all that apply. Type to filter the list.
- **Notes**: optional.

The labeling config uses Label Studio **Taxonomy** tags so you can search and multi-select control IDs instead of typing them.

---

## Step 4: Export from Label Studio

Export → **JSON** to get annotations. You can then write a small script to merge `uae_ia_control_ids` and `iso27001_control_ids` (and notes) back into your policy dataset or a separate mapping table.

---

## Files involved

| File | Role |
|------|------|
| `data/03_label_studio_input/annotate_policy_controls.xml` | Label Studio labeling config (generated; use dropdowns with search). |
| `build_policy_controls_label_studio_xml.py` | Script to **rebuild** the XML from control ID lists (run after editing `uae_ia_control_ids.txt` or `iso27001_control_ids.txt`). |
| `data/03_label_studio_input/policy_controls_tasks.json` | All passages in one file (default). |
| `data/03_label_studio_input/policy_controls_by_policy/*.json` | One tasks JSON per policy document (use `--per-policy`). |
| `data/02_processed/policies/all_policies_for_mapping.json` | Parsed policy passages (source). |
| `data/03_label_studio_input/uae_ia_control_ids.txt` | UAE IA control ID list for annotators. |
| `data/03_label_studio_input/iso27001_control_ids.txt` | ISO 27001 Annex A control ID list. |
| `data/03_label_studio_input/adhics_control_ids.txt` | ADHICS (Abu Dhabi Healthcare Information and Cyber Security) control ID list. |
| **`generate_policy_controls_tasks.py`** | **Generates** tasks (combined or per-policy). |

**Extracting ADHICS controls from the ADHIC PDF (PDF only):**
Regulation policies are always PDF; extraction is from PDF only. ADHICS uses **domains**, **sub-domains**, and controls with **Control Demand**, **Control Criteria**, and **Applicability** (Basic/Transitional/Advanced). Section B uses **tables**. The extractor outputs:
`standard_name`, `version`, `controls`: each with `domain`, `sub_domain`, `control_id` (e.g. HR 1.1, AM 3.6), `control_demand`, `applicability`, `criteria[]`, `references[]`, `source_index[]`.

```bash
python3 extract_adhic_controls.py --pdf data/01_raw/regulation/ADHIC.pdf --update-control-ids
```
This writes `data/02_processed/adhics_controls_structured.json` and optionally updates `adhics_control_ids.txt`. Then run `build_policy_controls_label_studio_xml.py` to refresh the Label Studio dropdown.

**Regenerating the XML (after changing control lists):**
```bash
python3 build_policy_controls_label_studio_xml.py
```
This reads `uae_ia_control_ids.txt`, `iso27001_control_ids.txt`, and `adhics_control_ids.txt` (if present) and overwrites `annotate_policy_controls.xml` with Taxonomy dropdowns.
