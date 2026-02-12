# Project structure (modular)

The project is organized into **modules** and **entrypoint scripts** at the root. Implement step-by-step by moving logic into the right module and keeping entrypoints thin.

---

## Directory layout

```
regulatory_parsing2/
├── README.md                 # Project overview + quick links
├── PROJECT_STRUCTURE.md      # This file
├── requirements.txt
│
├── docs/                     # Documentation only
│   ├── README.md             # Index of docs
│   ├── HOW_TO_RUN.md         # How to run pipeline + annotation
│   ├── ANNOTATE_MAPPINGS_STEPS.md
│   ├── MAPPING_STRATEGY_AND_REGNLP.md
│   └── claude_analysis.md    # RegNLP roadmap
│
├── src/                      # Modular packages (implement step-by-step)
│   ├── pipeline/             # Regulation parsing (existing)
│   │   ├── parser.py
│   │   ├── classifier.py
│   │   ├── control_to_json_converter.py
│   │   └── mapper.py
│   ├── utils/                # Shared helpers
│   │   └── label_studio.py
│   ├── controls/             # Control extraction + Label Studio convert
│   │   └── __init__.py
│   ├── policies/             # Policy extraction
│   │   └── __init__.py
│   ├── mapping/              # Compliance mapping (retrieval, NLI, pipeline)
│   │   └── __init__.py
│   └── annotation/           # Label Studio task gen/export
│       └── __init__.py
│
├── data/                     # Inputs and outputs
│   ├── 01_raw/               # Raw PDF/DOCX
│   ├── 02_processed/         # Controls JSON, policies JSON
│   ├── 03_label_studio_input/# Task JSONs, XML configs
│   └── 06_compliance_mappings/# mappings.json, by_policy/, reports
│
├── config/                   # Config files
├── configs/
│
└── [Entrypoint scripts at root - see below]
```

---

## Modules and entrypoints

| Module        | Purpose | Current entrypoints (root) | Next steps |
|---------------|---------|----------------------------|------------|
| **controls**  | UAE IA control extraction; convert Label Studio export to our JSON | `improved_control_extractor.py`, `convert_label_studio_controls_to_json.py`, `validate_extraction_label_studio.py` (controls) | Move extraction + convert logic into `src/controls/`; keep scripts as thin CLI. |
| **policies**  | Policy document parsing → passages JSON | `policy_extractor.py`, `flexible_policy_extractor.py`, `process_all_documents.py` | Move extraction into `src/policies/`; single entrypoint script. |
| **mapping**   | Retrieval (BM25 + Dense + RRF), NLI, pipeline, save per-policy | `compliance_mapping_pipeline.py`, `quick_start_compliance.py`, `split_mappings_by_policy.py` | Move pipeline + retrieval into `src/mapping/`; `quick_start_compliance.py` imports and runs. |
| **annotation**| Label Studio: generate tasks, export annotated JSON | `annotate_mappings_label_studio.py`, `validate_extraction_label_studio.py` (export) | Move generate/export logic into `src/annotation/`; scripts as CLI. |
| **pipeline**  | Regulation parsing (PDF → structured controls) | Used by control extractors | Keep as-is or merge into `controls` when refactoring. |
| **utils**     | Shared (e.g. Label Studio helpers) | Used by various scripts | Add any shared IO/helpers here. |

---

## Implementation order (step-by-step)

1. **No code move yet** – Current entrypoints stay at root and work as today. Only docs and module stubs are in place.
2. **Phase 1 – Mapping** – Move `compliance_mapping_pipeline.py` (and retrieval/NLI) into `src/mapping/`; `quick_start_compliance.py` becomes a short script that imports and runs. Test end-to-end.
3. **Phase 2 – Controls** – Move control extraction and Label Studio convert into `src/controls/`; point root scripts to imports.
4. **Phase 3 – Policies** – Move policy extraction into `src/policies/`; single or few entrypoints.
5. **Phase 4 – Annotation** – Move task generate/export into `src/annotation/`; keep Label Studio XML/config in `data/03_label_studio_input/`.
6. **Phase 5 (optional)** – RePASs-style metrics, XRefRAG/multi-framework (see `docs/claude_analysis.md`).

---

## Data flow

```
data/01_raw/ (PDF, DOCX)
    → [controls]  → data/02_processed/uae_ia_controls_*.json
    → [policies]  → data/02_processed/policies/all_policies_for_mapping.json
    → [mapping]   → data/06_compliance_mappings/mappings.json, by_policy/*.json
    → [annotation]→ data/03_label_studio_input/*.json → Label Studio → mappings_annotated.json
```

---

## Where to add new code

- **New regulation/framework** → `src/controls/` (or `src/pipeline/` if it’s parsing-only).
- **New policy format/source** → `src/policies/`.
- **Retrieval/NLI/RePASs changes** → `src/mapping/`.
- **Label Studio config or export format** → `src/annotation/` + `data/03_label_studio_input/`.
- **Shared helpers** → `src/utils/`.
