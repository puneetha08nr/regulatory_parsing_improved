# Single-Policy Compliance Mapping (End-to-End)

Minimal project: **one policy document → mapping → evaluation** using the annotated golden set. No combined policies; easy to track and reproduce.

## What it does

1. **Load** one policy (e.g. `Asset Management Policy 6_corrected.json`) and UAE IA controls.
2. **Map** controls to passages (RegNLP-style: BM25 + Dense retrieval → NLI or reranker).
3. **Filter** the golden dataset to that policy’s annotations only.
4. **Evaluate** pipeline mappings vs. filtered golden (Precision, Recall, F1).

All inputs/outputs are under this folder or pointed from `config.py`.

## Approach (from `docs/`)

- **RegNLP-style**: One corpus per policy doc; retrieval (BM25 + Dense + RRF) then NLI/reranker on candidates.
- **Golden set**: Human-annotated (control, passage) pairs; use `corrected_control_id` when present.
- **Evaluation**: Pair-level P/R/F1 vs. golden positives/negatives.

See `docs/MAPPING_STRATEGY_AND_REGNLP.md` and `docs/PIPELINE_WORKFLOW_AND_ROADMAP.md`.

## Inputs (shared with main repo)

| Input | Default path | Description |
|-------|--------------|-------------|
| Policy (one doc) | `../data/02_processed/policies/Asset Management Policy 6_corrected.json` | List of passages (`id`, `text`, `section`, `metadata`) |
| Controls | `../data/02_processed/uae_ia_controls_clean.json` | UAE IA controls (structured JSON) |
| Golden set | `../data/07_golden_mapping/golden_mapping_dataset.json` | Annotated (control, passage) pairs; filtered to this policy for evaluation |

## Outputs (all under `output/`)

| File | Description |
|------|-------------|
| `output/mappings.json` | Pipeline predictions (control → passage, status, score) for this policy only |
| `output/golden_filtered.json` | Golden rows for this policy (for evaluation) |
| `output/evaluation.json` | P/R/F1 and TP/FP/FN counts |
| `output/retrieval_log.json` | Retrieved passage IDs per control (optional; for Recall@K) |

## Run (one command)

From **project root** (`regulatory_parsing_improved/`):

```bash
python3 single_policy_e2e/run.py
```

Or from this folder:

```bash
python3 run.py
```

To use another policy, set in `config.py` or override:

```bash
POLICY_JSON="../data/02_processed/policies/Third-Party Security Policy v2.0_corrected.json" python3 run.py
```

## Requirements

- Same as main pipeline: Python 3.8+, `rank_bm25`, `transformers`, `torch` (optional: `sentence-transformers` for dense retrieval). See root `requirements.txt`.
- Run from repo root so imports (`compliance_mapping_pipeline`, `scripts.evaluate_pipeline`) resolve.

## Layout

```
single_policy_e2e/
├── README.md           # This file
├── config.py           # Paths and policy selection
├── run.py              # End-to-end: load → map → evaluate
├── evaluate.py         # Evaluation only (filter golden + mappings by policy)
└── output/             # Created by run: mappings, golden_filtered, evaluation
```
