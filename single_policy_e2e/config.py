"""
Paths and policy selection for single-policy end-to-end run.
All paths are relative to the repo root (parent of this folder).
"""
import os
from pathlib import Path

# Repo root (parent of single_policy_e2e)
ROOT = Path(__file__).resolve().parent.parent

# One policy document (list of passages: id, text, section, metadata)
POLICY_JSON = os.environ.get(
    "POLICY_JSON",
    str(ROOT / "data/02_processed/policies/Asset Management Policy 6_corrected.json"),
)

# UAE IA controls (structured JSON)
CONTROLS_JSON = os.environ.get(
    "CONTROLS_JSON",
    str(ROOT / "data/02_processed/uae_ia_controls_clean.json"),
)

# Full golden dataset; we filter to this policy's rows for evaluation
GOLDEN_JSON = os.environ.get(
    "GOLDEN_JSON",
    str(ROOT / "data/07_golden_mapping/golden_mapping_dataset.json"),
)

# Outputs (all under single_policy_e2e/output/)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
