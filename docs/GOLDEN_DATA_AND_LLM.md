# Golden data for policy–control mapping: LLMs and accuracy

**Short answer:** No LLM can guarantee 100% accurate control→policy mappings. For **golden data** (ground truth for evaluation or training), the only way to get high accuracy is **human verification**. Use an LLM to **draft** mappings, then **humans review and correct** in Label Studio; the exported annotations are your golden set.

---

## Why no LLM gives “100% accurate” golden data

- **Hallucination and drift:** Models can invent or mis-assign control IDs, or be inconsistent across similar passages.
- **No ground truth at inference time:** The model was not trained on your exact UAE IA / ADHICS / ISO 27001 ↔ policy phrasing, so it can miss or over-match.
- **Ambiguity:** One passage often relates to several controls; “exact” mapping is partly subjective and needs domain judgment.

So “100% accurate” golden data is produced by **humans** (or humans + tight LLM-assisted workflow), not by an LLM alone.

---

## Recommended workflow: LLM draft → human review = golden data

| Step | Who/What | Output |
|------|----------|--------|
| 1. Draft | Strong LLM (or your retrieval+NLI pipeline) | Proposed control IDs per policy passage |
| 2. Import | Script | Label Studio tasks (passage + proposed controls) |
| 3. Review & correct | Human annotators | Accepted/edited control IDs per passage |
| 4. Export | Label Studio → JSON | **Golden data** (human-verified mappings) |

Golden data = **exported Label Studio annotations after human review**, not raw LLM output.

---

## Which LLM to use for the **draft** (best consistency, not “100%”)

Use a model that is strong at **instruction following**, **structured output**, and **long context** (full policy passages). These are good candidates for generating draft mappings that are then reviewed by humans.

| Model | Use for drafting | Notes |
|-------|-------------------|--------|
| **Claude 3.5 Sonnet / Opus** (Anthropic) | ✅ Yes | Strong at long documents, nuanced reasoning, and JSON; good for policy text. |
| **GPT-4o / GPT-4 Turbo** (OpenAI) | ✅ Yes | Reliable structured output and instruction following; good for control IDs and short justifications. |
| **Gemini 1.5 Pro** (Google) | ✅ Yes | Very long context; useful if you send whole policies or many passages at once. |
| **Llama 3.1 70B / 405B** (open-weight) | ✅ Possible | Can be fine-tuned on your mappings later; out-of-the-box less consistent than the above. |

For **golden data**, the main lever is **human review**, not which specific LLM you use for the draft. Pick one of the top-tier models above and standardize on it for reproducibility.

---

## Practical setup for golden data in this repo

1. **Generate tasks** (passages + optional draft controls):
   ```bash
   python3 generate_policy_controls_tasks.py \
     --policies data/02_processed/policies/all_policies_for_mapping.json \
     --per-policy --output-dir data/03_label_studio_input/policy_controls_by_policy
   ```
2. **(Optional)** Run an LLM or your retrieval+NLI pipeline to **propose** UAE IA / ADHICS / ISO 27001 control IDs per passage; merge those into the task JSON as pre-annotations or a “suggested” field so annotators can accept or change them.
3. **Label Studio:** Import tasks, use the Taxonomy dropdowns (`annotate_policy_controls.xml`) so annotators select the **exact** control IDs. Annotators correct any wrong or missing IDs.
4. **Export** from Label Studio → JSON. That exported file is your **golden dataset** for:
   - Evaluating automated mapping (retrieval + NLI or another LLM)
   - Training or fine-tuning a model later

---

## Summary

- **100% accurate golden data** → only from **human-verified** labels (e.g. in Label Studio).
- **LLM role** → **draft** mappings to speed up annotation; do **not** treat raw LLM output as golden data.
- **Model choice** → Claude 3.5 Sonnet/Opus, GPT-4o, or Gemini 1.5 Pro are good for drafting; keep one fixed for reproducibility.
- **Exact mapping** → Use **Taxonomy** in Label Studio with your control ID lists so annotators pick **exact** UAE IA / ADHICS / ISO 27001 IDs; export = golden set.
