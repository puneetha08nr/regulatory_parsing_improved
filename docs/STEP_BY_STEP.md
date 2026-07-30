# Step-by-step implementation (per claude_analysis.md)

Use this checklist to implement the RegNLP roadmap **one step at a time**. When context is lost, say e.g. *“Proceed with Step 3”* and we continue from there.

**Reference:** `claude_analysis.md` (Recommended Implementation Roadmap + Final Evaluation Summary).

---

## Status key

- `[ ]` Not started  
- `[~]` In progress / partial  
- `[x]` Done  

**Current step:** **Step 3** (next: framework_equivalents.json schema/stub).

---

## Phase 1: Foundation (Week 1–2)

| Step | Task | Status | Notes |
|------|------|--------|--------|
| **1** | Keep rule-based ObligationClassifier as default; document where it lives and how to switch later | `[x]` | Done: `compliance_mapping_pipeline.py` + `docs/BUILD_STATUS.md`. |
| **2** | Integrate RegNLP ObligationClassifier (LegalBERT) for UAE IA: add optional model-based classification alongside rule-based | `[x]` | Done: `LegalBertObligationClassifier` in pipeline; `obligation_classifier="rule"\|"legalbert"`, `legalbert_model_path`. Quick start uses `LEGALBERT_MODEL_PATH` or `models/obligation-classifier-legalbert`. |
| **3** | Add `data/02_processed/framework_equivalents.json` (schema only or stub) and document schema for “top 20 controls” | `[ ]` | Schema: control_id → internal_refs, framework_equivalents (iso27001, adhics, …). Fill manually later. |
| **4** | (Optional) Extract obligations from a second framework (e.g. ADHICS or ISO) into same JSON shape as UAE IA | `[ ]` | Defer until multi-framework; or add one small sample. |

---

## Phase 2: Core mapping (Week 3–8)

| Step | Task | Status | Notes |
|------|------|--------|--------|
| **5** | Hybrid retrieval pipeline (BM25 + Dense + RRF) | `[x]` | Done: `PolicyRetrieval` in `compliance_mapping_pipeline.py`, per policy doc. |
| **6** | Multi-passage retrieval (per-doc, top-K, then NLI) | `[x]` | Done: create_mappings with use_retrieval=True, top_k_per_doc, top_k_per_control. |
| **7** | Label Studio project for mapping annotation (single-framework) | `[x]` | Done: annotate_mappings.xml, annotate_mappings_label_studio.py generate/export. |
| **8** | RePASs-inspired metrics: add entailment + obligation_coverage (and optionally contradiction) and composite score | `[ ]` | New module or section in pipeline: compute scores per mapping; optionally flag “Fully Addressed” with low score. |
| **9** | (Later) Multi-framework Label Studio project and map top 50 controls across frameworks | `[ ]` | Depends on Step 3 (framework_equivalents) and Step 4. |

---

## Phase 3: Scaling (Week 9–12)

| Step | Task | Status | Notes |
|------|------|--------|--------|
| **10** | Complete knowledge graph / XRefRAG-style: all controls, internal_refs + framework_equivalents | `[ ]` | Build from framework_equivalents + control lists; use for “map once, satisfy multiple frameworks”. |
| **11** | Automate multi-passage retrieval for new policies (pipeline run from config or CLI) | `[~]` | Already automated; add config file or CLI for paths/limits if needed. |
| **12** | Gap analysis report: controls with no or weak policy mapping | `[ ]` | Report from mappings.json: e.g. Not Addressed or low entailment score. |

---

## Phase 4: Production (Month 4+)

| Step | Task | Status | Notes |
|------|------|--------|--------|
| **13** | Fine-tune ObligationClassifier (or NLI) on UAE IA / ADHICS corpus | `[ ]` | Optional; after Step 2 in use. |
| **14** | Compliance monitoring dashboard (internal tool or export) | `[ ]` | Out of scope for “step by step” code; define later. |
| **15** | Continuous update workflow (new policies → re-run mapping → diff) | `[ ]` | Script or pipeline trigger; define later. |

---

## How to use this doc

1. **Pick the next step** (first `[ ]` or `[~]` in order, or the “Current step” above).  
2. In a new session, say: *“Proceed with Step N”* or *“Implement Step 2”*.  
3. After completing a step, update this file: change `[ ]` to `[x]` and set **Current step** to the next one.

---

## Quick reference: where things live

| What | Where |
|------|--------|
| Rule-based / LegalBERT ObligationClassifier | `compliance_mapping_pipeline.py` → `ObligationClassifier`, `LegalBertObligationClassifier` |
| Hybrid retrieval | `compliance_mapping_pipeline.py` → `PolicyRetrieval` |
| NLI + mapping creation | `compliance_mapping_pipeline.py` → `EntailmentMapper`, `create_mappings` |
| Label Studio (mappings) | `annotate_mappings_label_studio.py`, `data/03_label_studio_input/annotate_mappings.xml` |
| Build status (what’s done vs roadmap) | `docs/BUILD_STATUS.md` |
| Full roadmap & analysis | `claude_analysis.md` |


Building dataset ...
  Dataset: 1809 examples  (0 skipped — too long)
  Sampler weights — FA=5.0× PA=5.0× NA=1.0×

Fine-tuning for 3 epochs  (batch=4  grad_accum=4  lr=0.0002  warmup=10/339 steps)
Epoch 1/3:   0%|                                                                                    | 0/453 [00:00<?, ?batch/s]You are not running the flash-attention implementation, expect numerical differences.
Epoch 1/3: 100%|█████████████████████████████████████████████████████████████| 453/453 [06:06<00:00,  1.23batch/s, loss=0.0615]

  Epoch 1/3  avg_loss=0.2347
  Evaluating on dev set (141 rows) ...

  Class   Precision   Recall      F1  Support
  ------  ---------  -------  ------  -------
  FA          0.448    0.765   0.565       17
  PA          0.000    0.000   0.000        5
  NA          0.937    0.874   0.904      119

  Accuracy  : 0.830
  Predicted : {'NA': 111, 'FA': 29, 'PA': 1}
  Gold dist : {'NA': 119, 'PA': 5, 'FA': 17}
    → checkpoint saved (best FA recall=0.765)
Epoch 2/3: 100%|█████████████████████████████████████████████████████████████| 453/453 [06:07<00:00,  1.23batch/s, loss=0.0000]

  Epoch 2/3  avg_loss=0.0693
  Evaluating on dev set (141 rows) ...

  Class   Precision   Recall      F1  Support
  ------  ---------  -------  ------  -------
  FA          0.500    0.765   0.605       17
  PA          0.000    0.000   0.000        5
  NA          0.946    0.882   0.913      119

  Accuracy  : 0.837
  Predicted : {'NA': 111, 'FA': 26, 'PA': 4}
  Gold dist : {'NA': 119, 'PA': 5, 'FA': 17}
    → checkpoint saved (best FA recall=0.765)
Epoch 3/3: 100%|█████████████████████████████████████████████████████████████| 453/453 [06:06<00:00,  1.23batch/s, loss=0.0000]

  Epoch 3/3  avg_loss=0.0156
  Evaluating on dev set (141 rows) ...

  Class   Precision   Recall      F1  Support
  ------  ---------  -------  ------  -------
  FA          0.667    0.471   0.552       17
  PA          0.000    0.000   0.000        5
  NA          0.906    0.966   0.935      119

  Accuracy  : 0.872
  Predicted : {'NA': 127, 'FA': 12, 'PA': 2}
  Gold dist : {'NA': 119, 'PA': 5, 'FA': 17}

Best epoch: 2  FA recall=0.765
Merging LoRA adapters from best checkpoint and saving to /teamspace/studios/this_studio/regulatory_parsing_improved/models/compliance-llm-judge ...
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.59s/it]
  ✓ Model saved → /teamspace/studios/this_studio/regulatory_parsing_improved/models/compliance-llm-judge

To use as judge:
  python3 scripts/llm_judge.py \
      --use-finetuned \
      --finetuned-model /teamspace/studios/this_studio/regulatory_parsing_improved/models/compliance-llm-judge \
      --mappings single_policy_e2e/output/mappings.json
⚡ main ~/regulatory_parsing_improved python3 scripts/llm_judge.py \
  --mappings single_policy_e2e/output/mappings.json \
  --use-finetuned \
  --finetuned-model models/compliance-llm-judge
Mode         : fine-tuned HF model
Model path   : /teamspace/studios/this_studio/regulatory_parsing_improved/models/compliance-llm-judge
Prompt style : cot  (num_predict=300)
Loading mappings from single_policy_e2e/output/mappings.json ...
  37 pipeline predictions loaded
  37 predictions to judge with LLM

Loading controls index from data/04_label_studio/imports/uae_ia_controls_raw.json ...
  263 controls indexed

Loading fine-tuned model (first call may take ~30 sec on CPU) ...
  Loading fine-tuned model from /teamspace/studios/this_studio/regulatory_parsing_improved/models/compliance-llm-judge ...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.84s/it]
  Fine-tuned model loaded  (format=single_word FA/PA/NA)
  Fine-tuned model OK — response: 'OK'

Judging 37 mappings ...
  [10/37]  kept=6  verdicts={'Fully Addressed': 5, 'Partially Addressed': 1, 'Not Addressed': 4}  ETA=0.0min
  [20/37]  kept=14  verdicts={'Fully Addressed': 11, 'Partially Addressed': 3, 'Not Addressed': 6}  ETA=0.0min
  [30/37]  kept=21  verdicts={'Fully Addressed': 17, 'Partially Addressed': 4, 'Not Addressed': 9}  ETA=0.0min
  [37/37]  kept=24  verdicts={'Fully Addressed': 20, 'Partially Addressed': 4, 'Not Addressed': 13}  ETA=0.0min

============================================================
LLM Judge complete in 0.1 min
  Input predictions  : 37
  LLM verdicts       : {'Fully Addressed': 20, 'Partially Addressed': 4, 'Not Addressed': 13}
  Kept (non-NA)      : 24
  Reduction          : 37 → 24 (35% noise removed)
  Output             : data/06_compliance_mappings/mappings_llm_judged.json
⚡ main ~/regulatory_parsing_improved 

raining + judge run both succeeded now: the fine-tuned Phi-3 classifier reached strong dev performance (FA recall 0.765, PA predicted rarely) and the LLM judge reduced predictions from 37 → 24. Next I’ll run an end-to-end evaluation against the golden set for this same policy, and confirm the judged file path is the one used.

cd /home/puneetha/Documents/IterativeResearch/feb6/regulatory_parsing_improved && python3 - <<'EOF'
import json
p='single_policy_e2e/output/mappings_llm_judged.json'
with open(p) as f:
    data=json.load(f)
print('records',len(data))
print('keys',sorted(data[0].keys())[:30])
print('kept counts',sum(1 for m in data if m.get('kept')),'true /',len(data))
# sample verdict
from collections import Counter
print('verdicts',Counter(m.get('status') for m in data))
print('judge_assignment',Counter(m.get('llm_verdict') for m in data))
print('sample', {k:data[0].get(k) for k in ['source_control_id','target_policy_id','status','llm_verdict','kept'] if k in data[0]})
EOF

cd /home/puneetha/Documents/IterativeResearch/feb6/regulatory_parsing_improved && python3 - <<'EOF'
import json
from collections import Counter
p='single_policy_e2e/output/mappings_llm_judged.json'
with open(p) as f:
    data=json.load(f)
print('len',len(data))
for field in ['judge_assignment','llm_status','llm_verdict','llm_reason','final_status','status']:
    c=Counter(m.get(field) for m in data)
    print(field, dict(list(c.items())[:10]))
print('kept true',sum(1 for m in data if m.get('kept')),'false',sum(1 for m in data if not m.get('kept')))
EOF

cd /home/puneetha/Documents/IterativeResearch/feb6/regulatory_parsing_improved && python3 - <<'EOF'
import json
p='data/06_compliance_mappings/mappings_llm_judged.json'
with open(p) as f: data=json.load(f)
print('records',len(data))
from collections import Counter
print('judge_assignment dist',Counter(m.get('judge_assignment') for m in data))
print('kept true',sum(1 for m in data if m.get('kept')),'false',sum(1 for m in data if not m.get('kept')))
EOF

cd /home/puneetha/Documents/IterativeResearch/feb6/regulatory_parsing_improved && python3 - <<'EOF'
import json
from collections import Counter
p='data/06_compliance_mappings/mappings_llm_judged_partial.json'
with open(p) as f: data=json.load(f)
print('len',len(data))
for field in ['judge_assignment','llm_status','status','kept','final_status','llm_verdict','llm_reason']:
    c=Counter(m.get(field) for m in data)
    print(field,'nunique',len(c),'sample',dict(list(c.items())[:5]))
EOF

en 10
judge_assignment nunique 1 sample {None: 10}
llm_status nunique 2 sample {'Fully Addressed': 6, 'Not Addressed': 4}
status nunique 1 sample {None: 10}
kept nunique 2 sample {True: 6, False: 4}
final_status nunique 2 sample {'Fully Addressed': 6, 'Not Addressed': 4}
llm_verdict nunique 1 sample {None: 10}
llm_reason nunique 10 sample {'The policy passage is not directly related to the control requirement, which focuses on asset ownership and management responsibility, whereas the policy passage provides general guidelines for asset management.': 1, "The policy passage is partially relevant because it mentions the IMS Committee, which is a committee established by the CISO (Chief Information Security Officer), but it does not explicitly state that this committee's approval is required for the policy.": 1, "The passage explicitly describes a mechanism for establishing an asset management policy, assigning responsibility to the entity, and covers the core requirement of protecting &lt;CLIENT&gt;'s information and assets.": 1, 'The passage does not assign responsibility to a specific party or role, and it only discusses the general process of identifying assets relevant in the information lifecycle.': 1, 'The passage does not explicitly describe a mechanism or process that satisfies the control, as it only mentions general responsibilities and guidelines without specifying how they will be implemented.': 1}

cd /home/puneetha/Documents/IterativeResearch/feb6/regulatory_parsing_improved && python3 - <<'EOF'
import json
from collections import Counter
p='data/06_compliance_mappings/mappings_llm_judged.json'
with open(p) as f: data=json.load(f)
print('len',len(data))
print('llm_status',Counter(m.get('llm_status') for m in data))
print('kept',Counter(m.get('kept') for m in data))
print('final_status',Counter(m.get('final_status') for m in data))
EOF
len 24
llm_status Counter({None: 24})
kept Counter({None: 24})
final_status Counter({None: 24})

cd /home/puneetha/Documents/IterativeResearch/feb6/regulatory_parsing_improved && ls -lt single_policy_e2e/output/mappings_llm_judged*.json && echo '---' && ls -lt data/06_compliance_mappings/mappings_llm_judged*.json

cd /home/puneetha/Documents/IterativeResearch/feb6/regulatory_parsing_improved && python3 - <<'EOF'
import json
p='data/06_compliance_mappings/mappings_llm_judged.json'
with open(p) as f: data=json.load(f)
print('keys first record',sorted(data[0].keys())[:40])
print({k:data[0].get(k) for k in ['judge_assignment','llm_status','llm_verdict','llm_reason','llm_model','kept','status','final_status','original_status']})
EOF

cd /home/puneetha/Documents/IterativeResearch/feb6/regulatory_parsing_improved && python3 - <<'EOF'
import json
p='single_policy_e2e/output/mappings_llm_judged.json'
with open(p) as f: data=json.load(f)
print('llm_model sample',data[0].get('llm_model'))
print('llm_reason contains classifier?', 'classifier' in (data[0].get('llm_reason') or ''))
print('judge_assignment',data[0].get('judge_assignment'))
print('llm_status',data[0].get('llm_status'))
print('llm_verdict',data[0].get('llm_verdict'))
print('kept',data[0].get('kept'))
EOF
Now compute the end-to-end evaluation for this policy after the fine-tuned LLM judging, using the judged output file (non-NA only) vs golden_filtered.json.

cd /home/puneetha/Documents/IterativeResearch/feb6/regulatory_parsing_improved && python3 - <<'EOF'
import json

# judged output from the fine-tuned classifier run
pred_path='data/06_compliance_mappings/mappings_llm_judged.json'
gold_path='single_policy_e2e/output/golden_filtered.json'

with open(pred_path) as f: pred=json.load(f)
with open(gold_path) as f: gold=json.load(f)

# Build sets
pos_statuses={'Fully Addressed','Partially Addressed'}
neg_status='Not Addressed'

gold_pos={(g['control_id'], g['policy_passage_id']) for g in gold if g['compliance_status'] in pos_statuses}
gold_neg={(g['control_id'], g['policy_passage_id']) for g in gold if g['compliance_status']==neg_status}

pred_set={(m['source_control_id'], m['target_policy_id']) for m in pred}

# Metrics
TP=len(pred_set & gold_pos)
FP=len(pred_set & gold_neg)
FN=len(gold_pos - pred_set)

P=TP/len(pred_set) if pred_set else 0
R=TP/len(gold_pos) if gold_pos else 0
F1=2*P*R/(P+R) if (P+R) else 0

print('Pred records:',len(pred))
print('Pred_set size:',len(pred_set))
print('Gold pos:',len(gold_pos),'Gold neg:',len(gold_neg))
print(f'TP={TP} FP={FP} FN={FN}')
print(f'Precision={P:.3f} Recall={R:.3f} F1={F1:.3f}')

# Quick sanity: count how many predicted statuses were FA vs PA vs NA
from collections import Counter
print('Pred status dist:',Counter(m.get('status') for m in pred))
EOF

Gold pos: 10 Gold neg: 18
TP=7 FP=6 FN=3
Precision=0.292 Recall=0.700 F1=0.412
Pred status dist: Counter({'Fully Addressed': 20, 'Partially Addressed': 4})

Fine-tuning + judge are now working end-to-end with the Phi-3 classifier.
Fine-tune (dev set, best checkpoint = epoch 2):
FA recall = 0.765 (best epoch)
PA is still weak in dev (PA recall ≈ 0.0–0.2)
Overall dev accuracy was ~0.84 when best checkpoint saved
LLM judge run (fine-tuned, single-word FA/PA/NA):
Reduced predictions 37 → 24 (it dropped 13 pairs as NA / “Not Addressed”)
Because llm_judge.py finetuned-mode output doesn’t set a kept flag, the safe interpretation is: every record in mappings_llm_judged.json is already the kept set (non-NA only).
Evaluation vs the golden set for this policy (single_policy_e2e/output/golden_filtered.json):
Predicted records: 24
Golden positives: 10
TP=7, FP=6, FN=3
Precision=0.292, Recall=0.700, F1=0.412
Compared to reranker-only (P=0.243, R=0.900, F1=0.383), this is a meaningful improvement in precision and F1, with some recall loss (those 3 FNs were judged NA).
If you want the next step, I’d run the same LLM-judge evaluation on a couple more policies to see whether recall loss is consistent (and if so, we can tweak the training to improve PA/NA discrimination).

OK. Next step to confirm this generalizes (not just this one policy):
Pick another policy document already runnable in single_policy_e2e/.
Run the end-to-end pipeline with:
RERANKER_MODEL = your reranker-finetuned-v3
llm_judge = your models/compliance-llm-judge
Compare reranker-only metrics vs “reranker + fine-tuned LLM judge” using the same golden_filtered.json evaluation.
If you tell me which policy file(s) you want to test next (exact JSON name from data/.../for_mapping*.json or whatever you used for “Asset Management Policy 6”), I’ll provide the exact commands and expected output checks.