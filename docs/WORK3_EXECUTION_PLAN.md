# Work 3 Execution Plan — Compliance Paper + Fundamentals

**Goal:** submit the AutoComply-IoMT paper to *Computers & Security*, and come out of it
genuinely strong in ML/LLM fundamentals — because the paper's pipeline IS the curriculum.

**Status at start:** Work 1 revision submitted ✅. UAE IA prototype working (Llama posF1=0.68).
Rejection feedback in hand: "no experimental proof" — so this plan is built around producing
that proof.

**Time budget:** ~12 hours/week, split every week the same way:

```
LEARN  4 hrs — fundamentals (study plan phases)
BUILD  6 hrs — the healthcare rerun (experiments)
WRITE  2 hrs — the paper, same week as the work it describes
```

The three lanes reinforce each other: what you learn Monday explains what you build
Wednesday, and what you build Wednesday goes into the paper Friday.

---

## The Big Picture — 16 Weeks

| Month | LEARN | BUILD | WRITE |
|---|---|---|---|
| **1** | ML basics (Phase 0) | Adjudicate labels · HIPAA atoms | Introduction + Related Work |
| **2** | Transformers + NLI (Phases 2–3) | Incident records · kappa study | Methodology |
| **3** | Fine-tuning (Phases 4–5) | Healthcare rerun · both judges | Results |
| **4** | Evaluation (Phase 7) | Error analysis · final tables | Discussion · polish · **submit** |

---

## Month 1 — Foundations + Data

### Week 1
- **LEARN:** Phase 0A maths (3 days' worth): vectors, dot product, cosine similarity,
  gradients. Watch 3Blue1Brown Neural Networks (4 videos, ~1 hr). Start Andrew Ng course.
- **BUILD:** Adjudicate the 70 disputed labels in `data/10_judge/adjudication.csv`
  (2–3 focused hours — highest-leverage act in this plan). Run
  `python3 scripts/adjudicate.py --apply`, then re-grade both judges against clean labels.
- **WRITE:** Nothing yet. Set up the paper skeleton: LaTeX or Word template for
  *Computers & Security*, section headings only.

### Week 2
- **LEARN:** Phase 0B — supervised vs self-supervised, overfitting, train/val/test,
  class imbalance, data leakage. **Hands-on:** train a logistic regression and a random
  forest in scikit-learn on any toy dataset.
- **BUILD:** Download CICIoMT2024 (`cicresearch.ca/IOTDataset/CICIoMT2024/`).
  Extract the 18 attack types from `Attack_info.csv`. Read 2–3 sample captures.
- **WRITE:** Introduction, first draft (1 page). The story: IoMT incident records contain
  compliance evidence; verification is manual; cloud APIs are prohibited for PHI.

### Week 3
- **LEARN:** Phase 0C — neurons, activations, loss, backprop, gradient descent.
  **Hands-on:** Karpathy micrograd video, typing along.
- **BUILD:** Start HIPAA atomisation. §164.312 Technical Safeguards, using the five
  splitting rules (below). Target: first 20 controls atomised.
- **WRITE:** Related Work skim: collect 15–20 papers. Search: "compliance mapping NLP",
  "regulatory requirement decomposition", "entailment-based compliance", ContractNLI.

### Week 4
- **LEARN:** Phase 1 — tokenization, BM25, sparse vs dense retrieval. HF NLP course ch. 1–2.
- **BUILD:** Finish HIPAA atomisation (~40 controls, ~120 atoms) + ISO 27799 clauses 9–12.
  Output format: same JSON schema as `data/02_processed/uae_ia_controls_structured.json`.
- **WRITE:** Related Work, first draft (1.5 pages). Position against: holistic classifiers,
  cloud-LLM approaches, ContractNLI.

**Month 1 exit:** clean adjudicated labels · CICIoMT2024 in hand · ~120 HIPAA atoms ·
Intro + Related Work drafted.

---

## Month 2 — The Transformer + The Annotation Study

### Week 5
- **LEARN:** Phase 2 begins — attention, Q/K/V, the Illustrated Transformer (read twice).
- **BUILD:** Write the first 10 IoMT incident records (DDoS scenarios). Use the schema:
  Incident ID / Timestamp / Affected Device / Protocol / Observed Behaviour /
  Actions Taken / Outcome / ATT&CK Technique / HIPAA Safeguard.
- **WRITE:** Methodology §1: the atomisation protocol (rules below) — this section can be
  written completely before any experiment runs.

### Week 6
- **LEARN:** Phase 2 continues — **Karpathy "Let's build GPT" (~2 hrs, typing along).**
  Non-negotiable. This is the week fundamentals click.
- **BUILD:** Write incident records 11–30 (recon, MQTT attacks, spoofing scenarios).
- **WRITE:** Methodology §2: pipeline description (retrieval → per-atom NLI → rollup).
  Reuse the architecture description you already have.

### Week 7
- **LEARN:** Phase 3 — BERT, NLI, bi- vs cross-encoders. Then **reread
  `scripts/nli_judge.py`** — it will read differently now.
- **BUILD:** Finish records 31–50. **Recruit the second annotator this week at the latest**
  (labmate / colleague — ~20 hrs of their time total).
- **WRITE:** Methodology §3: annotation protocol + kappa design.

### Week 8
- **LEARN:** Read ContractNLI (Koreeda & Manning 2021) properly — the closest published
  work; your Related Work must position against it convincingly.
- **BUILD:** **Kappa study round 1:** both annotators independently atomise the same 5
  clauses. Compute kappa. If κ < 0.75 → add a clarifying sub-rule, re-annotate.
  Then both annotate the 50 incident records independently.
- **WRITE:** Nothing new — buffer week for Methodology cleanup.

**Month 2 exit:** 50 incident records · two independent annotation sets · pre-adjudication
kappa measured · Methodology drafted.

---

## Month 3 — Fine-tuning + The Experiments

### Week 9
- **LEARN:** Phase 4 — decoders, sampling, instruction tuning. Reread
  `scripts/finetune_atom_compliance.py`.
- **BUILD:** Adjudicate annotator disagreements by discussion → final human answer key.
  Record post-adjudication agreement (expect > 0.75).
- **WRITE:** Results §1: the annotation study itself IS a result — report both kappas.
  The gap between them demonstrates annotation difficulty (motivates automation).

### Week 10
- **LEARN:** Phase 5 — LoRA, QLoRA, quantization, learning rates. You've already run this
  once; now understand every line of the config.
- **BUILD:** **The healthcare rerun.** Swap inputs, run in order:
  `build_gold_matrix.py` → `build_atom_dataset.py` → `nli_judge.py` → `grade_judge.py`.
  This gives the zero-shot NLI number on healthcare data.
- **WRITE:** Results §2 skeleton: the comparison table (rows: NLI zero-shot, Llama
  fine-tuned; columns: P, R, F1, kappa).

### Week 11
- **LEARN:** Phase 5 continues — read the LoRA paper (Hu et al. 2021).
- **BUILD:** Llama fine-tune on healthcare atoms (Lightning.ai, same workflow as before:
  push → train → pull). Grade against the human key.
- **WRITE:** Fill the Results table with real numbers. Write the comparison narrative.

### Week 12
- **LEARN:** Phase 7 — precision/recall/F1, kappa, calibration, why accuracy lies.
  Reread `scripts/grade_judge.py`.
- **BUILD:** Error analysis: read every disagreement between best judge and human key.
  Categorise: label issue / single-passage limitation / genuine model error.
- **WRITE:** Results §3: error analysis table + the NA-bias finding section
  (your sleeper contribution — give it space).

**Month 3 exit:** all experiments done · full Results section with real numbers.

---

## Month 4 — Analysis, Polish, Submit

### Week 13
- **LEARN:** Phase 7 continues — LLM-as-judge biases, circular evaluation (you lived it;
  now read about it).
- **BUILD:** Any experiment gaps found while writing (there are always 1–2 reruns).
- **WRITE:** Discussion: limitations (50 records, single institution's style, distributed
  evidence), the data-sovereignty argument, commercial implications (one paragraph).

### Week 14
- **LEARN:** Light week — revisit weak spots from the question bank
  (`docs/LLM_STUDY_PLAN.md`, Concept Question Bank).
- **BUILD:** Freeze the repo state; tag the commit the paper describes.
- **WRITE:** Abstract + Conclusion. Use the abstract seed in
  `docs/RESEARCH_DIRECTION_PHD.md` — update with real numbers, remove blockchain sentence.

### Week 15
- **WRITE-heavy week:** full read-through, figures, citation cleanup, formatting to
  *Computers & Security* requirements. Send to supervisor.

### Week 16
- Supervisor feedback → revisions → **submit**.
- Then: two weeks of pure LEARN catch-up (Phases 6, 8) before starting Work 2.

---

## Reference: The Atomisation Protocol (from the handoff doc — use verbatim)

Atom = smallest indivisible obligation: **one actor + one action + one object**.

| Rule | Split condition |
|---|---|
| 1 | Two verbs joined by "and" → two atoms |
| 2 | Two objects protected → two atoms |
| 3 | If/when conditions → keep attached to their atom |
| 4 | Definitions → never split |
| 5 | Unless/except exceptions → separate atom |

Quality check per atom: exactly one action · exactly one object · answerable Yes/No from
incident text alone · removing it would leave a compliance gap undetected.

Gate: two annotators atomise the same 5 clauses independently → κ > 0.75 before proceeding.

---

## The Paper's Three Claims (protect these)

1. **Per-atom decomposition of HIPAA controls** — first for a healthcare standard;
   reduces compliance verification to binary entailment a local model can do.
2. **NA-bias annotation finding** — human annotators systematically over-label
   "Not Addressed"; existing compliance benchmarks inherit this bias.
3. **Local-model viability under data sovereignty** — HIPAA prohibits routing PHI through
   cloud APIs; the constraint IS the framing, not a limitation.

---

## Rules for the Whole Plan

1. **Never skip the WRITE lane.** 2 hrs/week while the work is fresh beats 40 hrs of
   archaeology in month 4.
2. **Human labels before model claims.** Every published number is graded against the
   human-adjudicated key — nothing else.
3. **One blocker at a time.** If stuck > 2 days on anything, write it down, switch lanes,
   raise it with the supervisor that week.
4. **The product waits.** No autocomply repo, no RBI atoms, no Vikas features until this
   submits. Work 2 learning starts only after submission.
5. **After each LEARN phase, reread the matching script** (mapping in
   `docs/LLM_STUDY_PLAN.md`, "The One Rule"). Concepts land differently in code you own.

---

## Start Today (Week 1, Day 1)

1. Open `data/10_judge/adjudication.csv` — do the first 20 HIGH-priority rows
2. Watch 3Blue1Brown video 1 (~15 min)
3. Message a potential second annotator — their calendar is your longest lead time
