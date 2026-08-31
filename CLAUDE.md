# CLAUDE.md — PhD Research Context

Auto-loaded by Claude Code at session start. Read this before doing anything else.
Last updated: 2026-08-31.

---

## Who

Avinash — PhD student + technical founder. Three parallel tracks:
- **PhD** — 5-paper IoT security thesis
- **Vikas** — 45-agent AI content-ops platform (Tamil Nadu scheme outreach) — also Work 2's testbed
- **Elyts-EDGE** — OOH analytics, Bengaluru billboards

---

## The five works

| # | Topic | Status | Venue |
|---|---|---|---|
| 1 | IPFS healthcare IoT storage | ✅ Submitted (revision Jul 2026) | Computer Networks Q1 |
| 2 | Secure MCP for IoT agents | 🔧 Seeding in LEARN lane | IEEE TDSC / TIFS |
| 3 | AutoComply-IoMT per-atom compliance | 🔧 **Active build** | Computers & Security |
| 4 | Trusted edge intelligence | ⬜ Not started | IEEE IoTJ / JSA |
| 5 | Adaptive trust orchestration | ⬜ Not started | TDSC / FGCS |

---

## What Work 3 is

**One-sentence definition:**
Given an IoMT incident record and a regulatory control, the system decomposes the control into atomic obligations and uses a local language model to determine — for each atom — whether the incident record provides evidence that the obligation was met. The result is a per-atom binary coverage matrix that rolls up into a compliance verdict (Fully / Partially / Not Addressed) without routing protected health information through any external API.

**Why this is novel (NotebookLM + manual verification, 2026-08-30):**
- Closest precedents: GraphCompliance (CU decomposition), ContractNLI (clause-level NLI), PrivComp-KG, Chattoraj & Joshi 2025 (local Mistral for FDA compliance)
- Novel combination: per-atom binary coverage matrix applied to IoMT *runtime incident logs* under active PHI sovereignty constraint
- Architecture type: neuro-symbolic hybrid — atom checklist is symbolic (deterministic), per-atom verdict is neural (local NLI / fine-tuned Llama)
- Key differentiators: no graph construction overhead (vs GraphCompliance), no keyword confusion (vs end-to-end LLM), works on qualitative incident text (vs code-gen methods), no external API (vs cloud RAG)
- **ContractNLI must be positioned against explicitly in Related Work** — clause-level on contracts vs atom-level on incident records under healthcare sovereignty

**Three claims to protect — do not weaken these:**
1. Per-atom decomposition of HIPAA controls — first for a healthcare standard
2. NA-bias annotation finding — human annotators systematically over-label "Not Addressed"
3. Local-model viability under data sovereignty — HIPAA prohibits routing PHI through cloud APIs; this is the framing, not a limitation

---

## Current pipeline state

### What exists

| Artefact | Location | State |
|---|---|---|
| UAE IA evaluation set | `data/12_atoms/pairs_eval.json` | 162 pairs, 102 train / 60 test |
| Llama-1B QLoRA adapter | `atom_judge_adapter/` | Best result: posF1=0.687, P=0.793, R=0.605 |
| Grade report | `data/10_judge/grade_report.json` | κ=0.233 vs unadjudicated labels |
| Disputed labels | `data/10_judge/adjudication.csv` | 70 rows, NOT yet adjudicated |
| HIPAA §164.312 atoms | `data/13_healthcare/hipaa_312_atoms.json` | 27 atoms, 5 controls ✅ |
| ISO 27002 cl.9–12 atoms | `data/13_healthcare/iso27002_912_atoms.json` | 49 atoms, 15 controls ✅ |
| Incident record schema | `data/13_healthcare/incidents/SCHEMA.md` | ✅ written |
| Incident templates | `data/13_healthcare/incidents/templates.json` | 19 attack types, 54 target records ✅ |
| Incident records | `data/13_healthcare/incidents/records.json` | **50/50 records complete ✅** — 19 attack types, 17 contained / 33 breach/partial |

**Total atoms: 76 across 20 controls** (HIPAA 27 + ISO 27002 49)

### Pipeline execution order
```
build_gold_matrix.py → build_atom_dataset.py → nli_judge.py → grade_judge.py
Fine-tune: finetune_atom_compliance.py → Lightning.ai (push → train → pull)
Adjudication: scripts/adjudicate.py --apply
```

### Known metrics (do not recompute — verified 2026-07-30)
```
posF1 = 0.687   P = 0.793   R = 0.605   (Llama-1B QLoRA, UAE IA test set)
agreement = 0.543   kappa = 0.233   (vs unadjudicated labels — will improve after adjudication)
```

---

## What has been built this session (2026-08-30)

### Atom files
- **`data/13_healthcare/hipaa_312_atoms.json`** — §164.312 atomised from official law.cornell.edu text. 5 standards, 27 atoms. Each atom has `key`, `rule`, `claim`, `source_text`.
- **`data/13_healthcare/iso27002_912_atoms.json`** — ISO 27002:2013 cl. 9–12, IoMT-verifiable controls only (filtered out HR/policy/clear-desk controls). 15 controls, 49 atoms. Citable as "ISO/IEC 27002:2013 as adopted by ISO 27799:2016" in methodology.
- Both files use same schema as `data/02_processed/uae_ia_controls_structured.json`.

### Incident record templates
- **`data/13_healthcare/incidents/SCHEMA.md`** — field definitions + writing rules + quality bar
- **`data/13_healthcare/incidents/templates.json`** — 19 CICIoMT2024 attack types, each with: scenario seed, three outcome variants (contained/partial/breached), atoms likely evidenced, device suggestions

### Incident records (completed 2026-08-31)
- **`data/13_healthcare/incidents/records.json`** — 50 records complete
  - 19 attack types (all CICIoMT2024 categories covered)
  - 17 contained / 33 breach-or-partial (~34%/66% — slightly breach-heavy for NA-bias demonstration)
  - Statistics grounded in real CICIoMT2024 CSV flow counts, rates, and payload sizes
  - Device variety: infusion pumps, CGM, ECG, ventilators, pulse oximeters, DICOM gateways, MQTT brokers, telestroke, RTLS, anaesthesia workstations, pill dispensers, insulin pumps, NICU, eICU, surgical suites
  - Key NA-bias record: INC-018-B (ARP Spoofing, rogue CA + CGM exfiltration) has explicit callout
  - INC-009-A → INC-012-A: narratively linked ping-sweep → port-scan chain
  - INC-009-B: ping sweep missed by suppressed IDS rule, led to exploitation 3 days later

### Roadmap
- **`docs/WORK2_WORK3_ROADMAP.html`** — 24-week parallel plan, open in browser
- **`docs/SESSION_HANDOFF.md`** — fuller session context

---

## Atomisation protocol (five rules)

**Atom = one actor + one action + one object**

| Rule | Condition | Example |
|---|---|---|
| 1 | Two verbs joined by "and" | "identify **and** track" → two atoms |
| 2 | Two objects | "persons **or** software programs" → two atoms |
| 3 | If/when conditions | Keep attached to their atom |
| 4 | Definitions | Never split |
| 5 | Unless/except exceptions | Separate atom |

**Quality gate per atom — all four must be true:**
1. Exactly one action
2. Exactly one object
3. Answerable Yes/No from incident text alone
4. Removing it would leave a compliance gap undetected

---

## The sequenced task list (where we are)

| Step | Task | Status |
|---|---|---|
| 0 | Adjudicate 70 UAE disputed labels | ⬜ Not done — defer until step 4 review |
| 1 | Atomise HIPAA §164.312 | ✅ Done — 27 atoms |
| 1b | Atomise ISO 27002 cl. 9–12 | ✅ Done — 49 atoms |
| 2 | **Write 50 incident records** | ✅ Done — 50 records, 19 attack types, grounded in CICIoMT2024 CSV stats |
| 3 | Build gold matrix (atom × record pairs) | ⬜ After records done |
| 4 | Annotation study (second annotator + kappa) | ⬜ Annotator not yet recruited |
| 5 | Zero-shot NLI judge on healthcare data | ⬜ After gold matrix |
| 6 | Llama fine-tune on healthcare atoms | ⬜ After NLI baseline |
| 7 | Grade both judges vs human key | ⬜ After fine-tune |
| 8 | Error analysis + NA-bias table | ⬜ After grading |
| 9 | Paper draft | ⬜ Write each section same week as the work it describes |

**Step 2 is current.** User writes incident record prose into the template skeletons. Start with T-13 (MQTT-Malformed_Data) and T-18 (Spoofing-ARP_Spoofing). After 5 records written, bring back for quality gate check.

---

## Rules — always follow these

1. **Never skip the WRITE lane.** 2h/week writing while the work is fresh, both papers.
2. **Human labels before model claims.** Every published number graded against human-adjudicated key.
3. **One blocker at a time.** Stuck >2 days → write it down, switch lanes, raise with supervisor.
4. **Product waits.** No autocomply repo, no RBI atoms, no new Vikas features. Vikas work = Work 2 instrumentation only.
5. **Works 2 + 3 run in parallel, all three lanes.** Build split is temporal not proportional — Work 3 owns build by default; Work 2 takes it whole during Work 3's blocked weeks (annotators working, GPU training, supervisor review). Never slice 6 build hours into 3+3.
6. **When they collide, Work 3 wins.** Work 2 drops to 2h reading floor.
7. **Blockchain is out of Work 3.** Do not reintroduce.

---

## Known traps — read before proposing anything

- **Golden set is 162 pairs not 1,146 rows.** Evaluate and prompt only, never train.
- **`uae_ia_controls_structured.json` is incomplete** (~181 of 251 controls). Controls like T6.2.2 are real. Check the raw file before concluding a control doesn't exist. Multiple near-duplicate variants in `data/02_processed/` — confirm which one a script reads before editing.
- **Anthropic LLM judge is built but unrunnable** — `scripts/llm_judge_anthropic.py` exists; no API balance. Llama adapter is the working judge.
- **Repo root is messy** — many one-off scripts from earlier phases. The maintained pipeline is in `scripts/`. Treat root `.py` files as historical unless proven current.
- **ISO 27799 is paywalled** — using ISO 27002:2013 cl. 9–12 instead, citable as "as adopted by ISO 27799:2016".
- **NotebookLM citations need manual verification** — it confabulates paper details even when the paper exists. Verify GraphCompliance, PrivComp-KG, Galli et al. 2025 independently before citing.

---

## Files that matter

| Path | What |
|---|---|
| `CLAUDE.md` | This file — session context |
| `docs/WORK2_WORK3_ROADMAP.html` | 24-week parallel plan |
| `docs/WORK3_EXECUTION_PLAN.md` | Work 3 internals, LEARN phases, rules |
| `docs/LLM_STUDY_PLAN.md` | Fundamentals curriculum |
| `docs/SESSION_HANDOFF.md` | Fuller session context |
| `data/13_healthcare/hipaa_312_atoms.json` | HIPAA atoms |
| `data/13_healthcare/iso27002_912_atoms.json` | ISO 27002 atoms |
| `data/13_healthcare/incidents/templates.json` | Incident record templates |
| `data/13_healthcare/incidents/SCHEMA.md` | Writing rules for records |
| `data/10_judge/adjudication.csv` | 70 disputed UAE labels |
| `data/10_judge/grade_report.json` | Current metrics |
| `data/12_atoms/pairs_eval.json` | UAE evaluation set |
| `atom_judge_adapter/` | Trained LoRA adapter |
