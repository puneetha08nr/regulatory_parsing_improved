# PhD Session Handoff — Works 2 + 3

**Read this first in a new session.** Written 2026-08-25. Everything below was verified against
the repo on that date, except where marked *(from handoff doc)*.

Repo: `~/Documents/IterativeResearch/feb6/regulatory_parsing_improved` · branch `main` ·
last substantive commit `9f7d385` (2026-07-30).

---

## 1. Who and what

Avinash — PhD student and technical founder, running three tracks at once:

| Track | What |
|---|---|
| **PhD** | 5-paper IoT security thesis — *"Towards Trustworthy Autonomous Security in IoT Systems"* |
| **Vikas** | 45-agent AI content-ops platform, Tamil Nadu scheme outreach — **this is Work 2's testbed** |
| **Elyts-EDGE** | OOH advertising analytics, Bengaluru billboards — possible Work 4 testbed |

**This repo is Work 3.** Work 2 has no repo yet; it will live in the Vikas codebase.

### The five works

| # | Topic | Status | Target venue |
|---|---|---|---|
| 1 | IPFS healthcare IoT storage evaluation | ✅ **Submitted** (revision, Jul 2026) | Computer Networks Q1 |
| 2 | Secure MCP architecture for IoT agents | 🔧 **Starting now** | IEEE TDSC (primary) / TIFS |
| 3 | AutoComply-IoMT — per-atom compliance | 🔧 **Prototype, active** | Computers & Security |
| 4 | Trusted edge intelligence for IoT | ⬜ Not started | IEEE IoTJ / JSA |
| 5 | Adaptive trust orchestration | ⬜ Not started | TDSC / FGCS |

Patent filed after Work 5, covering orchestration across Works 2–5. *(from handoff doc)*

---

## 2. The current decision — read this before proposing anything

**Works 2 and 3 run in parallel, all three lanes each (learn / build / write), from week 1.**
Week 1 = 25 Aug 2026.

The user explicitly rejected an earlier proposal that gave Work 2 reading hours only — *"we need
not stop building, we can learn and build and write, else it is too much lag"*. Do not re-propose
a serial plan.

**The mechanism — the build split is temporal, not proportional.** Never slice the 6 build hours
into 3+3; deep work doesn't divide, and 3h/week finishes nothing in either paper. Instead Work 3
owns the build lane by default, and Work 2 takes it *whole* during weeks Work 3 is blocked on an
external dependency:

| Week | Work 3 blocked on | Work 2 sprint |
|---|---|---|
| 2 | CICIoMT2024 downloading | MCP reference harness |
| 8–9 | annotators labelling independently | signed tool calls + tamper-evident audit log |
| 12 | Llama fine-tune running on Lightning | attacks 3–5, corpus complete |
| 15–16 | supervisor holds the manuscript | zero-trust enforcement layer |

**Work 2 keeps a 2h writing lane every week** from week 3 — you write the section covering
whatever you designed that week. That is what removes the lag.

**Budget:** 14 hrs/week. If it has to drop to 12, both papers slide ~2 weeks.
**Targets:** Work 3 submits week 17. Work 2 first draft week 24.

**Collision rule:** only Work 3 has a deadline before week 17. When the two collide, Work 3 wins
and Work 2 falls back to a 2h reading floor — never to zero, never above the floor.

Full week-by-week grid: **`docs/WORK2_WORK3_ROADMAP.html`** (open in a browser).

> This overrides Rule 4 of `docs/WORK3_EXECUTION_PLAN.md` ("Work 2 learning starts only after
> submission"). That document is otherwise still the authority on Work 3's internals — the
> atomisation protocol, the LEARN phase mapping, and the paper's three claims.

---

## 3. Where Work 3 actually stands

Rejected once. **Reason was "no experimental proof" — a methodology gap, not a novelty gap.**
The architecture is correct and transferable; nothing needs rebuilding. The whole plan exists to
produce that missing proof.

### Verified state (2026-08-25)

**Current best result** — Llama-3.2-1B QLoRA per-atom judge, evaluated on 162 UAE IA pairs:

```
positive-class  F1 = 0.687   P = 0.793   R = 0.605   (tp 46 / fp 12 / fn 30)
overall         agreement = 0.543   kappa = 0.233
```

Source: `data/10_judge/grade_report.json`. Zero-shot NLI baseline was ~0.44 F1.
Trained adapter sits in `atom_judge_adapter/` (LoRA safetensors, 45 MB).

**Evaluation set** — `data/12_atoms/pairs_eval.json`, 162 control×passage pairs:

```
labels   Not Addressed 86  ·  Partially Addressed 59  ·  Fully Addressed 17
split    train 102  ·  test 60
atoms    atoms_train.jsonl 386  ·  atoms_test.jsonl 169
```

**Unfinished:** 70 disputed labels sit unadjudicated in `data/10_judge/adjudication.csv`.
The κ = 0.233 above is *against unadjudicated labels* — cleaning them is the single
highest-leverage act available and is week-1 work.

### The three claims to protect

1. **Per-atom decomposition of HIPAA controls** — first for a healthcare standard. Reduces
   compliance verification to binary entailment a local model can do.
2. **NA-bias annotation finding** — human annotators systematically over-label "Not Addressed";
   existing compliance benchmarks inherit the bias. The sleeper contribution; give it space.
3. **Local-model viability under data sovereignty** — HIPAA prohibits routing PHI through cloud
   APIs. The constraint *is* the framing, not a limitation.

### What Work 3 still needs

- Adjudicated labels (70 rows) → clean human answer key
- CICIoMT2024 downloaded, 18 attack types extracted from `Attack_info.csv`
- HIPAA §164.312 + ISO 27799 cl. 9–12 atomised → ~120 atoms
- 50 human-authored IoMT incident records
- Two-annotator kappa study, κ > 0.75 gate — **second annotator not yet recruited**
- Healthcare rerun of the full pipeline + fine-tune, graded against the human key

---

## 4. Where Work 2 stands

Nothing built. Research question is locked *(from handoff doc)*:

> *"How can a zero-trust security architecture for MCP-based multi-agent IoT systems prevent
> agent impersonation, prompt injection, memory tampering, and capability escalation?"*

**Testbed:** Vikas, 45 agents in production. The build work is instrumentation of a system that
already exists — capability manifests, signed tool calls, audit logging — not a simulator written
from scratch. That is why Work 2 build can start in month 1.

**Build order (each increment is independently a paper artefact):**

1. wk 2 — MCP reference harness (server + client you fully control)
2. wk 5–9 — instrumentation: per-agent capability manifests, signed tool calls, tamper-evident
   audit log. *Must precede the attacks* — without the log you cannot observe an attack succeeding.
3. wk 11–14 — one attack per STRIDE category, then measure the **undefended baseline**.
   Worthless if measured after hardening anything.
4. wk 15–20 — enforce zero trust, re-run the corpus, report the delta **and the latency/token cost**.

**Reading list:** MCP spec (`modelcontextprotocol.io`), NIST SP 800-207, OWASP LLM Top 10
(LLM01, LLM02), NIST SP 800-213. MCP is Nov 2024 and too new for most peer-reviewed venues —
arXiv cs.CR is mandatory.

**Production risk:** Vikas is live. Signed tool calls and enforcement on 45 running agents can
break Tamil Nadu outreach. Build behind a flag, shadow-mode first.

---

## 5. Traps — things that look true but aren't

- **The golden set is smaller than it looks.** 1,146 rows is really 162 pairs. Use it to
  *evaluate and prompt*, never to train.
- **`data/02_processed/uae_ia_controls_structured.json` is incomplete** relative to the 251-control
  source (it holds ~180 keys). Controls like T6.2.2 are real but missing. When a control looks
  absent, check the raw file before concluding it doesn't exist. Note there are several near-duplicate
  variants in that directory (`_clean`, `_corrected`, `_v2`, `_from_label_studio`) — confirm which
  one a script actually reads before editing.
- **The Anthropic LLM judge is built but unrunnable** — `scripts/llm_judge_anthropic.py` exists;
  `ANTHROPIC_API_KEY` has no balance. The Llama adapter is the working judge.
- **Repo root is messy** — many one-off extractors and JSON dumps from earlier phases. The
  maintained pipeline is in `scripts/`; treat root-level `.py` files as historical unless proven
  otherwise.

---

## 6. Files that matter

| Path | What |
|---|---|
| `docs/WORK2_WORK3_ROADMAP.html` | **The plan.** 24-week two-lane grid |
| `docs/WORK3_EXECUTION_PLAN.md` | Work 3 internals: atomisation protocol, LEARN phases, rules |
| `docs/LLM_STUDY_PLAN.md` | Fundamentals curriculum + concept question bank |
| `docs/handoff_phd_aug2026.docx` | Source handoff — thesis arc, Work 2 concepts, decisions log |
| `docs/RESEARCH_DIRECTION_PHD.md` | Abstract seed (update numbers, remove blockchain sentence) |
| `docs/STEP_BY_STEP.md` | How the training + eval runs were actually executed |
| `data/10_judge/adjudication.csv` | **70 disputed labels — week 1 task** |
| `data/10_judge/grade_report.json` | Current metrics + full disagreement list |
| `data/12_atoms/pairs_eval.json` | The 162-pair evaluation set |
| `atom_judge_adapter/` | Trained LoRA adapter (the working judge) |

**Pipeline order:** `build_gold_matrix.py` → `build_atom_dataset.py` → `nli_judge.py` →
`grade_judge.py`. Fine-tuning: `finetune_atom_compliance.py` (runs on Lightning.ai —
push → train → pull). Adjudication: `scripts/adjudicate.py --apply`.

---

## 7. Standing rules

1. **Never skip the WRITE lane.** 2 hrs/week while the work is fresh beats 40 hrs of archaeology
   in month 4. Applies to both papers.
2. **Human labels before model claims.** Every published number is graded against the
   human-adjudicated key — nothing else.
3. **One blocker at a time.** Stuck > 2 days → write it down, switch lanes, raise it with the
   supervisor that week.
4. **The product waits.** No autocomply repo, no RBI atoms, no new Vikas features. Vikas work is
   limited to Work 2 instrumentation.
5. **After each LEARN phase, reread the matching script.** Mapping is in `docs/LLM_STUDY_PLAN.md`.
6. Blockchain is out of Work 3 — decided, do not reintroduce.

---

## 8. Locked values — do not recompute or "correct"

Work 1 (already submitted, for consistency if it comes back for revision):

```
144 configs · 4,320 trials · 100% DHT · 35.8% aggregate · 66.7% WAN · 176× · 1,763 KB · 22.56 s
H = 444.64  (network condition)   H = 1626.82  (node count)   both p < 0.001
```

---

## 9. Do this first

1. **Message the second annotator.** The only task whose lead time you don't control — ~20 hrs of
   their time across weeks 7–9, and contribution #2 depends on it. If unconfirmed by week 3, the
   week-9 kappa gate slips.
2. **Start CICIoMT2024 downloading** — `cicresearch.ca/IOTDataset/CICIoMT2024/`. It gates month 2,
   and its download time is what makes week 2 a Work 2 sprint.
3. **Open `data/10_judge/adjudication.csv`**, work the first 20 HIGH-priority rows, then
   `python3 scripts/adjudicate.py --apply` and re-grade.
4. **Two hours on the MCP spec.** Open `docs/WORK2_LIT.md` with three columns — paper, threat
   covered, gap.

### Open blockers

| Blocker | Track |
|---|---|
| Second annotator not recruited | Work 3 — longest lead time |
| CICIoMT2024 not downloaded | Work 3 — blocks month 2 |
| HIPAA §164.312 not atomised | Work 3 |
| 70 labels unadjudicated | Work 3 — cheapest win |
| Work 2 has no repo/branch yet | Work 2 |
| `ANTHROPIC_API_KEY` has no balance | Work 3 — non-blocking, Llama judge works |
| Missing worker container, prod DB empty | Elyts-EDGE — out of scope |
| Meta/Kapso account | Vikas — out of scope |
