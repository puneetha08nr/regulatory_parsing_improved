# Research Direction — PhD Work 2 (AutoComply-IoMT)

## Where This Fits in the Thesis

```
Work 1 ✅  Can we trust decentralised storage for healthcare data?  (IPFS — published)
Work 2 ←   Can we trust automated compliance reasoning over IoMT incidents?
Work 3      Can we trust intelligence extracted from incident records without privacy leakage?
Work 4      Can we trust autonomous systems to respond safely with auditable decision chains?
```

Everything below is Work 2. The pipeline we built here is the prototype.

---

## Honest Gap Between Current Build and PhD Requirement

| Dimension | What we built | What the PhD needs |
|-----------|--------------|-------------------|
| Regulatory framework | UAE IA controls (251) | HIPAA + ISO 27799 + IEC 80001 |
| Input documents | Enterprise security policies (12 docs) | IoMT incident records (CICIoMT2024 grounded) |
| Domain | General enterprise | Healthcare IoT |
| Golden set size | 162 pairs, 12 policies | 500+ pairs, 20+ incident records |
| Annotators | 1 (you) | 2+ (inter-annotator kappa reportable) |
| Publication framing | Internal prototype | IEEE/ACM venue paper |

The **methodology is correct and transferable**. Everything retrains and reruns on the new
domain. No architectural rework needed.

---

## What Is Genuinely Novel — Keep These in the Paper

### 1. Per-Atom Decomposition of Regulatory Controls

Existing compliance NLP frames the task holistically:
`(document, control) → {Fully / Partially / Not Addressed}` — a 3-class classifier
that requires labelled training data and produces opaque judgments.

We reframe as:
`(document, atom_i) → {covered, not_covered}` for each sub-obligation independently,
then derive FA/PA/NA mechanically from atom coverage counts.

**Why this is novel for the paper:**
- Reduces compliance verification to binary entailment — what NLI models are designed for
- Errors are interpretable: you see which specific atom failed, not just "Partially Addressed"
- Enables mechanical FA/PA/NA derivation, removing holistic label subjectivity
- Makes the judge locally runnable — no cloud API dependency (critical for HIPAA data)

Verify this hasn't been published before with a targeted lit search on "compliance
mapping NLP", "regulatory requirement decomposition", "entailment-based compliance".

**Core claim:** "Decomposing HIPAA safeguard requirements into constituent atom obligations
reduces compliance verification to per-atom entailment, enabling a local model to achieve
F1=[X] without sending PHI-adjacent incident records to external APIs."

### 2. Conservative Annotation Bias in Compliance Labeling (Publishable Finding)

In our 162-pair UAE IA set: **60 of 70 human-AI disagreements were human=NA where
the passage plainly addressed the control**. Rolled up to control level: human key
showed 66 gaps; our per-atom judge found only 29. 43 of 66 "gaps" were actually covered.

This is a methodological finding with implications beyond this paper:
- Existing compliance benchmarks are likely NA-biased — any model trained on them
  learns the NA prior, not compliance semantics
- This explains why prior classifier approaches showed near-zero PA recall
- Per-atom decomposition partially corrects for this by forcing mechanical coverage judgment
  rather than holistic conservative-leaning annotation

In the HIPAA paper version: get kappa between two human annotators before adjudication.
A kappa < 0.5 on the initial annotation (expected) demonstrates the problem empirically.
Post-adjudication kappa should recover to > 0.7.

### 3. Local-Model Viability Under Healthcare Data Sovereignty Constraints

Healthcare organisations **cannot** send incident records to external APIs (HIPAA §164.312
prohibits disclosure of PHI without patient authorisation). The local deployment constraint
is not a limitation to apologise for — it is the correct framing of the problem:

> "We evaluate compliance verification under the data sovereignty constraints of
> healthcare environments, where routing incident records through cloud APIs is
> prohibited by the same regulations the system is assessing compliance with."

Results so far on UAE IA prototype:
- Zero-shot NLI cross-encoder (CPU, no training): F1 = 0.53
- Fine-tuned Llama-1B per-atom (GPU, Colab): TBD — expected higher

This comparison is the primary empirical contribution table of Work 2.

---

## What Needs to Change — Execution Order

### Step 1 — Download CICIoMT2024 and Extract Attack Taxonomy (1 day)

```
URL: cicresearch.ca/IOTDataset/CICIoMT2024/
Key file: Attack_info.csv — 18 attack types across 5 categories
```

Attack categories in CICIoMT2024:
- DDoS (UDP, TCP, ICMP flood)
- DoS (Slowloris, HTTP)
- Recon (Port scan, Ping sweep)
- MQTT attacks (Brute force, malformed packet)
- Spoofing (ARP, DNS)

Each attack type becomes a worknote scenario template. 2-3 worknotes per type.

### Step 2 — Atomise HIPAA / ISO 27799 Controls (3 days)

Target the HIPAA Security Rule Technical Safeguards (§164.312) and Physical Safeguards
(§164.310) — these map most directly to IoMT incident records. Also pull ISO 27799
clauses 9-12 (access control, cryptography, physical, operations).

Focus on ~40 controls with clear sub-requirements. Each sub-requirement = one atom.
Target ~120 atoms total (3 per control avg).

Output: JSON file in the same format as `data/02_processed/uae_ia_controls_structured.json`
```json
[{"control": {"id": "HIPAA-164.312.a.1", "name": "Access Control",
              "description": "...", "sub_controls": ["...", "..."]}}]
```

### Step 3 — Write 50 Human-Authored IoMT Incident Records (1 week)

Ground each in a CICIoMT2024 attack scenario. Use this schema (matches ServiceNow OT
structure confirmed in your earlier research):

```
Incident ID:     INC-XXXX
Timestamp:       [date/time]
Affected Device: [patient monitor / infusion pump / wearable / gateway + ID]
Protocol:        [MQTT / HTTP / HL7 / Modbus]
Observed Behaviour: [free text — what the analyst saw]
Actions Taken:   [free text — containment, remediation]
Outcome:         [resolved / escalated / under monitoring]
ATT&CK Technique: [ground truth — T1xxx]
HIPAA Safeguard: [ground truth — which §164.3xx was implicated]
```

Write 2-3 records per CICIoMT2024 attack type. Vary device type, severity, and analyst
writing style. 50 records is sufficient for a preliminary conference submission.

### Step 4 — Run Annotation Round (you + 1 colleague)

Annotate independently first. Report pre-adjudication kappa. Adjudicate by discussion.
This produces:
- A human-adjudicated answer key (not AI-derived — removes circularity from current build)
- A reportable kappa showing initial annotator disagreement (expected < 0.6)
- Post-adjudication agreement (expected > 0.75)

The kappa gap is a data point demonstrating the annotation difficulty — motivates the
automated approach.

### Step 5 — Re-run the Pipeline on Healthcare Data

The pipeline is already built. Swap the input files and re-run in order:

```bash
# 1. Pool candidates (build_gold_matrix.py) — HIPAA controls + incident records
# 2. Annotate → clean_answer_key.py
# 3. Build per-atom dataset → build_atom_dataset.py
# 4. Run NLI judge → nli_judge.py
# 5. Run Llama fine-tune → finetune_atom_compliance.py  (Colab)
# 6. Grade both → grade_judge.py
# 7. Compliance report → compliance_report.py
```

Steps 4-7 run in under 10 minutes on CPU once the data is ready.

### Step 6 — Add Hyperledger Anchoring (1 day)

Hash each assessed (control, incident, verdict, timestamp) tuple and anchor to Hyperledger
Fabric or Besu. One transaction per assessment. Store the transaction hash in the
compliance_report.json output.

This satisfies "blockchain-enabled" in your registered thesis title and provides
a tamper-proof HIPAA audit trail — a genuine, commercially relevant contribution.
Conference paper: SDK call is sufficient. Journal: full node deployment.

---

## Research Gap Claim (Abstract Seed)

> Healthcare IoT incident records contain rich evidence of compliance posture — which HIPAA
> safeguards were invoked, bypassed, or violated during a security event. Yet compliance
> verification remains manual and legally inadmissible without an immutable audit record.
> Existing automated approaches apply holistic classifiers requiring labelled data that
> does not exist in healthcare, or rely on cloud LLM APIs prohibited from processing
> PHI-adjacent incident data under HIPAA §164.312.
>
> We propose AutoComply-IoMT, a per-atom compliance verification pipeline that decomposes
> each HIPAA/ISO 27799 control into constituent sub-obligations, reduces verification to
> per-atom binary entailment, and derives FA/PA/NA mechanically — enabling a local model
> without external API dependence. We report a key methodological finding: human annotators
> exhibit systematic conservative bias in compliance annotation (NA over-labeling), inflating
> apparent error rates in prior work. On a benchmark of [N] IoMT incident records mapped to
> [K] HIPAA controls, our zero-shot NLI judge achieves F1=[X]; a fine-tuned Llama-1B
> achieves F1=[Y]. All verified compliance evidence is anchored to an immutable blockchain
> audit record, producing legally defensible HIPAA compliance documentation.

---

## One Critical Caveat: Circularity in Current Results

The current v2 labels (HIPAA adaptation will reuse these) were derived from Claude's own
inline judgments. Grading NLI/Llama against these means you are measuring "how close is
this model to Claude's assessment" — not "how close is this model to ground truth".

For the PhD paper this must be fixed:
- Human-adjudicated key only (Step 4 above)
- No AI-in-the-loop for label generation
- Claude inline judgments can inform but not constitute the answer key

The annotation under-labeling finding still holds and is citable — it came from comparing
Claude's atom-level reasoning against human holistic labels, which is a valid comparison.

---

## Target Venues (Work 2)

| Venue | Fit | Notes |
|-------|-----|-------|
| IEEE COMPSAC | Strong | Compliance automation + healthcare IoT |
| IEEE TrustCom | Strong | Trust + blockchain + IoT |
| CODASPY | Good | Data and application security/privacy |
| ACM SACMAT | Good | Access control — HIPAA maps directly |
| AMIA NLP | Niche | Healthcare NLP — clinical audience only |

Target COMPSAC or TrustCom. 8-page limit. Pipeline + annotation study + judge comparison
fits cleanly.

---

## What This Work Feeds Into Work 3 and Work 4

**Work 3** (threat intelligence) reuses:
- The same incident record corpus (50 → 500 over time)
- The same CICIoMT2024 attack taxonomy
- The HIPAA-assessed verdicts become labels for TTP extraction

**Work 4** (agentic response) reuses:
- AutoComply as the compliance evidence generation module
- The blockchain anchoring module as the audit trail
- The incident record format as the auto-generated output of the response agent

Work 2 is not a standalone paper — it is the compliance intelligence layer that both
Work 3 and Work 4 depend on. Build it robustly.
