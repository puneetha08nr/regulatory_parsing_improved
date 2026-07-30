# Golden Set Annotation Rubric — FA / PA / NA

This rubric defines how a `(control, policy passage)` pair is labelled **Fully
Addressed / Partially Addressed / Not Addressed**. It exists so the label is
*derived from evidence*, not from a gut call — which is what makes Partially
Addressed reproducible and gives us a measurable human-agreement ceiling.

The unit of judgement is the control's **atomic obligations** (`sub_controls`,
e.g. `M1.1.1.a`, `M1.1.1.b`, …). You mark which atoms the passage covers; the
label falls out of that.

---

## 1. The decision, in one line

> **Does *this passage*, on its own, satisfy the obligation atoms of *this control*?**

- **Fully Addressed (FA)** — the passage covers **every** atom of the control.
- **Partially Addressed (PA)** — the passage covers **at least one but not all** atoms.
- **Not Addressed (NA)** — the passage covers **no** atom.

"Covers" is judged **per passage**, not per policy. A control that is fully
addressed *across three passages* is PA on each of those passages individually.
Combining passages into a control-level verdict is a separate aggregation step
(Phase 3) — do **not** do it here.

---

## 2. What "covers an atom" means

An atom is covered when the passage states a requirement, mechanism, or
assignment that **satisfies the obligation the atom expresses** — not merely
mentions the same topic.

| Situation | Covered? |
|---|---|
| Passage mandates the specific action the atom requires (same obligation) | ✅ yes |
| Passage assigns the responsibility / establishes the mechanism the atom names | ✅ yes |
| Passage restates the atom in different words but same substance | ✅ yes |
| Passage mentions the **topic** but states no obligation ("assets are important") | ❌ no |
| Passage covers an **adjacent** obligation, not this atom | ❌ no |
| Passage is boilerplate (scope, purpose, confidentiality, ToC) | ❌ no |

**Topic overlap is not coverage.** The single most common labelling error is
marking an atom covered because the passage is *about the same subject*. Require
that the passage would let an auditor tick the obligation off.

**Evidence is mandatory for every covered atom.** If you cannot quote the exact
span of the passage that covers the atom, the atom is **not** covered.

---

## 3. Deriving the label from atoms

Let `n` = number of atoms in the control, `k` = atoms you marked covered.

| Condition | Label |
|---|---|
| `k == n` (and `n > 0`) | **Fully Addressed** |
| `0 < k < n` | **Partially Addressed** |
| `k == 0` | **Not Addressed** |

To record NA you must **positively confirm** it: put `-` in `covered_atoms`. A
**blank** cell means "not yet reviewed" and the compiler will refuse to freeze
the sheet until it's gone — this is deliberate, so a half-finished pass can
never masquerade as "everything else is NA".

### Controls with no atoms (`n == 0`)

Some controls have an obligation only in the top-level `description` and no
`sub_controls`. Treat the description as a single atom and judge FA/NA on it
(PA is possible only if the description itself is a compound obligation you can
split — note the split in the `note` column). Put the verdict in
`label_manual`.

---

## 4. Tie-break and edge rules

1. **Vague gestures don't count.** "The entity shall manage assets
   appropriately" does not cover a specific atom like "maintain an asset
   inventory". Require specificity matching the atom.
2. **Cross-references.** A passage that says "see the Access Control Policy"
   does **not** cover the atom — the obligation lives in the other document.
   Mark NA for this passage.
3. **Forward-looking / aspirational language** ("we aim to", "should consider")
   still counts as an obligation for coverage purposes *if* it names the
   mechanism; note the weak modality in `note`.
4. **Partial atom coverage.** An atom is binary — covered or not. If a passage
   covers half of a compound atom, mark it **not** covered and leave a `note`.
   Do not invent half-atoms.
5. **Wrong control entirely.** If the pooled pair is a bad candidate (passage
   addresses a *different* control), that's just NA for this control. If you
   spot the control it *should* map to, record it in `note` — it becomes a new
   positive candidate, not a correction to this row.
6. **Duplicated boilerplate** (same confidentiality/scope block repeated across
   passages) is NA everywhere. These are also blocklist candidates.

---

## 5. Confidence

Set `confidence` 1–5 on any pair where you hesitated. Low-confidence pairs are
the ones a second annotator should re-label first — they concentrate where the
human ceiling actually sits.

---

## 6. Worked examples

**Control M1.1.3** — "responsibilities and authorities of roles for information
security are assigned and communicated." Atoms: `a` assign responsibilities,
`b` communicate them.

- Passage: *"The CISO is responsible for X; role responsibilities are published
  in the security charter and communicated to all staff."* → covers `a` and `b`
  → **FA**. Evidence: the two quoted clauses.
- Passage: *"The CISO is responsible for X."* → covers `a`, not `b` → **PA**.
- Passage: *"&lt;CLIENT&gt; shall ensure the security of assets by identifying
  information assets."* → topic is assets, no role assignment → **NA**.

---

## 7. Column reference (annotation CSV)

| Column | Who fills | Meaning |
|---|---|---|
| `covered_atoms` | annotator | Comma-separated atom keys covered, e.g. `a,c`. `ALL` = every atom. **`-` = reviewed, none covered (confirmed NA)**. **Blank = not yet annotated** — the compiler refuses to freeze a sheet with blanks, so mark every NA row with `-`. |
| `evidence` | annotator | Quoted span(s) justifying the covered atoms. Required if any atom is covered. |
| `label_manual` | annotator | Only for `n_atoms == 0` controls, or to override the atom-derived label (explain in `note`). |
| `confidence` | annotator | 1–5, optional; set it when you hesitated. |
| `note` | annotator | Anything: the control it *should* map to, weak modality, boilerplate flag. |
| `label_prior` | pre-filled | Existing golden label for this pair, if any — review it, don't trust it. |
| `source` | pre-filled | Why this pair is in the pool: `prediction` / `golden` / `retrieval` / `neg_sample`. |

`source == prediction` pairs **must** all be labelled — they are what makes
precision honest. `source == neg_sample` pairs are the random unpooled cells
that estimate how many true positives the pool missed (recall honesty).
