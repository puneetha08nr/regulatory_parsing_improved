I've read through all the docs (PIPELINE_WORKFLOW_AND_ROADMAP.md, RUN_COMPARISON_AND_DIAGNOSIS.md, cursor_progress.md, BUILD_STATUS.md, MAPPING_STRATEGY_AND_REGNLP.md, STEP_BY_STEP.md, CHANGES.md) plus the latest batch_results.json. Here's my assessment.

What the project is

A RegNLP-style pipeline that maps UAE IA regulatory controls to internal policy passages: rule-based/LegalBERT obligation filter → per-document hybrid retrieval (BM25 + dense + RRF) → cross-encoder reranker → score thresholds → Fully/Partially/Not Addressed, with an optional fine-tuned Phi-3 LLM judge on top, and a Label Studio annotation loop feeding a golden set (~1,146 pairs).

The headline finding: retrieval works, judgment doesn't

Recall@5 is 81.5% and Recall@20 hit 1.0 on some policies — the right passages are being found. Everything downstream of retrieval is where accuracy dies. End-to-end you're at P=0.29/R=0.70/F1=0.41 on the best single-policy run, and the latest batch eval shows per-policy F1 between 0.0 and 0.33.

Caveats in the current setup (things the docs gloss over)

1. Your precision numbers aren't measuring what you think. The golden set is not exhaustive — it only labels pairs the pipeline itself surfaced as candidates. A "false positive" may just be an unlabeled true positive, and controls the retriever never surfaced were never annotated at all (circularity: the model generates its own exam questions). The Run 3 "2.1% precision" over 386 predictions is against ~89 known positives; most predictions are simply unjudged.
2. The eval slices are statistically meaningless. Batch eval policies have 1–6 golden positives each. With one positive, recall is 0% or 100% — you cannot detect a 90%-accuracy system, let alone tune toward one. The dev set for every retrain is 141 rows with 17 FA / 5 PA.
3. The golden data is 90.5% Not Addressed, 3.5% PA, plus 150 malformed rows (the "No" family — broken control IDs, pure NA noise, still in training data per cursor_progress.md). Families T4/M1 have zero PA examples. PA recall = 0.000 across every training run is not a modeling failure — the class barely exists in the data.
4. The models are mismatched to the task. bge-reranker-base / ms-marco-MiniLM are web-search relevance models. "Does this policy passage satisfy this legal obligation" is an entailment/coverage judgment, not topical relevance. That's why scores compress into 0.3–0.6 and no global threshold separates classes — the thresholds (0.45/0.25) are compensating for a model that fundamentally can't express the distinction.
5. Fine-tuning has repeatedly hurt. MSE regression collapsed ranking (Spearman 0.22→0.14, R@5 81.5→70.4); the Phi-3 judge learned family priors instead of semantics (T8 is 93% NA in training → it rejected 86% of T8 predictions, dropping true positives). With ~109 positives total, every fine-tune is fitting noise.
6. Known corpus defects are irreducible error: gold pairs pointing at missing policy files (Access Control, Logging & Monitoring v3), three controls (T6.2.2, T1.2.3, T2.2.6) silently filtered out by the rule-based obligation classifier before retrieval — those are guaranteed FNs no model change can fix.
7. Moving-target methodology: the reranker base model changed between runs (bge-reranker-base → MiniLM-L-6 → MiniLM-L-2), thresholds changed, blocklists auto-apply — run-to-run comparisons in the docs aren't apples-to-apples.

Why it's not touching 90%

Because three ceilings stack multiplicatively, and each is currently well below 0.95:

- Data ceiling — you can't learn a 3-way legal judgment from ~109 positives and 40 PA examples, and you can't measure 90% against a non-exhaustive, pipeline-biased golden set with no inter-annotator agreement number. Even the definition of "accuracy" is unstable: at a 90% NA base rate, always-predict-NA scores 90% accuracy while being useless. The real metric is F1 on positives, and PA vs FA is genuinely ambiguous without a written rubric — the human ceiling itself may be ~80–85% agreement.
- Model ceiling — a 22M-parameter search reranker + a global score threshold is structurally incapable of "this passage covers 3 of the 4 sub-obligations in M2.1.1." Coverage is compositional; a scalar relevance score isn't.
- Pipeline ceiling — obligation filter drops controls, duplicate/missing policies corrupt the index, section-level passages (~19 per policy) mix boilerplate with substance.

The project keeps pulling the one lever (fine-tune the reranker harder) that addresses none of these.

How I'd build it from scratch

1. Fix the ground truth before anything else. Write an explicit FA/PA/NA rubric (FA = every atomic obligation in the control is covered; PA = at least one but not all; with tie-break rules). Pick 3–5 policies and annotate the full control × passage matrix for them — exhaustively, not just pipeline candidates — with two annotators and a measured Cohen's kappa. That gives you: (a) an honest test set where precision is actually precision, (b) the human agreement ceiling, which is your realistic max. Freeze it.

2. Decompose both sides structurally. Parse each control into atomic obligations (actor / modal / action / object / condition) — an LLM does this reliably in one pass over 263 controls. Parse policies at clause level with section-hierarchy metadata, dedupe boilerplate by hash, and keep <CLIENT> normalization consistent.

3. Keep retrieval dumb and recall-oriented. BM25 + a modern embedding model (bge-m3 / e5-large), tuned only for Recall@30 ≈ 100% against the golden matrix. This stage already works in your pipeline; it needs no learning.

4. Replace threshold-on-a-score with an LLM judge doing rubric-based coverage checking — a frontier model (this is exactly the shape of task where Claude/GPT-class models are strong zero/few-shot), prompted per (control, candidate passage) to output, as structured JSON: which obligation atoms the passage covers, quoted evidence spans, and the derived FA/PA/NA label. The label is computed from atom coverage, not vibes — that's what makes PA tractable. At ~10 candidates × 263 controls this is a few thousand cheap calls, cached. This step alone would likely jump you from F1 ≈ 0.4 to 0.7+, because it removes the scalar-score bottleneck entirely.


5. Aggregate at control level, not pair level. The compliance question is "is control X addressed by the policy set," with the pair mappings as evidence. Multiple PA passages can jointly make a control FA — the current pipeline can't express that; an aggregation step over judged atoms can.

6. Only then distill for cost. Once the LLM judge + human corrections give you 5–10k silver labels, train a cross-encoder to imitate it for the easy bulk, and keep the LLM for the uncertain band (active learning on disagreements). Distillation from a good teacher with lots of labels works; fine-tuning a weak model on 109 positives — the current approach — doesn't.

7. Evaluate honestly. Held-out policies never used in prompt/threshold iteration, per-family breakdown, precision on the exhaustive matrix, and control-level gap reports as the product output.

The realistic target with this design is 85–92% agreement-with-human on FA/PA/NA — i.e., roughly at the inter-annotator ceiling — with the current architecture capped somewhere around F1 0.5 no matter how many reranker retrains you run, because its bottlenecks are the label supply and the scalar-score formulation, not the weights.