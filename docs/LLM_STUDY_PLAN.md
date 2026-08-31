# LLM Study Plan — grounded in this project

**Purpose:** learn LLM fundamentals from first principles, using this compliance-mapping
codebase as the worked example. Every concept below is anchored to a file, a number, or a
failure that actually happened in this repo.

**Why this works:** most people learn these concepts abstractly. You have a system where
each one has already bitten you. In an interview you can answer with *"in my project, X
happened, and here's why"* — which beats any certificate.

**Timeline:** ~3.5 months at ~10 hrs/week. Short on time? Do Phases 2, 5, 7 and the six
interview answers — three weeks, covers most of what gets asked.

---

## Progress Tracker

- [ ] Phase 0A — Maths you actually need
- [ ] Phase 0B — Machine learning fundamentals ← start here if new to ML
- [ ] Phase 0C — Neural networks
- [ ] Phase 1 — Classical NLP
- [ ] Phase 2 — The Transformer ← core
- [ ] Phase 3 — Encoder models (BERT family)
- [ ] Phase 4 — Decoder models (GPT / Llama family)
- [ ] Phase 5 — Fine-tuning ← most job-relevant
- [ ] Phase 6 — Retrieval & RAG
- [ ] Phase 7 — Evaluation ← most underrated
- [ ] Phase 8 — Prompting & Reasoning
- [ ] Phase 9 — Agents & Tools
- [ ] Phase 10 — Alignment & Safety
- [ ] Phase 11 — Recent Frontier
- [ ] Concept question bank — can answer from memory

**Books:** see the Book References section at the end. Four of the six recommended are free;
buy Géron and Raschka.

---

## Phase 0 — Foundations (2.5 weeks)

**If you are new to machine learning, this is real work, not a skim.** Everything after this
assumes it. Budget the time.

### 0A — Maths you actually need (3 days)

Far less than people claim. You need to *understand* these, not derive them.

| Concept | What it is | Why it matters |
|---|---|---|
| **Vector** | An ordered list of numbers, e.g. `[0.2, -1.4, 0.8]` | A piece of data as a point in space. An embedding is just a long vector |
| **Dot product** | Multiply elementwise, sum: `a·b = Σ aᵢbᵢ` | Measures alignment. Big positive = same direction |
| **Cosine similarity** | Dot product of normalised vectors | "How similar in meaning" — magnitude-independent |
| **Matrix** | A grid of numbers; a stack of vectors | Weights are matrices. A batch of data is a matrix |
| **Matrix multiplication** | Rows × columns | ~95% of neural network compute. GPUs exist for this |
| **Derivative / gradient** | Rate of change; which way is uphill | Tells you which direction to nudge each weight |
| **Probability distribution** | Numbers ≥ 0 summing to 1 | Every model output over classes or tokens |
| **Log / exp** | Inverse of each other | Log turns products into sums; used everywhere in loss functions |

You do **not** need: proofs, integrals, matrix decompositions by hand, or measure theory.

---

### 0B — Machine learning fundamentals (1 week)

**The core idea:** in traditional programming you write rules and get answers. In machine
learning you supply answers (data + labels) and the computer derives the rules. That's the
whole paradigm shift.

**Types of learning**

| Type | You give it | It learns | Example |
|---|---|---|---|
| **Supervised** | Inputs + correct labels | Input → label mapping | Spam / not spam |
| **Unsupervised** | Inputs only | Structure, groupings | Customer segmentation |
| **Self-supervised** | Raw data; labels derived from it | Representations | **All LLM pretraining** — the label is the next word |
| **Reinforcement** | An environment and rewards | A policy of actions | Game playing; RLHF |

Self-supervised is why LLMs work at scale: no human labelled the internet, but "predict the
next word" generates infinite labels from raw text for free.

**Task types**

| Task | Output | Example |
|---|---|---|
| **Binary classification** | One of 2 classes | Is this an obligation? |
| **Multi-class classification** | One of N classes | Fully / Partially / Not Addressed |
| **Multi-label classification** | Any subset of N | Which of 8 topics does this cover? |
| **Regression** | A continuous number | Predicted price |
| **Ranking** | An ordering | Search results |
| **Generation** | A sequence | Text, code, images |

**The vocabulary you must be fluent in**

| Term | Meaning |
|---|---|
| **Sample / instance / example** | One row of data |
| **Feature** | One input variable. In deep learning, learned rather than hand-designed |
| **Label / target / ground truth** | The correct answer for a sample |
| **Parameters** | What the model *learns* — weights and biases |
| **Hyperparameters** | What *you* set before training — learning rate, batch size, layer count |
| **Training set** | Data the model learns from |
| **Validation / dev set** | Data for tuning hyperparameters and choosing checkpoints |
| **Test set** | Touched once, at the end. Any tuning against it invalidates it |
| **Epoch** | One full pass through the training set |
| **Batch** | A group of samples processed together before one weight update |
| **Step / iteration** | One weight update (one batch) |
| **Inference** | Using a trained model to make predictions |
| **Checkpoint** | Saved model weights at a point in training |

**The central problem: generalisation**

| | Symptom | Cause | Fix |
|---|---|---|---|
| **Underfitting** | Bad on train *and* test | Model too simple; trained too little | Bigger model, train longer, better features |
| **Good fit** | Good on train, good on test | — | — |
| **Overfitting** | Great on train, bad on test | Model memorised the training data | More data, regularisation, early stopping, smaller model |

**Bias-variance trade-off:** bias is error from wrong assumptions (too simple — underfits);
variance is error from sensitivity to the particular training sample (too complex —
overfits). Total error has both, and reducing one usually raises the other. Modern deep
learning complicates this picture, but the intuition still guides debugging.

**Regularisation — techniques that fight overfitting**

| Technique | How |
|---|---|
| **L2 (weight decay)** | Penalise large weights. `weight_decay=0.01` in our training config |
| **L1** | Penalise absolute weight size; drives some to exactly zero (feature selection) |
| **Dropout** | Randomly zero some neurons during training so the network can't rely on any one. `lora_dropout=0.05` |
| **Early stopping** | Stop when validation loss stops improving |
| **Data augmentation** | Create more training variety |

**Cross-validation:** split data into k folds; train on k−1, validate on 1; rotate; average.
Use it when data is small and a single split would be high-variance. With 500 examples a
single test split can swing wildly by luck.

**Data problems that ruin everything**

| Problem | What it is | Why it's fatal |
|---|---|---|
| **Class imbalance** | One class dominates | Model learns to always predict the majority and scores high while being useless |
| **Data leakage** | Test information influences training | Metrics look great, production fails |
| **Label noise** | Wrong or inconsistent labels | Caps achievable accuracy; no model beats its labels |
| **Distribution shift** | Production data differs from training data | Silent degradation over time |
| **Selection bias** | Data isn't representative | Model works only on the sampled slice |

**Classical algorithms worth knowing by name and shape**

You will rarely use these for text now, but interviewers ask, and they teach the concepts
cleanly.

| Algorithm | Idea | Still used for |
|---|---|---|
| **Linear regression** | Fit a straight line | Baselines; interpretable regression |
| **Logistic regression** | Linear model + sigmoid → probability | Strong text baseline on top of embeddings |
| **k-Nearest Neighbours** | Predict from the k closest training points | The intuition behind vector search |
| **Decision tree** | Nested if/else learned from data | Interpretability |
| **Random forest** | Many trees, averaged | Robust tabular baseline |
| **Gradient boosting (XGBoost / LightGBM)** | Trees fixing previous trees' errors | **Still state of the art on tabular data** |
| **SVM** | Maximum-margin separating boundary | Small-data classification |
| **k-means** | Cluster into k groups | Unsupervised grouping |
| **PCA** | Project to fewer dimensions, keep variance | Dimensionality reduction, visualisation |

**Always build a baseline first.** Majority-class, random, or logistic-regression-on-
embeddings. If your sophisticated model can't beat the baseline, something is wrong — and
this happened repeatedly in this project.

**Resources**
- Andrew Ng, *Machine Learning Specialization* (Coursera) — the canonical starting point
- StatQuest with Josh Starmer (YouTube) — best short explanations of individual concepts
- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — the
  standard practical book
- scikit-learn user guide — unusually well written; read the "supervised learning" section

---

### 0C — Neural networks (4 days)

**A neuron:** take inputs, multiply each by a weight, add them up, add a bias, pass through
an activation function. `output = activation(Σ wᵢxᵢ + b)`. That's it.

**A network:** neurons in layers. Each layer's output is the next layer's input. "Deep"
just means many layers.

**Activation functions — why they exist:** without them, stacking linear layers collapses to
a single linear layer, so depth buys nothing. Non-linearity is what lets networks represent
complex functions.

| Function | Shape | Used for |
|---|---|---|
| **Sigmoid** | Squashes to (0,1) | Binary output probability. Saturates → vanishing gradients |
| **Tanh** | Squashes to (−1,1) | Older hidden layers |
| **ReLU** | `max(0,x)` | The default for years — cheap, no saturation for positive values |
| **GELU / SiLU** | Smooth ReLU variants | **What transformers use** |
| **Softmax** | Vector → probability distribution | Final layer for multi-class and next-token prediction |

**Loss functions — how "wrong" is measured**

| Loss | For |
|---|---|
| **Mean Squared Error** | Regression |
| **Binary cross-entropy** | Binary classification |
| **Categorical cross-entropy** | Multi-class. **This is LLM training loss** — cross-entropy over the vocabulary |

Cross-entropy heavily penalises being confidently wrong, which is exactly the behaviour you
want to discourage.

**How training actually works — the loop**

1. **Forward pass** — run a batch through, get predictions
2. **Compute loss** — how wrong were they
3. **Backward pass (backpropagation)** — chain rule, applied layer by layer backwards, to get
   each weight's contribution to the error
4. **Optimiser step** — nudge every weight slightly in the direction that reduces loss
5. Repeat for the next batch

Backpropagation is *just the chain rule from calculus*, applied efficiently. It is not a
learning algorithm — gradient descent is. Backprop only computes the gradients.

**Gradient descent variants**

| Variant | Update after | Trade-off |
|---|---|---|
| **Batch** | The entire dataset | Stable, impossibly slow at scale |
| **Stochastic (SGD)** | Every single sample | Fast, very noisy |
| **Mini-batch** | A batch of 8–512 | **What everyone uses** — the practical middle |

**Optimisers**

| Optimiser | Adds |
|---|---|
| **SGD** | The plain update rule |
| **+ Momentum** | Keeps some previous direction; smooths through noise |
| **Adam** | Per-parameter adaptive learning rates + momentum. The default |
| **AdamW** | Adam with correctly decoupled weight decay. **What transformer training uses** — `adamw_8bit` in our config |

**Learning rate — the most important hyperparameter**

Too high: loss oscillates or diverges to NaN. Too low: training crawls or stalls in a bad
place. **Warmup** starts small and ramps up, because early gradients are unreliable.
**Schedules** (cosine, linear decay) reduce it over training so you settle rather than bounce.
Our config: `learning_rate=2e-4`, `warmup_steps=10`, `lr_scheduler_type="cosine"`.

**Things that go wrong**

| Problem | Symptom | Mitigation |
|---|---|---|
| **Vanishing gradients** | Deep layers stop learning | ReLU-family activations, residual connections, normalisation |
| **Exploding gradients** | Loss → NaN | Gradient clipping, lower LR |
| **Dead ReLUs** | Neurons stuck at 0 forever | Leaky ReLU / GELU, better init |
| **Bad initialisation** | Won't converge | Xavier / He init — modern frameworks handle it |

**Normalisation:** BatchNorm normalises across the batch; **LayerNorm** normalises across
features within one sample. Transformers use LayerNorm because batch statistics are unstable
with variable-length sequences.

**Embeddings, properly:** an embedding layer is a lookup table — a matrix with one learned
row per vocabulary item. Token ID 4823 means "return row 4823." Those rows are *learned
parameters*, trained like any other weight. That's all an embedding is.

**Why deep learning replaced classical ML for text:** classical ML needs hand-engineered
features — someone decides that word counts, sentence length, and keyword presence are what
matter. Neural networks learn the features themselves from raw data. This is
*representation learning*, and it's why performance kept improving with more data instead of
plateauing.

**Resources**
- 3Blue1Brown, *Neural Networks* (4 videos, ~1 hour) — the best visual explanation that exists
- Andrej Karpathy, *"The spelled-out intro to neural networks and backpropagation"*
  (micrograd) — builds backprop from scratch in pure Python. Do this one with your hands
- fast.ai, *Practical Deep Learning for Coders* — top-down, code-first

---

## Phase 1 — Classical NLP (1 week)

The stuff that came before transformers. Still running in this pipeline.

| Concept | In this project |
|---|---|
| **Tokenization** | Splitting text into units — word-level, subword (BPE, WordPiece, SentencePiece) |
| **Bag of words / TF-IDF** | Counting words, weighting rare ones higher |
| **BM25** | Improved TF-IDF. Literally in the retrieval step — `rank_bm25` in `regnlp_rag_pipeline.py` |
| **Word2Vec / GloVe** | First "words as vectors" — king − man + woman ≈ queen |
| **Sequence labelling** | NER, POS tagging |
| **Sparse vs dense retrieval** | BM25 (keyword) vs embeddings (semantic). This pipeline uses **both**, fused with RRF |

**Interview trap — "Why not just use BM25?"**
BM25 matches exact words. A policy saying *"personnel must complete cybersecurity education
annually"* won't match a control saying *"staff shall undergo security awareness training"* —
zero shared keywords, same meaning. Dense retrieval catches that. But BM25 catches exact
regulatory terms and control IDs that embeddings blur. Hence hybrid + Reciprocal Rank Fusion.

**Resource:** Hugging Face NLP Course, ch. 1–2 — `huggingface.co/learn/nlp-course` (free)

---

## Phase 2 — The Transformer (2 weeks) ← the core

Single most important thing to understand deeply. Everything after 2017 is a variation.

| Concept | Explain it like this |
|---|---|
| **Attention** | Every word looks at every other word and decides how much to care |
| **Query, Key, Value** | Q = what I'm looking for; K = what I advertise; V = what I actually give you. Score = Q·K |
| **Self- vs cross-attention** | Within one sequence vs across two |
| **Multi-head attention** | Run attention 8–32× in parallel; each head learns a different relationship |
| **Positional encoding** | Attention has no sense of order, so position is injected. Modern models use **RoPE** (rotary) — Llama uses this |
| **Feed-forward network (FFN/MLP)** | Per-token processing between attention layers |
| **Residual connections + LayerNorm** | Why deep networks train at all |
| **Encoder / decoder / encoder-decoder** | BERT = encoder. GPT & Llama = decoder. T5 = both |
| **Causal masking** | Decoders can't see the future — this is what makes them generative |
| **Context window** | How many tokens fit. Ours: `max_seq_length=2048` |

**Resources, in this order:**
1. Jay Alammar, *"The Illustrated Transformer"* — read it three times
2. Andrej Karpathy, *"Let's build GPT: from scratch, in code, spelled out"* (YouTube, ~2h)
   — **code a transformer yourself. Non-negotiable for interviewing well.**
3. Vaswani et al. 2017, *"Attention Is All You Need"* — read *after* the above, not before

---

## Phase 3 — Encoder Models: the BERT family (1 week)

**This is what `nli_judge.py` and the obligation classifier are.**

| Concept | In this project |
|---|---|
| **Masked Language Modelling** | BERT pretraining: hide 15% of words, predict them |
| **`[CLS]` token** | Pooled representation used for classification — the obligation classifier reads this |
| **Fine-tuning for classification** | Add a head, train on labels → `models/obligation-classifier-legalbert-finetuned` |
| **Domain-adapted BERT** | LegalBERT, BioBERT, SciBERT — same architecture, different pretraining corpus. **We use LegalBERT** |
| **Bi-encoder vs cross-encoder** | Bi-encoder: embed separately, compare — fast, for retrieval. Cross-encoder: feed both together, one score — slow, accurate, for reranking. **This pipeline uses both** |
| **NLI (Natural Language Inference)** | premise + hypothesis → entailment / contradiction / neutral. **The entire basis of `nli_judge.py`** |
| **Sentence-BERT** | How to get sentence embeddings that are actually comparable |

**Key insight:** the per-atom decomposition exists to turn a hard 3-class judgment into an
easy binary NLI problem. Premise = policy passage. Hypothesis = one atomic obligation.
Entailment = covered.

**Papers:** Devlin et al. 2018 (BERT) · Reimers & Gurevych 2019 (Sentence-BERT) ·
Koreeda & Manning 2021 (**ContractNLI** — closest published work to what we're doing, read carefully)

---

## Phase 4 — Decoder Models: GPT / Llama family (1 week)

**This is what the Unsloth fine-tune is.**

| Concept | Detail |
|---|---|
| **Autoregressive generation** | Predict next token, append, repeat |
| **Next-token prediction as pretraining** | The entire objective — everything emerges from this |
| **Sampling: temperature, top-k, top-p** | How randomness is controlled. Our inference: `do_sample=False` = greedy = deterministic |
| **Logits → softmax → probabilities** | The output layer |
| **KV cache** | Why generation speeds up after the first token |
| **Instruction tuning** | Base → chat model. Why `Llama-3.2-1B-Instruct`, not `Llama-3.2-1B` |
| **Chat templates / special tokens** | `<\|begin_of_text\|>`, EOS. Our `format_row()` appends `eos` |
| **Scaling laws** | Kaplan et al. 2020 · Chinchilla (Hoffmann et al. 2022) — compute-optimal data/param ratio |
| **Mixture of Experts (MoE)** | Activate only some parameters per token — Mixtral, DeepSeek |

---

## Phase 5 — Fine-tuning (2 weeks) ← most job-relevant

Done four times in this repo. Now understand what happened.

| Concept | Our experience |
|---|---|
| **Full FT vs PEFT** | We used PEFT. Full FT of 1B params needs ~40GB VRAM; the T4 had 14.5GB |
| **LoRA** | Freeze the model, train two small matrices whose product is the weight update. `r=16, lora_alpha=32` |
| **What `r` (rank) means** | Adapter capacity. Higher = more expressive, more memory |
| **`target_modules`** | Which layers get adapters — we targeted `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **QLoRA** | LoRA over a 4-bit quantized base. `load_in_4bit=True` — why 1B fit on a T4 |
| **Quantization** | fp32 / fp16 / bf16 / int8 / int4 — precision vs memory. Our log: `Bfloat16 = FALSE` (T4 is pre-Ampere) |
| **Gradient accumulation** | Simulate batch 16 with batch 4 × 4 steps — exactly our config |
| **Gradient checkpointing** | Trade compute for memory |
| **LR, warmup, schedulers** | `2e-4`, `warmup_steps=10`, cosine. Why LoRA uses much higher LR than full FT |
| **Catastrophic forgetting** | Model loses general ability while learning the task |
| **Class imbalance** | 22% positives → 3× oversample of the covered class |
| **SFT vs RLHF vs DPO** | Supervised → preference-based. InstructGPT (Ouyang 2022), DPO (Rafailov 2023) |

**Papers:** Hu et al. 2021 (LoRA) · Dettmers et al. 2023 (QLoRA) · Sebastian Raschka's LoRA blog posts

### Interview gold — the four fine-tuning attempts

| Attempt | Approach | Result | Root cause |
|---|---|---|---|
| Reranker v2 (MarginMSE) | Fine-tune cross-encoder on relevance | Spearman +0.053 → −0.019 | MS-MARCO relevance geometry can't express legal coverage |
| Reranker v3 (BPR) | Pairwise ranking loss, fewer negatives | Spearman −0.018, ranking inverted | Same — architecture, not hyperparameters |
| Phi-3 holistic | 3-class FA/PA/NA directly | macro F1 = 0.191 | Learned family priors, not semantics |
| **Llama-1B per-atom** | **Binary yes/no per sub-obligation** | **posF1 = 0.68** | **Correct task reformulation** |

The lesson: loss decreased in every run. Task performance only improved when the *problem
formulation* changed. Loss ≠ task performance.

---

## Phase 6 — Retrieval & RAG (1.5 weeks)

| Concept | In this project |
|---|---|
| **Embeddings & vector spaces** | Dense retrieval |
| **Vector databases** | FAISS, pgvector, Chroma, Qdrant. Product plan uses pgvector |
| **ANN search** | HNSW, IVF — why search stays fast at scale |
| **Chunking strategies** | Fixed / sentence / semantic / hierarchical. Our section-level passages (~19 per policy) mix boilerplate with substance — a known defect |
| **Hybrid search + RRF** | Reciprocal Rank Fusion — already in use |
| **Reranking** | Cross-encoder second pass |
| **RAG** | Lewis et al. 2020 |
| **Recall@k / MRR / nDCG** | Retrieval metrics. Ours: R@5 = 81.5% |
| **Retrieval/generation split** | Key finding: retrieval works (R@5 = 81%), judgment was broken. Diagnose separately |
| **Multi-hop retrieval** | When evidence spans documents — **our 25 PA→NA failures.** This is why Tier 2 exists |
| **GraphRAG, ColBERT (late interaction)** | Modern variants — know by name |

---

## Phase 7 — Evaluation (1 week) ← most underrated

Already learned the hard way here.

| Concept | Our experience |
|---|---|
| **Precision / Recall / F1** | P=0.79 R=0.61 F1=0.69 |
| **Macro vs micro vs weighted F1** | Phi-3 macro F1 = 0.191 |
| **Why accuracy lies** | 90% NA base rate → always-predict-NA scores 90% and is useless |
| **Confusion matrix reading** | Ours shows PA is the broken class (6/59 correct) |
| **Cohen's kappa** | Chance-corrected agreement. Ours = 0.234 |
| **Inter-annotator agreement** | The human ceiling. If humans agree only 80%, no model beats 80% |
| **Calibration** | Does "0.8 confidence" mean right 80% of the time? |
| **Abstention / selective prediction** | Say "I don't know" instead of guessing — the product differentiator |
| **Circular evaluation** | **Key methodological lesson** — labels derived from Claude's judgments measure similarity-to-Claude, not correctness |
| **Non-exhaustive golden sets** | Our "false positives" may be unlabeled true positives |
| **LLM-as-judge** | And its biases: position, verbosity, self-preference |

### Current confusion matrix (Llama per-atom, 162 pairs)

| Human label | Total | AI correct | Main error |
|---|---|---|---|
| Fully Addressed | 17 | 8 (47%) | 5 → NA |
| **Partially Addressed** | **59** | **6 (10%)** | **28 → FA, 25 → NA** |
| Not Addressed | 86 | 74 (86%) | 12 false positives |

The 28 PA→FA disagreements are probable *label* errors (conservative annotation bias).
The 25 PA→NA disagreements are genuine model failures needing multi-hop evidence (Tier 2).

---

## Phase 8 — Prompting & Reasoning (1 week)

| Concept | Note |
|---|---|
| Zero-shot / few-shot / in-context learning | No weight updates; the model learns from the prompt |
| **Chain-of-Thought** | Wei et al. 2022 — "think step by step" |
| Self-consistency | Sample multiple reasoning paths, majority vote |
| Structured output / JSON mode / tool schemas | `llm_judge_anthropic.py` uses a JSON schema |
| Prompt injection | Security concern, especially when ingesting documents |
| System / user / assistant roles | |
| Context engineering | Managing what goes in the window |

---

## Phase 9 — Agents & Tools (1.5 weeks) ← where the market is

This is **Tier 2** of the cascade and **PhD Work 2**.

| Concept | Note |
|---|---|
| **Function / tool calling** | Model emits a structured call, you execute it, return the result |
| **ReAct** | Yao et al. 2022 — Reason + Act loop. The Tier 2 investigator |
| **Agent loops** | Plan → act → observe → repeat until done |
| **Multi-agent systems** | Tier 3 adversarial verifier is a second agent |
| **MCP (Model Context Protocol)** | Anthropic's open standard connecting models to tools/data. **PhD Work 2 is MCP security** |
| **Memory: short- vs long-term** | |
| **Guardrails & sandboxing** | |
| **Evaluating agents** | Much harder than evaluating single calls |

---

## Phase 10 — Alignment & Safety (few days)

| Concept |
|---|
| RLHF — InstructGPT (Ouyang et al. 2022) |
| Reward models |
| DPO (Rafailov et al. 2023) — simpler RLHF alternative, now dominant |
| Constitutional AI (Bai et al. 2022) |
| Hallucination: causes and mitigations |
| Red-teaming, jailbreaks |
| Bias, fairness, model cards |

---

## Phase 11 — Recent Frontier (ongoing)

Know each by name plus a one-line explanation.

| Topic |
|---|
| **Reasoning models** — RL on chain-of-thought; test-time compute scaling |
| **FlashAttention** (Dao et al. 2022) — IO-aware attention; why long contexts became affordable |
| **Long context** — 1M+ token windows, RoPE scaling, needle-in-haystack evals |
| **Speculative decoding** — small model drafts, big model verifies |
| **vLLM / PagedAttention** — production serving |
| **Distillation** — big teacher → small student. **The realistic product path here** |
| **Synthetic data generation** — already have `scripts/generate_synthetic_data.py` |
| **Multimodal** — vision-language models; relevant for scanned PDFs |
| **Small language models** — Phi, Gemma, Qwen. On-device / on-prem |
| **Model merging** |
| **Agentic RAG** — retrieval as a tool the agent chooses to call |

---

## Concept Question Bank

Questions about the *concepts*, not any particular project. Format: **question** followed by
the key points a good answer contains. If you can produce the bullets from memory, you know it.

### Machine Learning Fundamentals

**Q. What is machine learning, in one sentence?**
Deriving rules from data rather than writing them, so the system generalises to inputs it
hasn't seen.

**Q. Supervised vs unsupervised vs self-supervised?**
Supervised: labelled examples, learn input→output. Unsupervised: no labels, find structure.
Self-supervised: labels derived automatically from the data itself — mask a word and predict it,
or predict the next token. Self-supervised is how LLMs are pretrained, and it's why they scale:
raw text generates unlimited training signal with no human labelling.

**Q. Parameters vs hyperparameters?**
Parameters are learned by training (weights, biases). Hyperparameters are set by you before
training (learning rate, batch size, number of layers, LoRA rank). You tune hyperparameters on
validation data, never on test.

**Q. Why do you need three data splits rather than two?**
Train fits parameters. Validation selects hyperparameters and checkpoints. Test estimates true
generalisation. If you tune against the test set — even by picking the best of several runs — it
has become a validation set and no longer estimates anything honestly.

**Q. Explain overfitting and how you detect it.**
The model memorises training data including its noise, so it performs well on train and badly on
unseen data. Detect it by tracking train and validation loss together: overfitting shows as train
loss still falling while validation loss flattens or rises. Fixes: more data, regularisation,
early stopping, a smaller model, dropout.

**Q. Explain the bias-variance trade-off.**
Bias is error from over-simple assumptions — the model can't represent the true pattern
(underfitting). Variance is error from over-sensitivity to the particular training sample —
it fits noise (overfitting). Model capacity trades one against the other; total error is
minimised somewhere in between.

**Q. Name four regularisation techniques and what each does.**
L2/weight decay — penalise large weights, pushing toward simpler functions. L1 — penalise
absolute weight magnitude, driving some weights to exactly zero. Dropout — randomly zero
activations during training so no single unit is relied upon. Early stopping — halt when
validation stops improving.

**Q. What is cross-validation and when do you need it?**
Split into k folds, train on k−1 and validate on the held-out one, rotate, average the results.
Needed when data is small — a single split's estimate has high variance, so a good or bad number
may just be luck of the draw.

**Q. What is data leakage? Give three forms.**
Test information influencing training or model selection. Forms: overlapping or duplicated
examples across splits; features computed using future or target information; and tuning
thresholds or selecting checkpoints on the test set. In LLMs, benchmark contamination in
pretraining data is a fourth.

**Q. Why is class imbalance a problem, and what do you do about it?**
The model can achieve high accuracy by always predicting the majority class while learning
nothing. Handle it with per-class metrics rather than accuracy, resampling (oversample minority
or undersample majority), class weights in the loss, threshold tuning, or generating synthetic
minority examples.

**Q. Why always build a baseline first?**
It tells you whether your complex model is adding anything. A majority-class predictor, a
random predictor, or logistic regression on embeddings takes minutes. If the sophisticated
approach doesn't beat it, the problem is your data, your framing, or a bug — not your model
capacity.

**Q. What is still competitive with deep learning, and where?**
Gradient-boosted trees (XGBoost, LightGBM) on tabular data — frequently still state of the art.
Deep learning wins where representation learning matters: text, images, audio, anything where
hand-designing features is impractical.

---

### Neural Networks & Training

**Q. What does a single neuron compute?**
A weighted sum of its inputs plus a bias, passed through a non-linear activation:
`activation(Σ wᵢxᵢ + b)`.

**Q. Why do neural networks need non-linear activation functions?**
Composing linear functions gives a linear function, so a network of any depth without them
collapses to a single linear layer. Non-linearity is what makes depth meaningful.

**Q. Why did ReLU replace sigmoid in hidden layers?**
Sigmoid saturates — its gradient approaches zero for large positive or negative inputs, so
gradients vanish through deep stacks. ReLU has constant gradient 1 for positive inputs, is
cheap to compute, and trains far deeper networks. Transformers now use smooth variants (GELU,
SiLU) which perform slightly better.

**Q. What is backpropagation? Is it a learning algorithm?**
It's the efficient application of the chain rule to compute the gradient of the loss with
respect to every parameter, working backwards through the network. It is *not* a learning
algorithm — it only computes gradients. Gradient descent is what uses them to update weights.

**Q. Walk through one training step.**
Forward pass a batch to get predictions → compute the loss against the labels → backpropagate
to get gradients for every parameter → optimiser updates each parameter in the direction that
reduces loss → zero the gradients and repeat.

**Q. Batch vs stochastic vs mini-batch gradient descent?**
Batch uses the whole dataset per update — stable but infeasibly slow. Stochastic uses one sample
— fast but very noisy. Mini-batch (typically 8–512) is the practical compromise everyone uses,
and it maps well onto GPU parallelism.

**Q. What does batch size affect?**
Gradient noise (larger = smoother estimate), memory use, throughput, and effective learning
dynamics. Larger batches often need a larger learning rate. When memory limits batch size,
gradient accumulation simulates a larger one.

**Q. Why AdamW rather than SGD?**
Adam adapts a per-parameter learning rate from running estimates of gradient mean and variance,
which handles the very different gradient scales across a transformer's layers. AdamW fixes
Adam's incorrect coupling of weight decay into the adaptive term. SGD with momentum can
generalise slightly better on some vision tasks but converges more slowly and needs more tuning.

**Q. What does the learning rate control, and what happens at each extreme?**
The step size of each weight update. Too high: the loss oscillates or diverges to NaN. Too low:
training is slow and can stall in a poor region. It's usually the single most impactful
hyperparameter.

**Q. Why use learning-rate warmup and a decay schedule?**
Early in training, gradient estimates are unreliable and adaptive optimiser statistics haven't
stabilised, so large steps can destabilise the model — warmup ramps up gently. Decay later
(cosine or linear) lets the model settle into a minimum instead of bouncing around it.

**Q. Vanishing and exploding gradients — cause and mitigation?**
Gradients are products of many terms through the chain rule; repeated multiplication by values
below 1 shrinks them toward zero, above 1 blows them up. Mitigations: ReLU-family activations,
residual connections that give gradients a direct path, normalisation layers, careful
initialisation, and gradient clipping for the exploding case.

**Q. BatchNorm vs LayerNorm — why do transformers use LayerNorm?**
BatchNorm normalises each feature across the batch, so its statistics depend on batch
composition and size. LayerNorm normalises across features within a single sample, making it
independent of batch and sequence length. That independence is what NLP needs, since sequences
vary in length and inference often runs with batch size 1.

**Q. What is an embedding layer, mechanically?**
A learned lookup table — a matrix with one row per vocabulary entry. Given token ID *i*, it
returns row *i*. Those rows are ordinary trainable parameters updated by gradient descent.

**Q. What is the loss function used to train an LLM?**
Cross-entropy over the vocabulary at each position — the negative log probability the model
assigned to the token that actually came next, averaged over all positions.

**Q. Why is cross-entropy preferred over accuracy as a training loss?**
Accuracy is a step function — it's flat almost everywhere, so it has no useful gradient.
Cross-entropy is smooth and differentiable, and it penalises confident mistakes far more than
uncertain ones, which shapes calibration as well as correctness.

**Q. Your training loss is decreasing but validation loss is rising. What's happening and what do you do?**
Overfitting. Options: stop earlier (that's what early stopping automates), add regularisation
(dropout, weight decay), reduce model capacity, get more or more varied data, or reduce the
number of epochs. First, confirm it isn't a data-splitting bug.

**Q. Your loss becomes NaN. Diagnose it.**
Most commonly learning rate too high, causing divergence. Also: exploding gradients (add
clipping), numerical instability in the loss (a log of zero), bad or missing input normalisation,
corrupted data with infinities or NaNs, or mixed-precision overflow in fp16 — bf16 or loss
scaling helps.

**Q. What does "the model isn't learning — loss is flat" indicate?**
Learning rate too low or effectively zero; a bug where gradients aren't flowing (parameters
frozen, gradients not zeroed, optimiser not stepping, a detached tensor); dead activations;
labels shuffled relative to inputs; or a model with insufficient capacity for the task.

---

### Tokenization & Embeddings

**Q. What is a token, and why not just use words?**
Subword unit. Word-level vocabularies explode in size and can't handle unseen words; character-level
makes sequences too long. Subword (BPE / WordPiece / SentencePiece) balances both — common words
stay whole, rare words split into pieces. Consequences: token count ≠ word count; non-English text
often costs more tokens; numbers and code tokenize awkwardly.

**Q. How does BPE work?**
Start with characters. Repeatedly merge the most frequent adjacent pair into a new token. Stop at
target vocabulary size. The merge list *is* the tokenizer.

**Q. Difference between a word embedding and a contextual embedding?**
Word2Vec/GloVe give one fixed vector per word — "bank" has one vector. BERT gives a different vector
per occurrence depending on context. Contextual embeddings come from the model's hidden states.

**Q. Why cosine similarity rather than Euclidean distance for embeddings?**
Cosine measures direction only, ignoring magnitude. Embedding magnitude often reflects token
frequency or length rather than meaning. Normalised vectors make cosine and dot product equivalent.

---

### Attention & Transformers

**Q. Explain attention in one sentence.**
Every position computes a weighted average of all positions' values, where the weights come from
how well its query matches each key.

**Q. Write the attention formula and explain each part.**
`Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`. Q·Kᵀ scores every query against every key. Softmax
turns scores into weights that sum to 1. Multiply by V to get the weighted sum.

**Q. Why divide by √d_k?**
Dot products grow with dimension. Large values push softmax into a saturated regime where one weight
≈1 and the rest ≈0, producing vanishing gradients. Scaling keeps the variance stable.

**Q. Why multiple heads instead of one big attention?**
Different heads learn different relationships — syntactic, positional, coreference. One head must
average all these into a single attention distribution; multiple heads keep them separable.

**Q. What is the computational complexity of self-attention?**
O(n²·d) in sequence length — every token attends to every token. This is why long context was hard,
and why FlashAttention, sliding-window attention, and linear-attention variants exist.

**Q. Why do transformers need positional encoding?**
Attention is permutation-invariant — shuffle the input and the output shuffles identically. Position
must be injected explicitly. Original paper: sinusoidal. BERT: learned. Llama and most modern models:
RoPE (rotary), which encodes *relative* position directly into the Q·K dot product and extrapolates
better to longer sequences.

**Q. What do residual connections and LayerNorm do?**
Residuals give gradients a direct path to earlier layers, making deep stacks trainable. LayerNorm
stabilises activations. Pre-LN (normalise before the sublayer) trains more stably than post-LN and
is what most modern models use.

**Q. What does the feed-forward network contribute?**
Attention mixes information *across* positions; the FFN transforms each position independently. Most
of a transformer's parameters live in the FFN. It's where much factual knowledge is stored.

---

### Model Families

**Q. Encoder-only vs decoder-only vs encoder-decoder — when do you use each?**
Encoder-only (BERT): bidirectional, sees full context both directions, best for classification,
NER, embeddings, retrieval. Decoder-only (GPT/Llama): causal masking, can only see the past, best for
generation. Encoder-decoder (T5/BART): separate encode and decode, natural for translation and
summarisation. Decoder-only dominates now because one architecture handles everything via prompting.

**Q. What is causal masking and why does it matter?**
Setting attention scores to −∞ for future positions so a token can't see ahead. It's what makes
next-token-prediction training valid — otherwise the model would trivially copy the answer.

**Q. Bi-encoder vs cross-encoder?**
Bi-encoder embeds two texts separately and compares vectors — you can precompute the corpus, so it
scales to millions of documents. Cross-encoder feeds both texts jointly through the model and outputs
one score — far more accurate because tokens attend across both texts, but you must run the model per
pair, so it can't scale. Standard pattern: bi-encoder retrieves top-k, cross-encoder reranks them.

**Q. What is NLI and why is it useful beyond its original task?**
Natural Language Inference: given a premise and a hypothesis, output entailment / contradiction /
neutral. Useful because a huge number of tasks can be reframed as "does this text entail this claim?"
— fact verification, zero-shot classification, coverage checking.

---

### Pretraining & Training

**Q. What is BERT's pretraining objective?**
Masked Language Modelling — mask ~15% of tokens and predict them. Of the masked positions: 80%
replaced with `[MASK]`, 10% replaced with a random token, 10% left unchanged. The 10/10 split exists
because `[MASK]` never appears at fine-tuning time, so the model shouldn't over-rely on it. (Original
BERT also had Next Sentence Prediction; later work found it mostly unhelpful.)

**Q. What is GPT's pretraining objective?**
Next-token prediction. That's it. Every capability — translation, reasoning, code — emerges from
scaling this one objective.

**Q. What are scaling laws?**
Loss falls as a power law in model parameters, dataset size, and compute. Kaplan et al. 2020 showed
the relationship; Chinchilla (Hoffmann et al. 2022) corrected the compute-optimal ratio — models were
badly under-trained on data, and roughly 20 tokens per parameter is closer to optimal.

**Q. What is instruction tuning, and how does it differ from pretraining?**
Pretraining teaches the model to continue text. Instruction tuning fine-tunes on
(instruction, response) pairs so the model answers rather than continues. It's what turns a base model
into a usable chat model.

**Q. What is catastrophic forgetting?**
Fine-tuning on a narrow task degrades general capability, because the weights that encoded general
knowledge get overwritten. Mitigations: lower learning rate, fewer epochs, PEFT (which freezes the
base), mixing general data into the fine-tuning set.

---

### Fine-tuning & Efficiency

**Q. Explain LoRA.**
Freeze the pretrained weight matrix W. Learn a low-rank update: `W' = W + BA`, where B is d×r and A is
r×k with r much smaller than d or k. Only A and B train — typically <1% of parameters. A is randomly
initialised and B is zero-initialised, so BA = 0 at the start and the model begins identical to the
base. At inference you can merge BA into W for zero added latency.

**Q. What does the rank `r` control?**
The capacity of the adapter. Higher r means more expressive updates and more memory. Typical values
8–64. Beyond a point, more rank stops helping — the intuition behind LoRA is that task-specific
adaptation is intrinsically low-rank.

**Q. What is QLoRA and what does it add over LoRA?**
LoRA over a base model quantized to 4-bit. Three contributions: NF4 (a 4-bit datatype matched to the
normal distribution of weights), double quantization (quantizing the quantization constants), and
paged optimizers (to survive memory spikes). Result: a model that needed an A100 fits on a consumer GPU.

**Q. Quantization — what's actually happening, and what breaks?**
Storing weights in fewer bits (fp16 → int8 → int4). Memory falls proportionally; quality degrades
gradually. Post-training quantization is cheap but lossier; quantization-aware training is better but
requires training. Outlier activations are the main failure mode, which is why methods like
LLM.int8() handle outlier channels in higher precision.

**Q. Why does LoRA use a much higher learning rate than full fine-tuning?**
You're training a small number of freshly initialised parameters rather than nudging a huge set of
pretrained ones. Typical LoRA LR is 1e-4 to 3e-4; full fine-tuning is usually 1e-5 to 5e-5.

**Q. What is gradient accumulation and when do you need it?**
Run several forward/backward passes, sum the gradients, then step once. Simulates a larger batch than
fits in memory. Effective batch = per-device batch × accumulation steps × devices.

**Q. SFT vs RLHF vs DPO?**
SFT: supervised fine-tuning on demonstration data — imitate good answers. RLHF: train a reward model
on human preference pairs, then optimise the policy against it with RL (usually PPO). DPO: skip the
reward model — derive a loss directly on preference pairs that implicitly optimises the same
objective. DPO is simpler and more stable, which is why it has largely displaced PPO-based RLHF.

---

### Inference & Generation

**Q. What does temperature do, mechanically?**
Divides logits by T before softmax. T→0 makes the distribution peaked (greedy); T>1 flattens it
(more random). It changes the distribution, not the ranking.

**Q. Top-k vs top-p (nucleus) sampling?**
Top-k: sample from the k highest-probability tokens. Top-p: sample from the smallest set whose
cumulative probability exceeds p. Top-p adapts to the shape of the distribution — when the model is
confident it considers few tokens, when uncertain it considers more. Top-k is fixed regardless.

**Q. When should you use greedy decoding?**
When you want determinism and reproducibility — classification, extraction, evaluation. Anything
where you'd be uncomfortable getting a different answer on a re-run.

**Q. What is a KV cache and why does it exist?**
During generation, keys and values for previous tokens don't change, so recomputing them each step is
wasted work. Cache them. Cost: memory grows linearly with sequence length and batch size — often the
real constraint in serving, which is what MQA/GQA and PagedAttention address.

**Q. MHA vs MQA vs GQA?**
Multi-Head Attention: every head has its own K and V. Multi-Query: all heads share one K/V pair —
much smaller cache, some quality loss. Grouped-Query: heads share K/V within groups — the middle
ground, and what most modern models use.

**Q. What does FlashAttention actually do?**
It doesn't change the mathematics. It changes memory access: tile the computation so the N×N attention
matrix is never materialised in slow HBM, and recompute parts during the backward pass instead of
storing them. Same output, much faster and much less memory.

**Q. Explain speculative decoding.**
A small fast draft model proposes k tokens. The large model verifies all k in a single forward pass
and accepts the longest correct prefix. Output is identical to what the large model alone would
produce; throughput improves because one big forward pass yields several tokens.

---

### Retrieval & RAG

**Q. What problem does RAG solve?**
Models have a knowledge cutoff, can't see private data, and hallucinate when they don't know. RAG
supplies relevant text at inference time so answers are grounded in retrievable, citable sources —
and updating knowledge means updating an index, not retraining.

**Q. Sparse vs dense retrieval — when does each win?**
Sparse (BM25) matches exact terms: strong on rare words, IDs, product codes, names, and it needs no
training. Dense matches meaning: strong on paraphrase and synonym. They fail on opposite cases, which
is why hybrid retrieval plus fusion generally beats either.

**Q. What is Reciprocal Rank Fusion and why use it over score averaging?**
Combine ranked lists by summing 1/(k + rank) across lists. It uses only ranks, so you don't have to
calibrate scores from systems on completely different scales — BM25 scores and cosine similarities
aren't comparable numbers.

**Q. How do you choose chunk size?**
Trade-off: small chunks retrieve precisely but lose surrounding context; large chunks carry context
but dilute the embedding and waste tokens. Overlap reduces boundary loss. The real answer is to
measure Recall@k on your own data — there's no universal size.

**Q. What is ANN search and why do you need it?**
Approximate Nearest Neighbour. Exact search is O(N) per query. HNSW (a navigable small-world graph)
and IVF (inverted file with clustering) give sub-linear search for a small recall loss. Every vector
database is fundamentally an ANN index plus filtering and persistence.

**Q. How do you evaluate a retriever separately from the generator?**
Retrieval metrics on their own: Recall@k, MRR, nDCG. This matters because a bad final answer has two
possible causes — the right passage was never retrieved, or it was retrieved and the model ignored it.
Those need completely different fixes, so measure them separately.

---

### Evaluation

**Q. Why can accuracy be a misleading metric?**
Class imbalance. With a 90% majority class, always predicting the majority scores 90% while being
useless. Use per-class precision/recall/F1, and report the base rate.

**Q. Precision vs recall — when do you optimise for which?**
Precision when false positives are costly (flagging something as compliant when it isn't; medical
over-diagnosis). Recall when false negatives are costly (missing a security incident, missing a
disease). F1 is their harmonic mean — it punishes imbalance between them.

**Q. Macro vs micro vs weighted F1?**
Macro: average per-class F1 equally — small classes count as much as large ones. Micro: pool all
TP/FP/FN then compute once — dominated by large classes, equals accuracy in single-label settings.
Weighted: average weighted by class support. Report macro when minority-class performance matters.

**Q. What is Cohen's kappa and why not just use agreement percentage?**
κ = (pₒ − pₑ) / (1 − pₑ), where pₒ is observed agreement and pₑ is agreement expected by chance. Raw
agreement is inflated when one class dominates — two annotators who both mostly say "no" agree a lot
by accident. Kappa corrects for that.

**Q. ROC-AUC vs PR-AUC?**
ROC-AUC uses TPR vs FPR; with heavy class imbalance FPR stays low even for bad models, so ROC-AUC
looks optimistic. PR-AUC (precision vs recall) is more informative when positives are rare.

**Q. What is model calibration?**
Whether predicted confidence matches empirical accuracy — of everything predicted at 0.8 confidence,
about 80% should be correct. Measured with reliability diagrams and Expected Calibration Error.
Matters whenever you threshold on confidence or route low-confidence cases to humans.

**Q. What is perplexity and what are its limits?**
exp(average negative log-likelihood) — roughly, how surprised the model is by the text. Useful for
comparing language models on the same tokenizer and corpus. Says nothing about whether outputs are
useful, correct, or safe, and isn't comparable across different tokenizers.

**Q. What's wrong with BLEU and ROUGE for modern generation?**
They measure n-gram overlap with a reference. A correct answer worded differently scores badly; a
wrong answer that copies reference phrasing scores well. Fine for translation and extractive
summarisation, weak for open-ended generation.

**Q. What is LLM-as-a-judge and what biases does it have?**
Using a strong model to score outputs. Known biases: position (favouring the first or last option),
verbosity (longer looks better), self-preference (favouring its own family's outputs), and
sensitivity to prompt phrasing. Mitigations: randomise order, score against an explicit rubric,
use multiple judges, and validate against human labels on a sample.

**Q. What is data leakage in evaluation, and what forms does it take?**
Test information influencing training or model selection. Forms: overlapping examples between splits;
tuning thresholds on the test set; benchmark contamination in pretraining data; and *circular
evaluation*, where labels were themselves generated by a model, so you measure similarity to that
model rather than correctness.

---

### Prompting & Reasoning

**Q. Zero-shot vs few-shot vs fine-tuning — how do you choose?**
Zero-shot first: cheapest, no data. Few-shot when the task has a format the model needs to see, and
you have a handful of examples. Fine-tuning when you have hundreds-plus examples, need a consistent
style or format, need lower latency/cost per call, or need to run a small local model.

**Q. Why does chain-of-thought help?**
It gives the model more forward passes to compute the answer, and each generated reasoning token
conditions the next. Errors become visible and localisable. It costs tokens and doesn't always help
on simple tasks.

**Q. Is a chain of thought a faithful explanation of the model's reasoning?**
Not necessarily. The stated reasoning can be post-hoc rationalisation that doesn't reflect the
computation actually driving the answer. Treat it as a useful artifact and a debugging aid, not proof.

**Q. What is self-consistency?**
Sample several reasoning paths at nonzero temperature and take the majority answer. Trades compute for
accuracy on problems with a single correct answer.

**Q. Prompt injection vs jailbreaking?**
Jailbreaking: the *user* tries to make the model violate its own guidelines. Prompt injection: a
*third party* plants instructions in content the model ingests — a document, a web page, a tool result
— so the model follows an attacker's instructions instead of the operator's. Injection is the more
serious systems problem because it doesn't require a malicious user.

---

### Alignment & Safety

**Q. Why does RLHF exist — why isn't SFT enough?**
It's easier for humans to compare two outputs than to write the ideal one. Preference data is cheaper
to collect at scale than demonstrations, and it can express qualities like tone and helpfulness that
are hard to demonstrate exhaustively.

**Q. What is reward hacking?**
The policy finds outputs that score highly under the reward model without being genuinely good —
exploiting the proxy rather than the objective. Mitigations: KL penalty against the reference policy,
reward model ensembles, periodic retraining on fresh preferences.

**Q. Why do models hallucinate?**
Training rewards fluent plausible continuations, not calibrated uncertainty; the objective never
teaches "say you don't know." Retrieval gaps and long-tail facts make it worse. Mitigations:
grounding with retrieval, citation requirements, abstention options, verification passes.

**Q. What is Constitutional AI?**
Using a written set of principles plus model-generated critiques and revisions to supply alignment
feedback, reducing dependence on human labelling for harmlessness.

---

### System Design Questions

These are open-ended; interviewers want your reasoning, not a memorised answer.

**Q. Design a document Q&A system over 10 million internal documents.**
Cover: ingestion and parsing; chunking strategy; embedding model choice; hybrid index; ANN
configuration; metadata filtering and permissions (a user must not retrieve documents they can't
read); reranking; prompt construction and citation; caching; evaluation of retrieval and generation
separately; cost per query; and how you handle updates and deletions.

**Q. You have 500 labelled examples and need a classifier. What do you do?**
Baseline first — zero-shot or few-shot prompting to establish a floor. Then consider: embeddings plus
a simple classifier (very strong at this data size), or LoRA fine-tuning a small model. Full
fine-tuning is likely to overfit. Address class imbalance. Use cross-validation because a single test
split at that size has huge variance. Spend effort on label quality — at 500 examples, label noise
dominates model choice.

**Q. Your fine-tuned model performs worse than the base model. Diagnose it.**
Check in this order: label quality and leakage; whether the metric matches the objective (a falling
loss with a falling task metric means the loss is measuring the wrong thing); learning rate too high;
too many epochs (overfitting) or too few; class imbalance producing a degenerate constant predictor;
train/serve prompt-format mismatch; and whether the task formulation itself suits the architecture.

**Q. How would you reduce inference cost by 10× without losing much quality?**
Route easy cases to a smaller model and escalate only hard ones; distil the large model into a small
one; quantize; cache aggressively — exact-match and semantic caching, plus prompt caching for stable
prefixes; shorten prompts and cap output length; batch where latency allows; and only then consider
better serving infrastructure.

**Q. How do you know a model is good enough to ship?**
Define the metric that maps to the business cost before measuring. Build a held-out set that reflects
production traffic. Establish the human ceiling — inter-annotator agreement bounds what's achievable.
Decide the error budget and which error type is worse. Add abstention for low confidence. Ship behind
a flag with online monitoring, because offline metrics always drift from production.

---

## Suggested Schedule

| Weeks | Focus |
|---|---|
| 1 | Phase 0A + start 0B — maths refresher, Andrew Ng course opening, StatQuest for specific concepts |
| 2 | Phase 0B — ML fundamentals. **Train a logistic regression and a random forest in scikit-learn yourself** |
| 3 | Phase 0C — neural networks. 3Blue1Brown, then **Karpathy's micrograd video with your hands on the keyboard** |
| 4 | Phase 1 — classical NLP, HF course ch. 1–2 |
| 5–6 | **Phase 2 — code a transformer with Karpathy's "Let's build GPT"** |
| 7 | Phase 3 (BERT/NLI) → then re-read `scripts/nli_judge.py` |
| 8 | Phase 4 (decoders) → then re-read `scripts/finetune_atom_compliance.py` |
| 9–10 | Phase 5 (fine-tuning) → re-run the experiments with understanding |
| 11–12 | Phase 6 + 7 (RAG, evaluation) |
| 13 | Phase 8 (prompting) |
| 14–15 | Phase 9 (agents) → build a minimal Tier 2 |
| 16 | Phase 10 + 11, question-bank drilling |

**~4 months at 10 hrs/week including ML foundations.** If you already know ML, skip to week 4
and it's ~3 months.

**Do not skip the hands-on items in weeks 2 and 3.** Reading about gradient descent and
implementing backpropagation are different kinds of knowing, and interviews probe the second.

---

## The One Rule

After each phase, go back and re-read the relevant script in this repo.

| Phase | Re-read |
|---|---|
| 3 — Encoders | `scripts/nli_judge.py`, `scripts/obligation_filter.py` |
| 4 — Decoders | `scripts/finetune_atom_compliance.py` |
| 5 — Fine-tuning | `scripts/finetune_reranker.py` (the failure), `finetune_atom_compliance.py` (the success) |
| 6 — Retrieval | `regnlp_rag_pipeline.py`, `single_policy_e2e/run.py` |
| 7 — Evaluation | `scripts/grade_judge.py`, `scripts/adjudicate.py` |
| 9 — Agents | `docs/PRODUCT_SCOPE.md` §4 (the cascade) |

The concepts land differently once you see them in code you wrote.

---

## Book References

Organised by phase. **FREE** marks books legally readable online at no cost — several of the
best ones are free.

### If you buy only three

1. **Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*** (O'Reilly,
   3rd ed. 2022). The single best practical ML book. Covers Phases 0B and 0C completely, with
   code you run. If you read one book, this is it.
2. **Raschka — *Build a Large Language Model (From Scratch)*** (Manning, 2024). Implements a
   GPT-style model step by step in PyTorch — tokenizer, attention, training loop, fine-tuning.
   Exactly Phases 2, 4, 5.
3. **Huyen — *Designing Machine Learning Systems*** (O'Reilly, 2022). What happens after the
   model works — data, deployment, monitoring, drift. This is what separates people who can
   train models from people who can ship them.

---

### Phase 0A — Maths

| Book | Note |
|---|---|
| Deisenroth, Faisal & Ong — *Mathematics for Machine Learning* (Cambridge, 2020) | **FREE** at `mml-book.github.io`. Exactly the maths ML needs, nothing more. Start here |
| Strang — *Introduction to Linear Algebra* | The classic. Pair with his MIT OpenCourseWare lectures |
| Downey — *Think Stats* / *Think Bayes* | **FREE**. Statistics taught through Python code rather than proofs |
| Bruce, Bruce & Gedeck — *Practical Statistics for Data Scientists* (O'Reilly, 2nd ed. 2020) | Statistics filtered to what's actually used |

**Honest advice:** don't front-load six months of maths. Read the MML book's chapters 2–5 for
orientation, then return to specific topics when a concept blocks you.

---

### Phase 0B — Machine learning fundamentals

| Book | Note |
|---|---|
| **Géron — *Hands-On Machine Learning*** (O'Reilly, 3rd ed. 2022) | **The recommendation.** Part I covers classical ML end to end with scikit-learn |
| James, Witten, Hastie & Tibshirani — *An Introduction to Statistical Learning* | **FREE** at `statlearning.com`. The gentlest rigorous treatment. Python edition (ISLP, 2023) exists |
| Raschka, Liu & Mirjalili — *Machine Learning with PyTorch and Scikit-Learn* (Packt, 2022) | Strong alternative to Géron; more PyTorch-forward |
| Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning* | **FREE**. ISL's advanced sibling. Reference, not a read-through |
| Bishop — *Pattern Recognition and Machine Learning* (Springer, 2006) | **FREE** PDF. Rigorous, Bayesian-leaning, pre-deep-learning. Classic but not where a beginner starts |
| Murphy — *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022) | **FREE** draft. Comprehensive and modern; heavier maths |

---

### Phase 0C — Neural networks & deep learning

| Book | Note |
|---|---|
| **Prince — *Understanding Deep Learning*** (MIT Press, 2023) | **FREE** at `udlbook.github.io`. The best modern deep-learning textbook — clear diagrams, covers transformers and diffusion. Strongly recommended |
| Chollet — *Deep Learning with Python* (Manning, 2nd ed. 2021) | Written by the creator of Keras. Excellent intuition-building; Keras rather than PyTorch |
| Zhang, Lipton, Li & Smola — *Dive into Deep Learning* | **FREE** at `d2l.ai`. Interactive, runnable notebooks, PyTorch/TF/JAX versions |
| Fleuret — *The Little Book of Deep Learning* | **FREE**. ~160 small pages. Dense summary — good revision, poor first read |
| Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016) | **FREE** at `deeplearningbook.org`. Foundational and heavily cited, but **predates transformers**. Read for theory, not for current practice |
| Bishop & Bishop — *Deep Learning: Foundations and Concepts* (Springer, 2024) | Bishop's modern rewrite; covers transformers properly |

---

### Phase 1–3 — NLP, retrieval, encoder models

| Book | Note |
|---|---|
| **Jurafsky & Martin — *Speech and Language Processing*** (3rd ed. draft) | **FREE** at `web.stanford.edu/~jurafsky/slp3/`. The canonical NLP textbook; the draft is continuously updated and now covers transformers and LLMs |
| Manning, Raghavan & Schütze — *Introduction to Information Retrieval* (Cambridge, 2008) | **FREE** at `nlp.stanford.edu/IR-book/`. **Where BM25, TF-IDF and inverted indexes actually live.** Pre-neural but the retrieval fundamentals are unchanged |
| Tunstall, von Werra & Wolf — *Natural Language Processing with Transformers* (O'Reilly, 2022) | By the Hugging Face team. Practical BERT-era transformer work — classification, NER, QA |

---

### Phase 4–5 — LLMs and fine-tuning

| Book | Note |
|---|---|
| **Raschka — *Build a Large Language Model (From Scratch)*** (Manning, 2024) | **The recommendation for this phase.** Builds a GPT from nothing in PyTorch — tokenizer, attention, pretraining, instruction tuning. Companion GitHub repo |
| Alammar & Grootendorst — *Hands-On Large Language Models* (O'Reilly, 2024) | By the author of *The Illustrated Transformer*. Exceptionally strong visual explanations |
| Iusztin & Labonne — *LLM Engineer's Handbook* (Packt, 2024) | End-to-end LLM system build including fine-tuning and deployment |

---

### Phase 6–8 — RAG, evaluation, prompting, production

| Book | Note |
|---|---|
| **Huyen — *Designing Machine Learning Systems*** (O'Reilly, 2022) | **The recommendation.** Data engineering, feature stores, deployment, monitoring, drift. Not LLM-specific and better for it |
| Huyen — *AI Engineering* (O'Reilly, 2025) | The LLM-era successor — RAG, evaluation, fine-tuning decisions, inference optimisation. Closest single book to what this whole plan covers |
| Ameisen — *Building Machine Learning Powered Applications* (O'Reilly, 2020) | Idea → shipped product, with a strong focus on iteration and evaluation |
| Lakshmanan, Robinson & Munn — *Machine Learning Design Patterns* (O'Reilly, 2020) | 30 named patterns for recurring ML problems |
| Berryman & Ziegler — *Prompt Engineering for LLMs* (O'Reilly, 2024) | Prompting treated as engineering rather than folklore |

---

### Interview preparation

| Book | Note |
|---|---|
| Huyen — *Introduction to Machine Learning Interviews* | **FREE** at `huyenchip.com/ml-interviews-book/`. Question bank plus advice on how ML interviews are actually structured |
| Aminian & Xu — *Machine Learning System Design Interview* (ByteByteGo, 2023) | Worked ML system-design cases, which is where senior interviews concentrate |
| Huo & Singh — *Ace the Data Science Interview* | Broader — SQL, statistics, probability, product sense |

---

### Programming, if needed

| Book | Note |
|---|---|
| McKinney — *Python for Data Analysis* (O'Reilly, 3rd ed. 2022) | **FREE** at `wesmckinney.com/book/`. By the creator of pandas |
| Ramalho — *Fluent Python* (O'Reilly, 2nd ed. 2022) | Intermediate → advanced Python. Worth it once the basics are comfortable |

---

### Reading order for a beginner

```
Month 1   Géron Part I                          → classical ML with code
          (+ MML book chs. 2–5 as reference)
Month 2   Prince, Understanding Deep Learning   → neural networks properly
          chs. 1–9
Month 3   Raschka, Build an LLM From Scratch    → transformers by implementation
          (whole book, typing along)
Month 4   Huyen, AI Engineering                 → production LLM systems
          + Jurafsky & Martin as reference
Ongoing   Huyen, ML Interviews (free)           → drill alongside everything
```

Four of the six books above are free. Buy Géron and Raschka.

---

### A note on what to skip

- **Goodfellow's *Deep Learning*** is the most-cited book in the field and predates
  transformers entirely. Cite it, don't start with it.
- **ESL and Bishop 2006** are excellent references and poor first reads. Reach for them when
  you need depth on a specific method.
- **Anything promising "LLMs in 7 days"** — this space is moving fast enough that thin books
  age badly. Prefer official documentation and papers for anything post-2024.
- **Agent-specific books** are mostly too new to have settled. The documentation
  (`modelcontextprotocol.io`, framework docs) and Anthropic's *Building Effective Agents*
  are currently better than any book on that topic.

---

## Reference Index — papers cited above

| Paper | Authors, year | Why |
|---|---|---|
| Attention Is All You Need | Vaswani et al., 2017 | The transformer |
| BERT | Devlin et al., 2018 | Encoder pretraining |
| Sentence-BERT | Reimers & Gurevych, 2019 | Usable sentence embeddings |
| RAG | Lewis et al., 2020 | Retrieval-augmented generation |
| Scaling Laws | Kaplan et al., 2020 | Size/data/compute relationships |
| LoRA | Hu et al., 2021 | Parameter-efficient fine-tuning |
| ContractNLI | Koreeda & Manning, 2021 | Closest published analogue to this work |
| Chinchilla | Hoffmann et al., 2022 | Compute-optimal training |
| InstructGPT | Ouyang et al., 2022 | RLHF |
| Chain-of-Thought | Wei et al., 2022 | Step-by-step reasoning |
| ReAct | Yao et al., 2022 | Reason + Act agent loop |
| FlashAttention | Dao et al., 2022 | Efficient attention |
| Constitutional AI | Bai et al., 2022 | AI feedback for alignment |
| QLoRA | Dettmers et al., 2023 | 4-bit + LoRA |
| DPO | Rafailov et al., 2023 | Preference tuning without RL |
