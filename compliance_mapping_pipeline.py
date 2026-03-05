"""
UAE IA Regulation to Internal Policy Mapping Pipeline
Following RegNLP methodology for compliance mapping.

Strategy (RegNLP-style): For each UAE IA control, first RETRIEVE top-K policy
passages (BM25 + Dense + RRF), then run NLI only on those candidates. This
avoids checking each control against every passage. Output = mapping (status +
evidence), RePASs-style. See MAPPING_STRATEGY_AND_REGNLP.md.

Components:
1. Obligation Classification (RegNLP ObligationClassifier approach)
2. RegNLP-style retrieval (BM25 + Dense + RRF) over policy passages
3. Optional: Cross-Encoder reranker (replaces NLI for speed; ~95% accuracy)
4. Optional: Policy graph (same-doc passage neighbors) for candidate expansion
5. Entailment-based labeling (RePASs-style) or reranker scores on candidates
6. Compliance Status Tracking
"""

import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers torch")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import numpy as np
    DENSE_AVAILABLE = True
    RERANKER_AVAILABLE = True
except ImportError:
    DENSE_AVAILABLE = False
    RERANKER_AVAILABLE = False
    CrossEncoder = None
    np = None


@dataclass
class IAControl:
    """UAE IA Regulation Control (Source)"""
    control_id: str  # e.g., "M1.1.1", "T4.2.1"
    control_name: str
    control_text: str
    control_family: str  # M1-M6 or T1-T9
    control_subfamily: str
    is_obligation: bool
    obligation_text: str  # Extracted obligation statement
    metadata: Dict


@dataclass
class PolicyPassage:
    """Internal Policy Passage (Target)"""
    policy_id: str
    policy_name: str
    passage_text: str
    section: str
    metadata: Dict


@dataclass
class ComplianceMapping:
    """Mapping between IA Control and Policy Passage"""
    mapping_id: str
    source_control_id: str  # UAE IA Control ID
    target_policy_id: str  # Internal Policy ID
    status: str  # "Fully Addressed", "Partially Addressed", "Not Addressed"
    entailment_score: float  # NLI score
    evidence_text: str  # Policy text that addresses the control
    mapping_date: str
    reviewer: Optional[str] = None
    notes: Optional[str] = None


class ObligationClassifier:
    """
    Classify and extract obligations from UAE IA controls
    Based on RegNLP ObligationClassifier approach
    """
    
    OBLIGATION_KEYWORDS = [
        'shall', 'must', 'required', 'is required', 'mandatory',
        'shall not', 'must not', 'prohibited', 'forbidden',
        'ensure', 'implement', 'establish', 'maintain', 'document'
    ]
    
    PERMISSION_KEYWORDS = [
        'may', 'can', 'should', 'recommended', 'optional'
    ]
    
    def __init__(self):
        # Try to load LegalBERT or similar model for obligation classification
        self.model = None
        self.tokenizer = None
        # For now, use rule-based approach (can be enhanced with fine-tuned model)
    
    def classify_control(self, control_text: str) -> Tuple[bool, str]:
        """
        Classify if control contains obligation and extract obligation text
        
        Returns:
            (is_obligation, obligation_text)
        """
        control_lower = control_text.lower()
        
        # Check for obligation keywords
        has_obligation = any(keyword in control_lower for keyword in self.OBLIGATION_KEYWORDS)
        has_permission = any(keyword in control_lower for keyword in self.PERMISSION_KEYWORDS)
        
        # Obligation takes precedence
        is_obligation = has_obligation and not (has_permission and not has_obligation)
        
        # Extract obligation statement
        obligation_text = self._extract_obligation(control_text)
        
        return is_obligation, obligation_text
    
    def _extract_obligation(self, text: str) -> str:
        """Extract the obligation statement from control text"""
        # Find sentences with obligation keywords
        sentences = re.split(r'[.!?]\s+', text)
        
        obligation_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in self.OBLIGATION_KEYWORDS):
                obligation_sentences.append(sentence.strip())
        
        if obligation_sentences:
            return " ".join(obligation_sentences)
        
        # Fallback: return first sentence
        return sentences[0].strip() if sentences else text[:200]


class LegalBertObligationClassifier:
    """
    RegNLP-style obligation classifier using LegalBERT (sequence classification).
    Use a model trained with RegNLP/RePASs (train_model.py) or any HuggingFace
    model with 2 labels (0=not obligation, 1=obligation). Same interface as ObligationClassifier.
    """
    # Same keywords as rule-based for extracting obligation text (sentence extraction)
    OBLIGATION_KEYWORDS = [
        'shall', 'must', 'required', 'is required', 'mandatory',
        'shall not', 'must not', 'prohibited', 'forbidden',
        'ensure', 'implement', 'establish', 'maintain', 'document'
    ]

    def __init__(self, model_path: Optional[str] = None, model_name: Optional[str] = None):
        """
        Load LegalBERT obligation classifier.
        Args:
            model_path: Local path to saved model (e.g. RePASs models/obligation-classifier-legalbert).
            model_name: HuggingFace model id (e.g. nlpaueb/legal-bert-base-uncased); only used if model_path is None.
                        For trained obligation classifier, use model_path. Base model with num_labels=2 is untrained.
        """
        self.model = None
        self.tokenizer = None
        self._max_length = 128
        if not NLI_AVAILABLE:
            return
        load_path = model_path or model_name
        if not load_path:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load LegalBERT obligation classifier from {load_path}: {e}")
            self.model = None
            self.tokenizer = None

    def _extract_obligation_text(self, text: str) -> str:
        """Extract obligation sentences (for obligation_text); same logic as rule-based."""
        sentences = re.split(r'[.!?]\s+', text)
        obligation_sentences = [
            s.strip() for s in sentences
            if any(kw in s.lower() for kw in self.OBLIGATION_KEYWORDS)
        ]
        if obligation_sentences:
            return " ".join(obligation_sentences)
        return sentences[0].strip() if sentences else text[:200]

    def classify_control(self, control_text: str) -> Tuple[bool, str]:
        """
        Classify if control contains obligation (model) and extract obligation text (rules).
        Returns:
            (is_obligation, obligation_text)
        """
        obligation_text = self._extract_obligation_text(control_text)
        if self.model is None or self.tokenizer is None:
            # Fallback: rule-based obligation detection
            lower = control_text.lower()
            has_ob = any(kw in lower for kw in self.OBLIGATION_KEYWORDS)
            return has_ob, obligation_text
        try:
            import torch
            inputs = self.tokenizer(
                control_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding="max_length",
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            pred = logits.argmax(dim=-1).item()
            is_obligation = pred == 1
            return is_obligation, obligation_text
        except Exception:
            lower = control_text.lower()
            has_ob = any(kw in lower for kw in self.OBLIGATION_KEYWORDS)
            return has_ob, obligation_text


class EntailmentMapper:
    """
    Map IA controls to policy passages using Natural Language Inference
    Based on RePASs (Regulatory Passage Answer Stability Score) approach
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize NLI model for entailment checking
        
        If model_name is None, tries NLI-specific models first, then falls back.
        Recommended models (in order of preference):
        - microsoft/deberta-v3-base: Good balance (3-class NLI)
        - roberta-large-mnli: High accuracy (3-class NLI)
        - microsoft/deberta-v3-xsmall: Fast but may need fine-tuning (2-class)
        """
        if not NLI_AVAILABLE:
            raise ImportError("Transformers required for entailment mapping")
        
        # Try NLI-specific models first
        if model_name is None:
            nli_models = [
                "microsoft/deberta-v3-base",  # Usually has 3 classes
                "roberta-large-mnli",  # Trained for NLI
                "microsoft/deberta-v3-xsmall"  # Fallback
            ]
            
            model_loaded = False
            for model in nli_models:
                try:
                    print(f"Trying NLI model: {model}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model)
                    self.model.eval()
                    num_labels = self.model.config.num_labels
                    print(f"✓ Loaded {model} with {num_labels} output classes")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"  Could not load {model}: {e}")
                    continue
            
            if not model_loaded:
                raise RuntimeError("Could not load any NLI model")
        else:
            print(f"Loading specified NLI model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            num_labels = self.model.config.num_labels
            print(f"✓ Loaded {model_name} with {num_labels} output classes")
    
    def check_entailment(self, premise: str, hypothesis: str) -> Tuple[bool, float, str]:
        """
        Check if premise (policy) entails hypothesis (IA control requirement)
        
        Returns:
            (is_entailed, entailment_probability, label)
            Always returns entailment prob as second value so status thresholds use it.
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            num_classes = probs.shape[-1]
            entailment_prob = probs[0][0].item()  # Always index 0 = entailment

            # Handle different model configurations
            if num_classes == 3:
                neutral_prob = probs[0][1].item()
                contradiction_prob = probs[0][2].item()
                if entailment_prob > 0.5:
                    label, is_entailed = "entailment", True
                elif contradiction_prob > 0.3:
                    label, is_entailed = "contradiction", False
                else:
                    label, is_entailed = "neutral", False
            elif num_classes == 2:
                if entailment_prob > 0.5:
                    label, is_entailed = "entailment", True
                else:
                    label, is_entailed = "non-entailment", False
            else:
                if entailment_prob > 0.5:
                    label, is_entailed = "entailment", True
                else:
                    label, is_entailed = "non-entailment", False

        return is_entailed, entailment_prob, label
    
    def map_control_to_policy(self, 
                              control: IAControl,
                              policy_passages: List[PolicyPassage],
                              threshold_full: float = 0.6,
                              threshold_partial: float = 0.35) -> List[ComplianceMapping]:
        """
        Map a single IA control to relevant policy passages.
        Uses entailment probability (not label confidence) for status.
        """
        mappings = []
        
        for policy in policy_passages:
            # Check entailment: policy (premise) entails control requirement (hypothesis)
            # score is always entailment probability
            is_entailed, entailment_prob, label = self.check_entailment(
                premise=policy.passage_text,
                hypothesis=control.obligation_text
            )
            
            # Status from entailment probability only
            if entailment_prob >= threshold_full:
                status = "Fully Addressed"
            elif entailment_prob >= threshold_partial:
                status = "Partially Addressed"
            else:
                status = "Not Addressed"
            
            mapping = ComplianceMapping(
                mapping_id=f"{control.control_id}_{policy.policy_id}",
                source_control_id=control.control_id,
                target_policy_id=policy.policy_id,
                status=status,
                entailment_score=entailment_prob,
                evidence_text=policy.passage_text[:500],
                mapping_date=datetime.now().isoformat(),
                notes=f"NLI label: {label}, entailment_prob: {entailment_prob:.3f}"
            )
            
            mappings.append(mapping)
        
        mappings.sort(key=lambda x: x.entailment_score, reverse=True)
        return mappings


class PolicyRetrieval:
    """
    RegNLP-style retrieval over policy passages: BM25 + Dense + RRF.
    Used to get top-K candidate passages per control before NLI (avoids NLI on full corpus).
    """
    def __init__(self, policy_passages: List[PolicyPassage], dense_model: str = "all-MiniLM-L6-v2"):
        self.passages = policy_passages
        self.passage_texts = [p.passage_text for p in policy_passages]
        self.bm25 = None
        self.embedding_model = None
        self.passage_embeddings = None
        self._dense_model_name = dense_model

    def setup(self) -> None:
        """Build BM25 index and optionally dense embeddings."""
        if BM25_AVAILABLE:
            tokenized = [t.lower().split() for t in self.passage_texts]
            self.bm25 = BM25Okapi(tokenized)
        if DENSE_AVAILABLE:
            self.embedding_model = SentenceTransformer(self._dense_model_name)
            self.passage_embeddings = self.embedding_model.encode(
                self.passage_texts, show_progress_bar=True, batch_size=32
            )

    def bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        tokenized_q = query.lower().split()
        scores = self.bm25.get_scores(tokenized_q)
        indexed = [(i, float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.passage_embeddings is None or np is None:
            return []
        q_emb = self.embedding_model.encode([query])
        sim = np.dot(self.passage_embeddings, q_emb.T).ravel()
        # Cosine similarity (SentenceTransformer outputs are typically L2-normalized)
        p_norm = np.linalg.norm(self.passage_embeddings, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q_emb)
        sim = sim / (p_norm.ravel() * q_norm + 1e-9)
        top_idx = np.argsort(sim)[-top_k:][::-1]
        return [(int(i), float(sim[i])) for i in top_idx if sim[i] > 0]

    @staticmethod
    def rrf(bm25_results: List[Tuple[int, float]], dense_results: List[Tuple[int, float]], k: int = 60) -> List[Tuple[int, float]]:
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}
        all_idx = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        rrf_scores = {}
        for idx in all_idx:
            s = 0.0
            if idx in bm25_ranks:
                s += 1.0 / (k + bm25_ranks[idx])
            if idx in dense_ranks:
                s += 1.0 / (k + dense_ranks[idx])
            rrf_scores[idx] = s
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query: str, top_k: int = 20) -> List[int]:
        """Return top_k passage indices for query (control text). RegNLP: BM25 + Dense + RRF."""
        bm25_res = self.bm25_search(query, top_k=min(100, len(self.passages))) if self.bm25 else []
        dense_res = self.dense_search(query, top_k=min(100, len(self.passages))) if self.passage_embeddings is not None else []
        if bm25_res and dense_res:
            fused = self.rrf(bm25_res, dense_res, k=60)
        elif bm25_res:
            fused = [(i, s) for i, s in bm25_res]
        elif dense_res:
            fused = [(i, s) for i, s in dense_res]
        else:
            return []
        return [idx for idx, _ in fused[:top_k]]


class ControlRetrieval:
    """
    BM25 + Dense + RRF index over IA control texts.
    Symmetric counterpart of PolicyRetrieval — used for passage-centric mapping
    where the passage is the query and controls are the corpus.
    """
    def __init__(self, controls: List, dense_model: str = "all-MiniLM-L6-v2"):
        self.controls = controls
        # Build searchable text per control: id + name + obligation text
        self.control_texts = []
        for c in controls:
            parts = [f"{c.control_id}: {c.control_name}"]
            if getattr(c, "control_text", ""):
                parts.append(c.control_text[:400])
            self.control_texts.append(" ".join(parts))
        self.bm25 = None
        self.embedding_model = None
        self.control_embeddings = None
        self._dense_model_name = dense_model

    def setup(self) -> None:
        """Build BM25 index and dense embeddings over control texts."""
        if BM25_AVAILABLE:
            tokenized = [t.lower().split() for t in self.control_texts]
            self.bm25 = BM25Okapi(tokenized)
        if DENSE_AVAILABLE:
            self.embedding_model = SentenceTransformer(self._dense_model_name)
            self.control_embeddings = self.embedding_model.encode(
                self.control_texts, show_progress_bar=False, batch_size=32
            )

    def bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        indexed = [(i, float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.control_embeddings is None or np is None:
            return []
        q_emb = self.embedding_model.encode([query])
        sim = np.dot(self.control_embeddings, q_emb.T).ravel()
        c_norm = np.linalg.norm(self.control_embeddings, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q_emb)
        sim = sim / (c_norm.ravel() * q_norm + 1e-9)
        top_idx = np.argsort(sim)[-top_k:][::-1]
        return [(int(i), float(sim[i])) for i in top_idx if sim[i] > 0]

    def search(self, query: str, top_k: int = 20) -> List[int]:
        """Return top_k control indices for the given passage query. BM25 + Dense + RRF."""
        n = len(self.controls)
        bm25_res  = self.bm25_search(query,  top_k=min(100, n)) if self.bm25 else []
        dense_res = self.dense_search(query, top_k=min(100, n)) if self.control_embeddings is not None else []
        if bm25_res and dense_res:
            fused = PolicyRetrieval.rrf(bm25_res, dense_res, k=60)
        elif bm25_res:
            fused = bm25_res
        elif dense_res:
            fused = dense_res
        else:
            return list(range(min(top_k, n)))   # fallback: first N controls
        return [idx for idx, _ in fused[:top_k]]


def _policy_doc_id_from_target(target_policy_id: str) -> str:
    """Extract policy document id from target_policy_id (part before _passage_N)."""
    if not target_policy_id:
        return "_unknown_"
    return re.sub(r"_passage_\d+$", "", target_policy_id).strip()


def _safe_policy_filename(policy_doc_id: str) -> str:
    """Turn policy doc id into a safe filename."""
    s = re.sub(r'[\s\\/:*?"<>|]+', "_", policy_doc_id.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed_policy"


class CrossEncoderReranker:
    """
    Cross-Encoder reranker: replaces slow NLI with (query, passage) scoring.
    ~95% of NLI accuracy, much faster. Use after BM25+Dense retrieval.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        if RERANKER_AVAILABLE and CrossEncoder is not None:
            self.model = self._try_load(model_name)

    def _ensure_config_model_type(self, path: str) -> bool:
        """If config.json exists but lacks model_type, add it (for HF AutoModel). Returns True if ok to load."""
        import os
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(config_path):
            return True  # no config here, skip
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            if config.get("model_type"):
                return True
            # sentence-transformers sometimes saves without model_type; BGE is RoBERTa-based
            config["model_type"] = "roberta"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            return True
        except Exception:
            return False

    def _patch_all_configs_under(self, root: str) -> None:
        """Patch every config.json under root (including subdirs) to add model_type if missing."""
        import os
        for dirpath, _dirnames, filenames in os.walk(root):
            if "config.json" in filenames:
                self._ensure_config_model_type(dirpath)

    def _try_load(self, model_name: str):
        """Load CrossEncoder, handling sentence-transformers v3 subdirectory layout and missing config.model_type."""
        import os
        # Resolve to absolute path; if relative and not under cwd, try relative to this file (repo root)
        if not os.path.isabs(model_name):
            abs_cwd = os.path.abspath(model_name)
            if os.path.isdir(abs_cwd):
                model_name = abs_cwd
            else:
                repo_root = Path(__file__).resolve().parent
                abs_repo = (repo_root / model_name).resolve()
                model_name = str(abs_repo) if abs_repo.is_dir() else abs_cwd
        # Patch all config.json under the model dir before any load attempt
        if os.path.isdir(model_name):
            self._patch_all_configs_under(model_name)
        candidates = [model_name]
        # sentence-transformers v3 CrossEncoder.save() nests weights under 0_CrossEncoder/
        if os.path.isdir(model_name):
            for sub in ["0_CrossEncoder", "1_CrossEncoder", "best_model"]:
                p = os.path.join(model_name, sub)
                if os.path.isdir(p):
                    candidates.insert(0, p)
        last_err = None
        for path in candidates:
            if not os.path.isdir(path):
                continue
            try:
                m = CrossEncoder(path)
                print(f"   ✓ Reranker loaded from: {path}")
                return m
            except Exception as e:
                last_err = e
        print(f"Warning: Could not load reranker {model_name}: {last_err}")
        return None

    def is_available(self) -> bool:
        return self.model is not None

    def rerank(
        self,
        control: "IAControl",
        query: str,
        passages: List[PolicyPassage],
        top_k: int = 5,
        threshold_full: float = 0.6,
        threshold_partial: float = 0.35,
    ) -> List[ComplianceMapping]:
        """Score (query, passage) pairs; return ComplianceMapping list with reranker score as entailment_score."""
        if not self.model or not passages:
            return []
        pairs = [(query, (p.passage_text[:512] or "")) for p in passages]
        scores = self.model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif hasattr(scores, "__iter__") and not isinstance(scores, list):
            scores = list(scores)
        results = []
        for p, score in zip(passages, scores):
            s = float(score)
            if s >= threshold_full:
                status = "Fully Addressed"
            elif s >= threshold_partial:
                status = "Partially Addressed"
            else:
                status = "Not Addressed"
            results.append(ComplianceMapping(
                mapping_id=f"{control.control_id}_{p.policy_id}",
                source_control_id=control.control_id,
                target_policy_id=p.policy_id,
                status=status,
                entailment_score=s,
                evidence_text=p.passage_text[:500],
                mapping_date=datetime.now().isoformat(),
                notes=f"Reranker: {s:.3f}",
            ))
        results.sort(key=lambda x: x.entailment_score, reverse=True)
        return results[:top_k]


class PolicyGraph:
    """
    GraphRAG-style graph: control -> passages (from retrieval), passage -> passages (same doc).
    Enables multi-hop retrieval (expand by neighbor passages).
    """
    def __init__(self):
        self.control_to_passages: Dict[str, List[Tuple[str, float]]] = {}
        self.passage_neighbors: Dict[str, List[str]] = {}

    def add_control_passage_edges(self, control_id: str, passage_scores: List[Tuple[PolicyPassage, float]]):
        existing = self.control_to_passages.get(control_id, [])
        existing.extend([(p.policy_id, s) for p, s in passage_scores])
        self.control_to_passages[control_id] = existing

    def set_passage_neighbors_from_docs(self, policy_passages_by_doc: Dict[str, List[PolicyPassage]]):
        for doc_id, passages in policy_passages_by_doc.items():
            ids = [p.policy_id for p in passages]
            for pid in ids:
                self.passage_neighbors[pid] = [x for x in ids if x != pid]

    def get_candidates_with_expansion(
        self,
        control_id: str,
        policy_passages: List[PolicyPassage],
    ) -> List[PolicyPassage]:
        passage_by_id = {p.policy_id: p for p in policy_passages}
        out_ids = set()
        if control_id in self.control_to_passages:
            for pid, _ in self.control_to_passages[control_id]:
                out_ids.add(pid)
            for pid in list(out_ids):
                for n in self.passage_neighbors.get(pid, []):
                    out_ids.add(n)
        return [passage_by_id[pid] for pid in out_ids if pid in passage_by_id]


class ComplianceMappingPipeline:
    """
    Complete pipeline for mapping UAE IA Regulation to internal policies.

    Mapping logic (RegNLP-style, per-document corpus):
    - Master list = UAE IA controls. Each policy document is its own corpus.
    - For each control and each policy doc: retrieve top-K passages within that
      doc (BM25 + Dense + RRF), run NLI on those, keep top_k_per_doc; then
      merge across docs and keep top_k_per_control overall.
    - Results are saved per policy document (one JSON per doc).
    """
    
    def __init__(
        self,
        obligation_classifier: str = "rule",
        legalbert_model_path: Optional[str] = None,
        legalbert_model_name: Optional[str] = None,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base",
        use_graph: bool = False,
    ):
        """
        Args:
            obligation_classifier: "rule" (default) or "legalbert".
            legalbert_model_path: Local path to trained LegalBERT obligation classifier.
            legalbert_model_name: HuggingFace model id for LegalBERT.
            use_reranker: If True, use Cross-Encoder reranker instead of NLI (faster, ~95% accuracy).
            reranker_model: Cross-Encoder model name (e.g. BAAI/bge-reranker-base, cross-encoder/ms-marco-MiniLM-L-6-v2).
            use_graph: If True, use GraphRAG-style expansion (control->passages + 1-hop passage neighbors).
        """
        if obligation_classifier == "legalbert" and (legalbert_model_path or legalbert_model_name):
            self.obligation_classifier = LegalBertObligationClassifier(
                model_path=legalbert_model_path,
                model_name=legalbert_model_name,
            )
            if self.obligation_classifier.model is None:
                print("Warning: LegalBERT classifier not available; falling back to rule-based.")
                self.obligation_classifier = ObligationClassifier()
        else:
            self.obligation_classifier = ObligationClassifier()
        self.entailment_mapper = None
        self.retrieval = None
        self.retrievals_by_doc = {}
        self.ia_controls = []
        self.policy_passages = []
        self.policy_passages_by_doc = {}
        self.mappings = []
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        self.use_graph = use_graph
        self.reranker = CrossEncoderReranker(reranker_model) if use_reranker else None
        self.policy_graph = PolicyGraph() if use_graph else None
    
    def load_ia_controls(self, controls_path: str):
        """Load UAE IA controls from structured JSON"""
        with open(controls_path, 'r', encoding='utf-8') as f:
            controls_data = json.load(f)
        
        self.ia_controls = []
        for control_data in controls_data:
            control_id = control_data.get('control', {}).get('id', '')
            control_name = control_data.get('control', {}).get('name', '')
            description = control_data.get('control', {}).get('description', '')
            
            # Classify obligation
            is_obligation, obligation_text = self.obligation_classifier.classify_control(
                description
            )
            
            control = IAControl(
                control_id=control_id,
                control_name=control_name,
                control_text=description,
                control_family=control_data.get('control_family', {}).get('number', ''),
                control_subfamily=control_data.get('control_subfamily', {}).get('number', ''),
                is_obligation=is_obligation,
                obligation_text=obligation_text,
                metadata={
                    'sub_controls': control_data.get('control', {}).get('sub_controls', []),
                    'guidelines': control_data.get('control', {}).get('implementation_guidelines', '')
                }
            )
            
            self.ia_controls.append(control)
        
        print(f"✓ Loaded {len(self.ia_controls)} IA controls")
    
    def load_policy_passages_from_list(self, policy_data: list):
        """Load policy passages from an in-memory list (e.g. from load_all_policies_from_dir)."""
        self.policy_passages = []
        for item in (policy_data or []):
            passage = PolicyPassage(
                policy_id=item.get('id', ''),
                policy_name=item.get('name', ''),
                passage_text=item.get('text', ''),
                section=item.get('section', ''),
                metadata=item.get('metadata', {})
            )
            self.policy_passages.append(passage)
        self.policy_passages_by_doc = {}
        for p in self.policy_passages:
            doc_id = _policy_doc_id_from_target(p.policy_id)
            self.policy_passages_by_doc.setdefault(doc_id, []).append(p)
        print(f"✓ Loaded {len(self.policy_passages)} policy passages in {len(self.policy_passages_by_doc)} document(s)")

    def load_policy_passages(self, policy_path: str):
        """Load internal policy passages and group by policy document (corpus per doc)."""
        # Support multiple formats
        if policy_path.endswith('.json'):
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy_data = json.load(f)
            if not isinstance(policy_data, list):
                policy_data = [policy_data]
            self.policy_passages = []
            for item in policy_data:
                passage = PolicyPassage(
                    policy_id=item.get('id', ''),
                    policy_name=item.get('name', ''),
                    passage_text=item.get('text', ''),
                    section=item.get('section', ''),
                    metadata=item.get('metadata', {})
                )
                self.policy_passages.append(passage)
        
        elif policy_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(policy_path)
            self.policy_passages = []
            for _, row in df.iterrows():
                passage = PolicyPassage(
                    policy_id=row.get('policy_id', ''),
                    policy_name=row.get('policy_name', ''),
                    passage_text=row.get('text', ''),
                    section=row.get('section', ''),
                    metadata={}
                )
                self.policy_passages.append(passage)
        
        # Group by policy document (id without _passage_N)
        self.policy_passages_by_doc = {}
        for p in self.policy_passages:
            doc_id = _policy_doc_id_from_target(p.policy_id)
            self.policy_passages_by_doc.setdefault(doc_id, []).append(p)
        print(f"✓ Loaded {len(self.policy_passages)} policy passages in {len(self.policy_passages_by_doc)} document(s)")
    
    def initialize_entailment_mapper(self):
        """Initialize NLI model for entailment checking"""
        if not NLI_AVAILABLE:
            print("⚠️  NLI not available. Using rule-based mapping only.")
            return
        
        self.entailment_mapper = EntailmentMapper()
        print("✓ Entailment mapper ready")
    
    def load_confirmed_negatives(self, path: str) -> None:
        """Load confirmed negative pairs from golden export.

        These are (control_id, policy_passage_id) pairs annotated as 'Not Addressed'
        with high confidence and a mismatch reason (e.g. keyword overlap). They are
        excluded from create_mappings() output so the pipeline does not surface them.

        File: golden_mapping_dataset.json (from create_golden_set_tasks --mode export).
        """
        import json as _json
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = _json.load(f)
        except Exception as e:
            print(f"Warning: could not load confirmed negatives from {path}: {e}")
            return
        self._confirmed_negatives = set()
        for r in rows:
            if r.get("is_hard_negative") or (
                r.get("compliance_status", "").lower().startswith("not")
                and (r.get("confidence") or 0) >= 4
                and r.get("mismatch_reason")
                and r.get("mismatch_reason", "").lower() != "correct match"
            ):
                cid = r.get("control_id", "")
                pid = r.get("policy_passage_id", "")
                if cid and pid:
                    self._confirmed_negatives.add((cid, pid))
        print(f"  Loaded {len(self._confirmed_negatives)} confirmed negative pairs (will be excluded from mappings)")

    def load_not_applicable_passages(self, path: str) -> None:
        """Load passage IDs that should never be matched to any control.

        These are boilerplate/administrative sections (Scope, Purpose, Introduction,
        Table of Contents, Policy Enforcement, Monitoring and Review, etc.) that
        contain generic language but carry no specific obligation coverage.

        They are excluded from retrieval entirely — not just blocked per-control-pair.
        This prevents them resurfacing when new controls are added.

        File: data/07_golden_mapping/not_applicable_passages.json
             (dict of passage_id -> {policy_name, section, reason})
        Also accepts golden_mapping_dataset.json directly — will derive the list
        from passages with mismatch_reason='scope' annotated 2+ times.
        """
        import json as _json
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = _json.load(f)
        except Exception as e:
            print(f"Warning: could not load not-applicable passages from {path}: {e}")
            return

        if isinstance(raw, dict):
            # not_applicable_passages.json format: {passage_id: {meta}}
            self._not_applicable_passages = set(raw.keys())
        elif isinstance(raw, list):
            # golden_mapping_dataset.json format: derive from scope annotations
            from collections import Counter
            scope_counts: Counter = Counter()
            for r in raw:
                if (r.get("compliance_status", "").lower().startswith("not")
                        and r.get("mismatch_reason", "") == "scope"):
                    pid = r.get("policy_passage_id", "")
                    if pid:
                        scope_counts[pid] += 1
            self._not_applicable_passages = {pid for pid, cnt in scope_counts.items() if cnt >= 2}
        else:
            self._not_applicable_passages = set()

        print(f"  Loaded {len(self._not_applicable_passages)} not-applicable passages "
              f"(boilerplate sections — excluded from all mappings)")

    def create_mappings(self, 
                       filter_obligations_only: bool = True,
                       top_k_per_control: int = 5,
                       use_retrieval: bool = True,
                       top_k_retrieve: int = 20,
                       top_k_per_doc: int = 3,
                       top_k_rerank: int = 50,
                       threshold_full: float = 0.45,
                       threshold_partial: float = 0.25):
        """
        Create compliance mappings (RegNLP-style, per-document corpus).

        When use_reranker=True (default): retrieve top_k_rerank, then Cross-Encoder
        rerank -> top_k_per_doc per doc (much faster than NLI). When use_reranker=False
        or reranker unavailable: use NLI on retrieved candidates.
        When use_graph=True: expand candidates via graph (control->passages + 1-hop).
        Call load_confirmed_negatives() before this to suppress known false matches.
        Call load_not_applicable_passages() before this to exclude boilerplate passages.

        Args:
            filter_obligations_only: Only map controls that are obligations
            top_k_per_control: Keep top K policy matches per control
            use_retrieval: Use BM25 + Dense + RRF before rerank/NLI
            top_k_retrieve: Passages to retrieve per control per document (first stage)
            top_k_per_doc: Max mappings to keep per control per document
            top_k_rerank: When using reranker, how many to pass to reranker (then take top_k_per_doc)
        """
        use_reranker_now = self.use_reranker and self.reranker and self.reranker.is_available()
        if use_reranker_now:
            print("  Using Cross-Encoder reranker (fast path; no NLI).")
        else:
            if self.entailment_mapper is None:
                self.initialize_entailment_mapper()
            if self.entailment_mapper is None:
                print("⚠️  Cannot create mappings: NLI not available and reranker not used or unavailable.")
                return
        
        if not self.policy_passages_by_doc:
            print("⚠️  No policy passages by document. Run load_policy_passages first.")
            return
        
        print(f"\nCreating compliance mappings (per-document corpus, RegNLP-style)...")
        print(f"  IA Controls: {len(self.ia_controls)}")
        print(f"  Policy documents: {len(self.policy_passages_by_doc)}")
        print(f"  use_retrieval={use_retrieval}, use_reranker={use_reranker_now}, use_graph={getattr(self, 'use_graph', False)}")
        print(f"  top_k_retrieve={top_k_retrieve}, top_k_rerank={top_k_rerank}, top_k_per_doc={top_k_per_doc}, top_k_per_control={top_k_per_control}")
        
        self.mappings = []
        # Retrieval log: control_id -> list of retrieved passage_ids (in rank order)
        # Saved at the end so evaluate_pipeline.py can compute Recall@K
        self._retrieval_log: dict = {}
        controls_to_map = [c for c in self.ia_controls if c.is_obligation] if filter_obligations_only else self.ia_controls

        # One retrieval index per policy document (can take several minutes: BM25 + Dense embeddings)
        if use_retrieval and BM25_AVAILABLE:
            self.retrievals_by_doc = {}
            doc_items = list(self.policy_passages_by_doc.items())
            total_docs = len(doc_items)
            print(f"  Building retrieval index (BM25 + Dense) for {total_docs} document(s)...")
            for idx, (doc_id, passages) in enumerate(doc_items, 1):
                print(f"  [{idx}/{total_docs}] Indexing: {doc_id[:60]}{'...' if len(doc_id) > 60 else ''} ({len(passages)} passages)", flush=True)
                r = PolicyRetrieval(passages)
                r.setup()
                if r.bm25 is not None or r.passage_embeddings is not None:
                    self.retrievals_by_doc[doc_id] = r
            if not self.retrievals_by_doc:
                use_retrieval = False
            else:
                print(f"✓ Retrieval index ready for {len(self.retrievals_by_doc)} document(s) (BM25 + Dense + RRF per doc)")
        else:
            if use_retrieval and not BM25_AVAILABLE:
                print("⚠️  rank_bm25 not installed; NLI on all passages per doc. pip install rank_bm25")
            use_retrieval = False
            self.retrievals_by_doc = {}

        if self.use_graph and self.policy_graph:
            self.policy_graph.set_passage_neighbors_from_docs(self.policy_passages_by_doc)

        num_controls = len(controls_to_map)
        num_docs = len(self.policy_passages_by_doc)
        stage = "Reranker" if use_reranker_now else "NLI"
        print(f"  Mapping {num_controls} controls ({stage} per control)...", flush=True)
        if use_reranker_now and num_docs > 1:
            print(f"  (First control: {num_docs} docs × rerank — may take a minute)", flush=True)
        for i, control in enumerate(controls_to_map):
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Progress: {i+1}/{num_controls}", flush=True)
            
            query = control.obligation_text or control.control_text
            control_mappings_all = []
            _per_doc_candidates: list = []  # all retrieved passages across docs (for retrieval log)
            doc_idx = 0
            for doc_id, passages in self.policy_passages_by_doc.items():
                doc_idx += 1
                if i == 0 and use_reranker_now:
                    print(f"    Control 1: doc {doc_idx}/{num_docs}", flush=True)
                if use_retrieval and doc_id in self.retrievals_by_doc:
                    retrieval = self.retrievals_by_doc[doc_id]
                    k_retrieve = min(top_k_rerank if use_reranker_now else top_k_retrieve, len(passages))
                    indices = retrieval.search(query, top_k=k_retrieve)
                    candidate_passages = [passages[j] for j in indices if 0 <= j < len(passages)]
                    if not candidate_passages:
                        candidate_passages = passages[:k_retrieve]
                else:
                    candidate_passages = passages[:min(top_k_rerank if use_reranker_now else top_k_retrieve, len(passages))]

                if self.use_graph and self.policy_graph and candidate_passages:
                    seen = {p.policy_id for p in candidate_passages}
                    for p in list(candidate_passages):
                        for neighbor_id in self.policy_graph.passage_neighbors.get(p.policy_id, []):
                            if neighbor_id not in seen:
                                neighbor = next((x for x in passages if x.policy_id == neighbor_id), None)
                                if neighbor:
                                    seen.add(neighbor_id)
                                    candidate_passages.append(neighbor)
                                    if len(candidate_passages) >= top_k_rerank:
                                        break
                        if len(candidate_passages) >= top_k_rerank:
                            break

                # Stash per-doc candidate passage IDs for retrieval log (built globally after reranking)
                _per_doc_candidates.extend(candidate_passages)

                if use_reranker_now and self.reranker and candidate_passages:
                    doc_mappings = self.reranker.rerank(
                        control, query, candidate_passages,
                        top_k=top_k_per_doc,
                        threshold_full=threshold_full,
                        threshold_partial=threshold_partial,
                    )
                    if self.use_graph and self.policy_graph and doc_mappings:
                        passage_scores = []
                        for m in doc_mappings:
                            p = next((x for x in self.policy_passages if x.policy_id == m.target_policy_id), None)
                            if p:
                                passage_scores.append((p, m.entailment_score))
                        if passage_scores:
                            self.policy_graph.add_control_passage_edges(control.control_id, passage_scores)
                else:
                    doc_mappings = self.entailment_mapper.map_control_to_policy(
                        control, candidate_passages
                    )
                    doc_mappings = doc_mappings[:top_k_per_doc]
                control_mappings_all.extend(doc_mappings)

            # ── Build retrieval log in global score order (fixes Recall@K measurement) ───
            # Use cross-encoder scores from control_mappings_all (sorted by score desc) as
            # the global ranking proxy.  This makes Recall@K measure: "was the gold passage
            # in the globally top-K reranked passages?" — independent of document iteration order.
            cid = control.control_id
            seen_in_log: set = set()
            self._retrieval_log[cid] = []
            for m in sorted(control_mappings_all, key=lambda x: x.entailment_score, reverse=True):
                if m.target_policy_id not in seen_in_log:
                    self._retrieval_log[cid].append(m.target_policy_id)
                    seen_in_log.add(m.target_policy_id)
            # Append any retrieved-but-not-reranked passages (scored below top_k_per_doc per doc)
            for p in _per_doc_candidates:
                if p.policy_id not in seen_in_log:
                    self._retrieval_log[cid].append(p.policy_id)
                    seen_in_log.add(p.policy_id)

            control_mappings_all.sort(key=lambda x: x.entailment_score, reverse=True)
            # Filter out globally not-applicable passages (boilerplate sections)
            not_applicable = getattr(self, "_not_applicable_passages", set())
            if not_applicable:
                control_mappings_all = [
                    m for m in control_mappings_all
                    if m.target_policy_id not in not_applicable
                ]
            # Filter out confirmed negatives (known false matches from human annotation)
            confirmed_neg = getattr(self, "_confirmed_negatives", set())
            if confirmed_neg:
                control_mappings_all = [
                    m for m in control_mappings_all
                    if (m.source_control_id, m.target_policy_id) not in confirmed_neg
                ]
            # Only include pairs that score above partial threshold — avoids inflating
            # "predicted positives" with passages the reranker itself labels Not Addressed.
            control_mappings_all = [
                m for m in control_mappings_all
                if m.status in ("Fully Addressed", "Partially Addressed")
            ]
            self.mappings.extend(control_mappings_all[:top_k_per_control])
        
        print(f"✓ Created {len(self.mappings)} compliance mappings")
    
    def save_mappings(self, output_path: str, format: str = "csv"):
        """Save mappings to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            # Save as CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'mapping_id', 'source_control_id', 'control_name', 'control_family',
                    'target_policy_id', 'policy_name', 'status', 'entailment_score',
                    'evidence_text', 'mapping_date', 'notes'
                ])
                writer.writeheader()
                
                for mapping in self.mappings:
                    # Get control and policy info
                    control = next((c for c in self.ia_controls if c.control_id == mapping.source_control_id), None)
                    policy = next((p for p in self.policy_passages if p.policy_id == mapping.target_policy_id), None)
                    
                    writer.writerow({
                        'mapping_id': mapping.mapping_id,
                        'source_control_id': mapping.source_control_id,
                        'control_name': control.control_name if control else '',
                        'control_family': control.control_family if control else '',
                        'target_policy_id': mapping.target_policy_id,
                        'policy_name': policy.policy_name if policy else '',
                        'status': mapping.status,
                        'entailment_score': mapping.entailment_score,
                        'evidence_text': mapping.evidence_text,
                        'mapping_date': mapping.mapping_date,
                        'notes': mapping.notes or ''
                    })
        
        elif format == "json":
            # Save as JSON (single combined file)
            mappings_data = [asdict(m) for m in self.mappings]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mappings_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(self.mappings)} mappings to {output_path}")
    
    def save_mappings_per_policy(self, output_dir: str, also_combined_path: Optional[str] = None):
        """
        Save mappings into one JSON file per policy document under output_dir.
        Uses target_policy_id to group (policy doc = id without _passage_N).
        Optionally write a single combined file to also_combined_path.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        by_policy: Dict[str, List[ComplianceMapping]] = {}
        for m in self.mappings:
            doc_id = _policy_doc_id_from_target(m.target_policy_id)
            by_policy.setdefault(doc_id, []).append(m)
        index = []
        for doc_id, group in sorted(by_policy.items()):
            name = _safe_policy_filename(doc_id)
            path = output_path / f"{name}.json"
            data = [asdict(m) for m in group]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            index.append({"policy_doc_id": doc_id, "file": f"{name}.json", "count": len(group)})
        index.sort(key=lambda x: -x["count"])
        with open(output_path / "_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        print(f"✓ Saved {len(by_policy)} per-policy mapping files under {output_dir}/")
        for entry in index:
            print(f"    {entry['count']:5}  {entry['file']}")
        if also_combined_path:
            self.save_mappings(also_combined_path, format="json")
    
    def create_passage_to_control_mappings(
        self,
        output_path: str,
        controls: Optional[List] = None,
        top_k_retrieve: int = 20,
        threshold_full: float = 0.60,
        threshold_partial: float = 0.35,
        batch_size: int = 64,
        filter_obligations_only: bool = True,
    ) -> None:
        """Passage-centric mapping: BM25+Dense retrieve top-K controls per passage,
        then Cross-Encoder scores (passage, control) pairs.

        Symmetric counterpart of create_mappings() (which is control-centric):
          control-centric:  query=control  → BM25 over passages  → CE(control, passage)
          passage-centric:  query=passage  → BM25 over controls  → CE(passage, control)

        Same computational cost as create_mappings() — top_k_retrieve CE calls per passage,
        not passage × all_controls. Produces genuine coverage scores with no keyword-overlap
        artefacts from one-sided retrieval.

        Output: mappings_by_passage.json
        [
          {
            "passage_id": "...", "policy_doc": "...", "section": "...",
            "passage_text": "...",
            "mapped_controls": [
              {"control_id": "M1.1.1", "control_name": "...", "family": "...",
               "status": "Fully Addressed", "score": 0.82},
              ...  sorted by score desc, only >= threshold_partial
            ]
          }, ...
        ]
        """
        if not self.reranker or not self.reranker.is_available():
            print("⚠️  Cross-Encoder not available — falling back to regrouped view.")
            self.save_mappings_per_passage(output_path)
            return

        controls_to_use = controls or self.ia_controls
        if filter_obligations_only:
            controls_to_use = [c for c in controls_to_use if c.is_obligation]

        not_applicable = getattr(self, "_not_applicable_passages", set())
        passages_to_map = [
            p for p in self.policy_passages
            if p.policy_id not in not_applicable and p.passage_text.strip()
        ]

        print(f"\n[Passage-centric] BM25+Dense → Cross-Encoder (passage → controls)")
        print(f"  {len(passages_to_map)} passages,  {len(controls_to_use)} obligation controls")
        print(f"  top_k_retrieve={top_k_retrieve}  → {len(passages_to_map) * top_k_retrieve:,} CE calls total")

        # Build retrieval index over control texts (one-time cost)
        print("  Building control retrieval index (BM25 + Dense)...")
        ctrl_index = ControlRetrieval(
            controls_to_use,
            dense_model=getattr(self, "_dense_model_name", "all-MiniLM-L6-v2"),
        )
        ctrl_index.setup()
        print("  Control index ready.")

        rows = []
        for p_idx, passage in enumerate(passages_to_map):
            if p_idx % 50 == 0:
                print(f"  passage {p_idx + 1}/{len(passages_to_map)} ...", flush=True)

            passage_text = passage.passage_text[:512]

            # Step 1: retrieve top-K candidate controls for this passage
            candidate_idxs = ctrl_index.search(passage_text, top_k=top_k_retrieve)
            candidate_controls = [controls_to_use[i] for i in candidate_idxs]

            if not candidate_controls:
                continue

            # Step 2: cross-encode (passage, control) pairs in batches
            pairs = [
                (passage_text, ctrl_index.control_texts[i][:512])
                for i in candidate_idxs
            ]
            all_scores: List[float] = []
            for i in range(0, len(pairs), batch_size):
                raw = self.reranker.model.predict(pairs[i: i + batch_size])
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                all_scores.extend([float(s) for s in raw])

            # Step 3: keep controls above threshold
            mapped = []
            for c, score in zip(candidate_controls, all_scores):
                if score < threshold_partial:
                    continue
                status = "Fully Addressed" if score >= threshold_full else "Partially Addressed"
                mapped.append({
                    "control_id":   c.control_id,
                    "control_name": c.control_name,
                    "family":       getattr(c, "control_family", ""),
                    "status":       status,
                    "score":        round(score, 4),
                })

            if not mapped:
                continue

            mapped.sort(key=lambda x: -x["score"])
            rows.append({
                "passage_id":      passage.policy_id,
                "policy_doc":      _policy_doc_id_from_target(passage.policy_id),
                "section":         getattr(passage, "section_title", ""),
                "passage_text":    passage.passage_text,
                "mapped_controls": mapped,
            })

        rows.sort(key=lambda r: -len(r["mapped_controls"]))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        total_mapped = sum(len(r["mapped_controls"]) for r in rows)
        print(f"\n✓ Passage-centric mapping complete:")
        print(f"  {len(rows)} passages cover {total_mapped} (passage, control) pairs")
        print(f"  thresholds: full>={threshold_full}  partial>={threshold_partial}")
        print(f"  Saved → {output_path}")

    def save_mappings_per_passage(
        self,
        output_path: str,
        min_score: float = 0.40,
    ) -> None:
        """Regroup the control-centric mappings into a passage-first view.

        Note: this does NOT re-run the cross-encoder — it only reshuffles the
        results already computed by create_mappings(). For a true passage-centric
        cross-encoder pass use create_passage_to_control_mappings() instead.
        """
        passage_lookup = {p.policy_id: p for p in self.policy_passages}

        by_passage: Dict[str, list] = {}
        for m in self.mappings:
            if m.entailment_score < min_score:
                continue
            by_passage.setdefault(m.target_policy_id, []).append(m)

        rows = []
        for pid, mappings in sorted(by_passage.items()):
            p = passage_lookup.get(pid)
            rows.append({
                "passage_id":   pid,
                "policy_doc":   _policy_doc_id_from_target(pid),
                "section":      getattr(p, "section_title", "") if p else "",
                "passage_text": p.passage_text if p else "",
                "mapped_controls": sorted(
                    [
                        {
                            "control_id":      m.source_control_id,
                            "control_name":    m.control_name,
                            "status":          m.status,
                            "entailment_score": round(m.entailment_score, 4),
                        }
                        for m in mappings
                    ],
                    key=lambda x: -x["entailment_score"],
                ),
            })

        rows.sort(key=lambda r: -len(r["mapped_controls"]))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        covered = len(rows)
        total_pairs = sum(len(r["mapped_controls"]) for r in rows)
        print(f"✓ Passage-centric view (regrouped): {covered} passages, {total_pairs} pairs")
        print(f"  (score filter >= {min_score} | {len(self.mappings) - total_pairs} weak matches excluded)")
        print(f"  Saved → {output_path}")

    def generate_compliance_report(self, output_path: str):
        """Generate compliance status report"""
        # Count by status
        status_counts = {}
        family_counts = {}
        
        for mapping in self.mappings:
            status = mapping.status
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Get control family
            control = next((c for c in self.ia_controls if c.control_id == mapping.source_control_id), None)
            if control:
                family = control.control_family
                if family not in family_counts:
                    family_counts[family] = {'Fully Addressed': 0, 'Partially Addressed': 0, 'Not Addressed': 0}
                family_counts[family][status] = family_counts[family].get(status, 0) + 1
        
        # Generate report
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_mappings': len(self.mappings),
                'total_controls': len(self.ia_controls),
                'total_policies': len(self.policy_passages),
                'status_breakdown': status_counts
            },
            'by_family': family_counts,
            'compliance_rate': {
                'fully_addressed': status_counts.get('Fully Addressed', 0) / len(self.mappings) * 100 if self.mappings else 0,
                'partially_addressed': status_counts.get('Partially Addressed', 0) / len(self.mappings) * 100 if self.mappings else 0,
                'not_addressed': status_counts.get('Not Addressed', 0) / len(self.mappings) * 100 if self.mappings else 0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Compliance report saved to {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("COMPLIANCE SUMMARY")
        print("=" * 60)
        print(f"Total Mappings: {len(self.mappings)}")
        print(f"\nStatus Breakdown:")
        for status, count in status_counts.items():
            percentage = count / len(self.mappings) * 100 if self.mappings else 0
            print(f"  {status}: {count} ({percentage:.1f}%)")
        
        print(f"\nCompliance Rate:")
        print(f"  Fully Addressed: {report['compliance_rate']['fully_addressed']:.1f}%")
        print(f"  Partially Addressed: {report['compliance_rate']['partially_addressed']:.1f}%")
        print(f"  Not Addressed: {report['compliance_rate']['not_addressed']:.1f}%")


def main():
    """Main compliance mapping pipeline"""
    print("=" * 60)
    print("UAE IA Regulation to Policy Compliance Mapping")
    print("=" * 60)
    import os
    legalbert_path = os.environ.get("LEGALBERT_MODEL_PATH") or (
        str(Path("models/obligation-classifier-legalbert").resolve()) if Path("models/obligation-classifier-legalbert").exists() else None
    )
    pipeline = ComplianceMappingPipeline(
        obligation_classifier="legalbert" if legalbert_path else "rule",
        legalbert_model_path=legalbert_path,
    )
    
    # Step 1: Load IA Controls
    print("\n[Step 1] Loading UAE IA Controls...")
    pipeline.load_ia_controls("data/02_processed/uae_ia_controls_structured.json")
    
    # Step 2: Load Policy Passages
    print("\n[Step 2] Loading Internal Policy Passages...")
    # TODO: Replace with your actual policy file path
    # pipeline.load_policy_passages("data/policies/internal_policies.json")
    print("⚠️  Policy file not specified. Please update the path.")
    print("   Example: pipeline.load_policy_passages('data/policies/policies.json')")
    
    # Step 3: Initialize Entailment Mapper
    print("\n[Step 3] Initializing Entailment Mapper...")
    pipeline.initialize_entailment_mapper()
    
    # Step 4: Create Mappings
    if pipeline.policy_passages:
        print("\n[Step 4] Creating Compliance Mappings...")
        pipeline.create_mappings(filter_obligations_only=True, top_k_per_control=5)
        
        # Step 5: Save Mappings (per-policy JSONs + optional combined + CSV)
        print("\n[Step 5] Saving Mappings...")
        pipeline.save_mappings("data/06_compliance_mappings/mappings.csv", format="csv")
        pipeline.save_mappings_per_policy(
            "data/06_compliance_mappings/by_policy",
            also_combined_path="data/06_compliance_mappings/mappings.json",
        )
        
        # Step 6: Generate Report
        print("\n[Step 6] Generating Compliance Report...")
        pipeline.generate_compliance_report("data/06_compliance_mappings/compliance_report.json")
    else:
        print("\n⚠️  Cannot create mappings without policy passages.")


if __name__ == "__main__":
    main()
