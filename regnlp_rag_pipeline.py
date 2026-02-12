"""
RegNLP MultiPassage-RegulatoryRAG Implementation
Following the exact methodology from RegNLP's MultiPassage-RegulatoryRAG repository

Key Components:
1. BM25 (sparse/lexical retrieval)
2. Dense retrieval (semantic embeddings)
3. Reciprocal Rank Fusion (RRF)
4. Learning-to-Rank (LTR) with features
5. Graph-derived information from document structure
6. Domain-specific filtering
7. Citation-grounded answer generation
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Install: pip install sentence-transformers scikit-learn")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Install: pip install rank-bm25")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    LTR_AVAILABLE = True
except ImportError:
    LTR_AVAILABLE = False
    print("Install: pip install scikit-learn")


class RegNLPRetrieval:
    """
    RegNLP's Multi-Passage Retrieval System
    Implements BM25 + Dense + RRF + Learning-to-Rank
    """
    
    def __init__(self, passages_path: str = "data/05_golden_dataset/passages.json"):
        self.passages = []
        self.passage_texts = []
        self.passage_embeddings = None
        self.bm25 = None
        self.embedding_model = None
        self.ltr_model = None
        self.scaler = None
        
        # Load passages
        self._load_passages(passages_path)
        
        # Build document structure graph
        self._build_structure_graph()
    
    def _load_passages(self, passages_path: str):
        """Load passages from JSON"""
        with open(passages_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.passages = data['passages']
        self.passage_texts = [p['passage_text'] for p in self.passages]
        
        print(f"✓ Loaded {len(self.passages)} passages")
    
    def _build_structure_graph(self):
        """
        Build graph structure from document hierarchy
        RegNLP uses graph-derived information for ranking
        """
        self.control_hierarchy = {}
        self.control_families = {}
        self.control_subfamilies = {}
        
        for passage in self.passages:
            control_id = passage.get('control_id')
            if control_id:
                # Extract family (M1, T1, etc.)
                family_match = re.match(r'([MT]\d+)', control_id)
                if family_match:
                    family = family_match.group(1)
                    self.control_families[control_id] = family
                
                # Extract subfamily (M1.1, T2.3, etc.)
                subfamily_match = re.match(r'([MT]\d+\.\d+)', control_id)
                if subfamily_match:
                    subfamily = subfamily_match.group(1)
                    self.control_subfamilies[control_id] = subfamily
                
                # Build hierarchy
                parts = control_id.split('.')
                if len(parts) >= 2:
                    parent = '.'.join(parts[:-1])
                    self.control_hierarchy[control_id] = parent
    
    def setup_bm25(self):
        """Setup BM25 sparse retrieval (RegNLP uses this for lexical signals)"""
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 required")
        
        print("Setting up BM25 retrieval...")
        tokenized_passages = [text.lower().split() for text in self.passage_texts]
        self.bm25 = BM25Okapi(tokenized_passages)
        print("✓ BM25 ready")
    
    def setup_dense_retrieval(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Setup dense retrieval (RegNLP uses semantic embeddings)
        For better results, consider: 'sentence-transformers/all-mpnet-base-v2'
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required")
        
        print(f"Setting up dense retrieval with {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        
        print("Creating embeddings...")
        self.passage_embeddings = self.embedding_model.encode(
            self.passage_texts,
            show_progress_bar=True,
            batch_size=32
        )
        print("✓ Dense retrieval ready")
    
    def bm25_search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """BM25 search returning (index, score) pairs"""
        if self.bm25 is None:
            raise ValueError("BM25 not setup. Call setup_bm25() first")
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def dense_search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Dense search returning (index, score) pairs"""
        if self.passage_embeddings is None:
            raise ValueError("Dense retrieval not setup. Call setup_dense_retrieval() first")
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.passage_embeddings)[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]
    
    def reciprocal_rank_fusion(self, 
                              bm25_results: List[Tuple[int, float]],
                              dense_results: List[Tuple[int, float]],
                              k: int = 60) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF) - RegNLP combines BM25 and dense retrieval
        RRF score = sum(1 / (k + rank)) for each retrieval method
        """
        # Create rank dictionaries
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}
        
        # Get all unique passage indices
        all_indices = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for idx in all_indices:
            score = 0.0
            if idx in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[idx])
            if idx in dense_ranks:
                score += 1.0 / (k + dense_ranks[idx])
            rrf_scores[idx] = score
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def extract_features(self, query: str, passage_idx: int) -> Dict[str, float]:
        """
        Extract features for Learning-to-Rank
        RegNLP uses: lexical signals, semantic signals, graph-derived information
        """
        passage = self.passages[passage_idx]
        passage_text = self.passage_texts[passage_idx]
        control_id = passage.get('control_id', '')
        
        features = {}
        
        # 1. Lexical features (BM25 score)
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_score = self.bm25.get_scores(tokenized_query)[passage_idx]
            features['bm25_score'] = float(bm25_score)
        else:
            features['bm25_score'] = 0.0
        
        # 2. Semantic features (dense similarity)
        if self.passage_embeddings is not None:
            query_embedding = self.embedding_model.encode([query])
            similarity = cosine_similarity(
                query_embedding,
                [self.passage_embeddings[passage_idx]]
            )[0][0]
            features['dense_similarity'] = float(similarity)
        else:
            features['dense_similarity'] = 0.0
        
        # 3. Graph-derived features
        # Control ID match (exact match in query)
        query_lower = query.lower()
        control_id_lower = control_id.lower()
        features['control_id_in_query'] = 1.0 if control_id_lower in query_lower else 0.0
        
        # Family match
        family = self.control_families.get(control_id, '')
        features['family_match'] = 1.0 if family and family.lower() in query_lower else 0.0
        
        # Passage length (normalized)
        features['passage_length'] = len(passage_text) / 1000.0  # Normalize
        
        # 4. Text overlap features
        query_words = set(query.lower().split())
        passage_words = set(passage_text.lower().split())
        overlap = len(query_words & passage_words)
        features['word_overlap'] = overlap / max(len(query_words), 1)
        features['word_overlap_ratio'] = overlap / max(len(passage_words), 1)
        
        # 5. Keyword matches (regulatory terms)
        regulatory_keywords = [
            'shall', 'must', 'required', 'ensure', 'implement',
            'control', 'policy', 'procedure', 'risk', 'security'
        ]
        keyword_matches = sum(1 for kw in regulatory_keywords if kw in passage_text.lower())
        features['regulatory_keyword_count'] = keyword_matches
        
        return features
    
    def train_learning_to_rank(self, train_path: str = "data/05_golden_dataset/train.json"):
        """
        Train Learning-to-Rank model using training data
        RegNLP uses feature-based LTR for final ranking
        """
        if not LTR_AVAILABLE:
            print("⚠️  Learning-to-Rank not available. Skipping LTR training.")
            return
        
        print("Training Learning-to-Rank model...")
        
        # Load training data
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Extract features and labels
        X = []  # Features
        y = []  # Relevance labels (1 if correct passage, 0 otherwise)
        
        # Create passage_id to index mapping
        passage_id_to_idx = {p['passage_id']: idx for idx, p in enumerate(self.passages)}
        
        for question_data in train_data['questions']:
            query = question_data['question']
            correct_passage_ids = set(p['passage_id'] for p in question_data['passages'])
            
            # Get candidate passages (top 50 from RRF)
            bm25_results = self.bm25_search(query, top_k=50)
            dense_results = self.dense_search(query, top_k=50)
            rrf_results = self.reciprocal_rank_fusion(bm25_results, dense_results, k=60)
            
            # Extract features for top candidates
            for passage_idx, rrf_score in rrf_results[:50]:
                features_dict = self.extract_features(query, passage_idx)
                # Add RRF score as feature
                features_dict['rrf_score'] = rrf_score
                
                # Create feature vector (ordered)
                feature_vector = [
                    features_dict['bm25_score'],
                    features_dict['dense_similarity'],
                    features_dict['rrf_score'],
                    features_dict['control_id_in_query'],
                    features_dict['family_match'],
                    features_dict['passage_length'],
                    features_dict['word_overlap'],
                    features_dict['word_overlap_ratio'],
                    features_dict['regulatory_keyword_count']
                ]
                
                X.append(feature_vector)
                
                # Label: 1 if this is a correct passage, 0 otherwise
                passage_id = self.passages[passage_idx]['passage_id']
                label = 1.0 if passage_id in correct_passage_ids else 0.0
                y.append(label)
        
        if len(X) == 0:
            print("⚠️  No training data extracted. Skipping LTR.")
            return
        
        # Train model
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest (RegNLP uses similar ensemble methods)
        self.ltr_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.ltr_model.fit(X_scaled, y)
        
        print(f"✓ LTR model trained on {len(X)} examples")
    
    def search(self, query: str, top_k: int = 10, use_ltr: bool = True) -> List[Dict]:
        """
        Main search function following RegNLP's approach:
        1. BM25 retrieval
        2. Dense retrieval
        3. RRF combination
        4. Learning-to-Rank re-ranking (optional)
        """
        # Step 1 & 2: Get results from both methods
        bm25_results = self.bm25_search(query, top_k=100)
        dense_results = self.dense_search(query, top_k=100)
        
        # Step 3: Combine with RRF
        rrf_results = self.reciprocal_rank_fusion(bm25_results, dense_results, k=60)
        
        # Step 4: Re-rank with LTR if available
        if use_ltr and self.ltr_model is not None:
            # Extract features and re-rank
            scored_results = []
            for passage_idx, rrf_score in rrf_results[:50]:  # Top 50 for re-ranking
                features_dict = self.extract_features(query, passage_idx)
                features_dict['rrf_score'] = rrf_score
                
                feature_vector = [
                    features_dict['bm25_score'],
                    features_dict['dense_similarity'],
                    features_dict['rrf_score'],
                    features_dict['control_id_in_query'],
                    features_dict['family_match'],
                    features_dict['passage_length'],
                    features_dict['word_overlap'],
                    features_dict['word_overlap_ratio'],
                    features_dict['regulatory_keyword_count']
                ]
                
                # Predict relevance score
                X_scaled = self.scaler.transform([feature_vector])
                ltr_score = self.ltr_model.predict(X_scaled)[0]
                
                scored_results.append((passage_idx, ltr_score))
            
            # Sort by LTR score
            scored_results.sort(key=lambda x: x[1], reverse=True)
            final_results = scored_results[:top_k]
        else:
            # Use RRF results directly
            final_results = rrf_results[:top_k]
        
        # Format results
        results = []
        for passage_idx, score in final_results:
            passage = self.passages[passage_idx].copy()
            passage['retrieval_score'] = float(score)
            results.append(passage)
        
        return results


class RegNLPAnswerGenerator:
    """
    Answer generation following RegNLP's approach:
    - Domain-specific filtering
    - Obligation-centric prompting
    - Citation-grounded answers
    """
    
    def __init__(self):
        self.obligation_keywords = [
            'shall', 'must', 'required', 'ensure', 'mandatory',
            'prohibited', 'forbidden', 'shall not', 'must not'
        ]
    
    def generate_answer(self, 
                       query: str,
                       retrieved_passages: List[Dict],
                       use_citations: bool = True) -> Dict:
        """
        Generate answer from retrieved passages
        Following RegNLP's citation-grounded approach
        """
        # Filter for obligation-centric content
        relevant_passages = self._filter_obligation_centric(retrieved_passages)
        
        if not relevant_passages:
            relevant_passages = retrieved_passages[:3]  # Fallback to top 3
        
        # Combine passage texts
        passage_texts = [p['passage_text'] for p in relevant_passages]
        combined_context = "\n\n".join([
            f"[Passage {i+1} - {p.get('passage_id', 'unknown')}]: {text}"
            for i, (p, text) in enumerate(zip(relevant_passages, passage_texts))
        ])
        
        # Generate answer (template-based, can be replaced with LLM)
        answer = self._extract_answer_from_passages(query, relevant_passages)
        
        # Add citations
        citations = []
        if use_citations:
            citations = [
                {
                    'passage_id': p.get('passage_id'),
                    'control_id': p.get('control_id'),
                    'text': p['passage_text'][:200]  # Snippet
                }
                for p in relevant_passages
            ]
        
        return {
            'answer': answer,
            'citations': citations,
            'num_passages_used': len(relevant_passages),
            'context': combined_context
        }
    
    def _filter_obligation_centric(self, passages: List[Dict]) -> List[Dict]:
        """Filter passages that contain obligation keywords"""
        filtered = []
        for passage in passages:
            text_lower = passage['passage_text'].lower()
            if any(keyword in text_lower for keyword in self.obligation_keywords):
                filtered.append(passage)
        return filtered if filtered else passages[:3]  # Fallback
    
    def _extract_answer_from_passages(self, query: str, passages: List[Dict]) -> str:
        """Extract answer from passages (can be enhanced with LLM)"""
        # Simple extraction: find sentences that match query keywords
        query_words = set(query.lower().split())
        
        best_sentences = []
        for passage in passages:
            text = passage['passage_text']
            sentences = re.split(r'[.!?]\s+', text)
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words & sentence_words)
                if overlap > 0:
                    best_sentences.append((sentence, overlap))
        
        # Sort by overlap and take top sentences
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        answer_sentences = [s[0] for s in best_sentences[:3]]
        
        return " ".join(answer_sentences) if answer_sentences else passages[0]['passage_text'][:300]


def evaluate_regnlp_approach(retrieval: RegNLPRetrieval,
                            answer_gen: RegNLPAnswerGenerator,
                            test_path: str = "data/05_golden_dataset/test.json"):
    """Evaluate RegNLP approach on test set"""
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['questions']
    
    print(f"\n{'='*60}")
    print("Evaluating RegNLP MultiPassage-RegulatoryRAG Approach")
    print(f"{'='*60}")
    print(f"Test questions: {len(questions)}")
    
    recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    exact_match = 0
    
    for i, question_data in enumerate(questions):
        query = question_data['question']
        correct_passage_ids = set(p['passage_id'] for p in question_data['passages'])
        correct_answer = question_data['answer']
        
        # Retrieve passages
        retrieved = retrieval.search(query, top_k=10, use_ltr=True)
        retrieved_ids = [r['passage_id'] for r in retrieved]
        
        # Calculate Recall@K
        for k in [1, 3, 5, 10]:
            top_k_ids = set(retrieved_ids[:k])
            if correct_passage_ids & top_k_ids:
                recall_at_k[k] += 1
        
        # Generate answer
        answer_result = answer_gen.generate_answer(query, retrieved, use_citations=True)
        generated_answer = answer_result['answer']
        
        # Simple exact match check
        if correct_answer.lower() in generated_answer.lower() or \
           generated_answer.lower() in correct_answer.lower():
            exact_match += 1
        
        # Print example
        if i < 3:
            print(f"\nExample {i+1}:")
            print(f"  Query: {query}")
            print(f"  Correct passages: {list(correct_passage_ids)}")
            print(f"  Retrieved (top 3): {retrieved_ids[:3]}")
            print(f"  Generated answer: {generated_answer[:150]}...")
    
    # Print metrics
    print(f"\n{'='*60}")
    print("Retrieval Metrics:")
    for k in [1, 3, 5, 10]:
        recall = recall_at_k[k] / len(questions)
        print(f"  Recall@{k}: {recall:.2%}")
    
    print(f"\nAnswer Generation:")
    em_score = exact_match / len(questions)
    print(f"  Exact Match: {em_score:.2%}")


def main():
    """Main RegNLP RAG pipeline"""
    print("=" * 60)
    print("RegNLP MultiPassage-RegulatoryRAG Implementation")
    print("=" * 60)
    
    # Initialize retrieval system
    retrieval = RegNLPRetrieval()
    
    # Setup retrieval methods
    retrieval.setup_bm25()
    retrieval.setup_dense_retrieval()
    
    # Train Learning-to-Rank (optional but recommended)
    print("\n" + "=" * 60)
    retrieval.train_learning_to_rank()
    
    # Initialize answer generator
    answer_gen = RegNLPAnswerGenerator()
    
    # Example query
    print("\n" + "=" * 60)
    query = "What are the requirements for access control?"
    print(f"Query: {query}")
    print("=" * 60)
    
    # Search
    results = retrieval.search(query, top_k=5, use_ltr=True)
    print(f"\nRetrieved {len(results)} passages:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['passage_id']}] Score: {r['retrieval_score']:.4f}")
        print(f"   {r['passage_text'][:150]}...")
    
    # Generate answer
    answer_result = answer_gen.generate_answer(query, results, use_citations=True)
    print(f"\n{'='*60}")
    print("Generated Answer:")
    print(f"{'='*60}")
    print(answer_result['answer'])
    print(f"\nCitations ({len(answer_result['citations'])} passages):")
    for cit in answer_result['citations']:
        print(f"  - {cit['passage_id']}: {cit['text'][:100]}...")
    
    # Evaluate on test set
    test_path = Path("data/05_golden_dataset/test.json")
    if test_path.exists():
        evaluate_regnlp_approach(retrieval, answer_gen)


if __name__ == "__main__":
    main()
