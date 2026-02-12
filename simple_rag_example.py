"""
Simple RAG Pipeline Example
Demonstrates how to use your golden dataset for retrieval and question answering
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Install with: pip install sentence-transformers scikit-learn")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Install with: pip install rank-bm25")


class SimpleRAG:
    """Simple RAG pipeline using your golden dataset"""
    
    def __init__(self, passages_path: str = "data/05_golden_dataset/passages.json"):
        self.passages = []
        self.passage_texts = []
        self.passage_embeddings = None
        self.bm25 = None
        self.embedding_model = None
        
        # Load passages
        self._load_passages(passages_path)
    
    def _load_passages(self, passages_path: str):
        """Load passages from JSON"""
        with open(passages_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.passages = data['passages']
        self.passage_texts = [p['passage_text'] for p in self.passages]
        
        print(f"✓ Loaded {len(self.passages)} passages")
    
    def setup_dense_retrieval(self, model_name: str = "all-MiniLM-L6-v2"):
        """Setup dense retrieval using sentence transformers"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required")
        
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        
        print("Creating embeddings for passages...")
        self.passage_embeddings = self.embedding_model.encode(
            self.passage_texts,
            show_progress_bar=True
        )
        
        print("✓ Dense retrieval ready")
    
    def setup_sparse_retrieval(self):
        """Setup sparse retrieval using BM25"""
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 required")
        
        print("Creating BM25 index...")
        # Tokenize passages
        tokenized_passages = [text.lower().split() for text in self.passage_texts]
        self.bm25 = BM25Okapi(tokenized_passages)
        
        print("✓ Sparse retrieval ready")
    
    def dense_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using dense embeddings"""
        if self.passage_embeddings is None:
            raise ValueError("Dense retrieval not setup. Call setup_dense_retrieval() first")
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.passage_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'passage': self.passages[idx],
                'score': float(similarities[idx])
            })
        
        return results
    
    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using BM25"""
        if self.bm25 is None:
            raise ValueError("Sparse retrieval not setup. Call setup_sparse_retrieval() first")
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'passage': self.passages[idx],
                'score': float(scores[idx])
            })
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid search combining dense and sparse retrieval
        alpha: weight for dense retrieval (1-alpha for sparse)
        """
        dense_results = self.dense_search(query, top_k=top_k * 2)
        sparse_results = self.sparse_search(query, top_k=top_k * 2)
        
        # Create passage_id to score mapping
        passage_scores = {}
        
        # Normalize and combine scores
        if dense_results:
            max_dense = max(r['score'] for r in dense_results)
            for r in dense_results:
                pid = r['passage']['passage_id']
                normalized_score = r['score'] / max_dense if max_dense > 0 else 0
                passage_scores[pid] = passage_scores.get(pid, 0) + alpha * normalized_score
        
        if sparse_results:
            max_sparse = max(r['score'] for r in sparse_results)
            for r in sparse_results:
                pid = r['passage']['passage_id']
                normalized_score = r['score'] / max_sparse if max_sparse > 0 else 0
                passage_scores[pid] = passage_scores.get(pid, 0) + (1 - alpha) * normalized_score
        
        # Sort by combined score
        sorted_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k
        results = []
        passage_dict = {p['passage_id']: p for p in self.passages}
        
        for pid, score in sorted_passages[:top_k]:
            results.append({
                'passage': passage_dict[pid],
                'score': score
            })
        
        return results


def evaluate_retrieval(rag: SimpleRAG, test_path: str = "data/05_golden_dataset/test.json"):
    """Evaluate retrieval on test set"""
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = test_data['questions']
    
    print(f"\nEvaluating on {len(questions)} test questions...")
    print("=" * 60)
    
    recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    
    for i, question in enumerate(questions[:10]):  # Test on first 10
        query = question['question']
        correct_passage_ids = set(p['passage_id'] for p in question['passages'])
        
        # Get retrieved passages
        retrieved = rag.hybrid_search(query, top_k=10)
        retrieved_ids = set(r['passage']['passage_id'] for r in retrieved)
        
        # Calculate recall@k
        for k in [1, 3, 5, 10]:
            top_k_ids = set(r['passage']['passage_id'] for r in retrieved[:k])
            if correct_passage_ids & top_k_ids:  # At least one correct
                recall_at_k[k] += 1
        
        # Print example
        if i == 0:
            print(f"\nExample Question {i+1}:")
            print(f"  Question: {query}")
            print(f"  Correct passages: {list(correct_passage_ids)}")
            print(f"  Retrieved (top 5): {[r['passage']['passage_id'] for r in retrieved[:5]]}")
    
    # Print results
    print(f"\n{'='*60}")
    print("Retrieval Results (on first 10 questions):")
    for k in [1, 3, 5, 10]:
        recall = recall_at_k[k] / min(10, len(questions))
        print(f"  Recall@{k}: {recall:.2%}")


def main():
    """Main example"""
    print("=" * 60)
    print("Simple RAG Pipeline Example")
    print("=" * 60)
    
    # Initialize RAG
    rag = SimpleRAG()
    
    # Setup retrieval methods
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        rag.setup_dense_retrieval()
    
    if BM25_AVAILABLE:
        rag.setup_sparse_retrieval()
    
    # Example query
    query = "What are the requirements for access control?"
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Dense search
    if rag.passage_embeddings is not None:
        print("\nDense Retrieval Results (top 3):")
        results = rag.dense_search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['passage']['passage_id']}] {r['passage']['passage_text'][:100]}...")
            print(f"     Score: {r['score']:.4f}")
    
    # Sparse search
    if rag.bm25 is not None:
        print("\nBM25 Retrieval Results (top 3):")
        results = rag.sparse_search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['passage']['passage_id']}] {r['passage']['passage_text'][:100]}...")
            print(f"     Score: {r['score']:.4f}")
    
    # Hybrid search
    if rag.passage_embeddings is not None and rag.bm25 is not None:
        print("\nHybrid Retrieval Results (top 3):")
        results = rag.hybrid_search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['passage']['passage_id']}] {r['passage']['passage_text'][:100]}...")
            print(f"     Score: {r['score']:.4f}")
    
    # Evaluate on test set
    if SENTENCE_TRANSFORMERS_AVAILABLE and BM25_AVAILABLE:
        test_path = Path("data/05_golden_dataset/test.json")
        if test_path.exists():
            evaluate_retrieval(rag)


if __name__ == "__main__":
    main()
