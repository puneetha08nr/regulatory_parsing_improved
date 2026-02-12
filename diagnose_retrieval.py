"""
Diagnostic script to analyze retrieval issues
"""

import json
from regnlp_rag_pipeline import RegNLPRetrieval

# Initialize
retrieval = RegNLPRetrieval()
retrieval.setup_bm25()
retrieval.setup_dense_retrieval()

# Test query
query = "What are the requirements for access control?"

print("=" * 60)
print("DIAGNOSTIC ANALYSIS")
print("=" * 60)
print(f"Query: {query}\n")

# 1. Check BM25 results
print("1. BM25 Results (top 10):")
bm25_results = retrieval.bm25_search(query, top_k=10)
for i, (idx, score) in enumerate(bm25_results[:10], 1):
    passage = retrieval.passages[idx]
    print(f"   {i}. [{passage['passage_id']}] Score: {score:.4f}")
    print(f"      Text: {passage['passage_text'][:80]}...")
    if passage['passage_id'].startswith('T4'):
        print(f"      ✓ T4 (Access Control) passage found!")

# 2. Check Dense results
print("\n2. Dense Retrieval Results (top 10):")
dense_results = retrieval.dense_search(query, top_k=10)
for i, (idx, score) in enumerate(dense_results[:10], 1):
    passage = retrieval.passages[idx]
    print(f"   {i}. [{passage['passage_id']}] Score: {score:.4f}")
    print(f"      Text: {passage['passage_text'][:80]}...")
    if passage['passage_id'].startswith('T4'):
        print(f"      ✓ T4 (Access Control) passage found!")

# 3. Check RRF results
print("\n3. RRF Results (top 10):")
rrf_results = retrieval.reciprocal_rank_fusion(bm25_results, dense_results, k=60)
for i, (idx, score) in enumerate(rrf_results[:10], 1):
    passage = retrieval.passages[idx]
    print(f"   {i}. [{passage['passage_id']}] Score: {score:.4f}")
    print(f"      Text: {passage['passage_text'][:80]}...")
    if passage['passage_id'].startswith('T4'):
        print(f"      ✓ T4 (Access Control) passage found!")

# 4. Check if LTR was trained
print("\n4. Learning-to-Rank Status:")
if retrieval.ltr_model is None:
    print("   ⚠️  LTR model NOT trained!")
    print("   Solution: Run retrieval.train_learning_to_rank() first")
else:
    print("   ✓ LTR model is trained")
    
    # Test with LTR
    print("\n5. Final Search with LTR (top 5):")
    final_results = retrieval.search(query, top_k=5, use_ltr=True)
    for i, r in enumerate(final_results, 1):
        print(f"   {i}. [{r['passage_id']}] Score: {r['retrieval_score']:.4f}")
        print(f"      Text: {r['passage_text'][:80]}...")
        if r['passage_id'].startswith('T4'):
            print(f"      ✓ T4 (Access Control) passage found!")

# 6. Check T4 passages in dataset
print("\n6. T4 Passages in Dataset:")
t4_passages = [p for p in retrieval.passages if p.get('passage_id', '').startswith('T4')]
print(f"   Found {len(t4_passages)} T4 passages")
if len(t4_passages) > 0:
    print("   Sample T4 passages:")
    for p in t4_passages[:5]:
        print(f"   - {p['passage_id']}: {p['passage_text'][:60]}...")

# 7. Check features for a T4 passage
if len(t4_passages) > 0:
    print("\n7. Feature Analysis for T4 Passage:")
    t4_passage = t4_passages[0]
    t4_idx = retrieval.passages.index(t4_passage)
    features = retrieval.extract_features(query, t4_idx)
    print(f"   Passage: {t4_passage['passage_id']}")
    print(f"   Features:")
    for key, value in features.items():
        print(f"     {key}: {value:.4f}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if retrieval.ltr_model is None:
    print("1. ⚠️  Train LTR model: retrieval.train_learning_to_rank()")
    print("2. Check if T4 passages are in top 50 RRF results")
    print("3. Verify BM25 and dense retrieval are finding T4 passages")
else:
    print("1. ✓ LTR is trained")
    print("2. Check if T4 passages have good feature scores")
    print("3. Consider adjusting LTR hyperparameters if needed")
