"""
Fix retrieval by training LTR and testing
"""

from regnlp_rag_pipeline import RegNLPRetrieval, RegNLPAnswerGenerator

print("=" * 60)
print("FIXING RETRIEVAL - Training LTR Model")
print("=" * 60)

# Initialize
retrieval = RegNLPRetrieval()
retrieval.setup_bm25()
retrieval.setup_dense_retrieval()

# CRITICAL: Train LTR model
print("\nTraining Learning-to-Rank model...")
print("This may take a few minutes...")
retrieval.train_learning_to_rank("data/05_golden_dataset/train.json")

# Test query
query = "What are the requirements for access control?"

print("\n" + "=" * 60)
print("TESTING RETRIEVAL WITH LTR")
print("=" * 60)
print(f"Query: {query}\n")

# Search with LTR
results = retrieval.search(query, top_k=10, use_ltr=True)

print(f"Retrieved {len(results)} passages:\n")
t4_count = 0
for i, r in enumerate(results, 1):
    is_t4 = r['passage_id'].startswith('T4')
    marker = "✓ T4" if is_t4 else "  "
    if is_t4:
        t4_count += 1
    print(f"{i}. {marker} [{r['passage_id']}] Score: {r['retrieval_score']:.4f}")
    print(f"   {r['passage_text'][:100]}...")

print(f"\n{'='*60}")
print(f"RESULTS: {t4_count} T4 passages in top {len(results)}")
print(f"{'='*60}")

if t4_count >= 3:
    print("✅ SUCCESS! LTR is working - T4 passages are ranked correctly")
else:
    print("⚠️  Still some issues - may need to adjust LTR or features")

# Test answer generation
print("\n" + "=" * 60)
print("TESTING ANSWER GENERATION")
print("=" * 60)

answer_gen = RegNLPAnswerGenerator()
answer_result = answer_gen.generate_answer(query, results[:5], use_citations=True)

print(f"\nGenerated Answer:")
print(f"{answer_result['answer']}")
print(f"\nCitations ({len(answer_result['citations'])} passages):")
for cit in answer_result['citations']:
    print(f"  - {cit['passage_id']}: {cit['text'][:80]}...")
