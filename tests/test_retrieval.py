import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import HierarchicalPolicyChunker
from src.retrieval import HybridRetriever, CrossEncoderReranker


def test_pipeline():
    chunker = HierarchicalPolicyChunker()
    chunks = []
    policy_dir = PROJECT_ROOT / "data" / "policies"
    for p in sorted(policy_dir.glob("*.md")):
        chunks.extend(chunker.chunk_file(p))

    print(f"Total chunks extracted: {len(chunks)}")
    assert len(chunks) > 0, "No chunks extracted!"

    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    query = "Prior authorization CPT 22612 lumbar fusion physical therapy 6 weeks"
    candidates = retriever.retrieve(query, top_k=5)
    print(f"\nTop 5 Hybrid RRF candidates for query '{query}':")
    for c in candidates:
        print(f"  [{c['chunk_id']}] RRF: {c['rrf_score']} | Dense: {c['dense_score']} | BM25: {c['bm25_score']}")
        print(f"     Hierarchy: {c['section_hierarchy']}")

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, candidates, top_k=3)

    print("\nTop 3 Re-ranked with Cross-Encoder:")
    for c in reranked:
        print(f"  Rank {c['rerank_position']}: {c['chunk_id']} | Final Score: {c['cross_encoder_score']}")
        print(f"     {c['section_hierarchy']}")

    top_chunk = reranked[0]
    assert "LCD-L35452" in top_chunk["policy_id"]
    print("\n[SUCCESS] Ingestion, Hybrid Retrieval & Re-ranking passed flawlessly!")


if __name__ == "__main__":
    test_pipeline()
