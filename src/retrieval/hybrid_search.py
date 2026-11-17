"""
Hybrid Retrieval Engine for Clinical Policy Documents.
Combines Dense Semantic Vector Search (Qdrant + FastEmbed) with
Sparse Lexical Search (BM25) using Reciprocal Rank Fusion (RRF).
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.ingestion.hierarchical_chunker import PolicyChunk


class ClinicalTokenizer:
    """Tokenizer tailored for healthcare text preserving alphanumeric codes (ICD/CPT)."""
    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Preserve alphanumeric codes like M54.5, L4-L5, CPT 22612, J0135
        normalized = text.lower()
        # Replace non-alphanumeric except dots and hyphens within words
        tokens = re.findall(r'[a-z0-9]+(?:[\.\-][a-z0-9]+)*', normalized)
        return [t for t in tokens if len(t) > 1 or t.isdigit()]


class HybridRetriever:
    """
    Executes dense vector search and sparse BM25 search over hierarchical policy chunks,
    combining ranked results via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, collection_name: str = "payer_clinical_policies", embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.qdrant = QdrantClient(":memory:")
        self.chunks: List[PolicyChunk] = []
        self.chunk_map: Dict[str, PolicyChunk] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []
        self._embedder = None
        self.vector_dim = 384  # Default dimension for bge-small-en-v1.5

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
                self._embedder = TextEmbedding(model_name=self.embedding_model_name)
            except Exception:
                self._embedder = None
        return self._embedder

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        embedder = self._get_embedder()
        if embedder is not None:
            embeddings = list(embedder.embed(texts))
            return [emb.tolist() for emb in embeddings]
        
        # Fallback pseudo-dense hash embedding for offline/testing if model is not loaded
        vectors = []
        for text in texts:
            tokens = ClinicalTokenizer.tokenize(text)
            vec = np.zeros(self.vector_dim, dtype=np.float32)
            for token in tokens:
                idx = abs(hash(token)) % self.vector_dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors

    def index_chunks(self, chunks: List[PolicyChunk]):
        """Indexes policy chunks into both Qdrant Vector Store and BM25 Sparse Index."""
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}

        # 1. Prepare BM25 Corpus
        self.corpus_tokens = [ClinicalTokenizer.tokenize(c.full_text_for_embedding) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # 2. Prepare Vector Embeddings
        texts_to_embed = [c.full_text_for_embedding for c in chunks]
        embeddings = self._compute_embeddings(texts_to_embed)
        self.vector_dim = len(embeddings[0]) if embeddings else 384

        # 3. Create and populate Qdrant collection
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )

        points = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=emb,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "policy_id": chunk.policy_id,
                        "section_hierarchy": chunk.section_hierarchy,
                        "cpt_codes": chunk.cpt_codes,
                        "icd_codes": chunk.icd_codes,
                        "content": chunk.content,
                    },
                )
            )

        self.qdrant.upsert(collection_name=self.collection_name, points=points)

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. Dense Vector search via Qdrant
        2. Sparse BM25 keyword search
        3. Reciprocal Rank Fusion (RRF) to merge and re-score candidates.
        """
        if not self.chunks or self.bm25 is None:
            return []

        # 1. Dense Search
        query_vector = self._compute_embeddings([query])[0]
        query_response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k * 2,
        )
        dense_hits = query_response.points

        dense_ranked_ids = []
        dense_scores_map = {}
        for rank, hit in enumerate(dense_hits):
            chunk_id = hit.payload["chunk_id"]
            dense_ranked_ids.append(chunk_id)
            dense_scores_map[chunk_id] = hit.score

        # 2. Sparse BM25 Search
        query_tokens = ClinicalTokenizer.tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_sparse_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: top_k * 2]

        sparse_ranked_ids = []
        sparse_scores_map = {}
        for idx in top_sparse_indices:
            if bm25_scores[idx] > 0:
                cid = self.chunks[idx].chunk_id
                sparse_ranked_ids.append(cid)
                sparse_scores_map[cid] = float(bm25_scores[idx])

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_Score = sum(weight / (k + rank))
        rrf_scores: Dict[str, float] = {}

        for rank, cid in enumerate(dense_ranked_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_weight / (rrf_k + rank + 1)

        for rank, cid in enumerate(sparse_ranked_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + sparse_weight / (rrf_k + rank + 1)

        # Sort candidate chunks by RRF score descending
        sorted_candidates = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for rank, cid in enumerate(sorted_candidates[:top_k]):
            chunk = self.chunk_map[cid]
            results.append({
                "rank": rank + 1,
                "chunk_id": cid,
                "rrf_score": round(rrf_scores[cid], 5),
                "dense_score": round(dense_scores_map.get(cid, 0.0), 4),
                "bm25_score": round(sparse_scores_map.get(cid, 0.0), 4),
                "policy_id": chunk.policy_id,
                "section_hierarchy": chunk.section_hierarchy,
                "content": chunk.content,
                "full_text": chunk.full_text_for_embedding,
                "cpt_codes": chunk.cpt_codes,
                "icd_codes": chunk.icd_codes,
                "page_number": chunk.page_number,
                "source_file": chunk.source_file,
            })

        return results
