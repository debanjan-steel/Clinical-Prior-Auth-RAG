"""
Cross-Encoder Re-Ranker for Prior Authorization Policy Retrieval.
Refines candidate policy chunks retrieved via Hybrid RRF, guaranteeing the
highest-fidelity criteria clauses enter the LLM reasoning context.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class CrossEncoderReranker:
    """
    Re-ranks top candidate chunks using a cross-encoder model
    (e.g., BAAI/bge-reranker-base or jinaai/jina-reranker-v1-tiny-en).
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v1-tiny-en"):
        self.model_name = model_name
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self._reranker = TextCrossEncoder(model_name=self.model_name)
            except Exception:
                self._reranker = None
        return self._reranker

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        code_boost: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks candidates against the clinical query.
        Applies a cross-encoder interaction score plus clinical code alignment boost.
        """
        if not candidates:
            return []

        reranker = self._get_reranker()
        documents = [c.get("content", "") for c in candidates]

        scores = []
        if reranker is not None:
            try:
                raw_scores = list(reranker.rerank(query, documents))
                scores = [float(s) for s in raw_scores]
            except Exception:
                scores = []

        # Fallback scoring if cross-encoder model is not initialized/offline
        if not scores or len(scores) != len(candidates):
            scores = []
            query_lower = query.lower()
            for cand in candidates:
                # Combine RRF base score with text term overlap
                base = cand.get("rrf_score", 0.0) * 10.0
                content_lower = cand.get("content", "").lower()
                query_words = set(query_lower.split())
                overlap = sum(1 for w in query_words if len(w) > 3 and w in content_lower)
                scores.append(base + (overlap * 0.05))

        # Healthcare specific code-alignment boost
        adjusted_results = []
        query_upper = query.upper()
        for idx, (cand, score) in enumerate(zip(candidates, scores)):
            final_score = score
            matched_cpt = [c for c in cand.get("cpt_codes", []) if c.upper() in query_upper]
            matched_icd = [i for i in cand.get("icd_codes", []) if i.upper() in query_upper]

            # Strong clinical boost for matching exact billing / diagnostic codes
            if matched_cpt:
                final_score += 2.5
            if matched_icd:
                final_score += 1.5

            # Prioritize actionable criteria / requirement sections over generic title headers
            hierarchy_lower = cand.get("section_hierarchy", "").lower()
            if any(k in hierarchy_lower for k in ["section 2.", "section 3.", "criteria", "requirements"]):
                final_score += 0.5

            cand_copy = dict(cand)
            cand_copy["cross_encoder_score"] = round(float(final_score), 4)
            cand_copy["matched_cpt"] = matched_cpt
            cand_copy["matched_icd"] = matched_icd
            adjusted_results.append(cand_copy)

        # Sort descending by cross-encoder score
        adjusted_results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        for new_rank, item in enumerate(adjusted_results[:top_k]):
            item["rerank_position"] = new_rank + 1

        return adjusted_results[:top_k]
