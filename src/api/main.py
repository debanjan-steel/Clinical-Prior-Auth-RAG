"""
FastAPI Service for Intelligent Prior Authorization & Utilization Management (UM) RAG.
Provides endpoints for evaluating patient prior authorization requests against payer policies.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import HierarchicalPolicyChunker
from src.retrieval import HybridRetriever, CrossEncoderReranker
from src.generator.um_evaluator import UMEvaluator, PriorAuthDecisionModel


app = FastAPI(
    title="Intelligent Prior Authorization UM RAG API",
    description="Enterprise-grade RAG evaluation pipeline for Healthcare Utilization Management & Clinical Prior Authorization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory pipeline state
chunker = HierarchicalPolicyChunker()
retriever = HybridRetriever()
reranker = CrossEncoderReranker()
evaluator = UMEvaluator()


@app.on_event("startup")
def startup_event():
    """Initializes and indexes clinical policy documents on server startup."""
    policy_dir = PROJECT_ROOT / "data" / "policies"
    chunks = []
    if policy_dir.exists():
        for file in sorted(policy_dir.glob("*.md")):
            chunks.extend(chunker.chunk_file(file))
        retriever.index_chunks(chunks)
        print(f"[API Startup] Successfully loaded {len(chunks)} policy chunks.")


class EvaluatePriorAuthRequest(BaseModel):
    patient_chart_text: str = Field(..., description="Full text of the patient medical record or chart note.")
    procedure_query: Optional[str] = Field(
        None, description="Optional query or procedure specification (e.g. CPT 22612 lumbar fusion)."
    )
    top_k_retrieval: int = Field(default=5, ge=1, le=15)


class PolicyChunkResponse(BaseModel):
    chunk_id: str
    policy_id: str
    section_hierarchy: str
    cross_encoder_score: float
    cpt_codes: List[str]
    icd_codes: List[str]


class PriorAuthResponse(BaseModel):
    decision: PriorAuthDecisionModel
    retrieved_chunks: List[PolicyChunkResponse]


@app.get("/health", tags=["System"])
def health_check():
    """Returns service health status and indexed policy count."""
    return {
        "status": "healthy",
        "indexed_chunks_count": len(retriever.chunks),
        "embedding_model": retriever.embedding_model_name,
        "reranker_model": reranker.model_name,
    }


@app.get("/policies", tags=["Knowledge Base"])
def list_policies():
    """Lists indexed clinical policy bulletins and their hierarchical section structure."""
    if not retriever.chunks:
        return {"policies": []}

    policies = {}
    for c in retriever.chunks:
        pid = c.policy_id
        if pid not in policies:
            policies[pid] = {
                "policy_id": pid,
                "title": c.policy_title,
                "sections": [],
                "cpt_codes": set(),
                "icd_codes": set(),
            }
        policies[pid]["sections"].append(c.section_hierarchy)
        policies[pid]["cpt_codes"].update(c.cpt_codes)
        policies[pid]["icd_codes"].update(c.icd_codes)

    result = []
    for pid, data in policies.items():
        result.append({
            "policy_id": pid,
            "title": data["title"],
            "section_count": len(data["sections"]),
            "cpt_codes": sorted(list(data["cpt_codes"])),
            "icd_codes": sorted(list(data["icd_codes"])),
        })
    return {"policies": result}


@app.post("/evaluate-prior-auth", response_model=PriorAuthResponse, tags=["Utilization Management"])
def evaluate_prior_auth(req: EvaluatePriorAuthRequest):
    """
    Evaluates a patient's medical record against the knowledge base of clinical policies.
    Executes Hybrid Retrieval (Dense + BM25) with Cross-Encoder re-ranking,
    and returns a citation-grounded determination checklist.
    """
    if not req.patient_chart_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Patient chart text cannot be empty.")

    query = req.procedure_query
    if not query:
        # Extract query hints from chart
        metadata = evaluator._extract_patient_metadata(req.patient_chart_text)
        query = f"Prior authorization for {metadata['requested_procedure']} {metadata['primary_diagnosis']}"

    # 1. Retrieve candidates
    candidates = retriever.retrieve(query, top_k=10)
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant clinical policies found in knowledge base.",
        )

    # 2. Re-rank with Cross-Encoder
    reranked = reranker.rerank(query, candidates, top_k=req.top_k_retrieval)

    # 3. Grounded evaluation
    decision = evaluator.evaluate(req.patient_chart_text, reranked)

    retrieved_summary = [
        PolicyChunkResponse(
            chunk_id=c["chunk_id"],
            policy_id=c["policy_id"],
            section_hierarchy=c["section_hierarchy"],
            cross_encoder_score=c.get("cross_encoder_score", 0.0),
            cpt_codes=c.get("cpt_codes", []),
            icd_codes=c.get("icd_codes", []),
        )
        for c in reranked
    ]

    return PriorAuthResponse(decision=decision, retrieved_chunks=retrieved_summary)
