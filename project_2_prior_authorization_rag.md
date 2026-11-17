# Project 2: Intelligent Prior Authorization & Utilization Management (UM) RAG Evaluator

> **Target Role**: Generative AI Data Scientist at Sagility  
> **Sagility Business Alignment**: Clinical UM/CM (Utilization Management / Care Management) Practice & Payer Operations  
> **Primary Tech Stack**: Python 3.13, LlamaIndex / LangChain, Qdrant / ChromaDB, BioLinkBERT / BGE Embeddings, Cross-Encoder Re-ranker, Ragas, FastAPI, Docker

---

## 1. Executive Summary & Problem Statement

Sagility supports 6 of the top US health plans and processes millions of member transactions annually. A central pillar of health plan operations is **Utilization Management (UM)**, specifically **Prior Authorization (PA)**. 

### The Operational Challenge
* **Complex Clinical Guidelines**: Health plans publish hundreds of Clinical Policy Bulletins (CPBs) and National Coverage Determinations (NCDs) spanning dozens of pages of legalistic and medical criteria (e.g., criteria for MRI authorization, biological therapies, or joint surgeries).
* **High Manual Review Time**: Clinical nurses and medical directors must review 50+ page patient Electronic Medical Records (EMRs) to find proof that a patient has failed conservative treatment (e.g., physical therapy for 6 weeks, NSAID trials) before approving an advanced procedure.
* **Denials Due to Lack of Information**: Over 30% of prior authorization delays or initial denials occur simply because documentation was overlooked in a voluminous patient chart.

### The Solution
Build an **Advanced Retrieval-Augmented Generation (RAG)** system that:
1. Ingests and hierarchically indexes complex Clinical Policy Bulletins (PDFs) from CMS and major payers.
2. Performs **Hybrid Retrieval** (Dense semantic vectors + BM25 keyword matching) to prevent missing technical medical codes (ICD-10, CPT, HCPCS).
3. Re-ranks candidates with a **Cross-Encoder** to guarantee the top criteria clauses enter the LLM context.
4. Executes strict chain-of-thought grounding against patient chart notes, verifying evidence criterion-by-criterion.
5. Emits an audit-proof recommendation (*Approved*, *Denied*, or *Additional Clinical Info Needed*) with verbatim policy citations.
6. Evaluates retrieval and generation quality quantitatively using **Ragas**.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Processing & Ingestion"]
        CPBPDFs["Clinical Policy Bulletins (CMS / Payer PDFs)"] --> DocParser["Layout-Aware Parser<br/>(PyMuPDF / pdfplumber)"]
        DocParser --> SemanticChunker["Hierarchical Section Chunker<br/>(Preserves Parent Section & Table Headers)"]
        SemanticChunker --> DenseEmbed["BGE-Large / BioLinkBERT Embeddings"]
        SemanticChunker --> SparseIndex["BM25 Lexical Keyword Index"]
        DenseEmbed --> VectorDB[("Qdrant / Chroma Vector Store")]
    end

    subgraph Retrieval["2. Hybrid Retrieval & Re-ranking"]
        PatientChart["Patient Medical Record / Prior Auth Request"] --> QueryGenerator["Clinical Criterion Query Generator"]
        QueryGenerator --> HybridEngine["Reciprocal Rank Fusion (RRF)<br/>Dense Similarity + BM25 Score"]
        VectorDB --> HybridEngine
        SparseIndex --> HybridEngine
        HybridEngine --> TopCandidates["Top 25 Retrieved Policy Chunks"]
        TopCandidates --> CrossEncoder["Cross-Encoder Re-Ranker<br/>(BAAI/bge-reranker-large)"]
        CrossEncoder --> TopKContext["Top 3-5 Highest Scoring Policy Chunks"]
    end

    subgraph Decision["3. Grounded Decision & Benchmark"]
        TopKContext --> LLMEvaluator["Reasoning LLM (Strict Grounding)"]
        PatientChart --> LLMEvaluator
        LLMEvaluator --> DecisionOutput["Structured Decision Report:<br/>• Determination (Approve / Deny / Info Needed)<br/>• Criteria Satisfaction Checklist<br/>• Section & Page Number Citations"]
        DecisionOutput --> RagasBench["Ragas Automated Evaluation<br/>(Faithfulness, Context Precision, Recall)"]
    end
```

---

## 3. Step-by-Step Implementation Guide

### Step 3.1: Policy Bulletins Ingestion & Hierarchical Chunking
* **Data Sources**: Download public **CMS National Coverage Determinations (NCDs)** and Local Coverage Determinations (LCDs) (e.g., *NCD 220.2 for PET Scans*, or *LCD L35452 for Lumbar Spinal Surgery*).
* Standard chunking (500 tokens with 50 token overlap) breaks when a sub-bullet relies on a parent requirement (e.g., "The patient must meet Criteria A, B, and one of the following:").
* Implement hierarchical chunking appending parent breadcrumbs:
  `[Policy: Lumbar Fusion -> Section 3: Indications -> Subsection B: Conservative Therapy]`

```python
# src/ingestion/chunker.py
from dataclasses import dataclass
from typing import List

@dataclass
class PolicyChunk:
    policy_id: str
    section_hierarchy: str
    content: str
    page_number: int

def chunk_policy_document(pdf_path: str) -> List[PolicyChunk]:
    # Extract text preserving markdown headings
    # Return chunks retaining full structural breadcrumbs
    ...
```

### Step 3.2: Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
Medical search requires matching both concept synonyms ("lower back pain" $\leftrightarrow$ "lumbago") and exact alphanumeric codes (ICD-10 `M54.5`, CPT `22612`).

```python
# src/retrieval/hybrid_search.py
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vector_client: QdrantClient, bm25_index: BM25Okapi, chunks: list):
        self.client = vector_client
        self.bm25 = bm25_index
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 20) -> list:
        # 1. Dense Vector Search
        dense_results = self.client.search(collection_name="payer_policies", query_text=query, limit=top_k)
        
        # 2. Sparse BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60
        for rank, res in enumerate(dense_results):
            rrf_scores[res.id] = rrf_scores.get(res.id, 0.0) + (1.0 / (k + rank + 1))
            
        for rank, idx in enumerate(sparse_top_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
            
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [self.chunks[i] for i in sorted_ids[:top_k]]
```

### Step 3.3: Cross-Encoder Re-Ranking
Pass the top 20 retrieved candidates through `BAAI/bge-reranker-large` or `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the top 3–5 policy criteria chunks with true semantic relevance.

### Step 3.4: Strict Grounding & Anti-Hallucination Reasoning Engine
Formulate prompt instructions that force the model to quote verbatim text from the patient's record to satisfy each policy criterion:

```python
# src/chains/evaluator_chain.py
SYSTEM_PROMPT = """
You are an expert Clinical Utilization Management reviewer. Your job is to assess whether a patient's prior authorization request satisfies the clinical policy criteria.

STRICT GROUNDING RULES:
1. For every requirement listed in the policy context, check if the patient record satisfies it.
2. If satisfied, you MUST quote the exact sentence from the patient chart and cite the section of the policy.
3. If not documented, mark the criterion as [UNSATISFIED - LACK OF CLINICAL EVIDENCE].
4. Output your determination as:
   - APPROVED: All mandatory criteria are fulfilled.
   - DENIED: Patient explicitly fails an exclusionary criterion.
   - PEND FOR RECORDS: Evidence is missing for a required criterion.
"""
```

---

## 4. Quantitative Evaluation Suite with Ragas

In Sagility's clinical operations, hallucinations lead to wrongful claim rejections or improper approvals. Standard RAG testing is insufficient; you must quantify pipeline performance.

```python
# evaluation/ragas_evaluation.py
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

def run_evaluation(test_records: list[dict]):
    # test_records contains: question, contexts, answer, ground_truth
    eval_dataset = Dataset.from_list(test_records)
    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    print(f"Faithfulness (Grounding): {results['faithfulness']:.4f}")
    print(f"Context Precision:        {results['context_precision']:.4f}")
    print(f"Answer Relevancy:         {results['answer_relevancy']:.4f}")
    return results
```

### Target Quality Benchmarks:
* **Faithfulness**: $> 0.95$ (ensures zero fabricated medical coverage criteria).
* **Context Precision**: $> 0.88$ (ensures retrieved clauses are directly applicable).
* **Answer Relevancy**: $> 0.90$.

---

## 5. Production Repository Blueprint

```
healthcare-prior-auth-rag/
├── README.md
├── Dockerfile
├── pyproject.toml
├── data/
│   ├── policies_pdf/          # CMS NCD / Payer clinical policy bulletins
│   └── synthetic_charts/     # Synthetic EMR clinical charts (SynPUF / Synthea)
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py
│   │   └── hierarchical_chunker.py
│   ├── retrieval/
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── generator/
│   │   ├── prompt_templates.py
│   │   └── um_evaluator.py
│   └── api/
│       └── main.py
├── evaluation/
│   ├── golden_dataset.json   # 50 verified ground-truth cases
│   └── ragas_benchmark.py
└── ui/
    └── streamlit_app.py
```

---

## 6. Resume Bullet Points & Interview Talking Points

### Resume Bullet Points
* *"Architected an Advanced RAG system for Healthcare Prior Authorization using hybrid retrieval (Dense + BM25) and cross-encoder re-ranking against CMS Clinical Policy Bulletins."*
* *"Engineered strict citation-grounded chain-of-thought verification for clinical criteria, achieving 0.96 Faithfulness and 0.91 Context Precision benchmarked via Ragas across 50+ complex medical scenarios."*

### Sagility Interview Talking Points
* **Question**: *"Why did you use Hybrid Search instead of pure vector search?"*  
  **Answer**: *"In healthcare, clinical policies depend heavily on exact identifiers like ICD-10 diagnostic codes, CPT procedure codes, and drug dosage units. Pure dense vector embeddings often compress these into semantic representations, missing exact code matches. By combining dense embeddings with BM25 via Reciprocal Rank Fusion, I ensured both semantic nuance and exact medical terminology were captured."*
* **Question**: *"How does this help Sagility's Clinical UM team?"*  
  **Answer**: *"UM nurses spend considerable time cross-referencing multi-page charts against payer policies. This solution acts as an intelligent co-pilot, automatically drafting the coverage checklist with exact page citations, accelerating turnaround times and reducing audit failure rates."*
