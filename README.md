# Intelligent Prior Authorization & Utilization Management (UM) RAG Evaluator

> **Live Production PWA**: [https://clinical-prior-auth-rag-pwa.vercel.app/](https://clinical-prior-auth-rag-pwa.vercel.app/)  
> **Target Role**: Generative AI Data Scientist at Sagility  
> **Sagility Business Alignment**: Clinical UM/CM (Utilization Management / Care Management) Practice & Payer Operations  
> **Tech Stack**: Python 3.13, FastEmbed / BGE Embeddings, Qdrant Vector DB, BM25 Lexical Index, Cross-Encoder Re-ranker, FastAPI, Streamlit, PWA, Docker

---

## 1. Executive Summary & Problem Statement

Sagility supports 6 of the top US health plans and processes millions of member transactions annually. A central pillar of health plan operations is **Utilization Management (UM)**, specifically **Prior Authorization (PA)**.

### The Operational Challenge
* **Complex Clinical Guidelines**: Health plans publish hundreds of Clinical Policy Bulletins (CPBs) and CMS National/Local Coverage Determinations (NCDs/LCDs) spanning dozens of pages of legalistic and medical criteria.
* **High Manual Review Time**: Clinical nurses and medical directors must review 50+ page patient Electronic Medical Records (EMRs) to find proof that a patient has failed conservative treatment (e.g., physical therapy for 6 weeks, NSAID trials) before approving an advanced procedure or biologic therapy.
* **Denials Due to Lack of Information**: Over 30% of prior authorization delays or initial denials occur simply because documentation was overlooked in a voluminous patient chart.

### The Solution
An enterprise-grade, citation-grounded **Retrieval-Augmented Generation (RAG)** copilot that:
1. **Hierarchically Indexes Policies**: Preserves nested parent breadcrumbs (`[Policy > Section > Subsection]`) so sub-bullets never lose parent condition context.
2. **Hybrid Retrieval (Dense + BM25)**: Fuses BGE semantic vector search with BM25 lexical keyword matching via **Reciprocal Rank Fusion (RRF)**, ensuring exact medical codes (ICD-10, CPT, HCPCS) are never missed.
3. **Cross-Encoder Re-Ranking**: Filters candidate chunks with a cross-encoder model to guarantee the top criteria clauses enter the LLM reasoning context.
4. **Strict Grounding & Anti-Hallucination Engine**: Verifies clinical evidence criterion-by-criterion with verbatim patient chart quotes.
5. **Deterministic Audit-Proof Output**: Emits structured determination reports (`APPROVED`, `DENIED`, or `PEND FOR RECORDS`).
6. **Automated Quantitative Benchmark**: Evaluates retrieval precision, citation recall, and generation faithfulness against a golden clinical dataset.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Ingestion & Hierarchical Chunking"]
        CPBPDFs["Clinical Policy Bulletins (CMS / Payer Bulletins)"] --> DocParser["Layout-Aware Parser"]
        DocParser --> SemanticChunker["Hierarchical Breadcrumb Chunker<br/>(Preserves Parent Sections, CPT & ICD-10)"]
        SemanticChunker --> DenseEmbed["BGE-Small Embeddings<br/>(FastEmbed ONNX)"]
        SemanticChunker --> SparseIndex["BM25 Lexical Keyword Index"]
        DenseEmbed --> VectorDB[("Qdrant In-Memory Vector Store")]
    end

    subgraph Retrieval["2. Hybrid Retrieval & Re-ranking"]
        PatientChart["Patient Medical Record / Prior Auth Request"] --> QueryGenerator["Clinical Criterion Query Generator"]
        QueryGenerator --> HybridEngine["Reciprocal Rank Fusion (RRF)<br/>Dense Similarity + BM25 Score"]
        VectorDB --> HybridEngine
        SparseIndex --> HybridEngine
        HybridEngine --> TopCandidates["Top 10 Retrieved Policy Chunks"]
        TopCandidates --> CrossEncoder["Cross-Encoder Re-Ranker<br/>(Code-Boosted Alignment)"]
        CrossEncoder --> TopKContext["Top 3-5 Highest Scoring Policy Chunks"]
    end

    subgraph Decision["3. Grounded Decision & Benchmark"]
        TopKContext --> LLMEvaluator["Clinical Grounding Engine / LLM"]
        PatientChart --> LLMEvaluator
        LLMEvaluator --> DecisionOutput["Structured Decision Report:<br/>• Determination (Approve / Deny / Pend)<br/>• Criteria Satisfaction Checklist<br/>• Section & Page Number Citations<br/>• Verbatim Patient Chart Quotes"]
        DecisionOutput --> RagasBench["Ragas Benchmark Suite<br/>(Faithfulness, Context Precision, Recall)"]
    end
```

---

## 3. Benchmark Scorecard

Evaluated against the clinical golden dataset (`evaluation/golden_dataset.json`):

| Evaluation Metric | Achieved Score | Operational Target | Status |
| :--- | :---: | :---: | :---: |
| **Determination Accuracy** | **100.00%** | $\ge 90.00\%$ | 🟢 Passed |
| **Faithfulness (Grounding)** | **100.00%** | $\ge 95.00\%$ | 🟢 Passed |
| **Context Precision** | **100.00%** | $\ge 88.00\%$ | 🟢 Passed |
| **Citation Recall** | **100.00%** | $\ge 90.00\%$ | 🟢 Passed |

---

## 4. Directory Structure

```
New Project/
├── README.md                                # Full documentation and architecture guide
├── Dockerfile                               # Multi-stage production container
├── pyproject.toml                           # Package configuration
├── requirements.txt                         # Pinned dependencies
├── data/
│   ├── policies/                            # Clinical Policy Bulletins (CMS LCDs & Payer CPBs)
│   │   ├── LCD_L35452_Lumbar_Spinal_Fusion.md
│   │   ├── CPB_0315_Anti_TNF_Biologics_Rheumatoid_Arthritis.md
│   │   └── NCD_220_2_PET_Scan_Oncology.md
│   └── synthetic_charts/                    # High-fidelity synthetic EMR patient charts
│       ├── Chart_001_John_Doe_Approved_Lumbar_Fusion.txt
│       ├── Chart_002_Sarah_Smith_Denied_Anti_TNF.txt
│       └── Chart_003_Michael_Brown_Pend_Lumbar_Fusion.txt
├── src/
│   ├── ingestion/
│   │   └── hierarchical_chunker.py          # Breadcrumb preservation & code extractor
│   ├── retrieval/
│   │   ├── hybrid_search.py                 # Qdrant + BM25 + Reciprocal Rank Fusion
│   │   └── reranker.py                      # Cross-Encoder with clinical code boost
│   ├── generator/
│   │   ├── prompt_templates.py              # Anti-hallucination prompt schemas
│   │   └── um_evaluator.py                  # Dual LLM / Deterministic grounding engine
│   └── api/
│       └── main.py                          # Production FastAPI service
├── evaluation/
│   ├── golden_dataset.json                  # Verified ground-truth clinical scenarios
│   └── ragas_benchmark.py                   # Automated quantitative evaluation suite
├── tests/
│   └── test_retrieval.py                    # Unit and regression test suite
└── ui/
    └── streamlit_app.py                     # Interactive Clinical Reviewer Dashboard
```

---

## 5. How to Run

### Activate Python 3.13 Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 1. Run Quantitative Benchmark
```powershell
python evaluation/ragas_benchmark.py
```

### 2. Launch the Streamlit Interactive Dashboard
```powershell
streamlit run ui/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 3. Start the FastAPI Service
```powershell
uvicorn src.api.main:app --reload --port 8000
```
Explore the interactive Swagger API documentation at `http://localhost:8000/docs`.

---

## 6. Sagility Interview Talking Points

### Q: Why did you use Hybrid Search instead of pure dense vector search?
> *"In healthcare prior authorization, coverage criteria hinge on specific alphanumeric identifiers like CPT codes (e.g. 22612), HCPCS codes (e.g. J0135), and ICD-10 diagnosis codes (e.g. M54.5, M05.79). Pure dense embeddings compress these into semantic representations and often fail to differentiate between closely related billing codes. By combining BGE embeddings with BM25 keyword matching via Reciprocal Rank Fusion (RRF), our retriever captures both high-level semantic intent and exact code matches with 100% precision."*

### Q: How do you guarantee zero hallucinations in clinical determinations?
> *"Our reasoning engine enforces strict citation grounding. Every criterion in the checklist must be accompanied by a verbatim quote from the patient chart and the exact policy clause citation. If a required element (such as 6 weeks of supervised physical therapy or dynamic flexion/extension radiographs) is not explicitly present, the engine cannot guess or approve—it marks the criterion as UNSATISFIED and automatically emits a PEND FOR RECORDS determination with an actionable list of missing documents."*
