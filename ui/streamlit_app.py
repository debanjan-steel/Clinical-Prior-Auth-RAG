"""
Streamlit Clinical Prior Authorization & Utilization Management Copilot Dashboard.
Interactive UI for reviewing patient charts against clinical policy bulletins.
"""

import sys
import json
from pathlib import Path
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import HierarchicalPolicyChunker
from src.retrieval import HybridRetriever, CrossEncoderReranker
from src.generator.um_evaluator import UMEvaluator
from evaluation.ragas_benchmark import PriorAuthBenchmarkRunner


st.set_page_config(
    page_title="Prior Authorization RAG Evaluator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
    }
    .badge-approved {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #31C48D;
    }
    .badge-denied {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #F98080;
    }
    .badge-pend {
        background-color: #FEF08A;
        color: #713F12;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #EAB308;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag_engine():
    """Initializes and caches the policy chunker, vector store, and retriever."""
    chunker = HierarchicalPolicyChunker()
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    evaluator = UMEvaluator()

    policy_dir = PROJECT_ROOT / "data" / "policies"
    chunks = []
    if policy_dir.exists():
        for file in sorted(policy_dir.glob("*.md")):
            chunks.extend(chunker.chunk_file(file))
        retriever.index_chunks(chunks)

    return chunker, retriever, reranker, evaluator


chunker, retriever, reranker, evaluator = load_rag_engine()


# Header Banner
st.markdown('<div class="main-header">🏥 Intelligent Prior Authorization & Utilization Management (UM) RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI Clinical Reviewer & Evaluation Copilot | Payer Operations & Clinical Care Management</div>', unsafe_allow_html=True)

# Tabs
tab_eval, tab_retrieval, tab_bench, tab_policies = st.tabs([
    "📋 Clinical Evaluation",
    "🔍 Hybrid Search Inspector",
    "📊 Ragas Evaluation Benchmark",
    "📚 Payer Policy Knowledge Base",
])

# ----------------- TAB 1: CLINICAL EVALUATION -----------------
with tab_eval:
    col_input, col_output = st.columns([1, 1], gap="medium")

    sample_cases = {
        "Case 1: Lumbar Spinal Fusion (Approved) - Johnathan Doe": "data/synthetic_charts/Chart_001_John_Doe_Approved_Lumbar_Fusion.txt",
        "Case 2: Anti-TNF Biologic Adalimumab (Denied) - Sarah Smith": "data/synthetic_charts/Chart_002_Sarah_Smith_Denied_Anti_TNF.txt",
        "Case 3: Lumbar Fusion Incomplete (Pend for Records) - Michael Brown": "data/synthetic_charts/Chart_003_Michael_Brown_Pend_Lumbar_Fusion.txt",
        "Custom Patient Chart": None,
    }

    with col_input:
        st.subheader("1. Patient Medical Record / PA Submission")
        selected_case_name = st.selectbox("Select Test Patient Scenario:", list(sample_cases.keys()))

        if sample_cases[selected_case_name] is not None:
            chart_file_path = PROJECT_ROOT / sample_cases[selected_case_name]
            default_chart_text = chart_file_path.read_text(encoding="utf-8")
        else:
            default_chart_text = "Paste patient medical chart notes here..."

        chart_text = st.text_area("Patient Chart Notes (EMR):", value=default_chart_text, height=420)
        
        proc_override = st.text_input(
            "Procedure / Diagnostic Codes Query Hint (Optional):",
            placeholder="e.g. CPT 22612 lumbar spinal fusion or HCPCS J0135 Adalimumab",
        )

        top_k = st.slider("Top Policy Criteria Chunks for Reasoning Context:", min_value=2, max_value=8, value=4)
        run_btn = st.button("🚀 Evaluate Prior Authorization Request", type="primary", use_container_width=True)

    with col_output:
        st.subheader("2. Audit-Proof Clinical Determination")

        if run_btn and chart_text.strip():
            with st.spinner("Executing Hybrid Retrieval (Dense + BM25), Cross-Encoder Re-Ranking, and Clinical Grounding..."):
                # Determine query
                metadata = evaluator._extract_patient_metadata(chart_text)
                query = proc_override.strip() if proc_override.strip() else f"Prior authorization for {metadata['requested_procedure']} {metadata['primary_diagnosis']}"

                candidates = retriever.retrieve(query, top_k=10)
                reranked = reranker.rerank(query, candidates, top_k=top_k)
                decision = evaluator.evaluate(chart_text, reranked)

                # Determination badge
                det = decision.determination
                if det == "APPROVED":
                    badge_html = f'<span class="badge-approved">✅ {det}</span>'
                elif det == "DENIED":
                    badge_html = f'<span class="badge-denied">❌ {det}</span>'
                else:
                    badge_html = f'<span class="badge-pend">⚠️ {det}</span>'

                st.markdown(badge_html, unsafe_allow_html=True)
                st.caption(f"Patient: **{decision.patient_name}** | Procedure: **{decision.requested_procedure}** | Confidence: **{decision.confidence_score * 100:.1f}%**")

                st.info(f"**Executive Summary:** {decision.summary_rationale}")

                # Missing Information alert
                if decision.missing_information:
                    st.warning("⚠️ **Missing Documentation / Pend Reasons:**\n" + "\n".join([f"- {m}" for m in decision.missing_information]))

                # Criteria Checklist
                st.markdown("### 📋 Criteria Satisfaction Checklist")
                for item in decision.criteria_checklist:
                    status_emoji = "✅" if item.status == "SATISFIED" else ("❌" if item.status == "CONTRAINDICATED" else "⚠️")
                    with st.expander(f"{status_emoji} {item.criterion_name} ({item.status})", expanded=True):
                        st.markdown(f"**Policy Citation:** `{item.policy_citation}`")
                        st.markdown(f"**Patient Evidence (Verbatim Quote):**\n> *\"{item.patient_evidence_quote}\"*")
                        st.markdown(f"**Clinical Rationale:** {item.clinical_rationale}")

# ----------------- TAB 2: HYBRID RETRIEVAL INSPECTOR -----------------
with tab_retrieval:
    st.subheader("🔍 Inspect Dense + BM25 Hybrid Retrieval & Cross-Encoder Scores")
    test_query = st.text_input(
        "Search Knowledge Base with Clinical Query or Billing Code:",
        value="Prior authorization CPT 22612 lumbar spinal fusion failed conservative therapy",
    )

    if test_query:
        candidates = retriever.retrieve(test_query, top_k=8)
        reranked = reranker.rerank(test_query, candidates, top_k=5)

        st.markdown(f"**Retrieved & Re-ranked Chunks ({len(reranked)} shown):**")
        for c in reranked:
            with st.container():
                st.markdown(f"#### Rank {c['rerank_position']}: {c['chunk_id']} — `{c['section_hierarchy']}`")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Cross-Encoder Score", f"{c['cross_encoder_score']:.4f}")
                m_col2.metric("RRF Score", f"{c['rrf_score']:.4f}")
                m_col3.metric("Dense Similarity", f"{c['dense_score']:.4f}")
                m_col4.metric("BM25 Score", f"{c['bm25_score']:.2f}")

                st.markdown(f"**Content Preview:**\n```\n{c['content']}\n```")
                st.markdown(f"🏷️ **Applicable CPT:** `{', '.join(c['cpt_codes']) or 'None'}` | **ICD-10:** `{', '.join(c['icd_codes']) or 'None'}`")
                st.divider()

# ----------------- TAB 3: RAGAS BENCHMARK -----------------
with tab_bench:
    st.subheader("📊 Ragas-Style Quantitative Evaluation Suite")
    st.markdown("""
    In clinical operations, standard subjective testing is unacceptable. The Prior Auth Evaluator must achieve:
    - **Faithfulness > 0.95**: Every evidence claim must strictly quote patient records without fabrication.
    - **Context Precision > 0.88**: Retrieved clauses must strictly match the relevant coverage policy.
    - **Citation Recall > 0.90**: All mandatory criteria sections must be surfaced and cited.
    """)

    if st.button("⚡ Run Full Automated Benchmark Suite", type="primary"):
        with st.spinner("Executing benchmark across verified clinical golden dataset..."):
            runner = PriorAuthBenchmarkRunner()
            summary = runner.run_benchmark()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Determination Accuracy", f"{summary['determination_accuracy'] * 100:.1f}%", "Target: >=90%")
            c2.metric("Faithfulness (Grounding)", f"{summary['faithfulness'] * 100:.1f}%", "Target: >=95%")
            c3.metric("Context Precision", f"{summary['context_precision'] * 100:.1f}%", "Target: >=88%")
            c4.metric("Citation Recall", f"{summary['citation_recall'] * 100:.1f}%", "Target: >=90%")

            st.markdown("### Benchmark Scenario Details")
            for case in summary["cases"]:
                status_icon = "🟢" if case["correct"] else "🔴"
                st.write(f"{status_icon} **{case['case_id']} - {case['patient']}**: Predicted `{case['predicted']}` | Expected `{case['expected']}` | Faithfulness `{case['faithfulness'] * 100:.0f}%`")

# ----------------- TAB 4: KNOWLEDGE BASE -----------------
with tab_policies:
    st.subheader("📚 Loaded Clinical Policy Bulletins (CMS NCD / LCD / Payer CPBs)")
    policy_files = list((PROJECT_ROOT / "data" / "policies").glob("*.md"))
    for pf in sorted(policy_files):
        with st.expander(f"📄 {pf.name}", expanded=False):
            st.markdown(pf.read_text(encoding="utf-8"))
