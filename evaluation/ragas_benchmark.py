"""
Ragas-Style Quantitative Evaluation Suite for Prior Authorization RAG.
Computes Faithfulness, Context Precision, Citation Recall, and Determination Accuracy
across the golden clinical benchmark dataset.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import HierarchicalPolicyChunker
from src.retrieval import HybridRetriever, CrossEncoderReranker
from src.generator.um_evaluator import UMEvaluator


class PriorAuthBenchmarkRunner:
    """
    Executes automated quantitative evaluation of the Prior Auth RAG pipeline
    against ground-truth clinical scenarios.
    """

    def __init__(self, golden_dataset_path: str = "evaluation/golden_dataset.json"):
        self.dataset_path = PROJECT_ROOT / golden_dataset_path
        self.chunker = HierarchicalPolicyChunker()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.evaluator = UMEvaluator()
        self.chunks = []

    def setup_knowledge_base(self):
        """Indexes all clinical policy bulletins into the hybrid search engine."""
        policy_dir = PROJECT_ROOT / "data" / "policies"
        self.chunks = []
        for file in sorted(policy_dir.glob("*.md")):
            self.chunks.extend(self.chunker.chunk_file(file))
        self.retriever.index_chunks(self.chunks)
        print(f"[Knowledge Base] Successfully indexed {len(self.chunks)} hierarchical policy chunks.")

    def run_benchmark(self) -> Dict[str, Any]:
        """Runs the entire benchmark suite and computes evaluation metrics."""
        if not self.chunks:
            self.setup_knowledge_base()

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            cases: List[Dict[str, Any]] = json.load(f)

        total_cases = len(cases)
        correct_determinations = 0
        faithfulness_scores = []
        context_precisions = []
        citation_recalls = []
        case_results = []

        print(f"\n========================================================")
        print(f"  RUNNING PRIOR AUTH RAG EVALUATION BENCHMARK ({total_cases} CASES)")
        print(f"========================================================")

        for case in cases:
            case_id = case["case_id"]
            chart_file = PROJECT_ROOT / case["chart_file"]
            chart_text = chart_file.read_text(encoding="utf-8")
            query = case["query"]
            expected_det = case["expected_determination"]
            expected_policy = case["expected_policy_id"]
            expected_citations = case.get("expected_citations", [])
            ground_truth_quotes = case.get("ground_truth_quotes", [])

            # 1. Hybrid Retrieval & Re-ranking
            candidates = self.retriever.retrieve(query, top_k=10)
            reranked = self.reranker.rerank(query, candidates, top_k=5)

            # 2. Decision Generation
            decision = self.evaluator.evaluate(chart_text, reranked)

            # 3. Metric 1: Determination Accuracy
            is_det_correct = (decision.determination == expected_det)
            if is_det_correct:
                correct_determinations += 1

            # 4. Metric 2: Context Precision
            # Fraction of retrieved chunks belonging to the target policy
            relevant_retrieved = sum(1 for c in reranked if expected_policy in c.get("policy_id", ""))
            prec = relevant_retrieved / len(reranked) if reranked else 0.0
            context_precisions.append(prec)

            # 5. Metric 3: Citation Recall
            # Verify if expected policy ID and section number appear in cited policy sections
            cited_sections = " ".join([c.policy_citation for c in decision.criteria_checklist] + decision.retrieved_policy_citations)
            matched_citations = 0
            for exp in expected_citations:
                parts = exp.split(":")
                sec_identifier = parts[0].strip()  # e.g. "LCD-L35452 Section 2.1"
                tokens = sec_identifier.split()
                policy_tag = tokens[0]  # "LCD-L35452"
                section_tag = tokens[-1]  # "2.1" or "Section 2.1"
                if policy_tag in cited_sections and section_tag in cited_sections:
                    matched_citations += 1

            recall = matched_citations / len(expected_citations) if expected_citations else 1.0
            citation_recalls.append(recall)

            # 6. Metric 4: Faithfulness (Grounding)
            # Checks whether patient_evidence_quote is actually found verbatim in chart_text
            verified_quotes = 0
            total_quotes = len(decision.criteria_checklist)
            for item in decision.criteria_checklist:
                quote = item.patient_evidence_quote
                if "N/A" in quote:
                    verified_quotes += 1
                else:
                    # Split compound quotes
                    segments = [s.strip() for s in quote.split("|")]
                    if any(seg.lower() in chart_text.lower() for seg in segments):
                        verified_quotes += 1

            case_faithfulness = verified_quotes / total_quotes if total_quotes > 0 else 1.0
            faithfulness_scores.append(case_faithfulness)

            print(f"\n[{case_id}] {case['patient_name']} ({case['requested_procedure']})")
            print(f"  • Predicted: {decision.determination} | Expected: {expected_det} -> {'[MATCH]' if is_det_correct else '[MISMATCH]'}")
            print(f"  • Confidence: {decision.confidence_score:.2f}")
            print(f"  • Case Faithfulness: {case_faithfulness:.2%}")
            print(f"  • Context Precision: {prec:.2%}")

            case_results.append({
                "case_id": case_id,
                "patient": case["patient_name"],
                "predicted": decision.determination,
                "expected": expected_det,
                "correct": is_det_correct,
                "faithfulness": case_faithfulness,
                "precision": prec,
                "decision": decision.model_dump(),
            })

        mean_accuracy = correct_determinations / total_cases
        mean_faithfulness = sum(faithfulness_scores) / total_cases
        mean_precision = sum(context_precisions) / total_cases
        mean_recall = sum(citation_recalls) / total_cases

        summary = {
            "total_cases": total_cases,
            "determination_accuracy": round(mean_accuracy, 4),
            "faithfulness": round(mean_faithfulness, 4),
            "context_precision": round(mean_precision, 4),
            "citation_recall": round(mean_recall, 4),
            "cases": case_results,
        }

        print(f"\n========================================================")
        print(f"               FINAL BENCHMARK SCORECARD                ")
        print(f"========================================================")
        print(f" Determination Accuracy:  {mean_accuracy:.2%} (Target: >= 90%)")
        print(f" Faithfulness (Grounding): {mean_faithfulness:.2%} (Target: >= 95%)")
        print(f" Context Precision:       {mean_precision:.2%} (Target: >= 88%)")
        print(f" Citation Recall:         {mean_recall:.2%} (Target: >= 90%)")
        print(f"========================================================\n")

        return summary


if __name__ == "__main__":
    runner = PriorAuthBenchmarkRunner()
    runner.run_benchmark()
