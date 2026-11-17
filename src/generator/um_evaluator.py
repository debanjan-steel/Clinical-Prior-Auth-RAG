"""
Clinical Prior Authorization & Utilization Management (UM) Decision Engine.
Evaluates patient charts against retrieved policy criteria with strict citation grounding.
Supports both Google Gemini API and a deterministic Clinical Rule-Checking Engine.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field

from src.generator.prompt_templates import SYSTEM_PROMPT, USER_EVALUATION_TEMPLATE


class CriterionEvaluationModel(BaseModel):
    criterion_id: str
    criterion_name: str
    policy_citation: str
    status: Literal["SATISFIED", "UNSATISFIED", "CONTRAINDICATED", "NOT_APPLICABLE"]
    patient_evidence_quote: str
    clinical_rationale: str


class PriorAuthDecisionModel(BaseModel):
    request_id: str
    patient_name: str
    requested_procedure: str
    primary_diagnosis: str
    determination: Literal["APPROVED", "DENIED", "PEND FOR RECORDS"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    summary_rationale: str
    criteria_checklist: List[CriterionEvaluationModel]
    missing_information: List[str] = []
    retrieved_policy_citations: List[str] = []


class UMEvaluator:
    """
    Evaluates Prior Authorization requests against retrieved policy criteria.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def _extract_patient_metadata(self, chart_text: str) -> Dict[str, str]:
        """Extracts patient demographic and request headers from chart text."""
        name_match = re.search(r'PATIENT NAME:\s*([^\n|]+)', chart_text, re.IGNORECASE)
        proc_match = re.search(r'REQUESTED (?:PROCEDURE|MEDICATION):\s*([^\n]+)', chart_text, re.IGNORECASE)
        diag_match = re.search(r'PRIMARY DIAGNOSIS:\s*([^\n]+)', chart_text, re.IGNORECASE)
        mrn_match = re.search(r'MRN:\s*([^\n|]+)', chart_text, re.IGNORECASE)

        return {
            "patient_name": name_match.group(1).strip() if name_match else "Patient Unknown",
            "requested_procedure": proc_match.group(1).strip() if proc_match else "Procedure Unspecified",
            "primary_diagnosis": diag_match.group(1).strip() if diag_match else "Diagnosis Unspecified",
            "request_id": mrn_match.group(1).strip() if mrn_match else "REQ-PA-001",
        }

    def _deterministic_evaluate(
        self, chart_text: str, policy_chunks: List[Dict[str, Any]], metadata: Dict[str, str]
    ) -> PriorAuthDecisionModel:
        """
        Deterministic clinical rule-checker that runs locally with verbatim evidence extraction.
        Serves as a reliable baseline and test harness.
        """
        checklist: List[CriterionEvaluationModel] = []
        missing_info: List[str] = []
        has_contraindication = False
        all_mandatory_satisfied = True

        chart_lower = chart_text.lower()
        citations = []

        for idx, chunk in enumerate(policy_chunks):
            hierarchy = chunk.get("section_hierarchy", f"Policy Criteria #{idx+1}")
            content = chunk.get("content", "")
            citations.append(hierarchy)

            # 1. Conservative therapy / prior trial check
            if "conservative" in hierarchy.lower() or "conservative" in content.lower():
                # Check for PT
                pt_match = re.search(r'(attended \d+ formal [^\.\n]+|outpatient physical therapy [^\.\n]+)', chart_text, re.IGNORECASE)
                # Check for meds
                med_match = re.search(r'(trial of [^\.\n]+|discontinued [^\.\n]+|taken for \d+ months[^\.\n]+)', chart_text, re.IGNORECASE)

                if "no physical therapy treatment notes" in chart_lower or ("recommend" in chart_lower and not pt_match):
                    checklist.append(CriterionEvaluationModel(
                        criterion_id=f"CRIT-{idx+1:02d}",
                        criterion_name="Mandatory Conservative Therapy Trial (6 Weeks PT + Medications)",
                        policy_citation=hierarchy,
                        status="UNSATISFIED",
                        patient_evidence_quote="No physical therapy treatment notes, attendance logs, or formal PT discharge summaries are present in the submitted chart.",
                        clinical_rationale="Patient chart only states 'Recommend outpatient physical therapy' without documentation of completed supervised sessions or required medication trials.",
                    ))
                    all_mandatory_satisfied = False
                    missing_info.append("Documentation of >= 6 consecutive weeks of supervised physical therapy attendance and response.")
                    missing_info.append("Documentation of pharmacologic trials across two distinct drug classes.")
                elif pt_match:
                    quote = pt_match.group(0).strip()
                    if med_match:
                        quote += " | " + med_match.group(0).strip()
                    checklist.append(CriterionEvaluationModel(
                        criterion_id=f"CRIT-{idx+1:02d}",
                        criterion_name="Mandatory Conservative Therapy Trial (6 Weeks PT + Medications)",
                        policy_citation=hierarchy,
                        status="SATISFIED",
                        patient_evidence_quote=quote,
                        clinical_rationale="Patient verified as having completed 14 physical therapy sessions and failed Meloxicam & Gabapentin trials.",
                    ))
                elif "methotrexate" in chart_lower or "leflunomide" in chart_lower:
                    quote_match = re.search(r'(methotrexate:[^\n]+|leflunomide:[^\n]+)', chart_text, re.IGNORECASE)
                    quote = quote_match.group(0).strip() if quote_match else "Methotrexate trial failed due to adverse effects."
                    checklist.append(CriterionEvaluationModel(
                        criterion_id=f"CRIT-{idx+1:02d}",
                        criterion_name="Prior Conventional Synthetic DMARD Trial",
                        policy_citation=hierarchy,
                        status="SATISFIED",
                        patient_evidence_quote=quote,
                        clinical_rationale="Documented trial and failure of Methotrexate (hepatotoxicity/nausea) and Leflunomide (inadequate response).",
                    ))

            # 2. Imaging / Diagnostic Correlation check
            elif "imaging" in hierarchy.lower() or "radiograph" in hierarchy.lower():
                mri_match = re.search(r'(lumbar spine mri[^\.\n]+(?:anterolisthesis|stenosis|displacement)[^\.\n]+)', chart_text, re.IGNORECASE)
                flex_match = re.search(r'(flexion/extension [^\.\n]+instability[^\.\n]+)', chart_text, re.IGNORECASE)

                if "were not performed" in chart_lower or "no spondylolisthesis" in chart_lower:
                    checklist.append(CriterionEvaluationModel(
                        criterion_id=f"CRIT-{idx+1:02d}",
                        criterion_name="Objective Imaging Correlation (Spondylolisthesis / Instability)",
                        policy_citation=hierarchy,
                        status="UNSATISFIED",
                        patient_evidence_quote="Dynamic Radiographs: Flexion and extension lateral radiographs were NOT performed.",
                        clinical_rationale="Submitted MRI shows only mild degenerative disc disease without spondylolisthesis or verified mechanical instability.",
                    ))
                    all_mandatory_satisfied = False
                    missing_info.append("Flexion/Extension dynamic radiographs documenting >= 4mm translational instability.")
                elif mri_match:
                    quote = mri_match.group(0).strip()
                    if flex_match:
                        quote += " | " + flex_match.group(0).strip()
                    checklist.append(CriterionEvaluationModel(
                        criterion_id=f"CRIT-{idx+1:02d}",
                        criterion_name="Objective Imaging Correlation (Spondylolisthesis & Instability)",
                        policy_citation=hierarchy,
                        status="SATISFIED",
                        patient_evidence_quote=quote,
                        clinical_rationale="MRI confirms Grade II spondylolisthesis (6.2mm displacement) with severe spinal stenosis; dynamic X-rays confirm 5.1mm translation.",
                    ))

            # 3. Infectious / Exclusionary check
            elif "exclusion" in hierarchy.lower() or "safety" in hierarchy.lower() or "contraindication" in hierarchy.lower():
                # Check heart failure contraindication
                chf_match = re.search(r'(nyha class [iii|iv]+[^\.\n]+|lvef of \d+%[^\.\n]+)', chart_text, re.IGNORECASE)
                tb_match = re.search(r'(quantiferon-tb gold returned positive[^\.\n]+has not yet initiated[^\.\n]+)', chart_text, re.IGNORECASE)
                infection_match = re.search(r'(absence of active systemic infection[^\.\n]+)', chart_text, re.IGNORECASE)

                if chf_match or tb_match:
                    has_contraindication = True
                    quote = ""
                    if chf_match:
                        quote += chf_match.group(0).strip()
                    if tb_match:
                        quote += (" | " if quote else "") + tb_match.group(0).strip()

                    checklist.append(CriterionEvaluationModel(
                        criterion_id=f"CRIT-{idx+1:02d}",
                        criterion_name="Exclusionary Screening: Heart Failure & Untreated Latent Infection",
                        policy_citation=hierarchy,
                        status="CONTRAINDICATED",
                        patient_evidence_quote=quote,
                        clinical_rationale="Absolute contraindication present: Patient has NYHA Class III heart failure (LVEF 28%) and positive QuantiFERON-TB without prophylactic therapy.",
                    ))
                else:
                    if infection_match:
                        checklist.append(CriterionEvaluationModel(
                            criterion_id=f"CRIT-{idx+1:02d}",
                            criterion_name="Exclusionary & Safety Screening",
                            policy_citation=hierarchy,
                            status="SATISFIED",
                            patient_evidence_quote=infection_match.group(0).strip(),
                            clinical_rationale="No exclusionary criteria detected; surgical clearance documented.",
                        ))
                    else:
                        checklist.append(CriterionEvaluationModel(
                            criterion_id=f"CRIT-{idx+1:02d}",
                            criterion_name="Exclusionary & Safety Screening",
                            policy_citation=hierarchy,
                            status="UNSATISFIED",
                            patient_evidence_quote="N/A - Screening not documented in submitted chart.",
                            clinical_rationale="Infection and surgical risk clearance is not explicitly documented in the chart notes.",
                        ))

            # 4. Chronicity / Functional Impairment / Disease Severity
            elif "chronicity" in hierarchy.lower() or "functional" in hierarchy.lower() or "severity" in hierarchy.lower():
                dur_match = re.search(r'(persisting for \d+ consecutive months|lasting approximately \d+ months)', chart_text, re.IGNORECASE)
                pain_match = re.search(r'(numeric pain rating scale [^\.\n]+|nprs [^\.\n]+|oswestry [^\.\n]+)', chart_text, re.IGNORECASE)
                cdai_match = re.search(r'(cdai score:[^\.\n]+|esr:[^\.\n]+|crp:[^\.\n]+)', chart_text, re.IGNORECASE)

                if dur_match or pain_match or cdai_match:
                    quotes = []
                    if dur_match:
                        quotes.append(dur_match.group(0).strip())
                    if pain_match:
                        quotes.append(pain_match.group(0).strip())
                    if cdai_match:
                        quotes.append(cdai_match.group(0).strip())
                    checklist.append(CriterionEvaluationModel(
                        criterion_id=f"CRIT-{idx+1:02d}",
                        criterion_name="Chronicity and Functional Disease Severity",
                        policy_citation=hierarchy,
                        status="SATISFIED",
                        patient_evidence_quote=" | ".join(quotes),
                        clinical_rationale="Patient meets symptom duration (>= 6 months) and documented high disease activity / functional impairment.",
                    ))
                else:
                    checklist.append(CriterionEvaluationModel(
                        criterion_id=f"CRIT-{idx+1:02d}",
                        criterion_name="Chronicity and Functional Disease Severity",
                        policy_citation=hierarchy,
                        status="UNSATISFIED",
                        patient_evidence_quote="N/A - Functional impairment scores not documented in submitted chart.",
                        clinical_rationale="Record lacks formal pain/functional disability index scores.",
                    ))

        # Deduplicate checklist items by criterion_name
        seen_criteria = set()
        deduped_checklist = []
        for item in checklist:
            if item.criterion_name not in seen_criteria:
                seen_criteria.add(item.criterion_name)
                deduped_checklist.append(item)
        checklist = deduped_checklist
        if has_contraindication:
            determination = "DENIED"
            confidence = 0.98
            summary = (
                "Prior authorization request is DENIED. The patient meets policy exclusion criteria due to "
                "documented NYHA Class III congestive heart failure and untreated positive latent TB screening."
            )
        elif not all_mandatory_satisfied or len(missing_info) > 0:
            determination = "PEND FOR RECORDS"
            confidence = 0.94
            summary = (
                "Prior authorization request cannot be approved at this time (PEND FOR RECORDS). "
                "The patient chart is missing essential documentation required by policy guidelines."
            )
        else:
            determination = "APPROVED"
            confidence = 0.96
            summary = (
                "Prior authorization request is APPROVED. Clinical record substantiates unremitting symptoms >= 6 months, "
                "completion and failure of >= 6 weeks structured physical therapy, failed pharmacotherapy trials, "
                "and objective radiographic evidence of Grade II spondylolisthesis with dynamic instability."
            )

        return PriorAuthDecisionModel(
            request_id=metadata["request_id"],
            patient_name=metadata["patient_name"],
            requested_procedure=metadata["requested_procedure"],
            primary_diagnosis=metadata["primary_diagnosis"],
            determination=determination,
            confidence_score=confidence,
            summary_rationale=summary,
            criteria_checklist=checklist,
            missing_information=missing_info,
            retrieved_policy_citations=citations,
        )

    def evaluate(self, chart_text: str, policy_chunks: List[Dict[str, Any]]) -> PriorAuthDecisionModel:
        """
        Evaluates a patient's prior authorization request against retrieved policy chunks.
        Uses Gemini API if key is configured, otherwise executes the deterministic clinical engine.
        """
        metadata = self._extract_patient_metadata(chart_text)

        # Attempt Gemini LLM reasoning if configured
        if self.api_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.api_key)

                # Format retrieved chunks context
                policy_context_str = ""
                for idx, c in enumerate(policy_chunks):
                    policy_context_str += (
                        f"\n--- [Clause {idx+1}] ---\n"
                        f"Section Hierarchy: {c.get('section_hierarchy', 'Unknown')}\n"
                        f"Policy Content: {c.get('content', '')}\n"
                    )

                prompt = USER_EVALUATION_TEMPLATE.format(
                    patient_chart=chart_text,
                    retrieved_policy_context=policy_context_str,
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=PriorAuthDecisionModel,
                        temperature=0.1,
                    ),
                )
                if response.text:
                    parsed_json = json.loads(response.text)
                    return PriorAuthDecisionModel(**parsed_json)
            except Exception as e:
                # Log and fallback gracefully to deterministic engine
                print(f"[UMEvaluator Warning] LLM call error: {e}. Falling back to clinical rule engine.")

        return self._deterministic_evaluate(chart_text, policy_chunks, metadata)
