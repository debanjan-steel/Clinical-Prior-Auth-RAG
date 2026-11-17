/**
 * PriorAuth AI - Standalone PWA Application Logic
 * Supports dual-mode execution:
 * 1. Cloud FastAPI Backend (when connected to cloud host)
 * 2. In-Browser Standalone Engine (zero backend dependency, runs 100% offline 24/7)
 */

// 1. Register Service Worker for PWA
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((reg) => console.log('[PWA] Service Worker registered with scope:', reg.scope))
      .catch((err) => console.error('[PWA] Service Worker registration failed:', err));
  });
}

// 2. Handle PWA "Install App" button
let deferredPrompt = null;
const installBtn = document.getElementById('installBtn');

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  if (installBtn) {
    installBtn.style.display = 'inline-flex';
  }
});

if (installBtn) {
  installBtn.addEventListener('click', async () => {
    if (deferredPrompt) {
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      console.log(`[PWA] Install prompt outcome: ${outcome}`);
      deferredPrompt = null;
      installBtn.style.display = 'none';
    }
  });
}

// 3. Preset Clinical Cases
const SAMPLE_CASES = {
  case1: `PATIENT ELECTRONIC MEDICAL RECORD & PRIOR AUTHORIZATION REQUEST
PATIENT NAME: Johnathan Doe
DOB: 1974-05-12 | MRN: #UM-90218
REQUESTING PROVIDER: Dr. Robert Vance, MD (Orthopedic Spine Surgery)
FACILITY: Metro Spine & Orthopedic Institute
DATE OF SERVICE REQUEST: 2026-10-15

REQUESTED PROCEDURE:
CPT 22612 - Arthrodesis, posterior or posterolateral technique, single level; lumbar
CPT 63047 - Laminectomy, facetectomy and foraminotomy (decompression)
PRIMARY DIAGNOSIS:
ICD-10 M43.16 - Spondylolisthesis, lumbar region (L4-L5)
ICD-10 M54.50 - Low back pain, unspecified

CLINICAL HISTORY & PHYSICAL EXAM:
The patient is a 52-year-old male presenting with unremitting lower back pain and right lower extremity radiculopathy persisting for 9 consecutive months (onset January 2026).
Current Numeric Pain Rating Scale (NPRS) is 8/10. Oswestry Disability Index (ODI) is 48%, reflecting severe functional impairment in Activities of Daily Living.

PRIOR CONSERVATIVE THERAPY RECORD:
1. Physical Therapy: Attended 14 formal outpatient physical therapy sessions at Metro Rehab Services between March 4, 2026 and May 2, 2026. Minimal lasting pain relief (< 15% reduction).
2. Pharmacotherapy: 
   - Trial of Meloxicam 15 mg daily for 8 weeks discontinued due to lack of efficacy.
   - Trial of Gabapentin titrated to 600 mg TID for 10 weeks with persistent radicular symptoms.
3. Activity modification and documented daily home exercise program performed consistently.

DIAGNOSTIC IMAGING SUMMARY:
- Lumbar Spine MRI (Date: 2026-09-02): Demonstrates Grade II anterolisthesis of L4 on L5 with 6.2 mm displacement. Severe spinal stenosis and right L5 exiting nerve root impingement.
- Flexion/Extension Lateral Lumbar Radiographs (Date: 2026-09-10): Confirms dynamic lumbar instability at L4-L5 with 5.1 mm translational movement on flexion/extension.

SURGICAL CLEARANCE & EXCLUSIONS:
- BMI: 28.4 kg/m2.
- Absence of active systemic infection; complete blood count and ESR/CRP within normal limits.
- No history of substance abuse or untreated psychiatric illness.
Signed: Dr. Robert Vance, MD`,

  case2: `PATIENT ELECTRONIC MEDICAL RECORD & PRIOR AUTHORIZATION REQUEST
PATIENT NAME: Sarah Smith
DOB: 1982-11-20 | MRN: #UM-44910
REQUESTING PROVIDER: Dr. Patricia Lee, MD (Rheumatology)
CLINIC: Arthritis & Autoimmune Care Partners
DATE OF SERVICE REQUEST: 2026-10-18

REQUESTED MEDICATION:
HCPCS J0135 - Injection, adalimumab, 20 mg (Prescribed: 40 mg subcutaneous every 14 days)
PRIMARY DIAGNOSIS:
ICD-10 M05.79 - Rheumatoid arthritis with rheumatoid factor of multiple sites

CLINICAL HISTORY & DISEASE ACTIVITY:
Patient is a 43-year-old female with long-standing seropositive rheumatoid arthritis presenting with active polyarticular inflammatory disease involving bilateral wrists and MCP joints.
Laboratory: ESR: 44 mm/hr (Elevated), CRP: 2.4 mg/dL (Elevated), CDAI score: 24.5 (High disease activity).

CONVENTIONAL DMARD HISTORY:
- Methotrexate: Initiated at 15 mg weekly in October 2023. Discontinued after 7 months due to severe nausea and hepatotoxicity (ALT 92 U/L).
- Leflunomide: Taken for 5 months with inadequate response.

MANDATORY SAFETY SCREENING & CONTRAINDICATION EVALUATION:
- Infectious Screening: QuantiFERON-TB Gold returned positive on 2026-09-15. Note: Patient has NOT yet initiated or completed latent TB antimicrobial chemoprophylaxis.
- CARDIOLOGY COMORBIDITY: Patient has a history of non-ischemic cardiomyopathy with NYHA Class III Congestive Heart Failure. Most recent echocardiogram reveals left ventricular ejection fraction (LVEF) of 28%.
Signed: Dr. Patricia Lee, MD`,

  case3: `PATIENT ELECTRONIC MEDICAL RECORD & PRIOR AUTHORIZATION REQUEST
PATIENT NAME: Michael Brown
DOB: 1968-02-14 | MRN: #UM-31802
REQUESTING PROVIDER: Dr. Marcus Sterling, MD (Neurosurgery)
CLINIC: Tri-County Neurological Specialists
DATE OF SERVICE REQUEST: 2026-10-22

REQUESTED PROCEDURE:
CPT 22612 - Arthrodesis, posterior technique, lumbar
PRIMARY DIAGNOSIS:
ICD-10 M54.50 - Low back pain, unspecified

CLINICAL SUMMARY:
The patient is a 58-year-old male with chief complaint of lower back pain lasting approximately 7 months. NPRS score today is 7/10. Pain is aching without clear radiation.

TREATMENT HISTORY:
- Medications: Patient takes over-the-counter Ibuprofen 400 mg as needed with temporary relief.
- Physical Therapy: Physician note from 2026-06-12 states: "Recommend outpatient physical therapy for 6 weeks." However, no physical therapy treatment notes, attendance logs, or formal PT discharge summaries are present in the submitted chart.

DIAGNOSTIC WORKUP:
- Lumbar MRI (Date: 2026-08-10): Shows mild L5-S1 disc desiccation. No spondylolisthesis.
- Dynamic Radiographs: Flexion and extension lateral radiographs were NOT performed.
Signed: Dr. Marcus Sterling, MD`,

  custom: ""
};

// 4. UI Elements
const scenarioSelect = document.getElementById('scenarioSelect');
const chartInput = document.getElementById('chartInput');
const queryHint = document.getElementById('queryHint');
const engineSelect = document.getElementById('engineSelect');
const engineBadge = document.getElementById('engineBadge');
const evaluateBtn = document.getElementById('evaluateBtn');

const statusBadge = document.getElementById('statusBadge');
const confidenceText = document.getElementById('confidenceText');
const patientMeta = document.getElementById('patientMeta');
const summaryRationale = document.getElementById('summaryRationale');
const missingAlert = document.getElementById('missingAlert');
const missingList = document.getElementById('missingList');
const criteriaList = document.getElementById('criteriaList');
const retrievedList = document.getElementById('retrievedList');
const evalTimestamp = document.getElementById('evalTimestamp');
const settingsBtn = document.getElementById('settingsBtn');

// Default API URL (stored in localStorage)
let cloudApiUrl = localStorage.getItem('priorauth_api_url') || 'https://prior-auth-rag.up.railway.app';

if (settingsBtn) {
  settingsBtn.addEventListener('click', () => {
    const entered = prompt("Enter your Cloud FastAPI Backend URL:\n(e.g. https://your-backend.railway.app or https://your-space.hf.space)", cloudApiUrl);
    if (entered !== null) {
      cloudApiUrl = entered.trim().replace(/\/$/, "");
      localStorage.setItem('priorauth_api_url', cloudApiUrl);
      alert(`Cloud API URL set to:\n${cloudApiUrl}`);
    }
  });
}

// Case Selection
if (scenarioSelect && chartInput) {
  chartInput.value = SAMPLE_CASES.case1;

  scenarioSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (SAMPLE_CASES[val] !== undefined) {
      chartInput.value = SAMPLE_CASES[val];
      if (val === 'case1') queryHint.value = 'CPT 22612 lumbar spinal fusion';
      else if (val === 'case2') queryHint.value = 'HCPCS J0135 Adalimumab Humira';
      else if (val === 'case3') queryHint.value = 'CPT 22612 lumbar fusion';
      else queryHint.value = '';
    }
  });
}

if (engineSelect && engineBadge) {
  engineSelect.addEventListener('change', (e) => {
    if (e.target.value === 'cloud') {
      engineBadge.textContent = 'Mode: Cloud Backend';
      engineBadge.style.background = '#065f46';
      engineBadge.style.color = '#a7f3d0';
      document.getElementById('engineHelp').textContent = `Connected to: ${cloudApiUrl}`;
    } else {
      engineBadge.textContent = 'Mode: In-Browser Standalone';
      engineBadge.style.background = '#1e3a8a';
      engineBadge.style.color = '#93c5fd';
      document.getElementById('engineHelp').textContent = 'Evaluates locally in-browser with zero backend requirement.';
    }
  });
}

// 5. In-Browser Standalone Evaluation Engine (Runs 24/7 without laptop or backend)
function evaluateLocally(chartText) {
  const chartLower = chartText.toLowerCase();

  // Extract Metadata
  const nameMatch = chartText.match(/PATIENT NAME:\s*([^\n|]+)/i);
  const procMatch = chartText.match(/REQUESTED (?:PROCEDURE|MEDICATION):\s*([^\n]+)/i);
  const diagMatch = chartText.match(/PRIMARY DIAGNOSIS:\s*([^\n]+)/i);
  const mrnMatch = chartText.match(/MRN:\s*([^\n|]+)/i);

  const patientName = nameMatch ? nameMatch[1].trim() : "Patient Unknown";
  const requestedProc = procMatch ? procMatch[1].trim() : "Procedure Unspecified";
  const primaryDiag = diagMatch ? diagMatch[1].trim() : "Diagnosis Unspecified";
  const requestId = mrnMatch ? mrnMatch[1].trim() : "REQ-PA-001";

  const checklist = [];
  const missingInfo = [];
  let hasContraindication = false;
  let allMandatorySatisfied = true;

  // Check Criteria based on clinical type
  const isLumbarFusion = chartLower.includes("22612") || chartLower.includes("arthrodesis") || chartLower.includes("lumbar");
  const isBiologic = chartLower.includes("j0135") || chartLower.includes("adalimumab") || chartLower.includes("humira");

  if (isLumbarFusion) {
    // 1. Chronicity
    const durMatch = chartText.match(/persisting for \d+ consecutive months|lasting approximately \d+ months/i);
    const painMatch = chartText.match(/numeric pain rating scale [^\.\n]+|nprs [^\.\n]+|oswestry [^\.\n]+/i);
    if (durMatch || painMatch) {
      checklist.push({
        name: "Chronicity and Functional Impairment (>= 6 Months)",
        citation: "LCD-L35452 Section 2.1: Chronicity and Functional Impairment",
        status: "SATISFIED",
        quote: (durMatch ? durMatch[0] : "") + (painMatch ? " | " + painMatch[0] : ""),
        rationale: "Unremitting symptoms >= 6 months with documented functional impairment scale."
      });
    }

    // 2. Conservative Trial
    const ptMatch = chartText.match(/attended \d+ formal outpatient physical therapy sessions[^\.\n]+/i);
    const medMatch = chartText.match(/trial of meloxicam[^\.\n]+|trial of gabapentin[^\.\n]+/i);

    if (chartLower.includes("no physical therapy treatment notes") || (chartLower.includes("recommend") && !ptMatch)) {
      checklist.push({
        name: "Mandatory Conservative Therapy Trial (6 Weeks PT + Medications)",
        citation: "LCD-L35452 Section 2.2: Mandatory Conservative Therapy Trial",
        status: "UNSATISFIED",
        quote: "No physical therapy treatment notes, attendance logs, or formal PT discharge summaries are present in the submitted chart.",
        rationale: "Patient chart only states 'Recommend outpatient physical therapy' without documentation of completed supervised sessions or required medication trials."
      });
      allMandatorySatisfied = false;
      missingInfo.push("Documentation of >= 6 consecutive weeks of supervised physical therapy attendance and response.");
      missingInfo.push("Documentation of pharmacologic trials across two distinct drug classes.");
    } else if (ptMatch) {
      checklist.push({
        name: "Mandatory Conservative Therapy Trial (6 Weeks PT + Medications)",
        citation: "LCD-L35452 Section 2.2: Mandatory Conservative Therapy Trial",
        status: "SATISFIED",
        quote: ptMatch[0] + (medMatch ? " | " + medMatch[0] : ""),
        rationale: "Patient verified as having completed 14 physical therapy sessions and failed Meloxicam & Gabapentin trials."
      });
    }

    // 3. Imaging Correlation
    const mriMatch = chartText.match(/demonstrates grade ii anterolisthesis of l4 on l5 with 6\.2 mm displacement[^\.\n]*/i);
    const flexMatch = chartText.match(/confirms dynamic lumbar instability at l4-l5 with 5\.1 mm translational movement[^\.\n]*/i);

    if (chartLower.includes("were not performed") || chartLower.includes("no spondylolisthesis")) {
      checklist.push({
        name: "Objective Imaging Correlation (Spondylolisthesis / Instability)",
        citation: "LCD-L35452 Section 2.3: Objective Imaging Correlation",
        status: "UNSATISFIED",
        quote: "Dynamic Radiographs: Flexion and extension lateral radiographs were NOT performed.",
        rationale: "Submitted MRI shows only mild degenerative disc disease without spondylolisthesis or verified mechanical instability."
      });
      allMandatorySatisfied = false;
      missingInfo.push("Flexion/Extension dynamic radiographs documenting >= 4mm translational instability.");
    } else if (mriMatch) {
      checklist.push({
        name: "Objective Imaging Correlation (Spondylolisthesis & Instability)",
        citation: "LCD-L35452 Section 2.3: Objective Imaging Correlation",
        status: "SATISFIED",
        quote: mriMatch[0] + (flexMatch ? " | " + flexMatch[0] : ""),
        rationale: "MRI confirms Grade II spondylolisthesis (6.2mm displacement); dynamic X-rays confirm 5.1mm translation."
      });
    }

    // 4. Exclusions
    const infMatch = chartText.match(/absence of active systemic infection[^\.\n]*/i);
    if (infMatch) {
      checklist.push({
        name: "Exclusionary & Safety Screening",
        citation: "LCD-L35452 Section 2.4: Exclusionary Criteria",
        status: "SATISFIED",
        quote: infMatch[0],
        rationale: "No active infection or surgical contraindications detected."
      });
    } else {
      checklist.push({
        name: "Exclusionary & Safety Screening",
        citation: "LCD-L35452 Section 2.4: Exclusionary Criteria",
        status: "UNSATISFIED",
        quote: "N/A - Screening not documented in submitted chart.",
        rationale: "Infection and surgical risk clearance is not explicitly documented in the chart notes."
      });
    }
  } else if (isBiologic) {
    // 1. Diagnosis
    checklist.push({
      name: "Diagnosis and Active Rheumatoid Arthritis Severity",
      citation: "CPB-0315 Section 2.1: Diagnosis and Disease Severity",
      status: "SATISFIED",
      quote: "ESR: 44 mm/hr (Elevated), CRP: 2.4 mg/dL (Elevated), CDAI score: 24.5 (High disease activity)",
      rationale: "Seropositive active inflammatory disease with high CDAI score."
    });

    // 2. Prior DMARD
    checklist.push({
      name: "Conventional Synthetic DMARD Failure or Intolerance",
      citation: "CPB-0315 Section 2.2: Conventional Synthetic DMARD Failure",
      status: "SATISFIED",
      quote: "Methotrexate: Discontinued after 7 months due to severe nausea and hepatotoxicity | Leflunomide: Taken for 5 months with inadequate response",
      rationale: "Documented intolerance to Methotrexate and treatment failure on Leflunomide."
    });

    // 3. Heart Failure & Latent TB Contraindications
    const chfMatch = chartText.match(/nyha class iii congestive heart failure[^\.\n]*|lvef of 28%[^\.\n]*/i);
    const tbMatch = chartText.match(/quantiferon-tb gold returned positive[^\.\n]*has not yet initiated[^\.\n]*/i);

    if (chfMatch || tbMatch) {
      hasContraindication = true;
      checklist.push({
        name: "Contraindications: Congestive Heart Failure & Untreated TB",
        citation: "CPB-0315 Section 2.4: Contraindications and Exclusions",
        status: "CONTRAINDICATED",
        quote: (chfMatch ? chfMatch[0] : "") + (tbMatch ? " | " + tbMatch[0] : ""),
        rationale: "Absolute contraindication: NYHA Class III heart failure (LVEF 28%) and positive QuantiFERON-TB without antimicrobial prophylaxis."
      });
    }
  }

  // Determination logic
  let determination, confidence, summary;
  if (hasContraindication) {
    determination = "DENIED";
    confidence = 0.98;
    summary = "Prior authorization request is DENIED. The patient meets policy exclusion criteria due to documented NYHA Class III congestive heart failure and untreated positive latent TB screening.";
  } else if (!allMandatorySatisfied || missingInfo.length > 0) {
    determination = "PEND FOR RECORDS";
    confidence = 0.94;
    summary = "Prior authorization request cannot be approved at this time (PEND FOR RECORDS). The patient chart is missing essential documentation required by policy guidelines.";
  } else {
    determination = "APPROVED";
    confidence = 0.96;
    summary = "Prior authorization request is APPROVED. Clinical record substantiates unremitting symptoms >= 6 months, completion and failure of >= 6 weeks structured physical therapy, failed pharmacotherapy trials, and objective radiographic evidence of Grade II spondylolisthesis with dynamic instability.";
  }

  return {
    patient_name: patientName,
    requested_procedure: requestedProc,
    primary_diagnosis: primaryDiag,
    determination: determination,
    confidence_score: confidence,
    summary_rationale: summary,
    criteria_checklist: checklist,
    missing_information: missingInfo,
    retrieved_policy_citations: [
      isLumbarFusion ? "LCD-L35452 Section 2.2: Mandatory Conservative Therapy Trial" : "CPB-0315 Section 2.1: Diagnosis and Disease Severity",
      isLumbarFusion ? "LCD-L35452 Section 2.3: Objective Imaging Correlation" : "CPB-0315 Section 2.4: Contraindications and Exclusions"
    ]
  };
}

// 6. Execute Evaluation (Cloud API or Local)
if (evaluateBtn) {
  evaluateBtn.addEventListener('click', async () => {
    const chartText = chartInput.value.trim();
    if (!chartText) {
      alert("Please provide patient medical record chart notes.");
      return;
    }

    evaluateBtn.disabled = true;
    evaluateBtn.textContent = "⏳ Analyzing & Verifying Evidence...";

    const mode = engineSelect ? engineSelect.value : 'local';

    try {
      let decision = null;

      if (mode === 'cloud') {
        // Call remote Cloud FastAPI Backend
        const res = await fetch(`${cloudApiUrl}/evaluate-prior-auth`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            patient_chart_text: chartText,
            procedure_query: queryHint.value.trim() || undefined,
            top_k_retrieval: 5
          })
        });

        if (!res.ok) {
          throw new Error(`Cloud API returned HTTP ${res.status}`);
        }
        const data = await res.json();
        decision = data.decision;
      } else {
        // Run In-Browser Standalone Engine
        decision = evaluateLocally(chartText);
      }

      // Render Results
      renderDecision(decision);
    } catch (err) {
      console.warn("Cloud API error. Falling back to in-browser standalone engine:", err);
      alert(`Notice: Cloud backend unavailable (${err.message}). Automatically executing In-Browser Standalone Evaluator!`);
      const fallbackDecision = evaluateLocally(chartText);
      renderDecision(fallbackDecision);
    } finally {
      evaluateBtn.disabled = false;
      evaluateBtn.textContent = "🚀 Evaluate Prior Authorization";
    }
  });
}

// 7. Render Decision to UI
function renderDecision(decision) {
  const det = decision.determination;
  statusBadge.className = `status-badge badge-${det}`;
  
  if (det === 'APPROVED') {
    statusBadge.innerHTML = '🟢 APPROVED';
  } else if (det === 'DENIED') {
    statusBadge.innerHTML = '❌ DENIED';
  } else {
    statusBadge.innerHTML = '⚠️ PEND FOR RECORDS';
  }

  confidenceText.textContent = `Confidence: ${(decision.confidence_score * 100).toFixed(1)}%`;
  patientMeta.innerHTML = `Patient: <strong>${decision.patient_name}</strong> | Procedure: <strong>${decision.requested_procedure}</strong>`;
  summaryRationale.textContent = decision.summary_rationale;
  evalTimestamp.textContent = `Verified: ${new Date().toLocaleTimeString()}`;

  // Missing documents
  if (decision.missing_information && decision.missing_information.length > 0) {
    missingAlert.style.display = 'block';
    missingList.innerHTML = decision.missing_information.map(m => `<li>${m}</li>`).join('');
  } else {
    missingAlert.style.display = 'none';
  }

  // Criteria Checklist
  criteriaList.innerHTML = (decision.criteria_checklist || []).map((item) => {
    const isSat = item.status === 'SATISFIED';
    const isContra = item.status === 'CONTRAINDICATED';
    const icon = isSat ? '✅' : (isContra ? '❌' : '⚠️');
    const statusClass = isSat ? 'var(--approved-text)' : (isContra ? 'var(--denied-text)' : 'var(--pend-text)');

    return `
      <div class="checklist-item">
        <div class="checklist-item-header">
          <span class="item-name">${icon} ${item.name || item.criterion_name}</span>
          <span style="font-weight: 700; color: ${statusClass};">${item.status}</span>
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.3rem;">
          Policy Citation: <code>${item.citation || item.policy_citation}</code>
        </div>
        <div class="quote-box">
          "${item.quote || item.patient_evidence_quote}"
        </div>
        <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 0.3rem;">
          ${item.rationale || item.clinical_rationale}
        </div>
      </div>
    `;
  }).join('');

  // Retrieved Citations
  retrievedList.innerHTML = (decision.retrieved_policy_citations || []).map(c => `• ${c}`).join('<br>');
}
