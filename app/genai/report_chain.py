"""LCEL chain: prediction -> grounded structured report. Falls back on API failure."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.genai.llm import get_llm
from app.genai.embeddings import get_vectorstore
from app.core.config import DIFFERENTIAL_GAP_THRESHOLD, LOW_CONFIDENCE_THRESHOLD

_REPORT_PROMPT = ChatPromptTemplate.from_template(
    """You are a specialized neuro-radiology clinical decision support assistant.
Write a concise, preliminary neuro-radiology screening report based strictly on macroscopic brain MRI patterns.

INPUT PARAMETERS:
- Neural Classification: {diagnosis}
- Classification Confidence: {confidence}%
- Differential Assessment: {differential_note}
- Reference Guidelines (WHO CNS v5): {context}

STRICT RADIOLOGICAL GUARDRAILS (ZERO HALLUCINATION TOLERANCE):
1. NEVER DECLARE GENOMIC OR MOLECULAR MUTATIONS:
   - Absolutely NEVER mention IDH status (e.g., "IDH-wildtype", "IDH-mutant"), 1p/19q codeletion, MGMT promoter methylation, or ATRX mutations. Standard structural MRI cannot detect molecular genetics without surgical biopsy and DNA sequencing.
2. NEVER DECLARE MICROSCOPIC HISTOPATHOLOGY:
   - Absolutely NEVER claim microscopic cellular findings (e.g., "microvascular proliferation", "pseudopalisading necrosis", "mitotic count", "nuclear pleomorphism").
3. RESTRICT TO MACROSCOPIC RADIOLOGICAL IMAGING PATTERNS:
   - Confine all findings strictly to visible macroscopic MRI features: signal intensity on T1/T2/FLAIR sequences, anatomical location (intra-axial, extra-axial, sellar/suprasellar), contrast enhancement patterns (homogeneous, heterogeneous, rim-enhancing), lesion margins, perilesional vasogenic edema, and mass effect/midline shift.
4. LOW CONFIDENCE & DIFFERENTIAL MANDATE:
   - If confidence is below 75% or differential note indicates a close call, you MUST NOT declare a single definitive pathology. Provide a prioritized differential diagnosis list instead of a single definitive declaration.

REQUIRED STRUCTURED SECTIONS:
- **Findings**: Visible macroscopic MRI imaging characteristics correlating with {diagnosis}.
- **Radiological Interpretation**: Working diagnostic impression (or prioritized differential if confidence is <75% or close call).
- **Recommended Next Steps**: Recommended follow-up imaging protocols (contrast-enhanced T1/T2/FLAIR, MR perfusion/spectroscopy) and neurosurgical consultation for definitive histopathological evaluation.

MANDATORY DISCLAIMER (Must be the final line):
Screening assistance only — confirm with a certified radiologist.

Keep the entire report concise, factual, and strictly under 150 words."""
)

def _differential_note(confidence: float, probabilities: dict[str, float]) -> str:
    top2 = sorted(probabilities.items(), key=lambda kv: -kv[1])[:2]
    if len(top2) == 2:
        gap = top2[0][1] - top2[1][1]
        if gap < DIFFERENTIAL_GAP_THRESHOLD or confidence < LOW_CONFIDENCE_THRESHOLD:
            return (f"Close differential alert: Primary {top2[0][0]} ({top2[0][1]*100:.1f}%) vs "
                    f"Secondary {top2[1][0]} ({top2[1][1]*100:.1f}%). Confidence is limited; "
                    "a prioritized differential diagnosis list is mandated.")
    return "Primary feature pattern identified. No immediate close secondary class."

def generate_report(diagnosis: str, confidence: float, probabilities: dict[str, float]) -> str:
    try:
        context_docs = get_vectorstore().as_retriever(search_kwargs={"k": 2}).invoke(diagnosis)
        context = " ".join(d.page_content for d in context_docs)
        chain = _REPORT_PROMPT | get_llm(max_tokens=1000) | StrOutputParser()
        raw_report = chain.invoke({
            "diagnosis": diagnosis,
            "confidence": round(confidence * 100, 1),
            "differential_note": _differential_note(confidence, probabilities),
            "context": context,
        })
        return str(raw_report).strip()
    except Exception as e:
        return (f"**Findings:**\n- Primary screening classification: {diagnosis} ({confidence*100:.1f}% confidence).\n\n"
                "**Radiological Interpretation:**\nPreliminary AI screening observation.\n\n"
                "**Recommended Next Steps:**\n- Contrast-enhanced multi-sequence MRI review.\n- Formal neuro-radiology evaluation.\n\n"
                "Screening assistance only — confirm with a certified radiologist.")