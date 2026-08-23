"""LCEL chain: prediction -> grounded structured report. Falls back on API failure."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.genai.llm import get_llm
from app.genai.embeddings import get_vectorstore
from app.core.config import DIFFERENTIAL_GAP_THRESHOLD

_REPORT_PROMPT = ChatPromptTemplate.from_template(
    """You are a radiology screening assistant. Write a brief structured report.
Classification: {diagnosis} ({confidence}% confidence)
Differential note: {differential_note}
Reference context: {context}
Sections: Findings, Interpretation, Recommended Next Steps.
End with: "Screening assistance only — confirm with a certified radiologist."
Keep it under 150 words."""
)

def _differential_note(probabilities: dict[str, float]) -> str:
    top2 = sorted(probabilities.items(), key=lambda kv: -kv[1])[:2]
    if len(top2) == 2 and (top2[0][1] - top2[1][1]) < DIFFERENTIAL_GAP_THRESHOLD:
        return f"Note: {top2[1][0]} ({top2[1][1]*100:.1f}%) cannot be fully excluded."
    return "None."

def generate_report(diagnosis: str, confidence: float, probabilities: dict[str, float]) -> str:
    try:
        context_docs = get_vectorstore().as_retriever(search_kwargs={"k": 2}).invoke(diagnosis)
        context = " ".join(d.page_content for d in context_docs)
        chain = _REPORT_PROMPT | get_llm() | StrOutputParser()
        return chain.invoke({
            "diagnosis": diagnosis, "confidence": round(confidence * 100, 1),
            "differential_note": _differential_note(probabilities), "context": context,
        })
    except Exception:
        return (f"Primary classification: {diagnosis} ({confidence*100:.1f}% confidence). "
                "AI report generation is temporarily unavailable — consult a certified radiologist.")