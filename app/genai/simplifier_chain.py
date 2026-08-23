from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.genai.llm import get_llm

_SIMPLIFY_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite this report in simple, everyday language for a patient. No jargon. Under 100 words.\n\n{report}"
)

def simplify_report(report: str) -> str:
    try:
        return (_SIMPLIFY_PROMPT | get_llm(max_tokens=200) | StrOutputParser()).invoke({"report": report})
    except Exception:
        return report  # fallback: original report, never fail silently