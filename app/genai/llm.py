"""Single LLM client — Groq's high-speed LPU inference, zero local RAM cost."""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.core.config import GROQ_API_KEY, MAX_TOKENS_REPORT, LLM_TIMEOUT_SECONDS

load_dotenv()

def get_llm(max_tokens: int = MAX_TOKENS_REPORT) -> ChatGroq:
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model_name="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=max_tokens,
        timeout=LLM_TIMEOUT_SECONDS,
    )