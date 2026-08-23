import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.genai.llm import get_llm
from app.genai.embeddings import get_vectorstore

_QA_PROMPT = ChatPromptTemplate.from_template(
    """You are NeuroScan AI, a clinical decision support assistant.
Answer the user's question accurately and concisely based on medical knowledge and the provided context.

Context:
{context}

Question: {question}

Instructions:
- Keep the response clear, helpful, and under 80 words.
- Provide factual medical information suitable for clinical screening assistance.

Answer:"""
)

def answer_question(question: str) -> str:
    try:
        # 1. Retrieve local context if vectorstore exists
        context = ""
        try:
            vs = get_vectorstore()
            if vs:
                docs = vs.as_retriever(search_kwargs={"k": 2}).invoke(question)
                context = " ".join(d.page_content for d in docs)
        except Exception as e:
            print(f"[VectorStore Warning] Could not retrieve docs: {e}")
            context = "General Neuro-oncology guidelines and brain MRI diagnostic reference."

        if not context or len(context.strip()) < 20:
            context = "Brain tumors include Gliomas, Meningiomas, and Pituitary adenomas categorized by WHO CNS criteria."

        # 2. Invoke Groq LLM
        chain = _QA_PROMPT | get_llm(max_tokens=200) | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    except Exception as e:
        # Terminal par exact error print hoga debugging ke liye
        print(f"[QA Chain Error]: {e}")
        traceback.print_exc()
        return "NeuroScan AI is currently processing. Please rephrase your question or check network connection."