"""Real-time grounding — scoped to trusted medical literature, not open web search."""
from langchain_core.tools import tool
from Bio import Entrez

Entrez.email = "your_email@example.com"  # required by NCBI, no key needed

@tool
def search_pubmed(query: str) -> str:
    """Search PubMed when local knowledge base lacks an answer to a medical question."""
    try:
        ids = Entrez.read(Entrez.esearch(db="pubmed", term=query, retmax=3))["IdList"]
        if not ids:
            return "No relevant PubMed results found."
        return Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text").read()[:1200]
    except Exception:
        return "PubMed lookup unavailable right now."