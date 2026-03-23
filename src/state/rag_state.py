from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RAGState(BaseModel):
    """State for the RAG pipeline, including self-reflection loop fields."""

    question: str = ""
    rewritten_query: str = ""
    retrieved_context: List[Document] = []
    answer: str = ""

    retry_count: int = 0
    max_retries: int = Field(default=2, description="Max reflection retries before accepting best-effort answer")
    reflection_score: int = 0
    reflection_critique: str = ""