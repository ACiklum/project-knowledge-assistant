from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.config.config import Config


class VectorStore:
    """Handle vector store operations. get_retriever(topic=...) filters by metadata['topic']."""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def create_vector_store(self, documents: List[Document]):
        """Create the FAISS index from documents."""
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

    def get_retriever(self, topic: Optional[str] = None):
        """Returns a retriever, optionally filtered by topic metadata."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        k = Config.RETRIEVER_K
        if topic:
            return self.vectorstore.as_retriever(
                search_kwargs={"filter": {"topic": topic}, "k": k}
            )
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    def retrieve_vector(self, query: str, k: int = 10, topic: Optional[str] = None) -> List[Document]:
        """Query the vector store. Use topic to restrict to deal or endpoints docs."""
        retriever = self.get_retriever(topic=topic)
        return retriever.invoke(query)
