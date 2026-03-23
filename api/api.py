import sys
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.config import Config
from src.document_Ingestion.document_processor import DocumentProcessor
from src.vectorstore.vector_store import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

graph_builder: GraphBuilder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load documents, build FAISS index, and compile LangGraph once at startup."""
    global graph_builder
    logger.info("Initializing RAG pipeline...")

    llm = Config.get_llm()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    vector_store = VectorStore()

    data_folder = Path(project_root) / Config.DATA_FOLDER
    documents = doc_processor.process_urls([str(data_folder)])
    vector_store.create_vector_store(documents)

    graph_builder = GraphBuilder(retriever=vector_store, llm=llm)
    graph_builder.build_graph()

    logger.info("RAG pipeline ready — %d chunks indexed.", len(documents))
    yield


app = FastAPI(
    title="Project Knowledge Assistant API",
    description="API for the AI-Agentic System Knowledge Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str


class SourceDocument(BaseModel):
    content: str
    source: str


class QueryMetadata(BaseModel):
    source: str = ""
    docs_retrieved: int = 0
    sources: List[SourceDocument] = []


class QueryResponse(BaseModel):
    answer: str
    metadata: Optional[QueryMetadata] = None


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Project Knowledge Assistant API. Use POST /ask to submit queries."
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    """Submit a question to the RAG pipeline and receive an answer with source documents."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = graph_builder.run_graph(request.query)

        sources = [
            {"content": doc.page_content[:500], "source": doc.metadata.get("source", "unknown")}
            for doc in result.get("retrieved_context", [])
        ]

        return QueryResponse(
            answer=result.get("answer", "No answer could be generated."),
            metadata={
                "source": "RAG Pipeline",
                "docs_retrieved": len(sources),
                "sources": sources,
            },
        )
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {str(e)}"
        )


# To run the API locally:
# uvicorn api.api:app --reload
