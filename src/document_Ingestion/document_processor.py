"""Document processing module for loading and splitting documents"""

from typing import List, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

TOPIC_REGISTRATION = "registration"
TOPIC_RESUBMISSION = "resubmission"
TOPIC_ENDPOINTS = "endpoints"
TOPIC_GENERAL = "general"


def get_topic_from_path(file_path: str) -> str:
    """Resolve topic from file path based on filename keywords."""
    path_lower = Path(file_path).name.lower()
    if "endpoints" in path_lower:
        return TOPIC_ENDPOINTS
    if "resubmission" in path_lower:
        return TOPIC_RESUBMISSION
    if any(x in path_lower for x in ("deal", "registration")):
        return TOPIC_REGISTRATION
    return TOPIC_GENERAL


def add_topic_metadata(documents: List[Document], source_path: str) -> None:
    """Set metadata['topic'] on each document. Mutates in place."""
    topic = get_topic_from_path(source_path)
    for d in documents:
        d.metadata["topic"] = topic


class DocumentProcessor:
    """Handle document loading and splitting"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the document processor
        Args:
            chunk_size: The size of each chunk
            chunk_overlap: The overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n---\n", "\n## ", "\n### ", "\n\n", "\n", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_documents(self, file_path: str) -> List[Document]:
        """Load a single text or PDF file.
        Args:
            file_path: The path to the file
        """
        loader = (
            TextLoader(file_path, encoding="utf-8")
            if file_path.endswith(".txt")
            else PyPDFLoader(file_path)
            if file_path.endswith(".pdf")
            else None
        )
        if loader is None:
            raise ValueError(f"Unsupported file type. Use .txt or .pdf, or use load_from_directory() for a folder.")
        docs = loader.load()
        add_topic_metadata(docs, file_path)
        return docs

    def load_from_directory(self, directory_path: str) -> List[Document]:
        """Load all .txt and .pdf documents from a directory.
        Args:
            directory_path: Path to the directory (e.g. 'data' or 'data/')
        """
        directory_path = Path(directory_path).resolve()
        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")

        documents = []

        # Load .txt files and add topic metadata from source path
        txt_loader = DirectoryLoader(
            str(directory_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in txt_loader.load():
            add_topic_metadata([doc], doc.metadata.get("source", ""))
            documents.append(doc)

        # Load .pdf files and add topic metadata from source path
        pdf_loader = DirectoryLoader(
            str(directory_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        for doc in pdf_loader.load():
            add_topic_metadata([doc], doc.metadata.get("source", ""))
            documents.append(doc)

        return documents

    def load(self, source: List[str]) -> List[Document]:
        """Load documents from a list of file paths
        Args:
            source: A list of file paths
        """
        docs: List[Document] = []
        for path in source:
            if Path(path).is_dir():
                docs.extend(self.load_from_directory(path))
            else:
                docs.extend(self.load_documents(path))
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks
        Args:
            documents: A list of documents
        """
        return self.splitter.split_documents(documents)
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        docs = self.load(urls)
        return self.split_documents(docs)