"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    LLM_MODEL = "openai:gpt-4o"
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval
    RETRIEVER_K = 15
    
    # Default URLs
    DATA_FOLDER = "data"
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)