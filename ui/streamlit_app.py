"""Streamlit UI for Agentic RAG System - Simplified Version"""

import logging
import streamlit as st
from pathlib import Path
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")

sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import Config
from src.document_Ingestion.document_processor import DocumentProcessor
from src.vectorstore.vector_store import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        data_folder = getattr(Config, "DATA_FOLDER", "data")
        documents = doc_processor.process_urls([data_folder])
        
        vector_store.create_vector_store(documents)
        
        graph_builder = GraphBuilder(
            retriever=vector_store,
            llm=llm
        )
        graph_builder.build_graph()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    """Main application"""
    init_session_state()
    
    st.title("🔍 Project Knowledge Assistant")
    st.markdown("Ask questions about the loaded documents")
    
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                result = st.session_state.rag_system.run_graph(question)
                elapsed_time = time.time() - start_time
                
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
                
                st.markdown("### 💡 Answer")
                st.success(result['answer'])
                
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result['retrieved_context'], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:1000] + "...",
                            height=100,
                            disabled=True
                        )
                
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()
