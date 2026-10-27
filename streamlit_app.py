
"""Streamlit UI for Agentic RAG System - Simplified Version"""

import streamlit as st
from pathlib import Path
import sys
import time
from langchain_huggingface import HuggingFaceEmbeddings
# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
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
def get_embeddings():
    """Load the embedding model once and reuse it everywhere."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()
        embeddings = get_embeddings()                       # <- change 1: load once

        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            embeddings=embeddings,                          # <- change 2a: share it
        )
        vector_store = VectorStore(embeddings=embeddings)   # <- change 2b: share it

        # Use default URLs
        urls = Config.DEFAULT_URLS

        # Process documents
        documents = doc_processor.process_url(urls)

        # Create vector store
        vector_store.create_vectorstore(documents)

        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
        # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                result = st.session_state.rag_system.run(question)
                elapsed_time = time.time() - start_time

            source = result.get("source", "")
            answer = result.get("answer") or "No answer was returned."

            st.session_state.history.append({
                'question': question,
                'answer': answer,
                'source': source,
                'time': elapsed_time,
            })

            # 1) The answer itself
            st.markdown("### 💡 Answer")
            st.markdown(answer)

            # 2) Where it came from
            if source == "wikipedia":
                st.info("🌐 Answered from Wikipedia (your documents didn't cover this)")
            elif source == "local_docs":
                st.info("📚 Answered from your documents")
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result["retrieved_docs"], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True,
                            key=f"doc_{len(st.session_state.history)}_{i}",
                        )
            else:
                st.warning(f"⚠️ Source label missing (got {source!r}).")

            st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                badge = "🌐 Wikipedia" if item.get('source') == "wikipedia" else "📚 Documents"
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"{badge} · Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()