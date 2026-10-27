"""Configuration module for Agentic RAG system"""


import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
# Load environment variables
load_dotenv()

class Config:
    EMBEDDING_MODEL="all-MiniLM-L6-v2"
    
    # Model Configuration
    LLM_MODEL = "llama3.2"
    
    # Document Processing
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 150
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    @staticmethod
    def get_llm():
        """Initialize and return the local Ollama LLM."""
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1
        )
