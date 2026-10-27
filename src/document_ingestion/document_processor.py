"""Document processing module with local Semantic Chunking using all-MiniLM-L6-v2"""

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class DocumentProcessor:
    """Handles document loading and semantic splitting using local embeddings"""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150, embeddings=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if embeddings is None:
            print("Loading local embedding model (all-MiniLM-L6-v2)...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.embeddings = embeddings

        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
    )
    
    # def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
    #     """
    #     Initialize document processor with local HuggingFace embeddings 
    #     for semantic chunking.
    #     """
    #     self.chunk_size = chunk_size
    #     self.chunk_overlap = chunk_overlap
        
    #     print("Loading local embedding model (all-MiniLM-L6-v2) for Semantic Chunking...")
    #     local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
    #     # Initialize SemanticChunker with local embeddings
    #     self.splitter = SemanticChunker(
    #         embeddings=local_embeddings,
    #         breakpoint_threshold_type="percentile",
    #         breakpoint_threshold_amount=95
    #     )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file   """
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or TXT files

        Args:
            sources: List of URLs, PDF folder paths, or TXT file paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
            else:
                path = Path(src)
                if path.is_dir():  # PDF directory
                    docs.extend(self.load_from_pdf_dir(path))
                elif path.is_file() and path.suffix.lower() == ".txt":
                    docs.extend(self.load_from_txt(path))
                else:
                    raise ValueError(
                        f"Unsupported source type: {src}. "
                        "Use URL, .txt file, or PDF directory."
                    )
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using semantic chunking
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of semantically split document chunks
        """
        return self.splitter.split_documents(documents)
    
    def process_url(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and semantically split documents
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed semantic document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)

    
# ==========================================
# Temporary test runner
# ==========================================
# if __name__ == "__main__":
#     print("Testing DocumentProcessor with Semantic Chunking...")
#     processor = DocumentProcessor()
    
#     # Test with a sample URL
#     test_urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    
#     try:
#         print(f"Loading and splitting URL: {test_urls[0]}")
#         chunks = processor.process_url(test_urls)
#         print(f"\nSuccessfully created {len(chunks)} semantic chunks!")
#         if chunks:
#             print("\n--- Sample Chunk 1 ---")
#             print(chunks[0].page_content[:300] + "...")
#     except Exception as e:
#         print(f"Error during test: {e}")
