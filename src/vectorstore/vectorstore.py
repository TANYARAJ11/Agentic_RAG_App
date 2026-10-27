# # # """vector store module for the document embedding and retrieval """

# # # from typing import List
# # # from langchain_community.vectorstores import FAISS
# # # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # # from langchain_core.documents import Document 
# # # from config.config import Config



# # # class VectorStore:
# # #     """manages vector store application"""
# # #     # def __init__(self):
# # #     #     self.embedding= GoogleGenerativeAIEmbeddings()
# # #     #     self.vectorstore=None
# # #     #     self.retriever=None
# # #     # def __init__(self):
# # #     #     # Explicitly pass the model name and API key
# # #     #     if not Config.GOOGLE_API_KEY:
# # #     #         raise ValueError("GOOGLE_API_KEY is not set in Config")
            
# # #     #     self.embedding = GoogleGenerativeAIEmbeddings(
# # #     #         model="model/text-embedding-004",
# # #     #         google_api_key=Config.GOOGLE_API_KEY
# # #     #     )
# # #     #     self.vectorstore = None
# # #     #     self.retriever_instance = None
# # #     def __init__(self):
# # #         # Force a refresh check of the API key
# # #         api_key = Config.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        
# # #         if not api_key:
# # #             raise ValueError("GOOGLE_API_KEY is missing! Check your .env file and restart the app.")

# # #         # Using models/embedding-001 with retrieval_document task type
# # #         # is the most stable configuration for the v1beta API in India.
# # #         self.embedding = GoogleGenerativeAIEmbeddings(
# # #             model="gemini-embedding-2",
# # #             google_api_key=api_key,
        
# # #         )
# # #         self.vectorstore = None
# # #         self.retriever_instance = None


# # #     def create_vectorstore(self, documents: List[Document]):
# # #         """
# # #         Create vector store from documents
        
# # #         Args:
# # #             documents: List of documents to embed
# # #         """
# # #         self.vectorstore = FAISS.from_documents(documents, self.embedding)
# # #         self.retriever = self.vectorstore.as_retriever()


# # #     def create_retriever(self,documents: List[Document]):
# # #         """
# # #         Create vectore store from documents
        
# # #         Args:
# # #             documents: List of documents to embed

# # #         """    
# # #         self.vectorstore=FAISS.from_documents(documents,self.embedding)
# # #         self.retriever=self.vectorestore.as_retriever()


# # #     def get_retriever(self):
# # #         """
# # #         Get the retriever instance

# # #         Returns:
# # #             Retreiver instance
# # #         """

# # #         if self.retriever is None:
# # #             raise ValueError("Vector store not initialized. Call create_vectorstore first.")
# # #         return self.retriever

# # #     def retriever(self,query: str, k: int =4 )->List[Document]:
# # #         """
# # #         Retriever relevant documents for a query

# # #         Args:
# # #             query:Search query
# # #             k: Number of documents to retrive

# # #         Returns:
# # #             List of relevant documents
# # #         """

# # #         if self.retriever is None:
# # #             raise ValueError("Vector store not initialized. Call create_vectorstore first.")
# # #         return self.retriever.invoke(query)
# # """vector store module for the document embedding and retrieval """

# # import os
# # import time
# # from typing import List
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from langchain_core.documents import Document 
# # from src.config.config import Config

# # class VectorStore:
# #     """manages vector store application"""
# #     def __init__(self):
# #         api_key = Config.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        
# #         if not api_key:
# #             raise ValueError("GOOGLE_API_KEY is missing!")

# #         self.embedding = GoogleGenerativeAIEmbeddings(
# #             model="text-embedding-004",
# #             google_api_key=api_key
# #         )
# #         self.vectorstore = None
# #         self.retriever_instance = None

# #     def create_vectorstore(self, documents: List[Document], batch_size: int = 5, delay: int = 2):
# #         """
# #         Create vector store from documents in small batches to avoid 429 rate limits.
        
# #         Args:
# #             documents: List of documents to embed
# #             batch_size: Number of chunks to send per request (Keep small for free tier)
# #             delay: Seconds to wait between batches
# #         """
# #         if not documents:
# #             raise ValueError("No documents provided to create vector store.")

# #         # Initialize FAISS with the first batch
# #         print(f"Embedding batch 1 of {(len(documents) + batch_size - 1) // batch_size}...")
# #         self.vectorstore = FAISS.from_documents(documents[:batch_size], self.embedding)
        
# #         # Add remaining documents in small batches with a delay
# #         for i in range(batch_size, len(documents), batch_size):
# #             batch_num = (i // batch_size) + 1
# #             total_batches = (len(documents) + batch_size - 1) // batch_size
# #             print(f"Embedding batch {batch_num} of {total_batches}...")
            
# #             time.sleep(delay)  # Wait to respect the rate limit
# #             batch_docs = documents[i:i + batch_size]
# #             self.vectorstore.add_documents(batch_docs)

# #         self.retriever_instance = self.vectorstore.as_retriever()

# #     def create_retriever(self, documents: List[Document]):
# #         """
# #         Create vector store from documents
# #         """    
# #         self.create_vectorstore(documents)

# #     def get_retriever(self):
# #         """
# #         Get the retriever instance
# #         """
# #         if self.retriever_instance is None:
# #             raise ValueError("Vector store not initialized. Call create_vectorstore first.")
# #         return self.retriever_instance

# #     def retrieve(self, query: str, k: int = 4) -> List[Document]:
# #         """
# #         Retrieve relevant documents for a query
# #         """
# #         if self.retriever_instance is None:
# #             raise ValueError("Vector store not initialized.")
# #         return self.retriever_instance.invoke(query)

# """vector store module for the document embedding and retrieval """

# import os
# import time
# from typing import List
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document 
# from src.config.config import Config

# class VectorStore:
#     """manages vector store application"""
#     def __init__(self):
#         api_key = Config.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        
#         if not api_key:
#             raise ValueError("GOOGLE_API_KEY is missing!")

#         self.embedding = GoogleGenerativeAIEmbeddings(
#             model="text-embedding-004",
#             google_api_key=api_key
#         )
#         self.vectorstore = None
#         self.retriever_instance = None

#     def create_vectorstore(self, documents: List[Document], batch_size: int = 5, delay: int = 2):
#         """
#         Create vector store from documents in small batches to avoid 429 rate limits.
        
#         Args:
#             documents: List of documents to embed
#             batch_size: Number of chunks to send per request (Keep small for free tier)
#             delay: Seconds to wait between batches
#         """
#         if not documents:
#             raise ValueError("No documents provided to create vector store.")

#         # Initialize FAISS with the first batch
#         print(f"Embedding batch 1 of {(len(documents) + batch_size - 1) // batch_size}...")
#         self.vectorstore = FAISS.from_documents(documents[:batch_size], self.embedding)
        
#         # Add remaining documents in small batches with a delay
#         for i in range(batch_size, len(documents), batch_size):
#             batch_num = (i // batch_size) + 1
#             total_batches = (len(documents) + batch_size - 1) // batch_size
#             print(f"Embedding batch {batch_num} of {total_batches}...")
            
#             time.sleep(delay)  # Wait to respect the rate limit
#             batch_docs = documents[i:i + batch_size]
#             self.vectorstore.add_documents(batch_docs)

#         self.retriever_instance = self.vectorstore.as_retriever()

#     def create_retriever(self, documents: List[Document]):
#         """
#         Create vector store from documents
#         """    
#         self.create_vectorstore(documents)

#     def get_retriever(self):
#         """
#         Get the retriever instance
#         """
#         if self.retriever_instance is None:
#             raise ValueError("Vector store not initialized. Call create_vectorstore first.")
#         return self.retriever_instance

#     def retrieve(self, query: str, k: int = 4) -> List[Document]:
#         """
#         Retrieve relevant documents for a query
#         """
#         if self.retriever_instance is None:
#             raise ValueError("Vector store not initialized.")
#         return self.retriever_instance.invoke(query)

"""vector store module for the document embedding and retrieval """

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document 

class VectorStore:
    """manages vector store application using local HuggingFace embeddings"""
    def __init__(self, embeddings=None):
        if embeddings is None:
            print("Loading local HuggingFace embeddings (all-MiniLM-L6-v2)...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.embedding = embeddings
        self.vectorstore = None
        self.retriever_instance = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents locally
        
        Args:
            documents: List of documents to embed
        """
        if not documents:
            raise ValueError("No documents provided to create vector store.")

        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever_instance = self.vectorstore.as_retriever()

    def create_retriever(self, documents: List[Document]):
        """
        Create vector store from documents
        """    
        self.create_vectorstore(documents)

    def get_retriever(self):
        """
        Get the retriever instance

        Returns:
            Retriever instance
        """
        if self.retriever_instance is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever_instance

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        if self.retriever_instance is None:
            raise ValueError("Vector store not initialized.")
        return self.retriever_instance.invoke(query)

# ==========================================
# Temporary test runner
# ==========================================
# if __name__ == "__main__":
#     from langchain_core.documents import Document
    
#     print("Testing VectorStore with local embeddings...")
#     vs = VectorStore()
    
#     # Create mock documents
#     test_docs = [
#         Document(page_content="LangChain is a framework for developing applications powered by language models."),
#         Document(page_content="HuggingFace provides state-of-the-art machine learning models and datasets.")
#     ]
    
#     vs.create_vectorstore(test_docs)
#     results = vs.retrieve("What is LangChain?", k=1)
    
#     print(f"\nRetrieved {len(results)} document(s):")
#     for doc in results:
#         print(f"- {doc.page_content}")