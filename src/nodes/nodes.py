"""Langgraph node for RAG workflow"""

from src.state.rag_state import RAGState

class RAGNodes:
    """Contains node functions for RAG workflow """

    def __init__(self,retriever, llm):
        """
        Initialize RAG nodes
        Args:
            retriever: Document retriever instance
            llm: Language model instance

        """
        self.retriever = retriever
        self.llm = llm


    def retrieve_docs(self,state:RAGState)->RAGState:
        """
        retrieve relevant document nodes

        Args:
            state: current RAG state

        Returns:
            Updated RAG state with retrieved documents    
        """

        docs = self.retriever.invoke(state.question)
        return RAGState(
         question=state.question,
        retrieved_docs=docs,
)

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer from retrieved documents

        Args:
            state: Current RAG State with retrieved documents

        Returns:
            Updated RAG state with generated answer

        """
        #combine retrieved documents into context

        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # Create prompt
        prompt = f"""Answer the question based on the context.

        Context:
        {context}

        Question: {state.question}"""

        response = self.llm.invoke(prompt)
       
        return RAGState(
         question=state.question,
         retrieved_docs=state.retrieved_docs,
         answer=response.content,
         source="local_docs",
)


