# # """Graph builder with Local RAG first, and Wikipedia ReAct Agent fallback"""

# from langgraph.graph import StateGraph, END
# from src.state.rag_state import RAGState
# from src.nodes.nodes import RAGNodes as LocalRAGNodes
# from src.nodes.reactnode import RAGNodes as ReactAgentNodes

# class GraphBuilder:
#     def __init__(self, retriever, llm):
#         self.retriever = retriever
#         self.llm = llm 
        
#         # Initialize both standard local nodes and ReAct agent nodes
#         self.local_nodes = LocalRAGNodes(retriever=self.retriever, llm=self.llm)
#         self.react_nodes = ReactAgentNodes(retriever=self.retriever, llm=self.llm)
#         self.graph = None

#     def _should_use_wikipedia(self, state: RAGState) -> str:
#         """
#         Condition: Check if local documents have enough info.
#         If no docs are retrieved or content is empty/irrelevant, route to 'wikipedia_agent'.
#         Otherwise, route to standard local 'responder'.
#         """
#         docs = state.retrieved_docs
#         if not docs or len(docs) == 0:
#             return "wikipedia_agent"
        
#         # Combine text to check substance
#         content = "\n".join([d.page_content for d in docs])
#         if len(content.strip()) < 30 or "no documents found" in content.lower():
#             return "wikipedia_agent"
            
#         return "resp"

#     def build(self):
#         """Build the conditional workflow graph"""
#         builder = StateGraph(RAGState)

#         # 1. Add nodes
#         builder.add_node("retriever", self.local_nodes.retrieve_docs)
#         builder.add_node("responder", self.local_nodes.generate_answer)
#         builder.add_node("wikipedia_agent", self.react_nodes.generate_answer)

#         # 2. Set entry point to always check local docs first
#         builder.set_entry_point("retriever")

#         # 3. Add conditional routing based on retrieved doc quality
#         builder.add_conditional_edges(
#             "retriever",
#             self._should_use_wikipedia,
#             {
#                 "responder": "responder",
#                 "wikipedia_agent": "wikipedia_agent"
#             }
#         )

#         # 4. Terminal edges
#         builder.add_edge("responder", END)
#         builder.add_edge("wikipedia_agent", END)

#         self.graph = builder.compile()
#         return self.graph

#     def run(self, question: str) -> dict:
#         if self.graph is None:
#             self.build()
#         initial_state = RAGState(question=question)
#         return self.graph.invoke(initial_state)

"""Graph builder: Local RAG first, Wikipedia ReAct agent as fallback"""

from typing import Optional

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes as LocalRAGNodes
from src.nodes.reactnode import RAGNodes as ReactAgentNodes

# Route names defined ONCE so a typo can never happen again
ROUTE_LOCAL = "responder"
ROUTE_WIKI = "wikipedia_agent"


def _to_text(content) -> str:
    """LLM replies can be a str or a list of blocks (some Gemini versions)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return str(content)


class GraphBuilder:
    def __init__(self, retriever, llm, force_route: Optional[str] = None):
        """
        force_route: None for normal behaviour (recommended),
                     "responder" or "wikipedia_agent" to test one path only.
        """
        self.retriever = retriever
        self.llm = llm
        self.force_route = force_route

        self.local_nodes = LocalRAGNodes(retriever=self.retriever, llm=self.llm)
        self.react_nodes = ReactAgentNodes(retriever=self.retriever, llm=self.llm)
        self.graph = None

    def _should_use_wikipedia(self, state: RAGState) -> str:
        """Decide which kitchen cooks the answer."""

        # Testing switch
        if self.force_route in (ROUTE_LOCAL, ROUTE_WIKI):
            print(f"[ROUTER] forced -> {self.force_route}")
            return self.force_route

        # Gate 1: nothing useful retrieved
        docs = state.retrieved_docs
        if not docs:
            print("[ROUTER] no docs -> wikipedia_agent")
            return ROUTE_WIKI

        context = "\n\n".join(d.page_content for d in docs)
        if len(context.strip()) < 30:
            print("[ROUTER] docs nearly empty -> wikipedia_agent")
            return ROUTE_WIKI

        # Gate 2: ask the LLM if these docs can really answer the question
        # (each chunk trimmed so the grader prompt stays small and fast)
        short_context = "\n\n".join(d.page_content[:800] for d in docs)
        prompt = f"""You are a strict relevance grader.

Question: {state.question}

Context:
{short_context}

Can the question be fully answered using ONLY this context?
Reply with exactly one word: YES or NO."""

        try:
            verdict = _to_text(self.llm.invoke(prompt).content).strip().upper()
        except Exception as e:
            # Unsure -> the agent is the safe choice (it also has the retriever tool)
            print(f"[ROUTER] grader failed ({e}) -> wikipedia_agent")
            return ROUTE_WIKI

        route = ROUTE_LOCAL if verdict.startswith("YES") else ROUTE_WIKI
        print(f"[ROUTER] verdict={verdict!r} -> {route}")
        return route

    def build(self):
        builder = StateGraph(RAGState)

        builder.add_node("retriever", self.local_nodes.retrieve_docs)
        builder.add_node(ROUTE_LOCAL, self.local_nodes.generate_answer)
        builder.add_node(ROUTE_WIKI, self.react_nodes.generate_answer)

        builder.set_entry_point("retriever")

        builder.add_conditional_edges(
            "retriever",
            self._should_use_wikipedia,
            {
                ROUTE_LOCAL: ROUTE_LOCAL,
                ROUTE_WIKI: ROUTE_WIKI,
            },
        )

        builder.add_edge(ROUTE_LOCAL, END)
        builder.add_edge(ROUTE_WIKI, END)

        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> dict:
        if self.graph is None:
            self.build()
        return self.graph.invoke(RAGState(question=question))