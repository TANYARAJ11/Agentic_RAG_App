"""Langgraph node for RAG workflow + ReAct Agent inside generate_answer"""

from typing import List, Optional

import requests
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool

from src.state.rag_state import RAGState

# Wikipedia asks API users to identify themselves with a User-Agent
WIKI_HEADERS = {"User-Agent": "DocRagApp/1.0 (tanyaraj381@gmail.com)"}


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
        return " ".join(p for p in parts if p).strip()
    return str(content)


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy init agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
        question=state.question,
        retrieved_docs=docs,
    )

    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found"
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        def wikipedia_tool_fn(query: str) -> str:
            """Search Wikipedia and return intro text of the top matches."""
            try:
                r = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "generator": "search",
                        "gsrsearch": query,
                        "gsrlimit": 3,
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "exlimit": 3,
                        "format": "json",
                    },
                    headers=WIKI_HEADERS,
                    timeout=15,
                )
                r.raise_for_status()
                pages = r.json().get("query", {}).get("pages", {})
                if not pages:
                    return "No Wikipedia results found."
                ordered = sorted(pages.values(), key=lambda p: p.get("index", 0))
                return "\n\n".join(
                    f"{p['title']}\n{p.get('extract', '')[:1500]}" for p in ordered
                )
            except Exception as e:
                # Return the problem as text so the app doesn't crash
                return f"Wikipedia lookup failed: {e}"

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from the user's indexed documents",
            func=retriever_tool_fn,
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge, people, places and definitions.",
            func=wikipedia_tool_fn,
        )

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """ReAct agent with tools"""
        if self._agent is not None:
            return

        tools = self._build_tools()
        system_prompt = (
            "You are a tool-using AI agent. You DO have access to external tools via tool calls.\n"
            "CRITICAL RULES:\n"
            "1. For questions about private documents, ALWAYS call the 'retriever' tool.\n"
            "2. For general knowledge, public facts, definitions, or world info (like 'What is LangChain?'), you MUST call the 'wikipedia' tool.\n"
            "3. NEVER tell the user you cannot access the internet. You have the 'wikipedia' tool - USE IT.\n"
            "4. Output your tool call before generating a final answer."
        )
        self._agent = create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using the ReAct agent with retriever + wikipedia."""
        print("[NODE] wikipedia agent running")
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer = _to_text(getattr(messages[-1], "content", ""))

        return RAGState(
            question=state.question,
            retrieved_docs=[],                      # local chunks weren't used, so don't show them
             answer=answer or "Could not generate answer",
            source="wikipedia",
)