import json
import logging

from typing import List
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent

from src.state.rag_state import RAGState

logger = logging.getLogger(__name__)

TOPIC_REGISTRATION = "registration"
TOPIC_RESUBMISSION = "resubmission"
TOPIC_ENDPOINTS = "endpoints"

REGISTRATION_SEARCH_DESCRIPTION = (
    "Search the deal registration and submission knowledge base. "
    "Use for: creating a new deal, registering a deal, base fields, participants, "
    "property data, loan options submission, waivers, automated pricing, "
    "final deal submission, Dealpath, DUS Gateway verification. "
    "Do NOT use for resubmission workflows or API endpoint URLs."
)

RESUBMISSION_SEARCH_DESCRIPTION = (
    "Search the deal resubmission workflow knowledge base. "
    "Use for: resubmission steps, NPulse, drafting a resubmission, resubmission waivers, "
    "resubmission loan options, resubmission status, failure troubleshooting, "
    "when to resubmit, resubmission API requirements. "
    "Do NOT use for initial deal registration or API endpoint URLs."
)

ENDPOINTS_SEARCH_DESCRIPTION = (
    "Search the Fannie Mae API endpoints reference. "
    "Use for: API endpoints, URLs, POST/GET/PATCH, which endpoint to call for "
    "deals, properties, participants, loan options, waivers, pricing. "
    "Do NOT use for process steps or workflows."
)

QUERY_REWRITE_PROMPT = """\
You are a query rewriter for a knowledge base search system about Fannie Mae deal registration and API endpoints.

Rewrite the user's question to improve retrieval. You should:
1. Expand abbreviations and vague terms using domain-specific vocabulary
2. Add synonyms that the knowledge base might use (e.g. "loan" → "loan option", "endpoint" → "API endpoint URL")
3. Keep the original intent intact

Domain terms to consider: deal registration, resubmission, loan options, waivers, \
participants, borrowers, key principals, sponsors, guarantors, properties, \
regulatory restrictions, AMI levels, automated pricing, prereview waivers, \
API endpoints, POST, GET, PATCH, DELETE, DUS Gateway, Dealpath, NPulse.

Return ONLY the rewritten query, no explanation."""

REFLECTION_PROMPT = """\
You are a quality reviewer for a RAG (Retrieval-Augmented Generation) system.

Given the following:
- USER QUESTION: {question}
- RETRIEVED CONTEXT: {context}
- GENERATED ANSWER: {answer}

Evaluate the answer on these criteria (1-5 scale each):
1. Relevance  — Does it directly answer the question?
2. Groundedness — Is it supported by the retrieved context?
3. Completeness — Does it cover all important aspects?

Respond with ONLY a valid JSON object (no markdown fences, no extra text):
{{"overall_score": <average of 3 scores rounded to nearest int, 1-5>, "critique": "<brief critique explaining gaps>", "retry": <true if overall_score < 3, else false>}}"""


class RAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.rag_agent = None

    def create_rag_tool(self, name: str, desc: str, retriever) -> Tool:
        def tool_func(query: str) -> str:
            results = retriever.invoke(query)
            return "\n\n".join(
                f"Chunk {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(results[:5])
            )

        return Tool(name=name, description=desc, func=tool_func)

    def build_rag_tools(self) -> List[Tool]:
        registration_retriever = self.retriever.get_retriever(TOPIC_REGISTRATION)
        resubmission_retriever = self.retriever.get_retriever(TOPIC_RESUBMISSION)
        endpoints_retriever = self.retriever.get_retriever(TOPIC_ENDPOINTS)
        return [
            self.create_rag_tool("registration_search", REGISTRATION_SEARCH_DESCRIPTION, registration_retriever),
            self.create_rag_tool("resubmission_search", RESUBMISSION_SEARCH_DESCRIPTION, resubmission_retriever),
            self.create_rag_tool("endpoints_search", ENDPOINTS_SEARCH_DESCRIPTION, endpoints_retriever),
        ]

    AGENT_PROMPT = (
        "You are a knowledge assistant. Pick the most relevant tool based on its description. "
        "If a question could apply to more than one tool, search both."
    )

    # ── Node 0: Query Rewrite ──────────────────────────────────────────

    def rewrite_query(self, state: RAGState) -> RAGState:
        response = self.llm.invoke(
            f"{QUERY_REWRITE_PROMPT}\n\nUser question: {state.question}"
        )
        rewritten = response.content if hasattr(response, "content") else str(response)
        logger.info("Query rewrite: %r → %r", state.question, rewritten.strip())
        return RAGState(
            question=state.question,
            rewritten_query=rewritten.strip(),
        )

    # ── Node 1: Retrieve ────────────────────────────────────────────────

    def retrieve_docs(self, state: RAGState) -> RAGState:
        tools = self.build_rag_tools()
        self.rag_agent = create_react_agent(self.llm, tools, prompt=self.AGENT_PROMPT)

        query = state.rewritten_query or state.question
        if state.retry_count > 0 and state.reflection_critique:
            query = (
                f"{query}\n\n"
                f"(Previous answer was inadequate: {state.reflection_critique}. "
                f"Search more broadly or with different terms.)"
            )

        result = self.rag_agent.invoke(
            {"messages": [HumanMessage(content=query)]}
        )

        retrieved_docs = [
            Document(
                page_content=str(msg.content).strip(),
                metadata={"source": getattr(msg, "name", "tool")},
            )
            for msg in result["messages"]
            if isinstance(msg, ToolMessage) and str(msg.content).strip()
        ]

        return RAGState(
            question=state.question,
            retrieved_context=retrieved_docs,
            retry_count=state.retry_count,
            max_retries=state.max_retries,
            reflection_critique=state.reflection_critique,
        )

    # ── Node 2: Generate answer ─────────────────────────────────────────

    def generate_answer(self, state: RAGState) -> RAGState:
        context = "\n\n".join(
            f"Chunk {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(state.retrieved_context)
        )

        retry_hint = ""
        if state.retry_count > 0 and state.reflection_critique:
            retry_hint = (
                f"\n\nIMPORTANT — A previous answer was rejected by the quality reviewer:\n"
                f'"{state.reflection_critique}"\n'
                f"Address these gaps in your new answer.\n"
            )

        prompt = (
            f"Answer the question using ONLY the context below.\n"
            f"Provide a detailed, self-contained answer. Describe each step fully — "
            f"do NOT reference section numbers or say 'see Section X'. "
            f"The reader has no access to the source document.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{state.question}\n"
            f"{retry_hint}\n"
            f"Answer:\n"
        )

        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return RAGState(
            question=state.question,
            retrieved_context=state.retrieved_context,
            answer=answer,
            retry_count=state.retry_count,
            max_retries=state.max_retries,
        )

    # ── Node 3: Reflect / self-critique ─────────────────────────────────

    def reflect_on_answer(self, state: RAGState) -> RAGState:
        context = "\n\n".join(
            f"Chunk {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(state.retrieved_context)
        )

        prompt = REFLECTION_PROMPT.format(
            question=state.question,
            context=context,
            answer=state.answer,
        )

        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        try:
            result = json.loads(content)
            score = int(result.get("overall_score", 3))
            critique = result.get("critique", "")
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Reflection returned non-JSON; treating as passing score. Raw: %s", content)
            score = 3
            critique = content

        should_retry = score < 3 and state.retry_count < state.max_retries
        logger.info(
            "Reflection — score=%d, retry_count=%d/%d, will_retry=%s",
            score, state.retry_count, state.max_retries, should_retry,
        )

        return RAGState(
            question=state.question,
            retrieved_context=state.retrieved_context,
            answer=state.answer,
            retry_count=state.retry_count + (1 if should_retry else 0),
            max_retries=state.max_retries,
            reflection_score=score,
            reflection_critique=critique,
        )