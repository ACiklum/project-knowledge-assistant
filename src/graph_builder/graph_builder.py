import logging

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes

logger = logging.getLogger(__name__)


def _should_retry(state: RAGState) -> str:
    """Conditional edge after the reflector node.

    Returns the next node name:
      - "retriever" if the answer quality is too low and retries remain
      - "__end__"   otherwise (good enough or out of retries)
    """
    if state.reflection_score >= 3:
        return END
    if state.retry_count >= state.max_retries:
        logger.warning(
            "Max retries (%d) reached with score %d — returning best-effort answer.",
            state.max_retries, state.reflection_score,
        )
        return END
    return "retriever"


class GraphBuilder:
    """Builds and manages the LangGraph for the RAG pipeline.
    """

    def __init__(self, retriever, llm):
        self.graph = None
        self.nodes = RAGNodes(retriever, llm)

    def build_graph(self):
        builder = StateGraph(RAGState)

        builder.add_node("query_rewriter", self.nodes.rewrite_query)
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        builder.add_node("reflector", self.nodes.reflect_on_answer)

        builder.set_entry_point("query_rewriter")

        builder.add_edge("query_rewriter", "retriever")
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", "reflector")
        builder.add_conditional_edges("reflector", _should_retry)

        self.graph = builder.compile()
        logger.info("Graph compiled: %s", self.graph)
        return self.graph

    def run_graph(self, question: str) -> dict:
        if self.graph is None:
            self.build_graph()
        initial_state = RAGState(question=question)
        result = self.graph.invoke(initial_state)
        return result