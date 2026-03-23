# Project Knowledge Assistant

A RAG-based Q&A system that reads project documents and answers questions about Fannie Mae deal registration workflows and API endpoints. It uses LangGraph to route questions to the right document set, rewrites queries for better retrieval, and has a self-reflection loop that checks answer quality before returning results.

## What it does

- Loads `.txt` and `.pdf` files from the `data/` folder
- Splits them into chunks using section-aware separators (`---`, `##`) and stores them in a FAISS vector index
- Tags each chunk with a topic (`registration`, `resubmission`, or `endpoints`) based on the filename
- Rewrites the user's query using domain-specific vocabulary to improve retrieval accuracy
- A ReAct Agent picks the right tool(s) — `registration_search`, `resubmission_search`, or `endpoints_search` — and retrieves relevant chunks
- The LLM generates a detailed, self-contained answer from those chunks
- A reflector node scores the answer (1-5) on relevance, groundedness, and completeness. If the score is below 3, it retries up to 2 times with the critique fed back in

## Tech used

- Python 3.12
- LangChain + LangGraph for the agent pipeline
- OpenAI gpt-4o for LLM and embeddings
- FAISS for vector storage
- FastAPI for the REST API
- Streamlit for the UI
- Pydantic for state and request/response models

## Folder structure

```
data/                          -- source documents (deal workflows, API endpoint reference)
src/config/config.py           -- API keys, model name, chunk params
src/document_Ingestion/        -- loads and chunks documents, tags topic metadata
src/vectorstore/               -- FAISS index, topic-filtered retrieval
src/state/rag_state.py         -- state model (question, rewritten_query, answer, reflection fields)
src/nodes/nodes.py             -- graph nodes: query rewrite, retrieve, generate, reflect
src/graph_builder/             -- wires the LangGraph with conditional retry edges
api/api.py                     -- FastAPI server
ui/streamlit_app.py            -- Streamlit UI (standalone, no API needed)
architecture.mmd               -- Mermaid diagram of the system
```

## Setup

1. Clone the repo and cd into it

2. Create a virtual environment and activate it:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Mac/Linux
   ```

3. Install dependencies:
   ```
   pip install -e .
   ```

4. Create a `.env` file in the root with your OpenAI key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

## How to run

**Quickest way** -- standalone Streamlit (no API server):

```
.venv\Scripts\activate

streamlit run ui/streamlit_app.py
```
Open http://localhost:8501.

**With the API** -- start FastAPI first, then query via curl or Swagger:
```
.venv\Scripts\activate
uvicorn api.api:app --reload
```
API docs at http://localhost:8000/docs.

**API only** -- test with curl:
```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"What are the steps to resubmit a deal?\"}"
```

## How the pipeline works

1. Documents are loaded from `data/`, split using section-aware separators (`---`, `##`), and tagged with topic metadata (`registration`, `resubmission`, `endpoints`)
2. Chunks are embedded and stored in FAISS
3. User asks a question
4. **Query Rewriter** expands vague terms into domain vocabulary (e.g. "loan endpoints" becomes "loan option API endpoints")
5. A **ReAct Agent** picks `registration_search`, `resubmission_search`, or `endpoints_search` (or multiple) based on the rewritten query
6. Retrieved chunks (top 5 per tool) are passed to the LLM to generate a detailed, self-contained answer
7. The **Reflector** node scores the answer on relevance, groundedness, completeness (1-5 scale)
8. If score < 3, it loops back to step 5 with the critique (max 2 retries)
9. Final answer + sources are returned

## Config

Edit `src/config/config.py` to change:

- `LLM_MODEL` -- default `openai:gpt-4o`
- `CHUNK_SIZE` -- default 1000
- `CHUNK_OVERLAP` -- default 200
- `RETRIEVER_K` -- default 10 (number of chunks retrieved)
- `DATA_FOLDER` -- default `data`

## Demo questions

**Deal registration workflow** (uses registration_search):
- "What are the steps to register a new deal in Dealpath?"
- "What happens after a deal is submitted to Fannie Mae?"

**Resubmission workflow** (uses resubmission_search):
- "What are the steps of deal resubmission?"
- "What is the resubmission workflow and when can I resubmit a deal?"

**API endpoints** (uses endpoints_search):
- "Which API endpoints do I need to create a deal and add participants?"
- "What are waiver endpoints and how do I submit a waiver?"
- "I want loan endpoints and when we need to create loan?"

**Cross-topic questions** (uses multiple tools):
- "How do I register a deal and what API endpoints should I call for registration?"
- "What are the steps to submit a deal and which endpoint updates the deal status?"
