# 🤖 AI-Agentic System — Final Assignment Project Plan

> **Stack:** Python · LangChain · LangGraph · OpenAI API · FAISS / ChromaDB · FastAPI · Streamlit
> **Goal:** Build a self-reflective, tool-using AI agent that can reason, retrieve, and act, accessible via API and UI.

---

## 📋 Table of Contents

1. [Project Idea & Problem Statement](#1-project-idea--problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Folder Structure](#4-folder-structure)
5. [Step 1 — Data Preparation & Contextualization](#5-step-1--data-preparation--contextualization)
6. [Step 2 — RAG Pipeline Design](#6-step-2--rag-pipeline-design)
7. [Step 3 — Reasoning & Self-Reflection](#7-step-3--reasoning--self-reflection)
8. [Step 4 — Tool-Calling Mechanisms](#8-step-4--tool-calling-mechanisms)
9. [Step 5 — LangGraph Agent Orchestration](#9-step-5--langgraph-agent-orchestration)
10. [Step 6 — API Endpoints (FastAPI)](#10-step-6--api-endpoints-fastapi)
11. [Step 7 — UI Integration (Streamlit)](#11-step-7--ui-integration-streamlit)
12. [Step 8 — Evaluation](#12-step-8--evaluation)
13. [Step 9 — Demo & Delivery](#13-step-9--demo--delivery)
14. [Implementation Checklist](#14-implementation-checklist)

---

## 1. Project Idea & Problem Statement

### 🎯 Project Knowledge Assistant

**Problem:**  
Client team members (developers, QAs, PMs) frequently have questions about project workflows, processes, onboarding steps, and internal conventions. Finding answers requires digging through scattered docs, Confluence pages, or asking senior colleagues — which is slow and inefficient.

**Solution:**  
An AI-Agentic Knowledge Assistant that ingests project-specific documents (flow diagrams, wikis, process docs, onboarding guides) and allows users to ask natural language questions and get accurate, cited answers instantly. The assistant will be exposed via a user-friendly web UI.

**Users:** Client project team members — developers, QAs, BAs, PMs  
**Data Ingested:** Project flow docs, onboarding wikis, process guides, FAQs, Confluence exports, Markdown files, PDFs

**Value over plain LLM or search:**
- Grounded answers from *your* project docs (no hallucination about your specific processes)
- Self-reflects if the answer seems incomplete and retries with better retrieval
- Can call tools to fetch live info or escalate when docs don't have the answer
- Cites the source document/section for every answer (builds user trust)
- Easily accessible through a dedicated REST API and Streamlit UI

---

## 2. System Architecture

```text
       User (Browser / App)
               │
               ▼
     ┌───────────────────┐
     │        UI         │
     │   (Streamlit)     │
     └─────────┬─────────┘
               │ HTTP / REST
               ▼
     ┌───────────────────┐
     │   API Endpoint    │
     │     (FastAPI)     │
     └─────────┬─────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Agent Graph               │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────┐  │
│  │  Router  │──▶│ RAG Node │──▶│ Reasoning Node  │  │
│  └──────────┘   └──────────┘   └────────┬────────┘  │
│        │                                │            │
│        │                     ┌──────────▼────────┐   │
│        │                     │  Tool-Call Node   │   │
│        │                     │ (search, calc,    │   │
│        │                     │  file read, etc)  │   │
│        │                     └──────────┬────────┘   │
│        │                                │            │
│        │                     ┌──────────▼────────┐   │
│        └────────────────────▶│ Reflection Node   │   │
│                               │ (self-critique)   │   │
│                               └──────────┬────────┘  │
└──────────────────────────────────────────┼───────────┘
                                           │
                                    Final Answer
```

**Key flows:**
- **RAG path** → Retrieve → Reason → Answer
- **Tool path** → Route to tool → Execute → Reflect → Answer
- **Reflection loop** → If answer quality is low, re-route and retry (max N times)
- **API & UI Layer** → Accept queries securely from HTTP endpoints and visualize in a chat UI

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Agent Framework | LangGraph |
| LLM abstraction | LangChain |
| LLM Provider | OpenAI (`gpt-4o` or `gpt-4o-mini`) |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Vector Store | FAISS (local, simple) or ChromaDB |
| Document Loaders | LangChain `PyPDFLoader`, `TextLoader`, `CSVLoader` |
| Text Splitter | `RecursiveCharacterTextSplitter` |
| Tool calling | LangChain `@tool` decorator |
| API Framework | FastAPI + Uvicorn |
| User Interface | Streamlit |
| Evaluation | Custom scorer + `langchain.evaluation` |
| Environment | `.env` + `python-dotenv` |

---

## 4. Folder Structure

```text
ai-agentic-system/
│
├── data/                         # Raw documents, PDFs, CSVs
│   └── your_documents.pdf
│
├── vectorstore/                  # Persisted FAISS / Chroma index
│
├── agent/
│   ├── 01_data_prep.py           # Step 1: load, chunk, embed, store
│   ├── 02_rag_pipeline.py        # Step 2: retriever setup & query
│   ├── 03_tools.py               # Step 4: custom @tool definitions
│   ├── 04_agent_graph.py         # Step 5: LangGraph agent graph
│   ├── 05_reflection.py          # Step 3: self-reflection node
│   └── 08_evaluation.py          # Step 8: evaluation harness
│
├── api/
│   └── 06_api.py                 # Step 6: FastAPI endpoints
│
├── ui/
│   └── 07_ui.py                  # Step 7: Streamlit frontend
│
├── notebooks/
│   └── demo.ipynb                # Interactive walkthrough / demo
│
├── PROJECT_PLAN.md               # This file
├── requirements.txt
└── .env                          # OPENAI_API_KEY etc. (do not commit!)
```

---

## 5. Step 1 — Data Preparation & Contextualization

**Goal:** Load your raw data, split it into meaningful chunks, and embed it into a vector store.

### What to do
1. Collect relevant documents (PDFs, markdown files, CSVs, web pages).
2. Load documents using LangChain loaders.
3. Split into overlapping chunks for better retrieval coverage.
4. Generate embeddings with OpenAI and store in FAISS/ChromaDB.
5. Persist the vector store to disk.

### Key code pattern
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load
loader = PyPDFLoader("data/your_doc.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed & Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore/")
```

### Deliverable
- `agent/01_data_prep.py` that produces a saved vector store

---

## 6. Step 2 — RAG Pipeline Design

**Goal:** Given a user query, retrieve the most relevant chunks and feed them to the LLM.

### What to do
1. Load the persisted vector store.
2. Create a retriever (similarity search, top-k).
3. Build a RAG chain: `retriever | prompt | llm | output_parser`.
4. Test with sample queries.

### Key code pattern
```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("vectorstore/", embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

result = rag_chain.invoke("What is the refund policy?")
print(result)
```

### Deliverable
- `agent/02_rag_pipeline.py` with a working RAG chain

---

## 7. Step 3 — Reasoning & Self-Reflection

**Goal:** After producing an answer, the agent critiques its own output and decides whether to retry.

### What to do
1. Define a **Reflection Node** in LangGraph.
2. The node uses the LLM to evaluate: *Is the answer relevant, complete, and grounded in the retrieved context?*
3. If quality score < threshold → loop back to retrieval or tool.
4. If quality is sufficient → route to final output.

### Reflection prompt template
```
You are a quality reviewer. Given:
- USER QUESTION: {question}
- RETRIEVED CONTEXT: {context}
- AGENT ANSWER: {answer}

Rate the answer on a scale 1–5 on:
1. Relevance (does it answer the question?)
2. Groundedness (is it supported by the context?)
3. Completeness (is anything important missing?)

Provide a score and a short critique. If score < 3, suggest how to improve.
Output JSON: {"score": int, "critique": str, "retry": bool}
```

### Deliverable
- `agent/05_reflection.py` with a `reflection_node(state)` function

---

## 8. Step 4 — Tool-Calling Mechanisms

**Goal:** Give the agent the ability to *act* — not just retrieve and respond.

### Tools to implement (pick 2–4)

| Tool | Description |
|---|---|
| `rag_search` | Search the vector store for relevant documents |
| `web_search` | Search the web (via Tavily or SerpAPI) |
| `calculator` | Evaluate math expressions |
| `read_file` | Read a local file's contents |
| `send_email` / `create_ticket` | Simulate taking a business action |
| `summarize_document` | Summarize a given document text |

### Key code pattern
```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluates a safe math expression and returns the result."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def rag_search(query: str) -> str:
    """Searches the internal knowledge base for information about the query."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)
```

### Deliverable
- `agent/03_tools.py` with all tools defined using `@tool`

---

## 9. Step 5 — LangGraph Agent Orchestration

**Goal:** Wire all components into a stateful, cyclical graph using LangGraph.

### Graph nodes

| Node | Role |
|---|---|
| `router` | Decides: RAG path, tool path, or direct answer |
| `retriever` | Runs RAG search and adds context to state |
| `reasoner` | LLM reasoning step using context + history |
| `tool_executor` | Runs selected tool, captures output |
| `reflector` | Scores answer, decides retry or finish |
| `final_output` | Formats and returns the final answer |

### Key code pattern
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    context: str
    tool_output: str
    answer: str
    reflection: dict
    retry_count: int

graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("retriever", retriever_node)
graph.add_node("reasoner", reasoner_node)
graph.add_node("tool_executor", tool_executor_node)
graph.add_node("reflector", reflection_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", route_decision)
graph.add_edge("retriever", "reasoner")
graph.add_edge("tool_executor", "reasoner")
graph.add_conditional_edges("reflector", should_retry)
graph.add_edge("reasoner", "reflector")

app = graph.compile()
result = app.invoke({"question": "What is our leave policy?", "retry_count": 0})
```

### Deliverable
- `agent/04_agent_graph.py` — the full LangGraph agent

---

## 10. Step 6 — API Endpoints (FastAPI)

**Goal:** Expose the LangGraph agent as a REST API so it can be safely consumed by web interfaces or external services.

### What to do
1. Create a FastAPI application.
2. Define a POST endpoint `/ask` that accepts a query.
3. Integrate the compiled LangGraph application into the endpoint logic.
4. Return the agent's finalized answer (and optionally references/metadata) as a JSON response.

### Key code pattern
```python
from fastapi import FastAPI
from pydantic import BaseModel
# from your_app import agent_app

app = FastAPI(title="Knowledge Assistant API")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    result = agent_app.invoke({"question": request.query, "retry_count": 0})
    return QueryResponse(answer=result.get("answer", "No answer found"))
```

### Deliverable
- `api/06_api.py` — A FastAPI server ready to run via `uvicorn api.06_api:app --reload`.

---

## 11. Step 7 — UI Integration (Streamlit)

**Goal:** Provide an interactive, chat-like frontend for users to communicate with the knowledge assistant visually.

### What to do
1. Set up a Streamlit app.
2. Initialize and maintain a chat history in the session state.
3. Prompt the user for input and display answers.
4. Connect the input prompt to the FastAPI backend (via `requests`) to retrieve real agent responses.

### Key code pattern
```python
import streamlit as st
import requests

st.title("🤖 Project Knowledge Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the project..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        # Make API call to FastAPI backend
        try:
            response = requests.post("http://localhost:8000/ask", json={"query": prompt})
            response.raise_for_status()
            answer = response.json().get("answer", "I could not find an answer.")
        except Exception as e:
            answer = f"Error connecting to API: {e}"
            
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
```

### Deliverable
- `ui/07_ui.py` — A Streamlit application runnable via `streamlit run ui/07_ui.py`.

---

## 12. Step 8 — Evaluation

**Goal:** Measure how well your agent performs objectively.

### Metrics to measure

| Metric | Description | How |
|---|---|---|
| Answer Relevance | Is the answer topically related to the question? | LLM-as-judge prompt |
| Groundedness | Is the answer supported by retrieved context? | LLM-as-judge prompt |
| Faithfulness | No hallucinated facts? | Compare answer vs docs |
| Task Success Rate | % of test questions correctly answered | Test set with ground truth |
| Reflection Impact | Did reflection improve failed answers? | Retry outcome tracking |

### Test set
Create 10–20 sample Q&A pairs for your domain:
```python
test_cases = [
    {"question": "What is the annual leave policy?", "expected": "25 days per year..."},
    {"question": "How do I submit an expense report?", "expected": "Use the HR portal..."},
]
```

### Evaluation loop
```python
for case in test_cases:
    result = app.invoke({"question": case["question"], "retry_count": 0})
    score = evaluate_answer(
        question=case["question"],
        answer=result["answer"],
        expected=case["expected"]
    )
    print(f"Q: {case['question']}\nScore: {score}\n")
```

### Deliverable
- `agent/08_evaluation.py` with evaluation harness and printed results table

---

## 13. Step 9 — Demo & Delivery

### Demo script outline
1. **Intro** (1 min) — describe the problem and why an agent is better than plain LLM
2. **Architecture walkthrough** (2 min) — show the LangGraph graph diagram and the API/UI flow
3. **Live demo via Streamlit UI** (3–4 min) — run 3–4 queries showing:
   - RAG retrieval in action
   - Tool being called
   - Reflection kicking in and improving an answer
4. **Evaluation results** (1–2 min) — show your metrics table
5. **Lessons learned** (1 min) — what worked, what didn't

### Deliverables checklist
- [ ] `requirements.txt` with all dependencies
- [ ] `agent/`, `api/`, and `ui/` with all 8 implementation files
- [ ] `notebooks/demo.ipynb` (optional, as we now have a Streamlit UI)
- [ ] Evaluation results (printed table or chart)
- [ ] Demo video (screen recording, 8–10 min)

---

## 14. Implementation Checklist

### Setup
- [ ] Create project folder and `requirements.txt`
- [ ] Set up `.env` with `OPENAI_API_KEY`
- [ ] Install dependencies: `pip install -r requirements.txt`

### Step 1 — Data Prep
- [ ] Collect and place documents in `data/`
- [ ] Implement `01_data_prep.py` (load → chunk → embed → store)
- [ ] Verify vector store is saved and loadable

### Step 2 — RAG Pipeline
- [ ] Implement `02_rag_pipeline.py`
- [ ] Test retrieval with 3 sample queries

### Step 3 — Tools
- [ ] Implement at least 2 tools in `03_tools.py`
- [ ] Test each tool independently

### Step 4 — Reflection
- [ ] Implement `05_reflection.py`
- [ ] Test it produces score + critique + retry decision

### Step 5 — LangGraph Agent
- [ ] Implement `04_agent_graph.py`
- [ ] Run end-to-end with 3 test queries

### Step 6 — API Endpoints
- [ ] Implement `06_api.py` with FastAPI
- [ ] Test endpoint using Postman / curl / Swagger UI

### Step 7 — UI Integration
- [ ] Implement `07_ui.py` with Streamlit
- [ ] Connect Streamlit to FastAPI and verify end-to-end chat flow

### Step 8 — Evaluation
- [ ] Create 10+ test cases
- [ ] Implement `08_evaluation.py`
- [ ] Run evaluation and record results

### Step 9 — Demo
- [ ] Record demo video (walking through the Streamlit UI and Architecture)

---

## 📦 requirements.txt (starter)

```
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langgraph>=0.2.0
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
openai>=1.0.0
pypdf>=4.0.0
tiktoken>=0.7.0
fastapi>=0.103.0
uvicorn>=0.23.2
streamlit>=1.26.0
requests>=2.31.0
```

---

> 💡 **Next step:** Fill in Section 1 with your real problem statement, then we'll start building step by step.
