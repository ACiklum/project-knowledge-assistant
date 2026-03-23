# Topic-Based Tool Routing Plan

**Goal:** Route user questions to the right knowledge source:
- **Deal** (registration or resubmission) → search only in deal docs
- **Endpoints** (API endpoints) → search only in endpoints docs

**Aligned with:** `PROJECT_PLAN.md` — Step 4 (Tools), Step 5 (LangGraph), Step 1 (Data Prep).

---

## 1. Overview of Changes

| Area | What changes |
|------|----------------|
| **Data / docs** | Assign a `topic` to each document (by filename) so chunks can be filtered. |
| **Document ingestion** | Add topic metadata when loading from `data/`. |
| **Vector store** | Support topic-filtered retrieval (one store, metadata filter). |
| **Tools** | New file `agent/03_tools.py`: `deal_search`, `endpoints_search`. |
| **Agent graph** | Router node + tool execution; state includes `selected_tool` and `context`. |
| **State** | Extend `RAGState` (or agent state) for tool choice and context. |

---

## 2. Topic Mapping (Data Conventions)

| File pattern / path | Topic |
|---------------------|--------|
| `*deal*`, `*resubmission*`, `*registration*` (e.g. `deal_registration_submission_sop.txt`, `deal_ resubmission_workflow.txt`) | `deal` |
| `*endpoints*` (e.g. `fannie_endpoints.txt`) | `endpoints` |

All other files can use a default (e.g. `general`) or be excluded from topic tools.

---

## 3. File-by-File Change Plan

### 3.1 Document ingestion — `src/document_Ingestion/document_processor.py`

**Changes:**
- Add a **topic resolution** helper: given a file path (or document source), return `"deal"` | `"endpoints"` | `"general"`.
- When loading documents (in `load_documents` and `load_from_directory`), set **metadata** on each `Document`: `metadata["topic"] = <resolved topic>`.
- Keep existing behavior (load/split) unchanged; only add metadata.

**Deliverable:** Documents (and chunks) carry `metadata["topic"]` so the vector store can filter by it.

---

### 3.2 Vector store — `src/vectorstore/vector_store.py`

**Changes:**
- Keep **one** FAISS index built from all chunks (with `topic` in metadata).
- Add a method to get a **topic-filtered retriever**: e.g. `get_retriever(topic: Optional[str] = None)`.
  - If `topic` is `None`, return the existing global retriever.
  - If `topic` is `"deal"` or `"endpoints"`, return a retriever that filters by `metadata["topic"] == topic` (FAISS supports `as_retriever(filter={"topic": "deal"})`).
- Optionally: `retrieve_vector(query, k, topic=None)` that uses the filtered retriever when `topic` is set.

**Deliverable:** Two “logical” retrievers (deal and endpoints) backed by the same index with metadata filter.

---

### 3.3 Tools — `agent/03_tools.py` (new file)

**Changes:**
- Create two LangChain `@tool` functions:
  - **`deal_search(query: str) -> str`**  
    - Description: e.g. “Search the deal registration and resubmission knowledge base. Use for questions about deal registration, resubmission workflow, waivers, loan options, submission steps.”
  - **`endpoints_search(query: str) -> str`**  
    - Description: e.g. “Search the API endpoints knowledge base. Use for questions about Fannie Mae API endpoints, URLs, POST/GET/PATCH, deal management APIs.”
- Each tool:
  - Gets the appropriate retriever from the vector store (deal or endpoints).
  - Runs `retriever.invoke(query)`.
  - Returns a string (e.g. concatenated `page_content` of retrieved docs) for the reasoner to use.
- Tools need access to the **vector store instance** (or a retriever factory). This can be:
  - Injected at init (e.g. when building the agent), or
  - Loaded from a shared module/singleton that creates the vector store once.

**Deliverable:** `agent/03_tools.py` with `deal_search` and `endpoints_search` callable by the agent.

---

### 3.4 Agent state — `src/state/rag_state.py` (or agent-specific state)

**Changes:**
- Extend state to support the graph:
  - `question: str`
  - `selected_tool: Optional[str]` — e.g. `"deal_search"` | `"endpoints_search"` (set by router).
  - `context: str` — result of the tool call (or RAG retrieval).
  - `answer: str`
  - Optional: `retry_count`, `reflection` for reflection loop later.
- If the project uses a separate agent state (e.g. in `agent/04_agent_graph.py`), define a `TypedDict` or Pydantic model there and keep `RAGState` for a simpler RAG-only flow if needed.

**Deliverable:** State schema that includes tool choice and context for the reasoner.

---

### 3.5 Agent graph — `agent/04_agent_graph.py`

**Changes:**
- **Router node:**  
  - Input: `question`.  
  - Use an LLM with a short prompt to classify: “Is this question about deal registration/resubmission, or about API endpoints?”  
  - Output: set `selected_tool` to `"deal_search"` or `"endpoints_search"`.
- **Tool-executor node (or “retriever” node):**  
  - Input: `question`, `selected_tool`.  
  - Call the corresponding tool (`deal_search(question)` or `endpoints_search(question)`).  
  - Set `context` to the tool’s string result.
- **Reasoner node:**  
  - Input: `question`, `context`.  
  - LLM generates the answer using `context` and sets `answer`.
- **Graph structure:**  
  - Entry → **router** → **tool_executor** → **reasoner** → end (reflection can be added later).
- **Integration:**  
  - Import tools from `03_tools.py` and the vector store (or retriever factory) so the tools can run.
- **`KnowledgeAgent.process_query`:**  
  - Invoke the compiled graph with `{"question": query}` and return `answer` (and optional metadata, e.g. `selected_tool`).

**Deliverable:** LangGraph that routes to the correct tool and returns an answer based on that tool’s output.

---

### 3.6 Data prep script (optional but recommended)

**File:** e.g. `agent/01_data_prep.py` or `scripts/build_vectorstore.py`

**Changes:**
- Load documents from `data/` using `DocumentProcessor` (with topic metadata).
- Split with `DocumentProcessor.split_documents`.
- Build the single FAISS index with `VectorStore.create_vector_store(documents)`.
- Persist the vector store (e.g. `vectorstore.save_local("vectorstore/")`).
- Ensure the vector store is loadable with the same metadata so filtered retrieval works.

**Deliverable:** One-time (or on-demand) script to rebuild the index with topic metadata; agent loads this index at startup.

---

## 4. Implementation Order

1. **Document processor** — add topic metadata when loading.
2. **Vector store** — add topic-filtered retriever (and ensure FAISS persistence includes metadata).
3. **State** — extend with `selected_tool`, `context`.
4. **Tools** — create `agent/03_tools.py` with `deal_search` and `endpoints_search`.
5. **Agent graph** — implement router, tool executor, reasoner, and wire `process_query`.
6. **Data prep** — script to build and save the index; ensure agent loads it.

---

## 5. Testing Suggestions

- **Deal question:** e.g. “What are the steps for resubmission?” → expect `deal_search` and answer from resubmission doc.
- **Endpoints question:** e.g. “Which endpoint do I use to create a deal?” → expect `endpoints_search` and answer from endpoints doc.
- **Edge case:** “What is the deal submission endpoint?” → could be routed to either; define in router prompt whether to prefer endpoints or deal.

---

## 6. Summary Checklist

- [ ] `document_processor.py`: add topic from path, set `metadata["topic"]` on docs.
- [ ] `vector_store.py`: add `get_retriever(topic=None)` and use metadata filter when topic is set.
- [ ] `agent/03_tools.py`: implement `deal_search` and `endpoints_search` using topic retrievers.
- [ ] `rag_state.py` or agent state: add `selected_tool`, `context`.
- [ ] `agent/04_agent_graph.py`: router → tool_executor → reasoner; `process_query` invokes graph.
- [ ] Data prep script: load from `data/` with topic metadata, build and save FAISS index.
- [ ] API/UI: no change (they already call `process_query`).
