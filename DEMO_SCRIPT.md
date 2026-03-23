# Demo Video Script

Target length: 8-10 minutes.

---

## Part 1 -- Introduction (1-1.5 min)

Open README.md in VS Code or show a title slide.

**Say:**

"This is the Project Knowledge Assistant. The problem it solves is simple: team members waste time digging through scattered documents to find answers about deal workflows and API endpoints. Instead of searching manually, they can ask this system a question in plain English and get a grounded answer pulled directly from the project docs.

It's built with Python, LangChain, LangGraph, OpenAI gpt-4o, FAISS for vector search, FastAPI for the backend, and Streamlit for the UI.

What makes it more than a basic RAG system is three things: a query rewriter that expands vague terms into domain vocabulary for better retrieval, topic-based tool routing with three separate tools -- registration, resubmission, and endpoints -- so the agent searches the right document set, and a self-reflection loop where the agent scores its own answer and retries if it's not good enough."

---

## Part 2 -- Architecture Walkthrough (2-2.5 min)

Open the architecture diagram (architecture.png or architecture.mmd rendered in mermaid.live / VS Code preview).

Walk through top to bottom:

**1. User Interface Layer**
"The user types a question in the Streamlit UI."

**2. API Layer**
"The question goes to a FastAPI endpoint at /ask. In standalone mode, the Streamlit app calls the graph directly without the API."

**3. LangGraph Agent Pipeline (spend the most time here)**
"This is the core of the system. There are four nodes:

- Query Rewriter: Before any retrieval happens, the LLM rewrites the user's question using domain-specific vocabulary. For example, if someone asks about 'loan endpoints', it expands that to 'loan option API endpoints' so it matches the terminology in our documents.

- Retriever: A ReAct Agent looks at the rewritten query and decides which tool to call -- registration_search for deal registration questions, resubmission_search for resubmission workflow questions, endpoints_search for API reference questions, or multiple tools if the question spans topics.

- Responder: The LLM takes the retrieved chunks and generates a detailed, self-contained answer. It's instructed not to reference section numbers since the reader has no access to the source document.

- Reflector: The LLM acts as a quality reviewer. It scores the answer from 1 to 5 on three criteria -- relevance, groundedness, and completeness. If the score is 3 or above, the answer goes to the user. If it's below 3, the system loops back to the retriever with the critique appended, so it can fetch better context and try again. This retry can happen up to 2 times."

**4. Data Layer**
"Documents from the data folder are loaded and split using section-aware separators -- the splitter respects '---' and '##' markdown boundaries so complete sections stay together. Chunks are tagged with a topic based on the filename -- registration, resubmission, or endpoints. They all go into one FAISS index, but retrieval is filtered by topic metadata so each tool only searches its own docs."

---

## Part 3 -- Code Walkthrough (2 min)

Open each file in VS Code. Spend about 20 seconds per file. Don't read code line by line -- just point at the key parts.

**File: src/document_Ingestion/document_processor.py**
Point at: `get_topic_from_path()` and the custom `separators` in the splitter.
Say: "This is where documents get tagged. Filenames with 'resubmission' get tagged as resubmission, 'deal' or 'registration' as registration, and 'endpoints' as endpoints. The splitter uses section-aware separators -- it splits on '---' and '##' boundaries first so complete sections stay intact."

**File: src/vectorstore/vector_store.py**
Point at: `get_retriever(topic=...)`.
Say: "One FAISS index for everything. When you pass a topic, it filters by metadata so you only search relevant docs."

**File: src/nodes/nodes.py**
Point at: the four methods -- `rewrite_query`, `retrieve_docs`, `generate_answer`, `reflect_on_answer`.
Say: "These are the four graph nodes. The query rewriter expands vague terms into domain vocabulary. The retriever uses a ReAct Agent to pick from three tools -- registration_search, resubmission_search, or endpoints_search. The responder generates a detailed answer. The reflector scores it on relevance, groundedness, and completeness."
Point at: the three tool descriptions and `build_rag_tools()`.
Say: "Each tool has a clear description so the agent knows when to use it. Registration for deal creation and submission, resubmission for NPulse workflows, endpoints for API URLs."

**File: src/graph_builder/graph_builder.py**
Point at: `build_graph()` method and `_should_retry`.
Say: "This is the graph wiring. Query rewriter to retriever to responder to reflector, then a conditional edge -- if score is 3 or above, we're done. If below 3 and retries remain, loop back to the retriever."

**File: src/state/rag_state.py**
Point at: the fields.
Say: "The state carries the question, the rewritten query, retrieved context, answer, plus the reflection fields -- retry count, max retries, score, and critique."

---

## Part 4 -- Live Demo (3-4 min)

Open the Streamlit UI in the browser. Keep the terminal visible alongside it so log output is visible (reflection scores, tool calls).

### Query 1: Registration question (single tool)

Type: "What are the steps to register a new deal in Dealpath?"

**While it loads, say:**
"This is a deal registration question, so the agent should pick the registration_search tool."

**After the answer appears:**
- Read the first few lines -- it should mention logging in, creating a deal, importing files, base fields, participants, property data, and setting Submit Registration to YES
- Click the Source Documents expander to show where the answer came from
- Point at the terminal log: "You can see the reflection score here -- it passed on the first attempt"

### Query 2: Resubmission question (single tool)

Type: "What are the steps of deal resubmission?"

**While it loads, say:**
"This is a resubmission question. Previously this would have pulled from the wrong document, but now we have a dedicated resubmission_search tool."

**After the answer appears:**
- Read through the steps -- it should cover: retrieve deal status, enter resubmission details, move to draft state, add waivers, add loan options, check API requirements, final submit
- Point out: "Notice how it describes each step in detail instead of just referencing section numbers. That's because the generation prompt instructs the LLM to write self-contained answers."

### Query 3: Endpoints question (single tool)

Type: "I want loan endpoints and when we need to create loan?"

**While it loads, say:**
"This is a vague question -- the user said 'loan' but the document uses 'loan option'. The query rewriter should expand this to match the document vocabulary."

**After the answer appears:**
- Show that it returned all three loan option endpoints: POST loanoptions, GET validationmessages, POST products
- Say: "Without the query rewriter, this would have returned generic results because 'loan' doesn't match 'loan option' well in embedding space."

### Query 4: Cross-topic question (multiple tools)

Type: "How do I register a deal and what API endpoints should I call for registration?"

**While it loads, say:**
"This one spans both registration and endpoints, so the agent should call multiple tools."

**After the answer appears:**
- Show the answer combines workflow steps with actual endpoint URLs like POST /deals and PATCH /deals/{{dealId}}
- Point at the terminal: "The agent called both registration_search and endpoints_search because the question spans both topics."

---

## Part 5 -- Wrap-up (1 min)

Switch back to VS Code or a closing slide.

**Say:**

"To summarize what we covered:
- The system ingests project documents, splits them using section-aware boundaries, and tags them by topic
- A query rewriter expands vague user terms into domain vocabulary for better retrieval
- A LangGraph ReAct agent routes questions to the right knowledge source -- registration, resubmission, or endpoints -- using topic-filtered tools
- The LLM generates detailed, self-contained answers grounded in the retrieved context
- A reflection loop scores the answer and retries if quality is too low

For next steps, I'd add vector store persistence so the index doesn't rebuild on every startup, an evaluation harness with ground-truth test cases, and potentially more document sources for broader coverage.

Thanks for watching."

---

## Recording Tips

- Use OBS Studio or the Windows Game Bar (Win+G) to record
- Layout: browser on the left (Streamlit UI), terminal on the right (logs). When showing code, use VS Code full screen
- Zoom into code with Ctrl+= in VS Code so it's readable on screen
- If a query takes a few seconds, fill the silence: "The agent is now retrieving context... generating the answer... and now reflecting on quality..."
- Do a dry run before recording -- run the 4 queries once so they're cached and you know the expected output
- Keep it under 10 minutes
