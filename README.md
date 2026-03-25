# Hierarchical RAG — Multi-Source Medical QA System

A production-grade **Hierarchical Retrieval-Augmented Generation** system for diabetes-domain question answering. Implements a distributed microservices architecture where specialized worker agents independently retrieve from domain-specific vector stores, and a master orchestrator synthesizes a single evidence-grounded answer.

---

## Architecture

```
User Query
    │
    ▼
Master Agent  ──────────────────────────────────────────────────
    │                                                           │
    ▼                                                           │
Query Synthesizer                                               │
(rewrites query into 6 domain-optimized variants)              │
    │                                                           │
    ├──────────────────┬──────────────────┐                    │
    ▼                  ▼                  ▼                    │
Books Worker    Clinical Worker    Research Worker             │
(Qdrant)        (Qdrant)           (Qdrant)                    │
Hybrid Search   Hybrid Search      Hybrid Search              │
    │                  │                  │                    │
    └──────────────────┴──────────────────┘                    │
                       │                                        │
                       ▼                                        │
              Context Aggregation                               │
              ([Books] + [Clinical] + [Research])              │
                       │                                        │
                       ▼                                        │
             Answer Synthesizer  ◄──────────────────────────────
                       │
                       ▼
               Final Answer
```

Each worker performs **hybrid retrieval** — combining dense (semantic) and sparse (BM25 keyword) search with Reciprocal Rank Fusion — to maximize recall across vocabulary gaps.

---

## Services

| Service | Port | Responsibility |
|---|---|---|
| `master-agent` | 8080 | Entry point, orchestrates all services |
| `query-synthesizer` | internal | Rewrites user query into domain-specific variants |
| `books-worker` | internal | Retrieves from medical book corpus |
| `clinical-worker` | internal | Retrieves from clinical trial reports |
| `paper-abstract-worker` | internal | Retrieves from research paper abstracts |
| `answer-synthesizer` | internal | Synthesizes final grounded answer |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) / IBM Granite 4 (Ollama) |
| Embeddings | `BAAI/bge-small-en` (384-dim, GPU-accelerated) |
| Vector DB | Qdrant (hybrid dense + sparse indexing) |
| Sparse Retrieval | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Orchestration | LangGraph state machine (backend), asyncio (main) |
| API Framework | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |

---

## Query Flow — Detailed

1. **Query Synthesis** — The user query is sent to the Query Synthesizer, which uses an LLM to produce 6 targeted query variants (dense + sparse for each of the 3 domains), preserving semantic intent while optimizing for each retriever type.

2. **Parallel Retrieval** — The master agent fans out to all three workers concurrently via `asyncio.gather`. Each worker:
   - Encodes the dense query with `BAAI/bge-small-en`
   - Runs BM25 sparse search over its Qdrant collection
   - Runs dense vector search over the same collection
   - Fuses results with RRF and returns top-k chunks

3. **Context Aggregation** — Results from all workers are labeled by source type (`[Books]`, `[Clinical]`, `[Research]`) and concatenated into a single context block.

4. **Answer Synthesis** — The Answer Synthesizer receives the original query + aggregated context and generates a concise, evidence-grounded response using only the provided context, preventing hallucination.

---

## Setup

### Prerequisites

- Docker & Docker Compose
- Qdrant running locally (or accessible on network)
- `GROQ_API_KEY` environment variable set

### 1. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Populate Vector Stores

Run the vector creation utility to chunk, encode, and upload documents to Qdrant:

```bash
python src/utils/create_vectors.py
```

This creates three Qdrant collections: `book_c`, `clinical_c`, `abstract_c`.

### 3. Start All Services

```bash
docker compose up --build
```

### 4. Query the System

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the effects of metformin on glucose control?"}'
```

---

## Vector Store Design

Each document chunk is stored with two representations in Qdrant:

- **Dense vector** — 384-dim embedding from `BAAI/bge-small-en`, capturing semantic meaning
- **Sparse vector** — BM25 token weights, capturing exact keyword matches

At query time, both search types run independently and are merged via RRF, ensuring neither vocabulary mismatch nor semantic drift causes relevant documents to be missed.

---

## Project Structure

```
hierarchical-rag/
├── docker-compose.yml
├── pyproject.toml
├── src/
│   ├── services/
│   │   ├── master_agent/          # Orchestrator (port 8080)
│   │   ├── query_synthesizer/     # LLM-based query rewriter
│   │   ├── answer_synthesizer/    # Final answer generation
│   │   └── workers/
│   │       ├── books/             # Book corpus retriever
│   │       ├── clinical/          # Clinical trial retriever
│   │       └── paper_abstract/    # Research abstract retriever
│   └── utils/
│       └── create_vectors.py      # Chunking + embedding + Qdrant upload
├── backend/                       # Alternate LangGraph implementation
│   ├── app/
│   │   ├── graph.py               # LangGraph state machine
│   │   ├── retriever.py           # Hybrid ensemble retriever
│   │   ├── prompts.py             # Domain-tuned system prompts
│   │   └── states.py              # Typed state definitions
│   └── config.py
└── data/
    ├── raw/                       # Source PDFs / documents
    ├── processed/                 # Extracted JSON / Markdown
    └── vectors/                   # Pre-built FAISS joblib files
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key for Groq LLM inference |

---

## Alternate Backend

The `backend/` directory contains an experimental LangGraph-based implementation using:

- IBM Granite 4 (via Ollama) as the LLM
- IBM Granite embeddings
- FAISS in-memory vector stores (pre-built joblib files)
- Contextual compression filtering (similarity threshold: 0.87)
- LangGraph state machine for explicit node-based orchestration

This version prioritizes local-first, offline operation without requiring Qdrant or a cloud LLM.
