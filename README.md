# Hierarchical RAG — Multi-Source Medical QA System

A production-grade **Hierarchical Retrieval-Augmented Generation** system for diabetes-domain question answering. Implements a distributed microservices architecture where specialized worker agents independently retrieve from domain-specific vector stores, and a master orchestrator synthesizes a single evidence-grounded answer. Achieves **100% context recall**, **0.93 answer similarity**, and **0.83 factual correctness (F1)** on RAGAS evaluation.

---

## Architecture

```
User Query
    │
    ▼
Master Agent
    │
    ▼
Query Synthesizer
(selects relevant workers, rewrites query into dense + sparse variants)
    │
    ├──── if selected ────┬──── if selected ────┐
    ▼                     ▼                     ▼
Books Worker       Clinical Worker       Research Worker
(Qdrant)           (Qdrant)              (Qdrant)
Hybrid Search      Hybrid Search         Hybrid Search
    │                     │                     │
    └─────────────────────┴─────────────────────┘
                          │
                     Cache Check (Redis)
                     hit? ─► return cached answer
                     miss? ▼
                Context Aggregation
                ([Books] + [Clinical] + [Research])
                          │
                          ▼
                Answer Synthesizer
                          │
                          ▼
                   Final Answer ──► cache in Redis
```

Each worker performs **hybrid retrieval** — combining dense (semantic) and sparse (BM25 keyword) search with Reciprocal Rank Fusion — to maximize recall across vocabulary gaps.

---

## Services

| Service | Port | Responsibility |
|---|---|---|
| `master-agent` | 8080 | Entry point, orchestrates all services |
| `query-synthesizer` | internal | Selects relevant workers, rewrites query into dense + sparse variants |
| `books-worker` | internal | Retrieves from medical book corpus |
| `clinical-worker` | internal | Retrieves from clinical trial reports |
| `paper-abstract-worker` | internal | Retrieves from research paper abstracts |
| `answer-synthesizer` | internal | Synthesizes final grounded answer |
| `redis` | 6379 | Caches answers keyed by normalized synthesized queries |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | `BAAI/bge-small-en` (384-dim, GPU-accelerated) |
| Vector DB | Qdrant (hybrid dense + sparse indexing) |
| Sparse Retrieval | BM25 via Qdrant (`Qdrant/bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Cache | Redis (TTL-based, keyed on synthesized queries) |
| API Framework | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| Evaluation | RAGAS (Faithfulness, Relevancy, Context Precision/Recall) |

---

## Query Flow

1. **Query Synthesis** — The user query is sent to the Query Synthesizer, which uses an LLM to decide which workers are relevant (1–3) and produces dense + sparse query variants per selected worker.

2. **Cache Check** — The master agent hashes the synthesized queries into a cache key and checks Redis. On a hit, the cached answer is returned immediately, skipping retrieval and answer synthesis.

3. **Parallel Retrieval** — On a cache miss, the master fans out to only the selected workers via `asyncio.gather`. Each worker:
   - Encodes the `dense_query` with `BAAI/bge-small-en`
   - Runs BM25 sparse search using the `sparse_query`
   - Fuses results with RRF and returns top-5 chunks as text

4. **Context Aggregation** — Results from workers are labeled by source type (`[Books]`, `[Clinical]`, `[Research]`) and concatenated.

5. **Answer Synthesis** — The Answer Synthesizer receives the original query + aggregated context and generates a concise, evidence-grounded response. The answer is cached in Redis before returning.

---

## Data Sync

The `sync.py` script manages the full pipeline from raw documents to indexed vectors:

```bash
python src/utils/sync.py
```

It:
1. Preprocesses documents from `data/fragmented/` → `data/processed/` using `unstructured`
2. Computes SHA-256 hash of each processed file
3. Compares against `data/manifest.json`
4. For changed/new files: deletes old chunks from Qdrant, re-chunks, encodes, and upserts
5. For removed files: cleans up Qdrant and manifest
6. Saves updated manifest with chunk IDs for future invalidation

---

## Evaluation

Run the RAGAS evaluation suite against the live system:

```bash
GROQ_API_KEY=... python src/utils/eval/run_eval.py
```

Uses the QA dataset at `src/eval/qa.json` (question + ground truth pairs). For each question, it queries the full pipeline and evaluates with:

- **Faithfulness** — is the answer grounded in retrieved context?
- **Response Relevancy** — does the answer address the question?
- **Context Precision** — are retrieved chunks relevant?
- **Context Recall** — did retrieval cover the ground truth?

LLM judge: Groq. Embeddings: Ollama. Results saved to `src/eval/eval_results.json`.

### Latest Results

| Metric | Score |
|---|---|
| Factual Correctness (F1) | 0.8300 |
| Answer Similarity | 0.9336 |
| Context Recall | 1.0000 |

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

### 2. Sync Documents to Vector Stores

```bash
python src/utils/sync.py
```

This preprocesses documents, chunks them, and upserts into three Qdrant collections: `book_c`, `clinical_c`, `abstract_c`.

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
- **Sparse vector** — BM25 token weights via `Qdrant/bm25`, capturing exact keyword matches

At query time, both search types run independently and are merged via RRF, ensuring neither vocabulary mismatch nor semantic drift causes relevant documents to be missed.

---

## Project Structure

```
hierarchical-rag/
├── docker-compose.yml
├── pyproject.toml
├── src/
│   ├── services/
│   │   ├── master_agent/          # Orchestrator (port 8080), Redis cache
│   │   ├── query_synthesizer/     # Worker selection + query rewriting
│   │   ├── answer_synthesizer/    # Final answer generation
│   │   └── workers/
│   │       ├── books/             # Book corpus retriever
│   │       ├── clinical/          # Clinical trial retriever
│   │       └── paper_abstract/    # Research abstract retriever
│   ├── utils/
│   │   ├── sync.py                # Document sync + Qdrant indexing
│   │   ├── create_vectors.py      # Standalone chunking + upload
│   │   └── eval/
│   │       └── run_eval.py        # RAGAS evaluation script
│   └── eval/
│       └── qa.json                # Evaluation QA dataset
└── data/
    ├── raw/                       # Source PDFs / documents
    ├── fragmented/                # Extracted text files
    ├── processed/                 # Parsed JSON (via unstructured)
    └── manifest.json              # Sync state (file hashes + chunk IDs)
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key for Groq LLM inference |
| `REDIS_URL` | No | Redis connection URL (default: `redis://redis:6379/0`) |
| `CACHE_TTL` | No | Cache expiry in seconds (default: `3600`) |
| `MASTER_AGENT_URL` | No | Master agent URL for eval script (default: `http://localhost:8080`) |
| `OLLAMA_EMBED_MODEL` | No | Ollama embedding model for eval (default: `nomic-embed-text`) |
