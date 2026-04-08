from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import groq
import os
from dotenv import load_dotenv
import json

load_dotenv()

SYSTEM_PROMPT = """
You are a query routing and rewriting system for a Retrieval-Augmented Generation (RAG) pipeline with multiple specialized vector stores.

Available vector stores:
- "books": Medical/scientific textbook content
- "clinical": Clinical trial reports and medical case data
- "paper_abstract": Research paper abstracts and academic findings

Your job:
1. Decide which vector stores are relevant to the user's query (at least 1, at most 3).
2. For each selected store, rewrite the query into optimized dense and sparse retrieval queries.

Rules:
- Only include stores that are relevant to the query
- Keep meaning identical across all queries
- Expand abbreviations if useful
- dense_query: natural language, semantic, sentence-form query optimized for embedding models
- sparse_query: keyword-focused, BM25-style query with important terms and phrases
- Do NOT answer the question
- Output ONLY valid JSON
- Do NOT add any backticks or other markers

Format:

{
  "workers": ["books", "clinical"],
  "books": {
    "dense_query": "semantic natural language query optimized for book content",
    "sparse_query": "keyword1 keyword2 relevant book terms"
  },
  "clinical": {
    "dense_query": "semantic natural language query optimized for clinical content",
    "sparse_query": "clinical keyword1 medical term symptom treatment"
  }
}

The "workers" array must list exactly the stores you included. Only include a store object if it appears in "workers".
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    assert os.environ.get("GROQ_API_KEY") is not None
    app.state.model = "llama-3.3-70b-versatile"
    app.state.groq = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
    yield


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    query: str


VALID_WORKERS = {"books", "clinical", "paper_abstract"}


class StoreQuery(BaseModel):
    dense_query: str
    sparse_query: str


class QuerySynthResponse(BaseModel):
    workers: list[str]
    books: StoreQuery | None = None
    clinical: StoreQuery | None = None
    paper_abstract: StoreQuery | None = None


@app.post("/query-synthesize", response_model=QuerySynthResponse)
def query_synthesize(req: Query):

    response = app.state.groq.chat.completions.create(
        model=app.state.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.query},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
        workers = [w for w in data.get("workers", []) if w in VALID_WORKERS]
        if not workers:
            workers = list(VALID_WORKERS)
        data["workers"] = workers
        validated = QuerySynthResponse(**data)
        return validated
    except Exception:
        print(content)
        fallback = StoreQuery(dense_query=req.query, sparse_query=req.query)
        return QuerySynthResponse(
            workers=list(VALID_WORKERS),
            books=fallback, clinical=fallback, paper_abstract=fallback
        )
