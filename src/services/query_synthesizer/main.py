from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import groq
import os
from dotenv import load_dotenv
import json

load_dotenv()

SYSTEM_PROMPT = """
/set nothink\n\n
You are a query rewriting system for a Retrieval-Augmented Generation (RAG) pipeline with multiple specialized vector stores.

Rewrite the user query into optimized retrieval queries for each vector store.

Rules:
- Keep meaning identical across all queries
- Expand abbreviations if useful
- dense_query: natural language, semantic, sentence-form query optimized for embedding models
- sparse_query: keyword-focused, BM25-style query with important terms and phrases
- Do NOT answer the question
- Output ONLY valid JSON
- Do NOT add any backticks or other markers

Format:

{
  "books": {
    "dense_query": "semantic natural language query optimized for book content",
    "sparse_query": "keyword1 keyword2 relevant book terms"
  },
  "clinical": {
    "dense_query": "semantic natural language query optimized for clinical content",
    "sparse_query": "clinical keyword1 medical term symptom treatment"
  },
  "paper_abstract": {
    "dense_query": "semantic natural language query optimized for research abstracts",
    "sparse_query": "research keyword1 methodology finding academic term"
  }
}
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


class StoreQuery(BaseModel):
    dense_query: str
    sparse_query: str


class QuerySynthResponse(BaseModel):
    books: StoreQuery
    clinical: StoreQuery
    paper_abstract: StoreQuery


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
        validated = QuerySynthResponse(**data)
        return validated
    except Exception:
        print(content)
        fallback = StoreQuery(dense_query=req.query, sparse_query=req.query)
        return QuerySynthResponse(
            books=fallback, clinical=fallback, paper_abstract=fallback
        )
