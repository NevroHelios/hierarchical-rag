from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import httpx
import os
import asyncio
import hashlib
import json
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

QUERY_SYNTHESIZER_URL = os.getenv("QUERY_SYNTHESIZER_URL", "http://query-synthesizer:8000")
ANSWER_SYNTHESIZER_URL = os.getenv("ANSWER_SYNTHESIZER_URL", "http://answer-synthesizer:8000")

BOOKS_WORKER_URL = os.getenv("BOOKS_WORKER_URL", "http://books-worker:8000")
CLINICAL_WORKER_URL = os.getenv("CLINICAL_WORKER_URL", "http://clinical-worker:8000")
PAPER_ABSTRACT_WORKER_URL = os.getenv("PAPER_ABSTRACT_WORKER_URL", "http://paper-abstract-worker:8000")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=60.0)
    app.state.redis = redis.from_url(REDIS_URL, decode_responses=True)
    yield
    await app.state.redis.aclose()
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


class UserQuery(BaseModel):
    query: str


WORKER_CONFIG = {
    "books": {"url": BOOKS_WORKER_URL, "label": "Books"},
    "clinical": {"url": CLINICAL_WORKER_URL, "label": "Clinical"},
    "paper_abstract": {"url": PAPER_ABSTRACT_WORKER_URL, "label": "Research"},
}


class MasterResponse(BaseModel):
    query: str
    answer: str
    contexts: list[str] = []


def build_cache_key(synthesized: dict) -> str:
    """Hash the normalized synthesized queries so similar user queries hit cache."""
    workers = sorted(synthesized.get("workers", []))
    key_parts = []
    for w in workers:
        store = synthesized.get(w, {})
        key_parts.append(f"{w}:{store.get('dense_query', '')}:{store.get('sparse_query', '')}")
    raw = "|".join(key_parts)
    return "rag:" + hashlib.sha256(raw.encode()).hexdigest()


async def fetch_worker(client: httpx.AsyncClient, url: str, payload: dict) -> str:
    try:
        response = await client.post(f"{url}/retrieve", json=payload)
        response.raise_for_status()
        return response.json().get("context", "")
    except Exception as e:
        print(f"Worker {url} failed: {e}")
        return ""


@app.post("/query", response_model=MasterResponse)
async def master_query(req: UserQuery):
    client: httpx.AsyncClient = app.state.client

    try:
        synth_response = await client.post(
            f"{QUERY_SYNTHESIZER_URL}/query-synthesize",
            json={"query": req.query},
        )
        synth_response.raise_for_status()
        synthesized = synth_response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Query synthesizer failed: {e}")

    cache_key = build_cache_key(synthesized)
    r: redis.Redis = app.state.redis

    cached = await r.get(cache_key)
    if cached:
        return MasterResponse(query=req.query, answer=cached)

    selected_workers = synthesized.get("workers", list(WORKER_CONFIG.keys()))

    tasks = []
    worker_labels = []
    for worker_name in selected_workers:
        if worker_name not in WORKER_CONFIG:
            continue
        config = WORKER_CONFIG[worker_name]
        payload = synthesized.get(worker_name)
        if not payload:
            continue
        tasks.append(fetch_worker(client, config["url"], payload))
        worker_labels.append(config["label"])

    results = await asyncio.gather(*tasks)

    combined_context = ""
    retrieved_contexts = []
    for label, ctx in zip(worker_labels, results):
        if ctx:
            combined_context += f"[{label}]\n{ctx}\n\n"
            retrieved_contexts.append(f"[{label}]\n{ctx}")

    if not combined_context.strip():
        combined_context = "No relevant context was retrieved from any source."

    try:
        answer_response = await client.post(
            f"{ANSWER_SYNTHESIZER_URL}/answer-synthesize",
            json={"query": req.query, "context": combined_context.strip()},
        )
        answer_response.raise_for_status()
        answer = answer_response.json().get("answer", "")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Answer synthesizer failed: {e}")

    await r.set(cache_key, answer, ex=CACHE_TTL)

    return MasterResponse(query=req.query, answer=answer, contexts=retrieved_contexts)