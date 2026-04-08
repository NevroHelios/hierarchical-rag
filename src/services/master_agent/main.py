from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import httpx
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

QUERY_SYNTHESIZER_URL = os.getenv("QUERY_SYNTHESIZER_URL", "http://query-synthesizer:8000")
ANSWER_SYNTHESIZER_URL = os.getenv("ANSWER_SYNTHESIZER_URL", "http://answer-synthesizer:8000")

BOOKS_WORKER_URL = os.getenv("BOOKS_WORKER_URL", "http://books-worker:8000")
CLINICAL_WORKER_URL = os.getenv("CLINICAL_WORKER_URL", "http://clinical-worker:8000")
PAPER_ABSTRACT_WORKER_URL = os.getenv("PAPER_ABSTRACT_WORKER_URL", "http://paper-abstract-worker:8000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=60.0)
    yield
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
    for label, ctx in zip(worker_labels, results):
        if ctx:
            combined_context += f"[{label}]\n{ctx}\n\n"

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

    return MasterResponse(query=req.query, answer=answer)