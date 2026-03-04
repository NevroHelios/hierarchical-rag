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

    books_payload = synthesized["books"]
    clinical_payload = synthesized["clinical"]
    paper_abstract_payload = synthesized["paper_abstract"]

    books_ctx, clinical_ctx, paper_ctx = await asyncio.gather(
        fetch_worker(client, BOOKS_WORKER_URL, books_payload),
        fetch_worker(client, CLINICAL_WORKER_URL, clinical_payload),
        fetch_worker(client, PAPER_ABSTRACT_WORKER_URL, paper_abstract_payload),
    )

    combined_context = ""
    if books_ctx:
        combined_context += f"[Books]\n{books_ctx}\n\n"
    if clinical_ctx:
        combined_context += f"[Clinical]\n{clinical_ctx}\n\n"
    if paper_ctx:
        combined_context += f"[Research]\n{paper_ctx}\n\n"

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