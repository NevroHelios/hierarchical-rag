from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

from qdrant_client import QdrantClient, models
from transformers.pipelines import pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL = "BAAI/bge-small-en"
    app.state.encoder = pipeline(
        "feature-extraction",
        model=MODEL,
        device="cuda",
        truncation=True,
        max_length=512,
    )
    app.state.client = QdrantClient(location="localhost", port=6333)

    yield
    app.state.encoder = None
    app.state.client = None


app = FastAPI(lifespan=lifespan)


class StoreQuery(BaseModel):
    dense_query: str
    sparse_query: str


class RetrieveResponse(BaseModel):
    context: str


@app.post("/retrieve", response_model=RetrieveResponse)
def get_result(req: StoreQuery, request: Request):
    encoder = request.app.state.encoder
    client = request.app.state.client
    assert encoder is not None
    assert client is not None

    encoded_text = encoder(req.dense_query)
    encoded_text = (
        encoded_text[0][0] if isinstance(encoded_text[0][0], list) else encoded_text[0]
    )

    hits = client.query_points(
        collection_name="abstract_c",
        prefetch=[
            models.Prefetch(
                query=models.Document(text=req.sparse_query, model="Qdrant/bm25"),
                using="sparse",
                limit=20,
            ),
            models.Prefetch(query=encoded_text, using="dense", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5,
    )

    chunks = [point.payload.get("text", "") for point in hits.points if point.payload]
    context = "\n\n".join(chunks)
    return RetrieveResponse(context=context)
