import json
import os
from pathlib import Path
from transformers import pipeline, AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict
import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from tqdm.auto import tqdm

MODEL = "BAAI/bge-small-en"
TOKEN_MIN_LENGTH = 200
TOKEN_MAX_LENGTH = 500


@dataclass
class Metadata:
    chunk_id: str  # uuid
    tokens: int
    source: str


@dataclass
class Chunk:
    text: str
    metadata: Metadata


@dataclass
class Contents:
    element_id: str
    metadata: Dict[str, str]  # irrelevant here
    text: str
    type: str


tokenizer = AutoTokenizer.from_pretrained(MODEL)

path = os.path.join(Path(os.getcwd()).parent, "data/processed")
filename = "abs.json"
src = os.path.join(path, filename)
# print(contents[0])


def build_chunks(filepath: str) -> List[Chunk]:
    with open(filepath, "r") as f:
        contents_data = json.load(f)

    current_len = 0
    current_text = ""
    chunks: List[Chunk] = []
    last_source = "unknown"

    for content_dict in contents_data:
        text = content_dict.get("text", "")
        tokens = tokenizer.encode(text)
        token_len = len(tokens)
        last_source = content_dict.get("metadata", {}).get("filename", "unknown")

        if current_len + token_len > TOKEN_MAX_LENGTH and current_text:
            metadata: Metadata = Metadata(
                chunk_id=str(uuid.uuid4()), source=last_source, tokens=current_len
            )
            chunk: Chunk = Chunk(text=current_text, metadata=metadata)
            chunks.append(chunk)
            current_len = 0
            current_text = ""

        current_len += token_len
        current_text += text

        if current_len >= TOKEN_MIN_LENGTH:
            metadata: Metadata = Metadata(
                chunk_id=str(uuid.uuid4()), source=last_source, tokens=current_len
            )
            chunk: Chunk = Chunk(text=current_text, metadata=metadata)
            chunks.append(chunk)
            current_len = 0
            current_text = ""

    # handle remaining text
    if current_text:
        metadata: Metadata = Metadata(
            chunk_id=str(uuid.uuid4()), source=last_source, tokens=current_len
        )
        chunk: Chunk = Chunk(text=current_text, metadata=metadata)
        chunks.append(chunk)

    return chunks


def encode_and_save(collection_name: str, chunks: List[Chunk]):
    pipe = pipeline(
        "feature-extraction",
        model=MODEL,
        device="cuda",
        truncation=True,
        max_length=512,
    )
    client = QdrantClient("localhost", port=6333)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    points = []
    for chunk in tqdm(chunks):
        embeddings = pipe(chunk.text)
        # flatten the nested list structure
        vector = (
            embeddings[0][0] if isinstance(embeddings[0][0], list) else embeddings[0]
        )

        point = models.PointStruct(
            id=chunk.metadata.chunk_id,
            vector={
                "dense": vector,
                "sparse": models.Document(
                    text=chunk.text,
                    model="Qdrant/bm25",
                ),
            },
            payload={
                "text": chunk.text,
                "source": chunk.metadata.source,
                "token_count": chunk.metadata.tokens,
            },
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)


def save_to_db(collection_name: str, src: str):
    chunks = build_chunks(src)
    encode_and_save(collection_name=collection_name, chunks=chunks)


if __name__ == "__main__":
    path = os.path.join(Path(os.getcwd()).parent, "data/processed")
    filename = "abs.json"
    src = os.path.join(path, filename)
    save_to_db(collection_name="abstract_c", src=src)

    filename = "book1.json"
    src = os.path.join(path, filename)
    save_to_db(collection_name="book_c", src=src)

    filename = "clinical_reports.json"
    src = os.path.join(path, filename)
    save_to_db(collection_name="clinical_c", src=src)
