import os
import json
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict

from unstructured.partition.text import partition_text
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from transformers import pipeline, AutoTokenizer
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from tqdm.auto import tqdm
from dataclasses import dataclass

MODEL = "BAAI/bge-small-en"
TOKEN_MIN_LENGTH = 200
TOKEN_MAX_LENGTH = 500

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRAGMENTED_DIR = BASE_DIR / "data" / "fragmented"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MANIFEST_PATH = BASE_DIR / "data" / "manifest.json"

FILE_COLLECTION_MAP = {
    "abs.json": "abstract_c",
    "book1.json": "book_c",
    "clinical_reports.json": "clinical_c",
}

FRAGMENTED_SOURCE_MAP = {
    "abstracts1.txt": "abs.json",
    "clinical_reports.txt": "clinical_reports.json",
}


@dataclass
class Metadata:
    chunk_id: str
    tokens: int
    source: str


@dataclass
class Chunk:
    text: str
    metadata: Metadata


tokenizer = AutoTokenizer.from_pretrained(MODEL)


def extract_to_json(filename: str):
    src = str(FRAGMENTED_DIR / filename)
    contents = partition_text(src)
    out = str(PROCESSED_DIR / f"{filename.split('.')[0]}.json")
    elements_to_json(contents, out)


def preprocess_all():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for frag_file, proc_file in FRAGMENTED_SOURCE_MAP.items():
        frag_path = FRAGMENTED_DIR / frag_file
        if frag_path.exists():
            print(f"Preprocessing {frag_file} -> {proc_file}")
            extract_to_json(frag_file)


def compute_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


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
            metadata = Metadata(
                chunk_id=str(uuid.uuid4()), source=last_source, tokens=current_len
            )
            chunks.append(Chunk(text=current_text, metadata=metadata))
            current_len = 0
            current_text = ""

        current_len += token_len
        current_text += text

        if current_len >= TOKEN_MIN_LENGTH:
            metadata = Metadata(
                chunk_id=str(uuid.uuid4()), source=last_source, tokens=current_len
            )
            chunks.append(Chunk(text=current_text, metadata=metadata))
            current_len = 0
            current_text = ""

    if current_text:
        metadata = Metadata(
            chunk_id=str(uuid.uuid4()), source=last_source, tokens=current_len
        )
        chunks.append(Chunk(text=current_text, metadata=metadata))

    return chunks


def ensure_collection(client: QdrantClient, collection_name: str):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )


def encode_and_upsert(client: QdrantClient, pipe, collection_name: str, chunks: List[Chunk]) -> List[str]:
    ensure_collection(client, collection_name)
    points = []
    chunk_ids = []
    for chunk in tqdm(chunks, desc=f"Encoding {collection_name}"):
        embeddings = pipe(chunk.text)
        vector = (
            embeddings[0][0] if isinstance(embeddings[0][0], list) else embeddings[0]
        )
        point = models.PointStruct(
            id=chunk.metadata.chunk_id,
            vector={
                "dense": vector,
                "sparse": models.Document(text=chunk.text, model="Qdrant/bm25"),
            },
            payload={
                "text": chunk.text,
                "source": chunk.metadata.source,
                "token_count": chunk.metadata.tokens,
            },
        )
        points.append(point)
        chunk_ids.append(chunk.metadata.chunk_id)

    if points:
        client.upsert(collection_name=collection_name, points=points)
    return chunk_ids


def delete_chunks(client: QdrantClient, collection_name: str, chunk_ids: List[str]):
    if not chunk_ids:
        return
    if not client.collection_exists(collection_name):
        return
    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(points=chunk_ids),
    )


def load_manifest() -> Dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: Dict):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def sync():
    preprocess_all()

    manifest = load_manifest()
    qdrant = QdrantClient("localhost", port=6333)
    pipe = pipeline(
        "feature-extraction",
        model=MODEL,
        device="cuda",
        truncation=True,
        max_length=512,
    )

    current_files = set()

    for filename, collection in FILE_COLLECTION_MAP.items():
        filepath = PROCESSED_DIR / filename
        if not filepath.exists():
            continue

        current_files.add(filename)
        file_hash = compute_file_hash(str(filepath))
        existing = manifest.get(filename, {})

        if existing.get("hash") == file_hash:
            print(f"[skip] {filename} unchanged")
            continue

        old_chunk_ids = existing.get("chunk_ids", [])
        if old_chunk_ids:
            print(f"[invalidate] {filename}: removing {len(old_chunk_ids)} old chunks from {collection}")
            delete_chunks(qdrant, collection, old_chunk_ids)

        print(f"[sync] {filename} -> {collection}")
        chunks = build_chunks(str(filepath))
        chunk_ids = encode_and_upsert(qdrant, pipe, collection, chunks)

        manifest[filename] = {
            "hash": file_hash,
            "collection": collection,
            "chunk_ids": chunk_ids,
        }

    removed = set(manifest.keys()) - current_files
    for filename in removed:
        entry = manifest[filename]
        print(f"[remove] {filename}: deleting from {entry['collection']}")
        delete_chunks(qdrant, entry["collection"], entry.get("chunk_ids", []))
        del manifest[filename]

    save_manifest(manifest)
    print("[done] manifest updated")


if __name__ == "__main__":
    sync()
