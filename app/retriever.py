import os
import pandas as pd
import joblib
import numpy as np
from operator import add
from functools import partial
from typing import TypedDict, List, Tuple, Literal, Dict, Annotated
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma

from app.build_db import get_client, build_collection



NAME = 'granite4:latest'
K = 10

def build_retriever(
        name: str,
        docs_and_vecs: List[Tuple[Document, np.ndarray]],
        k: int = K,
        weights: List[float] = [0.6, 0.4],
        similarity_threshold: float = 0.72
):
    embeddings = OllamaEmbeddings(model=NAME)
    client = get_client()
    collection_names = [c.name for c in client.list_collections()]
    if name not in collection_names:
        build_collection(name, docs_and_vecs, client)

    vector_store = Chroma(
        client=client,
        collection_name=name,
        embedding_function=embeddings
    )
    vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    keyword_retriever = BM25Retriever.from_documents([d for d, _ in docs_and_vecs])
    keyword_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=weights, # test the combine fn as well
    )

    filter_ = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)
    hybrid = ContextualCompressionRetriever(
        base_compressor=filter_,
        base_retriever=ensemble_retriever
    )

    return hybrid