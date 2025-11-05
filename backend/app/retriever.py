from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_huggingface import HuggingFaceEmbeddings

MODEL = "granite4:latest"
EMBEDDING_MODEL = "ibm-granite/granite-embedding-english-r2"
K = 10


def build_retriever(
    docs_and_vecs: List[Tuple[Document, List[float]]],
    k: int = K,
    ensemble_weights: List[float] = [0.6, 0.4],
    similarity_threshold: float = 0.87,
    embedding_model: str = EMBEDDING_MODEL,
):
    embeddings = HuggingFaceEmbeddings(model=embedding_model)

    vector_store = FAISS.from_embeddings(
        text_embeddings=[(d.page_content, v) for d, v in docs_and_vecs],
        embedding=embeddings,
        metadatas=[d.metadata for d, _ in docs_and_vecs],
    )
    vector_retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    keyword_retriever = BM25Retriever.from_documents([d for d, _ in docs_and_vecs])
    keyword_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=ensemble_weights,  # test the combine fn as well
    )

    filter_ = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=similarity_threshold
    )
    hybrid = ContextualCompressionRetriever(
        base_compressor=filter_, base_retriever=ensemble_retriever
    )

    return hybrid
