## running once will suffice

import chromadb
import joblib
from chromadb.config import Settings
from langchain_core.documents import Document
from typing import List, Tuple
from chromadb.api import ClientAPI
import numpy as np

def get_client(dir: str = './chroma_db') -> ClientAPI:
    return chromadb.PersistentClient(path=dir)


def build_collection(
        name: str,
        docs_and_vecs: List[Tuple[Document, np.ndarray]],
        client = None
):
    if client is None:
        client = get_client()
    
    try:
        client.delete_collection(name)
    except:
        pass

    collection = client.get_or_create_collection(name=name)
    documents = [d.page_content for d, _ in docs_and_vecs]
    embeddings = [v.tolist() for _, v in docs_and_vecs] 
    metadatas = [d.metadata for d, _ in docs_and_vecs]
    ids = [str(i) for i in range(len(docs_and_vecs))]

    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    return collection




# if __name__ == '__main__':
#     import os
#     from pathlib import Path
#     base_dir = Path(__file__).resolve().parent.parent
#     client = get_client()

#     data = joblib.load(os.path.join(base_dir, 'data\\vectors\\research_data_sample.joblib'))
#     build_collection('research', data, client)
#     data = joblib.load(os.path.join(base_dir, 'data\\vectors\\book_data_sample.joblib'))
#     build_collection('book', data, client)
#     data = joblib.load(os.path.join(base_dir, 'data\\vectors\\clinical_data_sample.joblib'))
#     build_collection('clinical', data, client)