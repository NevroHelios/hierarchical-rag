import os
from pathlib import Path
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

from app.retriever import build_retriever
from app.states import MasterAgentState, MasterQuery
from app.prompts import master_query_generator_prompt


base_dir = Path(__file__).resolve().parent.parent


def load_retrievers():
    research_data = joblib.load(os.path.join(base_dir, 'data\\vectors\\research_data_sample.joblib'))
    book_data = joblib.load(os.path.join(base_dir, 'data\\vectors\\book_data_sample.joblib'))
    clinical_data = joblib.load(os.path.join(base_dir, 'data\\vectors\\clinical_data_sample.joblib'))

    research = build_retriever("research", research_data, weights=[0.7, 0.3])
    book = build_retriever("book", book_data, weights=[0.5, 0.5])
    clinical = build_retriever("clinical", clinical_data, weights=[0.6, 0.4])

    return research, book, clinical


def master_query_node(state: MasterAgentState) -> MasterAgentState:
    query = state['question']

    prompt_master = master_query_generator_prompt + query
    
    initial_agent = llm.with_structured_output(MasterQuery)
    res = initial_agent.invoke(input=prompt_master)
    assert isinstance(res, MasterQuery), "Master query Failed"
    return {**state, 'queries': res.queries, 'contexts': [] }



def worker_node(state: MasterAgentState, worker_name: Literal['clinic', 'research', 'book']):
    query = state["queries"].get(worker_name)
    if query is None:
        return {**state}
    retriever = retrievers[worker_name]
    docs = retriever.invoke(query)
    context_chunks = []
    for doc in docs:
        source_info = doc.metadata['source']
        context_chunks.append(f"Source: {source_info}\nContent: {doc.page_content}")
    context = "\n\n---\n\n".join(context_chunks)
    return {'contexts': [(worker_name, context)]}


def master_synthesizer_node(state: MasterAgentState) -> MasterAgentState:
    question = state["question"]
    context_str = ""
    for agent_name, context in state['contexts']:
        context_str += f"--- Context from {agent_name} ---\n{context}\n\n"
    prompt = f"""
    You are an expert medical research assistant. You have received context from several specialized agents.
    - Prioritize evidence that directly addresses the user's question.
    - Explicitly mention which source each claim is drawn from.
    - Do NOT invent or infer beyond what is supported by the provided context.
    - If there is conflicting information, state it clearly.

    {context_str}

    Original Question: {question}
    Final, evidence-based answer:
    """

    
    res = llm.invoke(prompt).content
    assert isinstance(res, str), "Final answer generation failed"
    return {**state, "answer": res}

def route_to_workers(state: MasterAgentState) -> List[str]:
    return list(state["queries"].keys())


def compile_graph():
    global llm
    global retrievers
    research_retriever, book_retriever, clinical_retriever = load_retrievers()

    retrievers = {
        'research': research_retriever,
        'book': book_retriever,
        'clinic': clinical_retriever
    }
        
    MODEL = 'granite4:latest'
    llm = ChatOllama(model=MODEL, num_ctx=15768)
    embeddings = OllamaEmbeddings(model=MODEL)

    graph = StateGraph(MasterAgentState)

    graph.set_entry_point('master_query')
    graph.add_node("master_query", master_query_node)

    graph.add_node('clinic', partial(worker_node, worker_name='clinic'))
    graph.add_node('research', partial(worker_node, worker_name='research'))
    graph.add_node('book', partial(worker_node, worker_name='book'))

    graph.add_node('master_synthesizer', master_synthesizer_node)

    graph.add_conditional_edges(
        "master_query",
        route_to_workers,
        {
            'clinic': 'clinic',
            'research': 'research',
            'book': 'book'
        }
    )

    graph.add_edge('clinic', 'master_synthesizer')
    graph.add_edge('research', 'master_synthesizer')
    graph.add_edge('book', 'master_synthesizer')
    graph.set_finish_point('master_synthesizer')


    app = graph.compile()

    return app