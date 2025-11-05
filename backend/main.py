import torch
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.graph import RAGAgent
from config import CFG

torch.set_float32_matmul_precision('high')
rag_agent = None


@asynccontextmanager
async def prepare_rag_agent(app: FastAPI):
    global rag_agent
    rag_agent = RAGAgent(
        llm_model=CFG.MODEL,
        embedding_model=CFG.EMBEDDING_MODEL,
        ollama_base_url=CFG.LLM_API_URL
    )
    
    yield
    del rag_agent

app = FastAPI(lifespan=prepare_rag_agent)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_rag_agent( query: QueryRequest):
    assert isinstance(rag_agent, RAGAgent)
    answer = rag_agent.query(query.question) # return time taken as well
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)