from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import groq
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a final answer synthesizer for a Retrieval-Augmented Generation (RAG) pipeline.

You will receive a user query and retrieved context from multiple specialized vector stores (books, clinical literature, and research paper abstracts).

Your job is to synthesize a comprehensive, accurate, and concise answer based solely on the provided context.

Rules:
- Answer ONLY from the provided context, do not hallucinate
- If the context does not contain enough information, say so clearly
- Be concise but thorough
- Cite the type of source (book, clinical, research) when relevant
- Do NOT repeat the question
- Output plain text only, no JSON, no markdown unless necessary
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    assert os.environ.get("GROQ_API_KEY") is not None
    app.state.model = "llama-3.3-70b-versatile"
    app.state.groq = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
    yield


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    query: str
    context: str


class AnswerResponse(BaseModel):
    answer: str


@app.post("/answer-synthesize", response_model=AnswerResponse)
def answer_synthesize(req: Query):

    user_message = f"Query: {req.query}\n\nContext:\n{req.context}"

    response = app.state.groq.chat.completions.create(
        model=app.state.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        return AnswerResponse(answer="Could not generate an answer from the provided context.")

    return AnswerResponse(answer=content.strip())