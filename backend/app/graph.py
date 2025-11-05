import os
from pathlib import Path
import joblib
from functools import partial
from typing import List, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph


from app.retriever import build_retriever
from app.states import MasterAgentState, MasterQuery
from app.prompts import master_query_generator_prompt


base_dir = Path(__file__).resolve().parent.parent.parent
MODEL = "granite4:latest"
EMBEDDING_MODEL = "ibm-granite/granite-embedding-english-r2"


class RAGAgent:
    def __init__(
        self,
        llm_model: str = MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        k: int = 10,
        similarity_threshold: float = 0.87,
        ensemble_weights: List[float] = [0.6, 0.4],
        ollama_base_url: str = "http://172.26.166.119:11434",
    ) -> None:
        # TODO: structure it such that different retrievers can be set up for A/B testing
        self.llm = ChatOllama(model=llm_model, num_ctx=15768, base_url=ollama_base_url)
        self.embeddings_model = embedding_model

        research_retriever, book_retriever, clinical_retriever = self.load_retrievers(
            k=k,
            similarity_threshold=similarity_threshold,
            ensemble_weights=ensemble_weights,
            embedding_model=embedding_model,
        )

        self.retrievers = {
            "research": research_retriever,
            "book": book_retriever,
            "clinic": clinical_retriever,
        }

        self.app = self.compile_graph()

    def query(self, question: str) -> str:
        initial_state: MasterAgentState = {
            "question": question,
            "queries": {},
            "contexts": [],
            "answer": "",
        }
        final_state = self.app.invoke(initial_state)
        return final_state["answer"]

    def load_retrievers(
        self,
        k: int,
        similarity_threshold: float,
        ensemble_weights: List[float] = [0.6, 0.4],
        embedding_model: str = EMBEDDING_MODEL,
    ):
        research_data = joblib.load(
            os.path.join(base_dir, "data/vectors/research_data_sample.joblib")
        )
        book_data = joblib.load(
            os.path.join(base_dir, "data/vectors/book_data_sample.joblib")
        )
        clinical_data = joblib.load(
            os.path.join(base_dir, "data/vectors/clinical_data_sample.joblib")
        )

        research = build_retriever(
            research_data,
            ensemble_weights=ensemble_weights,
            k=k,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
        )
        book = build_retriever(
            book_data,
            ensemble_weights=ensemble_weights,
            k=k,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
        )
        clinical = build_retriever(
            clinical_data,
            ensemble_weights=ensemble_weights,
            k=k,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
        )

        return research, book, clinical

    def master_query_node(self, state: MasterAgentState) -> MasterAgentState:
        query = state["question"]
        prompt_master = master_query_generator_prompt + query

        initial_agent = self.llm.with_structured_output(MasterQuery)
        res = initial_agent.invoke(input=prompt_master)
        assert isinstance(res, MasterQuery), "Master query Failed"

        return {**state, "queries": res.queries, "contexts": []}

    def worker_node(
        self,
        state: MasterAgentState,
        worker_name: Literal["clinic", "research", "book"],
    ):
        query = state["queries"].get(worker_name)
        if query is None:
            return {**state}
        retriever = self.retrievers[worker_name]
        docs = retriever.invoke(query)

        context_chunks = []
        for doc in docs:
            source_info = doc.metadata["source"]
            context_chunks.append(f"Source: {source_info}\nContent: {doc.page_content}")
        context = "\n\n---\n\n".join(context_chunks)
        return {"contexts": [(worker_name, context)]}

    def master_synthesizer_node(self, state: MasterAgentState) -> MasterAgentState:
        question = state["question"]
        print(f"[MASTER SYNTHESIZER] Combining {len(state['contexts'])} contexts")

        context_str = ""
        for agent_name, context in state["contexts"]:
            context_str += f"--- Context from {agent_name} ---\n{context}\n\n"
        prompt = f"""
        You are an expert medical research assistant. You have received context from several specialized agents.
        - Prioritize evidence that directly addresses the user's question.
        - Do NOT invent or infer beyond what is supported by the provided context.
        - If there is conflicting information, state it clearly.
        - always answer within two sentences with key details.

        {context_str}

        Original Question: {question}
        Final, evidence-based answer:
        """

        res = self.llm.invoke(prompt).content
        assert isinstance(res, str), "Final answer generation failed"
        return {**state, "answer": res}

    def route_to_workers(self, state: MasterAgentState) -> List[str]:
        return list(state["queries"].keys())

    def compile_graph(self):
        graph = StateGraph(MasterAgentState)

        graph.set_entry_point("master_query")
        graph.add_node("master_query", self.master_query_node)

        graph.add_node("clinic", partial(self.worker_node, worker_name="clinic"))
        graph.add_node("research", partial(self.worker_node, worker_name="research"))
        graph.add_node("book", partial(self.worker_node, worker_name="book"))

        graph.add_node("master_synthesizer", self.master_synthesizer_node)

        graph.add_conditional_edges(
            "master_query",
            self.route_to_workers,
            {"clinic": "clinic", "research": "research", "book": "book"},
        )

        graph.add_edge("clinic", "master_synthesizer")
        graph.add_edge("research", "master_synthesizer")
        graph.add_edge("book", "master_synthesizer")
        graph.set_finish_point("master_synthesizer")

        app = graph.compile()

        return app
