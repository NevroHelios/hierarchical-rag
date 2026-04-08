import os
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "eval" / "qa.json"
RESULTS_PATH = BASE_DIR / "eval" / "eval_results.json"
MASTER_URL = os.getenv("MASTER_AGENT_URL", "http://localhost:8080")

GROQ_MODEL = os.getenv("GROQ_EVAL_MODEL", "llama-3.3-70b-versatile")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def load_dataset() -> dict:
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def query_system(question: str) -> dict:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{MASTER_URL}/query", json={"query": question})
        resp.raise_for_status()
        return resp.json()


def build_samples(data: dict) -> list[SingleTurnSample]:
    questions = data["questions"]
    ground_truths = data["ground_truths"]
    samples = []

    for question, ground_truth in zip(questions, ground_truths):
        print(f"Querying: {question[:80]}...")
        try:
            result = query_system(question)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        answer = result.get("answer", "")
        contexts = result.get("contexts", [])

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts if contexts else [""],
            reference=ground_truth,
        )
        samples.append(sample)

    return samples


def run():
    print(f"Loading dataset from {DATASET_PATH}")
    data = load_dataset()
    num_questions = len(data["questions"])
    print(f"Dataset: {num_questions} questions")

    print(f"\nCollecting responses from {MASTER_URL}...")
    samples = build_samples(data)
    print(f"\nCollected {len(samples)}/{num_questions} samples")

    if not samples:
        print("No samples collected, exiting")
        return

    eval_dataset = EvaluationDataset(samples=samples)

    llm = LangchainLLMWrapper(ChatGroq(
        model=GROQ_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
    ))

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    print("\nRunning ragas evaluation...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )

    print("\n=== Results ===")
    print(results)

    results_dict = {m.name: float(results[m.name]) for m in metrics}
    results_dict["num_samples"] = len(samples)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    try:
        results_df = results.to_pandas()
        results_df.to_csv(RESULTS_PATH.with_suffix(".csv"), index=False)
        print(f"Per-sample results saved to {RESULTS_PATH.with_suffix('.csv')}")
    except Exception:
        pass


if __name__ == "__main__":
    run()
