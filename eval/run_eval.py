import json
import os
import sys
import types
from dotenv import load_dotenv

# Compatibility shim: Ragas 0.4.3 has a legacy import for ChatVertexAI from langchain_community
# which was removed in langchain-community 0.3+. This dummy stub satisfies the import.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        _shim = types.ModuleType("langchain_community.chat_models.vertexai")
        _shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = _shim

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Add parent directory to sys.path so we can import from retrieval.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from retrieval import RAGService, CHAT_MODEL, EMBEDDING_MODEL


def main():
    # 1. Load the golden test dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Test dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # 2. Initialize your RAG service
    print("Loading RAG Service...")
    index_dir = os.path.join(PROJECT_ROOT, "faiss_index")
    rag = RAGService(index_dir=index_dir)
    rag.load()

    # 3. Collect outputs from the live RAG pipeline
    user_inputs = []
    responses = []
    retrieved_contexts = []
    references = []

    print(f"Running RAG pipeline for {len(test_cases)} questions...")
    for idx, item in enumerate(test_cases, 1):
        question = item["question"]
        ref = item["reference"]
        print(f" [{idx}/{len(test_cases)}] Querying: {question[:60]}...")

        # Retrieve documents
        docs = rag.retrieve(question)
        contexts = [doc.page_content for doc in docs]

        # Generate answer
        answer, _ = rag.answer(question)

        user_inputs.append(question)
        responses.append(answer)
        retrieved_contexts.append(contexts)
        references.append(ref)

    # 4. Prepare dataset in Ragas format
    eval_data = {
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": references,
    }
    dataset = Dataset.from_dict(eval_data)

    # 5. Configure Gemini as the Evaluator Judge
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    for metric in metrics:
        if hasattr(metric, "llm"):
            metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings

    # 6. Run evaluation
    print("\nEvaluating with RAGAS (using Gemini judge)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # 7. Print and save results
    df = results.to_pandas()
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    summary_cols = ["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available_cols = [c for c in summary_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    output_csv = os.path.join(os.path.dirname(__file__), "results.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved detailed results to: {output_csv}")

    # 8. CI/CD Quality Gate (Pass/Fail Thresholds)
    avg_faithfulness = df["faithfulness"].mean()
    avg_recall = df["context_recall"].mean()
    avg_relevancy = df["answer_relevancy"].mean()

    print("\n" + "=" * 60)
    print("QUALITY GATE CHECK")
    print("=" * 60)
    print(f"Avg Faithfulness:     {avg_faithfulness:.2f} (Target: >= 0.90)")
    print(f"Avg Context Recall:   {avg_recall:.2f} (Target: >= 0.85)")
    print(f"Avg Answer Relevancy: {avg_relevancy:.2f} (Target: >= 0.80)")

    failed = False
    if avg_faithfulness < 0.90:
        print("❌ FAILED: Faithfulness dropped below threshold!")
        failed = True
    if avg_recall < 0.85:
        print("❌ FAILED: Context Recall dropped below threshold!")
        failed = True
    if avg_relevancy < 0.80:
        print("❌ FAILED: Answer Relevancy dropped below threshold!")
        failed = True

    if not failed:
        print("✅ PASSED: All metrics meet quality thresholds!")
    else:
        print("\n⚠️  One or more quality gates failed. Review results.csv for details.")


if __name__ == "__main__":
    main()

