import json
import os
import sys
from dotenv import load_dotenv

# DeepEval imports
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from retrieval import RAGService, CHAT_MODEL


# 1. Custom Gemini Model for DeepEval
class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self, model_name=CHAT_MODEL):
        self.model_name = model_name
        self.model = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        if isinstance(response.content, str):
            return response.content
        if isinstance(response.content, list):
            return "".join(
                b if isinstance(b, str) else b.get("text", "")
                for b in response.content
            )
        return str(response.content)

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        if isinstance(response.content, str):
            return response.content
        if isinstance(response.content, list):
            return "".join(
                b if isinstance(b, str) else b.get("text", "")
                for b in response.content
            )
        return str(response.content)

    def get_model_name(self):
        return self.model_name


def main():
    # 2. Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # 3. Initialize RAG & Judge
    print("Loading RAG Service...")
    rag = RAGService(index_dir=os.path.join(PROJECT_ROOT, "faiss_index"))
    rag.load()
    judge = GeminiJudge()

    # Define DeepEval metrics with pass/fail thresholds
    faithfulness_metric = FaithfulnessMetric(threshold=0.8, model=judge)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    recall_metric = ContextualRecallMetric(threshold=0.8, model=judge)

    print(f"\nRunning DeepEval on {len(test_cases)} test cases...\n")

    for i, item in enumerate(test_cases, 1):
        q = item["question"]
        ref = item["reference"]

        # Run your pipeline
        docs = rag.retrieve(q)
        contexts = [doc.page_content for doc in docs]
        actual_ans, _ = rag.answer(q)

        # Build DeepEval test case
        case = LLMTestCase(
            input=q,
            actual_output=actual_ans,
            expected_output=ref,
            retrieval_context=contexts,
        )

        print("=" * 70)
        print(f"TEST CASE {i}: {q}")
        print("=" * 70)
        print(f"Generated Output: {actual_ans}\n")

        # Measure each metric
        for metric, name in [
            (faithfulness_metric, "Faithfulness"),
            (relevancy_metric, "Answer Relevancy"),
            (recall_metric, "Contextual Recall"),
        ]:
            metric.measure(case)
            status = "✅ PASS" if metric.is_successful() else "❌ FAIL"
            print(f"  [{status}] {name}: Score = {metric.score:.2f} (Threshold: {metric.threshold})")
            if hasattr(metric, "reason") and metric.reason:
                print(f"         Reasoning: {metric.reason}")
        print()


if __name__ == "__main__":
    main()
