import os
from dotenv import load_dotenv

load_dotenv()

from retrieval import RAGService
from langfuse import get_client

def test_tracing():
    print("Initializing Langfuse client...")
    langfuse = get_client()

    print("Checking Langfuse credentials...")
    if not langfuse.auth_check():
        print("❌ Langfuse authentication failed! Please verify LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST in your .env")
        return
    else:
        print("✅ Langfuse authentication successful!")

    print("\nLoading RAG Service...")
    rag = RAGService()
    rag.load()

    question = "What critical production points must have continuous video coverage?"
    print(f"\nAsking Question: '{question}'...")
    answer, sources = rag.answer(question)
    print(f"\nAnswer: {answer}\n")
    print(f"Sources: {sources}")

    print("\nFlushing traces to Langfuse...")
    langfuse.flush()
    print("🚀 Trace sent! Open your Langfuse Web Dashboard and navigate to the 'Traces' tab to view it.")

if __name__ == "__main__":
    test_tracing()
