import os

from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

SOURCE_FILE = "video_monitoring_requirements.md"
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Headers to split on -> each chunk's metadata carries its section titles,
# so "60 days retention" is still tagged with "Storage & Retention" etc.
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def build_index() -> None:
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Step 1: split by markdown headers so section context is preserved.
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    header_chunks = header_splitter.split_text(markdown_text)

    # Step 2: further split any oversized sections so chunks stay focused.
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = char_splitter.split_documents(header_chunks)

    # Prepend the section path to each chunk's text so the embedding and
    # the LLM both see the heading context, not just the bullet content.
    for doc in docs:
        breadcrumb = " > ".join(v for v in doc.metadata.values() if v)
        if breadcrumb:
            doc.page_content = f"{breadcrumb}\n{doc.page_content}"

    print(f"Split into {len(docs)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"FAISS index saved to ./{INDEX_DIR}")


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY in your .env file first.")
    build_index()