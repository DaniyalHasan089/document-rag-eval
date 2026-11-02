import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.1-flash-lite"
TOP_K = 1

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a video \
monitoring requirements document. Use ONLY the context below to answer. \
If the answer isn't in the context, say you don't know.

Things to keep in mind:
- Dont start with "According to the given context" or anything similar. Just answer the question directly.

Context:
{context}

Question: {question}

Answer:"""


class RAGService:
    """Loads the FAISS index once and answers questions against it."""

    def __init__(
        self,
        index_dir: str = INDEX_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        chat_model: str = CHAT_MODEL,
        top_k: int = TOP_K,
    ):
        self.index_dir = index_dir
        self.top_k = top_k
        self.vectorstore: FAISS | None = None
        self.llm: ChatGoogleGenerativeAI | None = None
        self._embedding_model = embedding_model
        self._chat_model = chat_model

    def load(self) -> None:
        if not os.path.isdir(self.index_dir):
            raise RuntimeError(
                f"FAISS index not found at '{self.index_dir}'. Run `python ingest.py` first."
            )
        embeddings = GoogleGenerativeAIEmbeddings(model=self._embedding_model)
        self.vectorstore = FAISS.load_local(
            self.index_dir, embeddings, allow_dangerous_deserialization=True
        )
        self.llm = ChatGoogleGenerativeAI(
            model=self._chat_model, 
            temperature=0.7,
            max_output_tokens=1024
     )

    def retrieve(self, question: str) -> list:
        if self.vectorstore is None:
            raise RuntimeError("RAGService.load() must be called before retrieving.")
        return self.vectorstore.similarity_search(question, k=self.top_k)

    def answer(self, question: str) -> tuple[str, list[str]]:
        docs = self.retrieve(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        response = self.llm.invoke(prompt)
        sources = sorted({doc.page_content.split("\n")[0] for doc in docs})
        return self._extract_text(response.content), sources

    @staticmethod
    def _extract_text(content) -> str:
        """Newer langchain-google-genai versions return content as a list of
        blocks (e.g. [{'type': 'text', 'text': '...'}]) instead of a plain
        string. Normalize either shape into a single string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return str(content)