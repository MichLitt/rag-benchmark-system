from collections import Counter

from src.retrieval.tokenize import simple_tokenize
from src.types import Document


class KeywordRetriever:
    """Simple lexical retriever used as a bootstrap baseline before FAISS/BM25."""

    def __init__(self, corpus: list[Document]) -> None:
        self._corpus = corpus
        self._doc_tokens = {doc.doc_id: Counter(simple_tokenize(doc.text)) for doc in corpus}

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        q_counter = Counter(simple_tokenize(query))
        scored: list[tuple[float, Document]] = []
        for doc in self._corpus:
            d_counter = self._doc_tokens[doc.doc_id]
            overlap = sum(min(q_counter[t], d_counter[t]) for t in q_counter)
            scored.append((float(overlap), doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0.0]
