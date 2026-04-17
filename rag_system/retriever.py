"""
retriever.py — Hybrid Retriever (Dense Vector + Sparse BM25)

Combines semantic similarity search (FAISS / ChromaDB) with lexical
BM25 retrieval.  Final scores are fused:

    final_score = alpha * vector_score + (1 - alpha) * bm25_score

Returns deduplicated, top-k ranked document chunks.
"""

import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# ─── BM25 Implementation ──────────────────────────────────────────────────────

class BM25:
    """
    Okapi BM25 scoring for sparse lexical retrieval.

    Lightweight, dependency-free implementation suitable for
    moderate-sized corpora (thousands of log chunks).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_dl = 0.0
        self.doc_freqs: Dict[str, int] = {}        # term -> number of docs containing it
        self.doc_lengths: List[int] = []
        self.term_freqs: List[Dict[str, int]] = []  # per-doc term frequencies
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[str]) -> "BM25":
        """Index a list of document strings."""
        self.corpus_size = len(documents)
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.term_freqs = []

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            tf: Dict[str, int] = defaultdict(int)
            seen = set()
            for t in tokens:
                tf[t] += 1
                if t not in seen:
                    self.doc_freqs[t] += 1
                    seen.add(t)
            self.term_freqs.append(dict(tf))

        self.avg_dl = sum(self.doc_lengths) / max(self.corpus_size, 1)
        self._compute_idf()
        return self

    def score(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_index, score) pairs sorted descending by BM25 score."""
        tokens = self._tokenize(query)
        scores: List[Tuple[int, float]] = []
        for idx in range(self.corpus_size):
            s = 0.0
            dl = self.doc_lengths[idx]
            tf_doc = self.term_freqs[idx]
            for t in tokens:
                if t not in tf_doc:
                    continue
                tf = tf_doc[t]
                idf = self.idf.get(t, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1e-6))
                s += idf * numerator / denominator
            if s > 0:
                scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _compute_idf(self):
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF with smoothing
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)


# ─── Scored document wrapper ──────────────────────────────────────────────────

@dataclass
class ScoredDocument:
    """A document with its retrieval scores."""
    document: Document
    vector_score: float = 0.0
    bm25_score: float = 0.0
    final_score: float = 0.0

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict:
        return self.document.metadata


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retriever combining dense vector search with BM25 sparse retrieval.

    Architecture:
        1. Dense path  — HuggingFace embeddings + ChromaDB similarity search
        2. Sparse path — BM25 over the same document corpus
        3. Score fusion — weighted combination with deduplication

    Parameters:
        alpha:      weight for vector scores (1-alpha for BM25)
        top_k:      number of documents to return after fusion
        model_name: HuggingFace embedding model
    """

    def __init__(
        self,
        alpha: float = 0.7,
        top_k: int = 12,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.alpha = alpha
        self.top_k = top_k
        self.model_name = model_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[Chroma] = None
        self.bm25: Optional[BM25] = None
        self.documents: List[Document] = []
        self._content_hash: str = ""

    # ── Index documents ────────────────────────────────────────────────────

    def index(self, documents: List[Document], collection_name: str = "telecom_logs") -> None:
        """
        Build both vector and BM25 indices from a list of Documents.

        Idempotent: re-indexes only if document content changes.
        """
        content_hash = hashlib.md5(
            "||".join(d.page_content for d in documents).encode()
        ).hexdigest()

        if content_hash == self._content_hash and self.vectorstore is not None:
            return  # already indexed

        self.documents = documents
        self._content_hash = content_hash

        # Dense index — ChromaDB
        import chromadb
        client = chromadb.Client()
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        self.vectorstore = Chroma.from_documents(
            documents, self.embeddings,
            collection_name=collection_name,
            client=client,
        )

        # Sparse index — BM25
        texts = [d.page_content for d in documents]
        self.bm25 = BM25().fit(texts)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[ScoredDocument]:
        """
        Hybrid retrieval: fuses vector + BM25 results.

        Returns a list of ScoredDocument objects sorted by final_score.
        """
        k = top_k or self.top_k
        if not self.vectorstore or not self.bm25:
            raise RuntimeError("Call index() before retrieve().")

        # 1. Dense retrieval with scores
        vector_results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )

        # 2. BM25 retrieval
        bm25_results = self.bm25.score(query, top_k=k)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max((s for _, s in bm25_results), default=1.0)
        if max_bm25 == 0:
            max_bm25 = 1.0

        # 3. Build score maps keyed by page_content (for deduplication)
        score_map: Dict[str, ScoredDocument] = {}

        for doc, vec_score in vector_results:
            key = doc.page_content
            if key not in score_map:
                score_map[key] = ScoredDocument(document=doc, vector_score=vec_score)
            else:
                score_map[key].vector_score = max(score_map[key].vector_score, vec_score)

        for idx, raw_score in bm25_results:
            doc = self.documents[idx]
            key = doc.page_content
            norm_score = raw_score / max_bm25
            if key not in score_map:
                score_map[key] = ScoredDocument(document=doc, bm25_score=norm_score)
            else:
                score_map[key].bm25_score = max(score_map[key].bm25_score, norm_score)

        # 4. Compute fused score
        for sd in score_map.values():
            sd.final_score = self.alpha * sd.vector_score + (1 - self.alpha) * sd.bm25_score

        # 5. Sort and return top-k
        ranked = sorted(score_map.values(), key=lambda x: x.final_score, reverse=True)
        return ranked[:k]

    def retrieve_vector_only(self, query: str, top_k: Optional[int] = None) -> List[ScoredDocument]:
        """Dense-only retrieval (for baseline comparison)."""
        k = top_k or self.top_k
        if not self.vectorstore:
            raise RuntimeError("Call index() before retrieve_vector_only().")

        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [
            ScoredDocument(document=doc, vector_score=score, final_score=score)
            for doc, score in results
        ]

    def retrieve_bm25_only(self, query: str, top_k: Optional[int] = None) -> List[ScoredDocument]:
        """BM25-only retrieval (for baseline comparison)."""
        k = top_k or self.top_k
        if not self.bm25:
            raise RuntimeError("Call index() before retrieve_bm25_only().")

        results = self.bm25.score(query, top_k=k)
        max_score = max((s for _, s in results), default=1.0) or 1.0
        return [
            ScoredDocument(
                document=self.documents[idx],
                bm25_score=raw / max_score,
                final_score=raw / max_score,
            )
            for idx, raw in results
        ]

    # ── Utility ────────────────────────────────────────────────────────────

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    @property
    def embedding_dim(self) -> int:
        return len(self.embeddings.embed_query("test"))

    def format_retrieved(self, scored_docs: List[ScoredDocument]) -> str:
        """Format retrieved documents for LLM context injection."""
        lines = []
        for sd in scored_docs:
            m = sd.metadata
            ts = m.get("timestamp", "")
            lines.append(
                f"[{m.get('severity', '?')}] [{m.get('source', '?')}:L{m.get('line', '?')}]"
                f" @{ts} {sd.content}"
            )
        return "\n".join(lines)
