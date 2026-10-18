"""
Hybrid Search Retriever: Dense Vector Search (Qdrant) + Sparse Keyword Search (BM25)
Merged via Reciprocal Rank Fusion (RRF) for enterprise-grade retrieval precision.
Eliminates pure-vector blind spots (exact numbers, codes, dates, acronyms).
"""

import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
from langchain_core.documents import Document

try:
    import bm25s
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from app.rag.vectorstore import get_existing_vectorstore


class HybridRetriever:
    """
    Ensemble Retriever combining Dense Embeddings (Semantic)
    and BM25 (Exact Keyword) via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, k_dense: int = 4, k_sparse: int = 4, rrf_k: int = 60):
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.rrf_k = rrf_k
        self._bm25_retriever = None
        self._corpus_docs: List[Document] = []

    def sync_corpus(self, docs: List[Document]):
        """Build or refresh in-memory BM25 index with corpus documents."""
        if not HAS_BM25 or not docs:
            return

        self._corpus_docs = docs
        corpus_texts = [d.page_content for d in docs]

        try:
            # Tokenize corpus for bm25s
            corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            self._bm25_retriever = retriever
        except Exception as e:
            print(f"Warning: BM25 indexing failed, using dense only: {e}")
            self._bm25_retriever = None

    def _dense_search(self, query: str, vectorstore) -> List[Document]:
        """Perform dense vector similarity search via Qdrant/FAISS."""
        try:
            return vectorstore.similarity_search(query, k=self.k_dense)
        except Exception as e:
            print(f"Dense search warning: {e}")
            return []

    def _sparse_search(self, query: str) -> List[Document]:
        """Perform sparse keyword search via BM25."""
        if not self._bm25_retriever or not self._corpus_docs:
            return []

        try:
            query_tokens = bm25s.tokenize([query], stopwords="en")
            results, scores = self._bm25_retriever.retrieve(
                query_tokens, k=min(self.k_sparse, len(self._corpus_docs))
            )
            matched_docs = []
            for idx in results[0]:
                if 0 <= idx < len(self._corpus_docs):
                    matched_docs.append(self._corpus_docs[idx])
            return matched_docs
        except Exception as e:
            print(f"BM25 search warning: {e}")
            return []

    def retrieve(self, query: str, vectorstore=None, top_k: int = 4) -> List[Document]:
        """
        Execute Hybrid Search using Reciprocal Rank Fusion (RRF).
        RRF Score = sum( 1 / (rrf_k + rank_i) )
        """
        if vectorstore is None:
            vectorstore = get_existing_vectorstore()

        if vectorstore is None:
            return []

        dense_docs = self._dense_search(query, vectorstore)
        sparse_docs = self._sparse_search(query)

        # If BM25 is not available or has no docs, return dense
        if not sparse_docs:
            return dense_docs[:top_k]

        # Reciprocal Rank Fusion
        doc_scores: Dict[str, Tuple[Document, float]] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            key = doc.page_content.strip()
            score = 1.0 / (self.rrf_k + rank)
            doc_scores[key] = (doc, doc_scores.get(key, (doc, 0.0))[1] + score)

        for rank, doc in enumerate(sparse_docs, start=1):
            key = doc.page_content.strip()
            score = 1.0 / (self.rrf_k + rank)
            doc_scores[key] = (doc, doc_scores.get(key, (doc, 0.0))[1] + score)

        # Sort by merged RRF score descending
        ranked_items = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_items[:top_k]]


# Global singleton instance
global_hybrid_retriever = HybridRetriever(k_dense=4, k_sparse=4, rrf_k=60)
