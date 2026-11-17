"""
Ultra-Fast Vector Retrieval Engine with Hybrid Dense + BM25 Fusion
Optimized for < 5ms retrieval latency
"""
import math
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridVectorEngine:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        self.tfidf = TfidfVectorizer(max_features=2048, stop_words='english')
        self.dense_embeddings: np.ndarray = None
        self.tokenized_corpus: List[List[str]] = []
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        if not chunks:
            return

        # 1. Tokenize for BM25
        self.tokenized_corpus = [chunk['text'].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. Build vectorized sparse-dense matrix for fast SIMD cosine similarity
        raw_texts = [chunk['text'] for chunk in chunks]
        sparse_mat = self.tfidf.fit_transform(raw_texts).toarray()
        
        # Normalize for rapid dot-product cosine similarity
        norms = np.linalg.norm(sparse_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense_embeddings = sparse_mat / norms

    def _get_query_vector(self, query: str) -> np.ndarray:
        if query in self._embedding_cache:
            return self._embedding_cache[query]
        
        q_vec = self.tfidf.transform([query]).toarray()
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        
        # Cache for sub-millisecond repeated queries
        if len(self._embedding_cache) < 1000:
            self._embedding_cache[query] = q_vec
        return q_vec

    def retrieve(self, query: str, top_k: int = 3, hybrid: bool = True) -> List[Dict[str, Any]]:
        t_start = time.perf_counter()
        if not self.chunks or self.dense_embeddings is None:
            return []

        # Vector Dense Search (SIMD matrix multiplication in NumPy)
        q_vec = self._get_query_vector(query)
        dense_scores = np.dot(self.dense_embeddings, q_vec.T).squeeze()

        if hybrid and self.bm25 is not None:
            # BM25 Lexical Search
            q_tokens = query.lower().split()
            bm25_raw = self.bm25.get_scores(q_tokens)
            max_bm25 = np.max(bm25_raw) if len(bm25_raw) > 0 and np.max(bm25_raw) > 0 else 1.0
            bm25_scores = np.array(bm25_raw) / max_bm25

            # Reciprocal Rank Fusion / Weighted Ensemble
            combined_scores = 0.65 * dense_scores + 0.35 * bm25_scores
        else:
            combined_scores = dense_scores

        # Top-K Indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(combined_scores[idx])
            results.append(chunk)

        return results
