import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from cortex.base import Retriever
from typing import List, Dict, Any

class HybridRetriever(Retriever):
    def __init__(self, index_file: str, meta_file: str, embed_index_file: str, embed_meta_file: str):
        self.index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.vectorizer = data["vectorizer"]

        self.embed_index = faiss.read_index(embed_index_file)
        with open(embed_meta_file, "rb") as f:
            self.embed_chunks = pickle.load(f)
        
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # Lexical search
        q_vec = self.vectorizer.transform([query]).toarray()
        distances, indices = self.index.search(q_vec, top_k)
        
        lexical_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                c = self.chunks[idx].copy()
                c["score"] = float(dist)
                c["score_type"] = "tfidf"
                lexical_results.append(c)

        # Semantic search
        q_emb = self.embed_model.encode([query]).astype("float32")
        distances, indices = self.embed_index.search(q_emb, top_k)
        
        semantic_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.embed_chunks):
                c = self.embed_chunks[idx].copy()
                c["score"] = float(dist)
                c["score_type"] = "semantic"
                semantic_results.append(c)

        combined = lexical_results + semantic_results
        
        if not combined:
            return []

        # Normalize and rank
        max_score = max(c["score"] for c in combined) or 1.0
        for c in combined:
            norm = c["score"] / max_score
            if c["score_type"] == "tfidf":
                c["final_score"] = 0.6 * norm
            else:
                c["final_score"] = 0.4 * norm

        # Deduplicate
        seen = set()
        final = []
        for c in combined:
            h = hash(c["text"][:200])
            if h not in seen:
                seen.add(h)
                final.append(c)
        
        return sorted(final, key=lambda x: x["final_score"])[:top_k]
