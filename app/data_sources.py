import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
from app.retrieval import BaseDataSource, BaseRetriever

class FAISSDataSource(BaseDataSource):
    """Local FAISS Index as a data source"""
    def __init__(self, index_path: str, metadata_path: str, dict_key: str = "chunks"):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            self.chunks = data[dict_key]
            self.vectorizer = data.get("vectorizer")
        else:
            self.chunks = data

    def search(self, query_vec: Any, top_k: int) -> List[Dict[str, Any]]:
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.array(query_vec).astype("float32")
        
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            c = self.chunks[idx].copy()
            c["score"] = float(dist)
            results.append(c)
        return results

class TFIDFRetriever(BaseRetriever):
    def __init__(self, data_source: FAISSDataSource):
        self.data_source = data_source
        self.vectorizer = data_source.vectorizer

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_vec = self.vectorizer.transform([query]).toarray()
        results = self.data_source.search(q_vec, top_k)
        for r in results:
            r["score_type"] = "tfidf"
        return results

class SemanticRetriever(BaseRetriever):
    def __init__(self, data_source: FAISSDataSource, embed_model: Any):
        self.data_source = data_source
        self.embed_model = embed_model

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_emb = self.embed_model.encode([query]).astype("float32")
        results = self.data_source.search(q_emb, top_k)
        for r in results:
            r["score_type"] = "semantic"
        return results
