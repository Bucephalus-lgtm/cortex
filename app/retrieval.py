import typing as t
from abc import ABC, abstractmethod

class BaseDataSource(ABC):
    @abstractmethod
    def search(self, query_vec: t.Any, top_k: int) -> t.List[t.Dict[str, t.Any]]:
        pass

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> t.List[t.Dict[str, t.Any]]:
        pass

class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers: t.List[t.Tuple[BaseRetriever, float]]):
        """
        :param retrievers: List of tuples (retriever_instance, weight)
        """
        self.retrievers = retrievers

    def retrieve(self, query: str, top_k: int = 3) -> t.List[t.Dict[str, t.Any]]:
        all_results = []
        for retriever, weight in self.retrievers:
            results = retriever.retrieve(query, top_k=top_k)
            for r in results:
                r["weight"] = weight
                all_results.append(r)
        
        if not all_results:
            return []
            
        # Normalize
        max_score = max(r.get("score", 1.0) for r in all_results) or 1.0
        
        final = []
        seen = set()
        
        for r in all_results:
            norm = r.get("score", 1.0) / max_score
            # The original logic used distance, so lower score = better
            # We assume lower norm is better. We weight it down using the weight factor.
            r["final_score"] = norm * r["weight"]
            
            h = hash(r["text"][:200])
            if h not in seen:
                seen.add(h)
                final.append(r)

        return sorted(final, key=lambda x: x["final_score"])[:top_k]
