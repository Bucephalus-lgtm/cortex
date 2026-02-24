from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class DataSource(ABC):
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        ...

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        ...

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        ...
