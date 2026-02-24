import json
from typing import List, Dict, Any
from cortex.base import DataSource

class LocalJSONDataSource(DataSource):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.file_path, "r") as f:
            return json.load(f)

class LocalRAGSource(DataSource):
    """Abstraction for the existing indexed data."""
    def __init__(self, meta_file: str):
        self.meta_file = meta_file

    def load_data(self) -> List[Dict[str, Any]]:
        import pickle
        with open(self.meta_file, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]
