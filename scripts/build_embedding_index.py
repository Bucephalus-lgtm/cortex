import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
INDEX_FILE = "index/faiss_embeddings.index"
META_FILE = "index/embeddings_metadata.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

def main():
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Embedding index built with {len(chunks)} chunks")

if __name__ == "__main__":
    main()
