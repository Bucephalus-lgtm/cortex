import json
import os
import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNKS_FILE = "chunks.json"
INDEX_DIR = "index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

os.makedirs(INDEX_DIR, exist_ok=True)

# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

print(f"Vectorizing {len(texts)} chunks with TF-IDF...")

# TF-IDF (fully local, zero downloads)
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)

vectors = vectorizer.fit_transform(texts).toarray()

# Build FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# Save index
faiss.write_index(index, INDEX_FILE)

# Save metadata + vectorizer
with open(META_FILE, "wb") as f:
    pickle.dump({
        "chunks": chunks,
        "vectorizer": vectorizer
    }, f)

print("FAISS index built using TF-IDF.")
