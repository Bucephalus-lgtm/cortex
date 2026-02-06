import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "index/faiss_embeddings.index"
META_FILE = "index/embeddings_metadata.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    chunks = pickle.load(f)

query = "Why did the payment service become slow?"

q_emb = model.encode([query]).astype("float32")
D, I = index.search(q_emb, 3)

print("\nTop semantic matches:\n")
for idx in I[0]:
    print("-----")
    print(chunks[idx]["text"][:400])
