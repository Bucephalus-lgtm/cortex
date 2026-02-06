import faiss
import pickle
import numpy as np

INDEX_FILE = "index/faiss.index"
META_FILE = "index/metadata.pkl"
TOP_K = 3


def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["vectorizer"]


def retrieve(query, index, chunks, vectorizer, k=TOP_K):
    query_vec = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vec, k)

    return [chunks[idx] for idx in indices[0]]


def generate_answer(query, retrieved_chunks):
    combined_text = " ".join(c["text"] for c in retrieved_chunks).lower()

    if "root cause" not in combined_text and "cause" not in combined_text:
        return "Answer not found in knowledge base."

    lines = combined_text.split(".")
    answer_lines = [
        l.strip()
        for l in lines
        if "root cause" in l or "involved" in l or "due to" in l
    ]

    # dedupe
    answer_lines = list(dict.fromkeys(answer_lines))

    if not answer_lines:
        return "Answer not found in knowledge base."

    return " ".join(answer_lines[:3])


if __name__ == "__main__":
    index, chunks, vectorizer = load_index()

    query = "Why did payment service timeout last quarter?"

    retrieved = retrieve(query, index, chunks, vectorizer)
    answer = generate_answer(query, retrieved)

    print("\n❓ Question:")
    print(query)

    print("\n📄 Retrieved Context:")
    for i, r in enumerate(retrieved, 1):
        print(f"\n--- Chunk {i} ---")
        print(r["text"][:300])

    print("\n✅ Answer:")
    print(answer)
