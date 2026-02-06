from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle

INDEX_FILE = "index/faiss.index"
META_FILE = "index/metadata.pkl"
TOP_K = 3

app = FastAPI(title="Production Knowledge Copilot")

# globals (loaded once)
index = None
chunks = None
vectorizer = None


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_resources():
    global index, chunks, vectorizer

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]
    vectorizer = data["vectorizer"]

    print("FAISS index and metadata loaded.")


def retrieve(query: str):
    query_vec = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vec, TOP_K)
    return [chunks[i] for i in indices[0]]


def generate_answer(retrieved_chunks):
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


@app.post("/ask")
def ask(req: AskRequest):
    retrieved = retrieve(req.question)
    answer = generate_answer(retrieved)

    return {
        "question": req.question,
        "answer": answer,
        "sources": [r["source"] for r in retrieved]
    }


@app.get("/health")
def health():
    return {"status": "ok"}
