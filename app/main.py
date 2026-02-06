from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_FILE = "index/faiss.index"
META_FILE = "index/metadata.pkl"
TOP_K = 3

embed_index = None
embed_chunks = None
embed_model = None

app = FastAPI(title="Production Knowledge Copilot")

index = None
chunks = None
vectorizer = None


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_resources():
    global index, chunks, vectorizer
    global embed_index, embed_chunks, embed_model

    # TF-IDF index
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    vectorizer = data["vectorizer"]

    # Embedding index
    embed_index = faiss.read_index("index/faiss_embeddings.index")
    with open("index/embeddings_metadata.pkl", "rb") as f:
        embed_chunks = pickle.load(f)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("TF-IDF + Embedding indexes loaded.")


def retrieve(query: str):
    q_vec = vectorizer.transform([query]).toarray()
    _, indices = index.search(q_vec, TOP_K)
    return [chunks[i] for i in indices[0]]


def detect_intent(question: str) -> str:
    q = question.lower()
    if "why" in q or "cause" in q:
        return "root cause"
    if "debug" in q or "investigate" in q:
        return "debug"
    if "prevent" in q or "avoid" in q:
        return "prevention"
    return "general"


def extract_section(text: str, intent: str) -> str:
    t = text.lower()

    if intent == "root cause":
        keys = ["root cause", "involved", "caused by"]
    elif intent == "debug":
        keys = ["debugging", "investigate", "steps"]
    elif intent == "prevention":
        keys = ["prevention", "avoid", "mitigation"]
    else:
        return text[:400]

    for k in keys:
        idx = t.find(k)
        if idx != -1:
            return text[idx:idx + 600]

    return text[:400]


def dedupe(texts):
    seen = set()
    result = []
    for t in texts:
        h = hash(t[:200])
        if h not in seen:
            seen.add(h)
            result.append(t)
    return result

def retrieve(query: str):
    q_vec = vectorizer.transform([query]).toarray()
    distances, indices = index.search(q_vec, TOP_K)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        c = chunks[idx].copy()
        c["score"] = float(dist)      # lower = better
        c["score_type"] = "tfidf"
        results.append(c)

    return results

def retrieve_semantic(query: str, top_k=3):
    q_emb = embed_model.encode([query]).astype("float32")
    distances, indices = embed_index.search(q_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        c = embed_chunks[idx].copy()
        c["score"] = float(dist)      # lower = better
        c["score_type"] = "semantic"
        results.append(c)

    return results

def hybrid_retrieve(query: str):
    tfidf_results = retrieve(query)
    semantic_results = retrieve_semantic(query)

    combined = tfidf_results + semantic_results

    # normalize
    max_score = max(c["score"] for c in combined) or 1.0

    for c in combined:
        norm = c["score"] / max_score
        if c["score_type"] == "tfidf":
            c["final_score"] = 0.6 * norm
        else:
            c["final_score"] = 0.4 * norm

    # dedupe
    seen = set()
    final = []
    for c in combined:
        h = hash(c["text"][:200])
        if h not in seen:
            seen.add(h)
            final.append(c)

    return sorted(final, key=lambda x: x["final_score"])[:TOP_K]

@app.post("/ask")
def ask(req: AskRequest):
    intent = detect_intent(req.question)
    retrieved = hybrid_retrieve(req.question)

    extracted = [
        extract_section(c["text"], intent)
        for c in retrieved
    ]

    extracted = dedupe(extracted)

    if not extracted:
        return {
            "question": req.question,
            "answer": "Answer not found in knowledge base.",
            "sources": []
        }

    return {
        "question": req.question,
        "answer": " ".join(extracted),
        "sources": list({c["source"] for c in retrieved})
    }


@app.get("/health")
def health():
    return {"status": "ok"}
