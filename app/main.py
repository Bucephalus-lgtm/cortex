from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle

INDEX_FILE = "index/faiss.index"
META_FILE = "index/metadata.pkl"
TOP_K = 3

app = FastAPI(title="Production Knowledge Copilot")

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
    print("FAISS index loaded.")


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


@app.post("/ask")
def ask(req: AskRequest):
    intent = detect_intent(req.question)
    retrieved = retrieve(req.question)

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
