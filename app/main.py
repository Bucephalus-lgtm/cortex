from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.data_sources import FAISSDataSource, TFIDFRetriever, SemanticRetriever
from app.retrieval import HybridRetriever
from app.llm import get_llm, truncate_context

INDEX_FILE = "index/faiss.index"
META_FILE = "index/metadata.pkl"
EMBED_INDEX_FILE = "index/faiss_embeddings.index"
EMBED_META_FILE = "index/embeddings_metadata.pkl"
TOP_K = 3

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app = FastAPI(title="Production Knowledge Copilot")

# Mount Static UI
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")

retriever_system = None
llm_provider = None

class AskRequest(BaseModel):
    question: str

@app.on_event("startup")
def load_resources():
    global retriever_system, llm_provider

    # Data Sources
    try:
        tfidf_ds = FAISSDataSource(INDEX_FILE, META_FILE, dict_key="chunks")
        embed_ds = FAISSDataSource(EMBED_INDEX_FILE, EMBED_META_FILE)
        
        # Models
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Retrievers
        tfidf_retriever = TFIDFRetriever(tfidf_ds)
        semantic_retriever = SemanticRetriever(embed_ds, embed_model)
        
        # Hybrid Retriever
        retriever_system = HybridRetriever([
            (tfidf_retriever, 0.6),
            (semantic_retriever, 0.4)
        ])
        print("Modular Retriever system loaded.")
    except Exception as e:
        print(f"Warning: Failed to load indexes. They may not exist yet. Error: {e}")
        retriever_system = None

    # LLM Initialization
    llm_provider = get_llm()

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
    if not retriever_system:
        return {
            "question": req.question,
            "answer": "System not initialized. Please build indexes first.",
            "sources": []
        }

    intent = detect_intent(req.question)
    retrieved = retriever_system.retrieve(req.question, top_k=TOP_K)

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
        
    # Context window management
    managed_context = truncate_context(extracted, max_tokens=3000)
    
    # LLM integration with prompt engineering
    answer = llm_provider.generate_answer(req.question, managed_context)

    return {
        "question": req.question,
        "answer": answer,
        "sources": list({c["source"] for c in retrieved})
    }

@app.get("/health")
def health():
    return {"status": "ok", "modular_backend": True}

