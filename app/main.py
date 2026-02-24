from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from cortex.retrieval.hybrid import HybridRetriever
from cortex.llm.providers import OpenAIProvider, GroqProvider, OllamaProvider, MockLLMProvider
from cortex.engine import CortexEngine

app = FastAPI(title="Cortex: Production Incident Copilot")

# Mount static files for the UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configuration
INDEX_FILE = "index/faiss.index"
META_FILE = "index/metadata.pkl"
EMBED_INDEX_FILE = "index/faiss_embeddings.index"
EMBED_META_FILE = "index/embeddings_metadata.pkl"

engine = None

class AskRequest(BaseModel):
    question: str

@app.on_event("startup")
def load_resources():
    global engine
    
    # Initialize Retriever
    try:
        retriever = HybridRetriever(
            index_file=INDEX_FILE,
            meta_file=META_FILE,
            embed_index_file=EMBED_INDEX_FILE,
            embed_meta_file=EMBED_META_FILE
        )
    except Exception as e:
        print(f"Error loading indexes: {e}")
        retriever = None

    # Determine LLM Provider based on available keys/config
    # 1. Groq (Best for speed + Zero local impact)
    if os.getenv("GROQ_API_KEY"):
        print("Using GroqProvider")
        llm_provider = GroqProvider()
    # 2. OpenAI
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAIProvider")
        llm_provider = OpenAIProvider()
    # 3. Ollama (Best for local offline - defaults to phi3 model)
    elif os.getenv("USE_OLLAMA") == "true":
        print("Using OllamaProvider")
        llm_provider = OllamaProvider(model=os.getenv("OLLAMA_MODEL", "phi3"))
    # 4. Fallback to Mock
    else:
        print("No LLM keys found. Using MockLLMProvider.")
        llm_provider = MockLLMProvider()

    # Initialize Engine
    if retriever:
        engine = CortexEngine(retriever=retriever, llm_provider=llm_provider)
    else:
        print("Engine NOT initialized due to missing retriever.")

@app.get("/")
def read_root():
    return FileResponse("app/static/index.html")

@app.post("/ask")
def ask(req: AskRequest):
    if not engine:
        return {
            "error": "Engine not initialized. Please check if indexes are built.",
            "suggestion": "Run scripts/build_index.py and scripts/build_embedding_index.py"
        }
    
    return engine.ask(req.question)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine_ready": engine is not None,
        "llm_provider": type(engine.llm_provider).__name__ if engine else None
    }
