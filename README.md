# Cortex 🧠🚨

Cortex is a production incident knowledge copilot that helps engineers quickly understand **why incidents happened** by querying historical **RCAs, runbooks, and operational documents**.

It provides **grounded, source-backed answers** using a Retrieval-Augmented Generation (RAG) style pipeline — without hallucinations and without relying on paid LLM APIs.

---

## ✨ What Cortex Does

- 🔍 Answers production incident questions (timeouts, Kafka lag, DB exhaustion, CPU spikes, Redis outages)
- 📚 Searches across historical RCAs and runbooks
- 🧠 Retrieves only relevant context using FAISS
- 🛑 Prevents hallucinations with strict grounding
- 🧾 Returns answers with **document-level source attribution**
- 💸 Fully offline, zero-cost setup

Example questions:
- *Why did the payment service timeout last quarter?*
- *How do we prevent cascading failures?*
- *Why did Kafka consumer lag occur?*
- *What happens if Redis goes down?*

---

## 🏗️ Architecture Overview

```

PDF Documents
↓
Text Extraction & Sanitization
↓
Chunking with Overlap
↓
TF-IDF Vectorization
↓
FAISS Vector Index
↓
FastAPI (/ask)
↓
Grounded Answer + Sources

```

---

## 🧠 Design Principles

- **Grounded answers only**  
  Cortex never invents information. If an answer is not present in the knowledge base, it responds with:
  > `Answer not found in knowledge base.`

- **No hallucinations**  
  Answers are deterministically generated from retrieved chunks.

- **Offline-first**  
  No OpenAI, no HuggingFace inference APIs, no paid services.

- **Production-style ingestion**  
  Documents are sanitized and anonymized before indexing.

---

## 🛠️ Tech Stack

- **Python**
- **FastAPI** – API layer
- **FAISS** – Vector similarity search
- **Scikit-learn (TF-IDF)** – Offline embeddings
- **NLTK** – Tokenization
- **Uvicorn** – ASGI server

---

## 📂 Project Structure

```

cortex/
├── app/
│   └── main.py          # FastAPI application
├── scripts/
│   ├── extract_text.py  # PDF → clean text
│   ├── chunk_docs.py    # Chunking logic
│   ├── build_index.py   # FAISS index builder
│   └── retrieve.py      # Local retrieval test
├── docs/
│   ├── raw_pdfs/        # Sanitized PDFs
│   └── clean_text/      # Extracted text
├── index/
│   ├── faiss.index
│   └── metadata.pkl
└── README.md

````

---

## 🚀 Getting Started

### 1️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Build the index

```bash
python scripts/build_index.py
```

### 4️⃣ Run the API

```bash
uvicorn app.main:app --reload
```

---

## 🔌 API Usage

### Health Check

```http
GET /health
```

Response:

```json
{ "status": "ok" }
```

---

### Ask a Question

```http
POST /ask
Content-Type: application/json
```

Request:

```json
{
  "question": "Why did payment service timeout last quarter?"
}
```

Response:

```json
{
  "question": "Why did payment service timeout last quarter?",
  "answer": "The root cause involved a mix of traffic spikes, resource saturation, configuration limits, and slow downstream calls.",
  "sources": [
    "payment-timeout-rca.pdf",
    "cascading-failure-runbook.pdf",
    "db-connection-exhaustion-rca.pdf"
  ]
}
```

---

## 🧪 Hallucination Guardrail Example

Request:

```json
{
  "question": "What is the company refund policy?"
}
```

Response:

```json
{
  "answer": "Answer not found in knowledge base."
}
```

---

## 🔒 Data Safety & Anonymization

* All documents are anonymized
* No real company identifiers are stored
* Indexed content contains sanitized text only
* Raw PDFs are used only during ingestion

---

## 🎯 Use Cases

* On-call engineers debugging incidents
* New team members learning from past outages
* SREs analyzing recurring failure patterns
* Backend engineers preparing postmortems

---

## 📌 Future Improvements

* Replace TF-IDF with dense embeddings
* Add optional LLM summarization layer
* Dockerize for deployment
* UI dashboard for search & analytics
* Role-based access control

---

## 🧑‍💻 Author

Built by **Bhargab Nath**

---

## ⭐ Why Cortex Matters

Cortex demonstrates how to build **production-safe GenAI systems** that:

* prioritize correctness over fluency
* avoid hallucinations by design
* integrate cleanly with backend architectures

```