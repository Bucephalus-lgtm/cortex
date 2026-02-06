import os
import json
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

TEXT_DIR = "docs/clean_text"
OUT_FILE = "chunks.json"

CHUNK_SIZE = 450      # tokens
OVERLAP = 70          # tokens

def chunk_text(tokens, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap

    return chunks


all_chunks = []

for file_name in os.listdir(TEXT_DIR):
    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(TEXT_DIR, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = word_tokenize(text)
    chunks = chunk_text(tokens, CHUNK_SIZE, OVERLAP)

    for idx, chunk in enumerate(chunks):
        all_chunks.append({
            "id": f"{file_name}_{idx}",
            "text": chunk,
            "source": file_name.replace(".txt", ".pdf")
        })

print(f"Total chunks created: {len(all_chunks)}")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)
