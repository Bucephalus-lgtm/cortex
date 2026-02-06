import pdfplumber
import os

RAW_DIR = "docs/raw_pdfs"
OUT_DIR = "docs/clean_text"

os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if len(line) < 5:
            continue
        if line.isdigit():
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


for pdf_file in os.listdir(RAW_DIR):
    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(RAW_DIR, pdf_file)
    out_path = os.path.join(OUT_DIR, pdf_file.replace(".pdf", ".txt"))

    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    full_text = clean_text(full_text)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Extracted & cleaned: {pdf_file}")
