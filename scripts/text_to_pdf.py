from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os
import textwrap

TEXT_DIR = "docs/clean_text"
PDF_DIR = "docs/raw_pdfs"

os.makedirs(PDF_DIR, exist_ok=True)

def create_pdf(text, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    x_margin = 40
    y_margin = 40
    y = height - y_margin

    for line in textwrap.wrap(text, 100):
        if y < y_margin:
            c.showPage()
            y = height - y_margin
        c.drawString(x_margin, y, line)
        y -= 14

    c.save()

for file in os.listdir(TEXT_DIR):
    if not file.endswith(".txt"):
        continue

    with open(os.path.join(TEXT_DIR, file), "r", encoding="utf-8") as f:
        text = f.read()

    pdf_name = file.replace(".txt", ".pdf")
    output_path = os.path.join(PDF_DIR, pdf_name)

    create_pdf(text, output_path)
    print(f"Updated PDF: {pdf_name}")
