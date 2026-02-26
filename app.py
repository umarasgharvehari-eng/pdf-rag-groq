import os, json
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

def read_pdf_pages(path: str):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append((i, page.extract_text() or ""))
    return pages

def chunk_text(text: str, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        start += (chunk_size - overlap)
    return chunks

pdf_files = [f for f in os.listdir() if f.lower().endswith(".pdf")]
assert pdf_files, "No PDFs found. Upload PDFs first."

chunks = []
meta = []

for pdf in pdf_files:
    pages = read_pdf_pages(pdf)
    for page_num, page_text in pages:
        for ci, c in enumerate(chunk_text(page_text, 1000, 200)):
            chunks.append(c)
            meta.append({"source": pdf, "page": page_num, "chunk_in_page": ci})

print("PDFs:", pdf_files)
print("Total chunks:", len(chunks))

# Embeddings (FREE)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True).astype("float32")

# FAISS
dim = emb.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(emb)

faiss.write_index(index, "index.faiss")
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump({"chunks": chunks, "meta": meta}, f, ensure_ascii=False)

print("Saved: index.faiss and chunks.json")
