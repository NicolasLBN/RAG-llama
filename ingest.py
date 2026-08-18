from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from rag import CHROMA_DIR, HEALTH_EMBED, ROOT, embed_texts, open_collection, wait_for

PDF_DIR = ROOT / "pdf"
MIN_CHARS = 300
MAX_CHARS = 500
OVERLAP = 50
BATCH_SIZE = 8


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?\:;])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    sentences = split_sentences(text) or [text]
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= MAX_CHARS:
            current = candidate
            continue
        if current:
            chunks.append(current)
            overlap = current[-OVERLAP:] if len(current) > OVERLAP else current
            current = f"{overlap} {sentence}".strip()
            if len(current) > MAX_CHARS:
                current = sentence[:MAX_CHARS]
        else:
            chunks.append(sentence[:MAX_CHARS])
            current = ""

    if current:
        chunks.append(current)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(merged[-1]) < MIN_CHARS and len(merged[-1]) + 1 + len(chunk) <= MAX_CHARS:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)
    return [chunk for chunk in merged if len(chunk) >= 40]


def extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    records = []
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk in chunk_text(page_text):
            records.append(
                {
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_index,
                }
            )
    return records


def chunk_id(record: dict) -> str:
    raw = f"{record['source']}|{record['page']}|{record['text']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def main() -> None:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Aucun PDF trouve dans {PDF_DIR}")

    print(f"Attente de llama-embed sur {HEALTH_EMBED} ...")
    wait_for(HEALTH_EMBED, "llama-embed")

    records: list[dict] = []
    for pdf in pdfs:
        chunks = extract_pdf_chunks(pdf)
        print(f"{pdf.name}: {len(chunks)} blocs")
        records.extend(chunks)

    if not records:
        raise SystemExit("Aucun texte extractible dans les PDF.")

    collection = open_collection(reset=True)
    total = 0
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        embeddings = embed_texts([item["text"] for item in batch])
        collection.add(
            ids=[chunk_id(item) for item in batch],
            documents=[item["text"] for item in batch],
            embeddings=embeddings,
            metadatas=[
                {"source": item["source"], "page": item["page"]}
                for item in batch
            ],
        )
        total += len(batch)
        print(f"Indexe {total}/{len(records)}")

    print(f"Termine. {len(records)} blocs dans {CHROMA_DIR}")


if __name__ == "__main__":
    main()
