from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pymupdf

from rag import CHROMA_DIR, HEALTH_EMBED, ROOT, embed_texts, open_collection, wait_for

PDF_DIR = ROOT / "pdf"
MIN_CHARS = 200
TARGET_MIN = 500
MAX_CHARS = 900
OVERLAP = 120
BATCH_SIZE = 8

HEADING_NUM = re.compile(r"^\d+(?:\.\d+){0,3}[\.\)]?\s+\S")


def detect_lang(pdf_path: Path) -> str:
    name = pdf_path.name.lower()
    if "-fr-" in name:
        return "fr"
    if "-en-" in name:
        return "en"
    return "unknown"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading(line: str) -> bool:
    line = line.strip()
    if len(line) < 4 or len(line) > 90:
        return False
    if HEADING_NUM.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) >= 0.75:
        return True
    return False


def split_paragraphs(text: str) -> list[tuple[str, str]]:
    lines = [line.strip() for line in text.splitlines()]
    heading = ""
    paragraphs: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = " ".join(buf).strip()
        buf.clear()
        if body:
            paragraphs.append((heading, body))

    for line in lines:
        if not line:
            flush()
            continue
        if is_heading(line):
            flush()
            heading = line
            continue
        buf.append(line)
    flush()
    return paragraphs


def pack_chunks(paragraphs: list[tuple[str, str]]) -> list[str]:
    chunks: list[str] = []
    current_heading = ""
    current = ""

    def with_heading(body: str, heading: str) -> str:
        body = body.strip()
        if heading and not body.startswith(heading):
            return f"{heading}\n{body}"
        return body

    for heading, para in paragraphs:
        if heading != current_heading and current:
            chunks.append(with_heading(current, current_heading))
            overlap = current[-OVERLAP:].strip() if len(current) > OVERLAP else ""
            current = overlap
            current_heading = heading
        elif heading:
            current_heading = heading

        candidate = f"{current} {para}".strip() if current else para
        if len(with_heading(candidate, current_heading)) <= MAX_CHARS:
            current = candidate
            continue

        if current:
            chunks.append(with_heading(current, current_heading))
            overlap = current[-OVERLAP:].strip() if len(current) > OVERLAP else ""
            current = f"{overlap} {para}".strip()
            if len(with_heading(current, current_heading)) > MAX_CHARS:
                current = para[:MAX_CHARS]
        else:
            chunks.append(with_heading(para[:MAX_CHARS], current_heading))
            current = ""

    if current:
        chunks.append(with_heading(current, current_heading))

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(merged[-1]) < TARGET_MIN and len(merged[-1]) + 1 + len(chunk) <= MAX_CHARS:
            merged[-1] = f"{merged[-1]}\n{chunk}".strip()
        else:
            merged.append(chunk)
    return [chunk for chunk in merged if len(chunk) >= MIN_CHARS]


def extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    lang = detect_lang(pdf_path)
    records = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_text = normalize_text(page.get_text("text") or "")
            for chunk in pack_chunks(split_paragraphs(page_text)):
                records.append(
                    {
                        "text": chunk,
                        "source": pdf_path.name,
                        "page": page_index,
                        "lang": lang,
                    }
                )
    return records


def chunk_id(record: dict) -> str:
    raw = f"{record['source']}|{record['page']}|{record['text']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexe les manuels PDF dans ChromaDB.")
    parser.add_argument(
        "--lang",
        default="fr",
        choices=["fr", "en", "all"],
        help="Langue des PDF a indexer (defaut: fr)",
    )
    args = parser.parse_args()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.lang != "all":
        pdfs = [pdf for pdf in pdfs if detect_lang(pdf) == args.lang]
    if not pdfs:
        raise SystemExit(f"Aucun PDF {args.lang} trouve dans {PDF_DIR}")

    print(f"Attente de llama-embed sur {HEALTH_EMBED} ...")
    wait_for(HEALTH_EMBED, "llama-embed")

    records: list[dict] = []
    for pdf in pdfs:
        chunks = extract_pdf_chunks(pdf)
        print(f"{pdf.name} [{detect_lang(pdf)}]: {len(chunks)} blocs")
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
                {
                    "source": item["source"],
                    "page": item["page"],
                    "lang": item["lang"],
                }
                for item in batch
            ],
        )
        total += len(batch)
        print(f"Indexe {total}/{len(records)}")

    print(f"Termine. {len(records)} blocs dans {CHROMA_DIR}")


if __name__ == "__main__":
    main()
