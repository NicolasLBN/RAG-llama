from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pymupdf

from rag import CHROMA_DIR, HEALTH_EMBED, ROOT, embed_texts, open_collection, wait_for

PDF_DIR = ROOT / "pdf"
MIN_CHARS = 50
MAX_CHARS = 900
BATCH_SIZE = 8

SECTION_NUM_ONLY = re.compile(r"^\d+(?:\.\d+){1,3}\.?$")
SECTION_WITH_TITLE = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s+(\S.{2,90})$")
CHAPTER = re.compile(r"^(\d+)\s+([A-ZÉÈÀÂÊÎÔÛÄËÏÖÜ].{3,80})$")
SPECIAL_HEADING = re.compile(r"^(NOTE|DANGER|ATTENTION|IMPORTANT|WARNING)[\s!:.]*$", re.I)
TOC_DOTS = re.compile(r"\.{4,}")
PAGE_MARK = re.compile(r"^\d+\s*/\s*\d+$")


def detect_lang(pdf_path: Path) -> str:
    name = pdf_path.name.lower()
    if "-fr-" in name:
        return "fr"
    if "-en-" in name:
        return "en"
    return "unknown"


def detect_doc_type(pdf_path: Path) -> str:
    name = pdf_path.name.lower()
    if "guide rapide" in name:
        return "quick_guide"
    if "profiling" in name:
        return "profiling"
    if "entretien" in name or "user guide" in name:
        return "manual"
    return "other"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise(line: str) -> bool:
    if not line:
        return True
    if PAGE_MARK.match(line):
        return True
    if TOC_DOTS.search(line):
        return True
    if re.fullmatch(r"[\d\s]+", line):
        return True
    return False


def is_heading(line: str) -> bool:
    if SPECIAL_HEADING.match(line):
        return True
    if SECTION_WITH_TITLE.match(line):
        return True
    if CHAPTER.match(line) and len(line) <= 80:
        return True
    return False


def merge_split_headings(lines: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if SECTION_NUM_ONLY.match(line) and index + 1 < len(lines):
            nxt = lines[index + 1]
            if not is_noise(nxt) and not SECTION_NUM_ONLY.match(nxt):
                merged.append(f"{line} {nxt}")
                index += 2
                continue
        merged.append(line)
        index += 1
    return merged


def split_sections(text: str) -> list[tuple[str, str]]:
    heading = ""
    sections: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = " ".join(buf).strip()
        buf.clear()
        if heading or body:
            sections.append((heading, body))

    lines = merge_split_headings([line.strip() for line in text.splitlines()])
    for line in lines:
        if is_noise(line):
            continue
        if is_heading(line):
            flush()
            heading = line
            continue
        buf.append(line)
    flush()
    return [(h, b) for h, b in sections if (h + " " + b).strip()]


def section_to_chunks(heading: str, body: str) -> list[str]:
    text = f"{heading}\n{body}".strip() if heading else body.strip()
    if len(text) <= MAX_CHARS:
        return [text] if len(text) >= MIN_CHARS else []
    chunks = []
    words = text.split()
    current: list[str] = []
    for word in words:
        current.append(word)
        if len(" ".join(current)) >= MAX_CHARS - 80:
            chunks.append(" ".join(current))
            current = current[-12:]
    if current:
        tail = " ".join(current)
        if chunks and len(tail) < MIN_CHARS:
            chunks[-1] = f"{chunks[-1]} {tail}".strip()
        else:
            chunks.append(tail)
    return [chunk for chunk in chunks if len(chunk) >= MIN_CHARS]


def extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    lang = detect_lang(pdf_path)
    doc_type = detect_doc_type(pdf_path)
    records = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_text = normalize_text(page.get_text("text") or "")
            for heading, body in split_sections(page_text):
                for chunk in section_to_chunks(heading, body):
                    records.append(
                        {
                            "text": chunk,
                            "source": pdf_path.name,
                            "page": page_index,
                            "lang": lang,
                            "doc_type": doc_type,
                            "heading": heading[:200],
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
                    "doc_type": item["doc_type"],
                    "heading": item["heading"],
                }
                for item in batch
            ],
        )
        total += len(batch)
        print(f"Indexe {total}/{len(records)}")

    print(f"Termine. {len(records)} blocs dans {CHROMA_DIR}")


if __name__ == "__main__":
    main()
