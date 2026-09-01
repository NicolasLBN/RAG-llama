from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

from rag import embed_texts, open_collection, wait_for
from settings import (
    CFG,
    CHROMA_DIR,
    HEALTH_EMBED,
    IMAGE_CAPTIONS_PATH,
    MANIFEST_PATH,
    OCR_REPORT_PATH,
    PDF_DIR,
    load_json,
    setup_logging,
)

LOGGER = logging.getLogger("rag.ingest")

MIN_CHARS = int(CFG.get("chunking", "min_chars", default=160))
MAX_CHARS = int(CFG.get("chunking", "max_chars", default=900))
OVERLAP_CHARS = int(CFG.get("chunking", "overlap_chars", default=140))
MIN_NATIVE_CHARS = int(CFG.get("chunking", "min_native_chars", default=80))
MIN_IMAGE_AREA_RATIO = float(CFG.get("chunking", "min_image_area_ratio", default=0.04))
TOC_DOT_LINES = int(CFG.get("chunking", "toc_dot_lines", default=8))
BATCH_SIZE = int(CFG.get("ingest", "batch_size", default=8))
OCR_DPI = int(CFG.get("ingest", "ocr_dpi", default=200))
DOCUMENT_PREFIX = str(CFG.get("embedding", "document_prefix", default=""))

SECTION_NUM_ONLY = re.compile(r"^\d+(?:\.\d+){1,3}\.?$")
SECTION_WITH_TITLE = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s+(\S.{1,120})$")
CHAPTER = re.compile(r"^(\d+)\s+([A-ZÉÈÀÂÊÎÔÛÄËÏÖÜÇ].{2,90})$")
SPECIAL_HEADING = re.compile(
    r"^(NOTE|DANGER|ATTENTION|IMPORTANT|WARNING)[\s!:.]*$",
    re.I,
)
TOC_DOTS = re.compile(r"\.{4,}")
PAGE_MARK = re.compile(r"^\d+\s*/\s*\d+$")
CAPTION = re.compile(
    r"^(?:fig(?:ure|\.)?|schema|schéma|illustration|photo)\b",
    re.I,
)
FILENAME_META = re.compile(
    r"^(?P<id>[\d.]+)-(?P<lang>fr|en)-v(?P<version>[\d.]+)",
    re.I,
)


@dataclass
class PageInfo:
    page: int
    text: str
    native_chars: int
    needs_ocr: bool
    ocr_used: bool
    image_ids: list[str]
    captions: list[str]


@dataclass
class Line:
    text: str
    page: int
    is_heading: bool
    heading_level: int
    heading_key: str
    image_ids: list[str] = field(default_factory=list)


def detect_lang(pdf_path: Path) -> str:
    name = pdf_path.name.lower()
    match = FILENAME_META.match(pdf_path.name)
    if match:
        return match.group("lang").lower()
    if "-fr-" in name or name.endswith("-fr.pdf"):
        return "fr"
    if "-en-" in name or name.endswith("-en.pdf"):
        return "en"
    return "unknown"


def detect_doc_type(pdf_path: Path) -> str:
    name = pdf_path.name.lower()
    if "guide rapide" in name or "quick user" in name:
        return "quick_guide"
    if "profiling" in name:
        return "profiling"
    if "entretien" in name or "user guide" in name or "maintenance" in name:
        return "manual"
    return "other"


def detect_document_id(pdf_path: Path) -> str:
    match = FILENAME_META.match(pdf_path.name)
    if match:
        return match.group("id")
    return pdf_path.stem


def detect_document_version(pdf_path: Path) -> str:
    match = FILENAME_META.match(pdf_path.name)
    if match:
        return f"v{match.group('version')}"
    return "unknown"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if re.fullmatch(r"[\d\s./]+", line):
        return True
    return False


def parse_heading(line: str) -> tuple[int, str] | None:
    if SPECIAL_HEADING.match(line):
        return 99, line.split()[0].upper() if line[:6].upper() in {"NOTE", "DANGER"} else line[:40]
    numbered = SECTION_WITH_TITLE.match(line)
    if numbered:
        number, title = numbered.group(1), numbered.group(2).strip()
        title = re.split(r"\s+a\)\s+", title, maxsplit=1)[0]
        if len(title) > 80:
            title = title[:80].rsplit(" ", 1)[0]
        return number.count(".") + 1, f"{number} {title}".strip()
    chapter = CHAPTER.match(line)
    if chapter and len(line) <= 90:
        return 1, line
    return None


def is_heading(line: str) -> bool:
    return parse_heading(line) is not None


def merge_wrapped_headings(lines: list[str]) -> list[str]:
    wrap_end = re.compile(
        r"(?:de|du|des|et|le|la|les|a|à|au|aux|un|une|d'|l'|the|and|of)$",
        re.I,
    )
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if parse_heading(line) and index + 1 < len(lines):
            nxt = lines[index + 1]
            continuation = (
                nxt
                and not is_heading(nxt)
                and not is_noise(nxt)
                and not nxt[:1].isdigit()
                and (wrap_end.search(line) or nxt[:1].islower())
            )
            if continuation:
                merged.append(f"{line} {nxt}")
                index += 2
                continue
        merged.append(line)
        index += 1
    return merged


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


def page_area(page: pymupdf.Page) -> float:
    rect = page.rect
    return max(float(rect.width * rect.height), 1.0)


def significant_images(page: pymupdf.Page, page_num: int, document_id: str) -> list[str]:
    image_ids: list[str] = []
    area = page_area(page)
    try:
        raw_images = page.get_images(full=True)
    except Exception as exc:
        LOGGER.warning("Lecture des images impossible p.%s: %s", page_num, exc)
        return []

    for index, image in enumerate(raw_images, start=1):
        xref = image[0]
        ratio = MIN_IMAGE_AREA_RATIO
        try:
            rects = page.get_image_rects(xref)
            if rects:
                ratio = max(abs(rect.width * rect.height) for rect in rects) / area
        except Exception:
            try:
                info = page.parent.extract_image(xref)
                width, height = info.get("width", 0), info.get("height", 0)
                ratio = (width * height) / max(area, 1.0)
            except Exception:
                ratio = MIN_IMAGE_AREA_RATIO
        if ratio < MIN_IMAGE_AREA_RATIO:
            continue
        image_ids.append(f"{document_id}-p{page_num}-img{index}")
    return image_ids


def extract_captions(text: str) -> list[str]:
    captions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if CAPTION.match(stripped) and 8 <= len(stripped) <= 180:
            captions.append(stripped)
    return captions


def try_ocr_page(page: pymupdf.Page, page_num: int) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        LOGGER.warning(
            "OCR demandé mais pytesseract/Pillow absents (PC de préparation uniquement)"
        )
        return ""
    try:
        pixmap = page.get_pixmap(dpi=OCR_DPI)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        text = pytesseract.image_to_string(image, lang="fra+eng")
        return normalize_text(text)
    except Exception as exc:
        LOGGER.error("OCR en échec page %s: %s", page_num, exc)
        return ""


def is_toc_page(text: str) -> bool:
    dotted = sum(1 for line in text.splitlines() if TOC_DOTS.search(line))
    return dotted >= TOC_DOT_LINES


def extract_page(
    page: pymupdf.Page,
    page_num: int,
    document_id: str,
    enable_ocr: bool,
) -> PageInfo:
    native = normalize_text(page.get_text("text") or "")
    native_chars = len(re.sub(r"\s+", "", native))
    image_ids = significant_images(page, page_num, document_id)
    large_visual = len(image_ids) > 0 and native_chars < MIN_NATIVE_CHARS
    needs_ocr = large_visual or (native_chars < MIN_NATIVE_CHARS and not native.strip())
    ocr_text = ""
    ocr_used = False
    if needs_ocr and enable_ocr:
        ocr_text = try_ocr_page(page, page_num)
        ocr_used = bool(ocr_text)
    text = ocr_text if ocr_used else native
    return PageInfo(
        page=page_num,
        text=text,
        native_chars=native_chars,
        needs_ocr=needs_ocr,
        ocr_used=ocr_used,
        image_ids=image_ids,
        captions=extract_captions(text),
    )


def page_lines(info: PageInfo) -> list[Line]:
    if is_toc_page(info.text):
        return []
    raw = merge_wrapped_headings(
        merge_split_headings([line.strip() for line in info.text.splitlines()])
    )
    lines: list[Line] = []
    for item in raw:
        if is_noise(item):
            continue
        heading = parse_heading(item)
        if heading:
            level, label = heading
            lines.append(
                Line(
                    text=label,
                    page=info.page,
                    is_heading=True,
                    heading_level=level,
                    heading_key=label,
                    image_ids=info.image_ids,
                )
            )
            continue
        lines.append(
            Line(
                text=item,
                page=info.page,
                is_heading=False,
                heading_level=0,
                heading_key="",
                image_ids=info.image_ids,
            )
        )
    return lines


def section_path_from_stack(stack: list[str]) -> str:
    return " > ".join(stack)


def collect_sections(lines: list[Line]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    stack: list[str] = []
    levels: list[int] = []
    heading = ""
    path = ""
    body: list[str] = []
    pages: list[int] = []
    image_ids: list[str] = []

    def flush() -> None:
        text = " ".join(body).strip()
        if not heading and not text:
            return
        sections.append(
            {
                "heading": heading,
                "section_path": path or heading,
                "body": text,
                "start_page": min(pages) if pages else 0,
                "end_page": max(pages) if pages else 0,
                "image_ids": list(dict.fromkeys(image_ids)),
            }
        )

    for line in lines:
        if line.is_heading:
            flush()
            body, pages, image_ids = [], [], []
            if line.heading_level == 99:
                heading = line.text
                path = section_path_from_stack(stack + [line.text])
            else:
                while levels and levels[-1] >= line.heading_level:
                    levels.pop()
                    stack.pop()
                stack.append(line.text)
                levels.append(line.heading_level)
                heading = line.text
                path = section_path_from_stack(stack)
            pages.append(line.page)
            image_ids.extend(line.image_ids)
            continue
        body.append(line.text)
        pages.append(line.page)
        image_ids.extend(line.image_ids)
    flush()
    return sections


def split_with_overlap(text: str, prefix: str) -> list[str]:
    usable_max = max(MIN_CHARS, MAX_CHARS - len(prefix))
    if len(text) <= usable_max:
        chunk = f"{prefix}{text}".strip()
        return [chunk] if len(chunk) >= MIN_CHARS or prefix else []

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def overlap_tail(parts: list[str]) -> list[str]:
        kept: list[str] = []
        size = 0
        for word in reversed(parts):
            extra = len(word) + (1 if kept else 0)
            if kept and size + extra > OVERLAP_CHARS:
                break
            kept.append(word)
            size += extra
        return list(reversed(kept))

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > usable_max:
            chunks.append(f"{prefix}{' '.join(current)}".strip())
            current = overlap_tail(current)
            current_len = len(" ".join(current))
        current.append(word)
        current_len += extra

    if current:
        tail = f"{prefix}{' '.join(current)}".strip()
        if chunks and len(tail) < MIN_CHARS:
            chunks[-1] = f"{chunks[-1]} {' '.join(current)}".strip()
        else:
            chunks.append(tail)
    return [chunk for chunk in chunks if len(chunk) >= MIN_CHARS]


def _keep_short(text: str) -> bool:
    return bool(SPECIAL_HEADING.match(text.split("\n", 1)[0].strip()))


def section_to_chunks(section: dict[str, Any]) -> list[str]:
    heading = section["heading"]
    path = section["section_path"]
    body = section["body"]
    prefix_parts = []
    if path:
        prefix_parts.append(path)
    if heading and heading not in path:
        prefix_parts.append(heading)
    prefix = ("\n".join(prefix_parts) + "\n") if prefix_parts else ""
    text = body.strip()
    if not text:
        combined = prefix.strip()
        return [combined] if combined and (len(combined) >= MIN_CHARS or _keep_short(combined)) else []
    chunks = split_with_overlap(text, prefix)
    if not chunks:
        combined = f"{prefix}{text}".strip()
        if combined and (len(combined) >= MIN_CHARS or _keep_short(heading) or _keep_short(path)):
            return [combined]
    return chunks


def load_image_captions() -> dict[str, str]:
    data = load_json(IMAGE_CAPTIONS_PATH, {})
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def extract_pdf_chunks(pdf_path: Path, enable_ocr: bool) -> tuple[list[dict], list[dict]]:
    lang = detect_lang(pdf_path)
    doc_type = detect_doc_type(pdf_path)
    document_id = detect_document_id(pdf_path)
    document_version = detect_document_version(pdf_path)
    document_hash = file_sha256(pdf_path)
    captions_map = load_image_captions()
    ocr_pages: list[dict] = []
    records: list[dict] = []

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"PDF illisible ou corrompu: {pdf_path.name}: {exc}") from exc

    with document:
        page_infos: list[PageInfo] = []
        lines: list[Line] = []
        for page_index, page in enumerate(document, start=1):
            info = extract_page(page, page_index, document_id, enable_ocr)
            page_infos.append(info)
            if info.needs_ocr:
                ocr_pages.append(
                    {
                        "source": pdf_path.name,
                        "page": page_index,
                        "native_chars": info.native_chars,
                        "ocr_used": info.ocr_used,
                        "image_count": len(info.image_ids),
                    }
                )
            lines.extend(page_lines(info))

    if not any(info.native_chars or info.ocr_used for info in page_infos):
        LOGGER.warning("PDF sans texte extractible: %s", pdf_path.name)

    for section_index, section in enumerate(collect_sections(lines)):
        extra_bits = [
            captions_map[image_id]
            for image_id in section["image_ids"]
            if image_id in captions_map
        ]
        if extra_bits:
            section["body"] = f"{section['body']} {' '.join(dict.fromkeys(extra_bits))}".strip()

        chunks = section_to_chunks(section)
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "text": chunk,
                    "document_id": document_id,
                    "document_hash": document_hash,
                    "source": pdf_path.name,
                    "page": section["start_page"],
                    "start_page": section["start_page"],
                    "end_page": section["end_page"],
                    "lang": lang,
                    "doc_type": doc_type,
                    "heading": section["heading"][:200],
                    "section_path": section["section_path"][:300],
                    "chunk_index": chunk_index,
                    "image_ids": section["image_ids"],
                    "document_version": document_version,
                    "section_index": section_index,
                }
            )
    return records, ocr_pages


def chunk_id(record: dict) -> str:
    raw = (
        f"{record['source']}|{record['start_page']}|{record['end_page']}|"
        f"{record['chunk_index']}|{record['heading']}|{record['text']}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def chroma_metadata(record: dict) -> dict[str, Any]:
    return {
        "document_id": record["document_id"],
        "document_hash": record["document_hash"],
        "source": record["source"],
        "page": int(record["page"] or 0),
        "start_page": int(record["start_page"] or 0),
        "end_page": int(record["end_page"] or 0),
        "lang": record["lang"],
        "doc_type": record["doc_type"],
        "heading": record["heading"],
        "section_path": record["section_path"],
        "chunk_index": int(record["chunk_index"]),
        "image_ids": ",".join(record["image_ids"]),
        "document_version": record["document_version"],
    }


def load_manifest() -> dict[str, Any]:
    data = load_json(MANIFEST_PATH, {"documents": {}})
    if not isinstance(data, dict):
        return {"documents": {}}
    data.setdefault("documents", {})
    return data


def save_manifest(manifest: dict[str, Any]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def delete_source(collection, source: str) -> int:
    try:
        existing = collection.get(where={"source": source}, include=["metadatas"])
    except Exception as exc:
        LOGGER.error("Lecture Chroma impossible pour %s: %s", source, exc)
        return 0
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def index_records(collection, records: list[dict]) -> None:
    total = 0
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        texts = [
            f"{DOCUMENT_PREFIX}{item['text']}" if DOCUMENT_PREFIX else item["text"]
            for item in batch
        ]
        embeddings = embed_texts(texts)
        collection.add(
            ids=[chunk_id(item) for item in batch],
            documents=[item["text"] for item in batch],
            embeddings=embeddings,
            metadatas=[chroma_metadata(item) for item in batch],
        )
        total += len(batch)
        print(f"Indexe {total}/{len(records)}")


def write_ocr_report(pages: list[dict]) -> None:
    OCR_REPORT_PATH.write_text(
        json.dumps({"pages": pages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Rapport OCR écrit: %s (%s pages)", OCR_REPORT_PATH, len(pages))


def selected_pdfs(lang: str) -> list[Path]:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if lang != "all":
        pdfs = [pdf for pdf in pdfs if detect_lang(pdf) == lang]
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indexe les manuels PDF dans ChromaDB (PC de préparation)."
    )
    parser.add_argument("--lang", default="fr", choices=["fr", "en", "all"])
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reconstruit toute la base vectorielle",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="N'indexe que les PDF nouveaux ou modifies (defaut)",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Tente un OCR sur les pages scannees (PC de preparation)",
    )
    parser.add_argument(
        "--report-ocr",
        action="store_true",
        help="Liste les pages necessitant un OCR sans indexer",
    )
    args = parser.parse_args()
    setup_logging()

    if not args.reset and not args.update:
        args.update = True

    pdfs = selected_pdfs(args.lang)
    if not pdfs:
        raise SystemExit(f"Aucun PDF {args.lang} trouve dans {PDF_DIR}")

    all_ocr_pages: list[dict] = []
    prepared: list[tuple[Path, str, list[dict]]] = []
    for pdf in pdfs:
        try:
            chunks, ocr_pages = extract_pdf_chunks(pdf, enable_ocr=args.ocr)
        except RuntimeError as exc:
            LOGGER.error("%s", exc)
            print(f"ERREUR: {exc}")
            continue
        all_ocr_pages.extend(ocr_pages)
        digest = file_sha256(pdf)
        print(
            f"{pdf.name} [{detect_lang(pdf)}]: {len(chunks)} blocs, "
            f"{len(ocr_pages)} pages OCR, hash={digest[:12]}"
        )
        prepared.append((pdf, digest, chunks))

    if args.report_ocr:
        write_ocr_report(all_ocr_pages)
        print(f"{len(all_ocr_pages)} pages a OCR. Rapport: {OCR_REPORT_PATH}")
        return

    if args.ocr and all_ocr_pages:
        write_ocr_report(all_ocr_pages)

    print(f"Attente de llama-embed sur {HEALTH_EMBED} ...")
    wait_for(HEALTH_EMBED, "llama-embed")

    collection = open_collection(reset=args.reset)
    manifest = {"documents": {}} if args.reset else load_manifest()
    indexed_names = set()
    total_chunks = 0

    for pdf, digest, chunks in prepared:
        indexed_names.add(pdf.name)
        previous = manifest["documents"].get(pdf.name, {})
        if args.update and not args.reset and previous.get("hash") == digest:
            print(f"Inchange: {pdf.name}")
            continue
        if not chunks:
            LOGGER.warning("Aucun chunk pour %s", pdf.name)
            continue
        deleted = delete_source(collection, pdf.name)
        if deleted:
            print(f"Remplace {deleted} anciens blocs: {pdf.name}")
        index_records(collection, chunks)
        manifest["documents"][pdf.name] = {
            "hash": digest,
            "lang": detect_lang(pdf),
            "document_id": detect_document_id(pdf),
            "document_version": detect_document_version(pdf),
            "chunks": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        total_chunks += len(chunks)

    if args.update:
        stale = [
            name
            for name, meta in list(manifest["documents"].items())
            if name not in indexed_names
            and (args.lang == "all" or meta.get("lang") == args.lang)
        ]
        for name in stale:
            deleted = delete_source(collection, name)
            manifest["documents"].pop(name, None)
            print(f"Supprime {deleted} blocs orphelins: {name}")

    save_manifest(manifest)
    print(f"Termine. {collection.count()} blocs dans {CHROMA_DIR}")
    if all_ocr_pages:
        print(f"Pages necessitant OCR: {len(all_ocr_pages)} -> {OCR_REPORT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
