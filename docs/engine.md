# ingest.py, rag.py, and evaluate.py

French version: [moteur.md](moteur.md).

These three files are the document core of the OPTIJET RAG. They do **not** load the GGUF models themselves. They talk to the Docker servers (`llama-embed` on port 8081, `llama-chat` on port 8080) and to the local store `chroma_db/`.

```
pdf/*.pdf
    │
    ▼
ingest.py  ── embeddings ──►  chroma_db  (text + vectors + metadata)
    │
    │                         rag.py
    │                           │
    │              retrieve → scores → Qwen
    │                           │
    ├───────────────────────────┼── ask.py / ui.py  (operator)
    └───────────────────────────┴── evaluate.py     (search quality)
```

Tunable values (chunk sizes, score weights, thresholds) live in `settings.py` and `config.json`.

---

## Are “vector” and “embedding” the same thing?

**Almost.** In this project they refer to the same object.

| Term | Meaning |
|---|---|
| **Vector** | A list of numbers, e.g. `[0.12, -0.45, 0.88, …]` (hundreds of dimensions). |
| **Embedding** | A vector that **represents the meaning of a piece of text**. |

The **embedding model** (Snowflake Arctic Embed on port 8081) **turns text into an embedding**. That embedding **is** a vector. We store it in Chroma and compare it with **cosine distance**: two texts with a similar meaning get vectors that sit close together, even if the words differ (“start the machine” vs “switch the unit on”).

Short usage in this repo:

- *embed a chunk* = send the paragraph to `:8081` and get its vector
- *vector search* = find the stored chunks whose embeddings are closest to the question’s embedding

The **raw text** is still kept. Qwen reads the text, not the numbers. The vector is only used to **find** the right paragraph.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation. Search the manuals first, then ask the LLM to write an answer **from those excerpts only**. |
| **LLM / Qwen** | The chat model on port 8080. It writes the operator-facing answer. It does not search the PDFs by itself. |
| **GGUF** | File format for a compressed (quantized) model that runs on CPU. |
| **Quantization (Q4_K_M)** | Smaller, faster model with less RAM (~2 GB for Qwen). Slightly less fluent than the full model. |
| **Chunk** | A short block of manual text (roughly 160–900 characters) stored as one searchable unit. |
| **Chunking** | Splitting a PDF into those blocks, following headings, with overlap so a sentence is not cut too harshly. |
| **Overlap** | Repeated words at the end of one chunk and the start of the next, so meaning is not lost at the cut. |
| **Metadata** | Extra fields on a chunk: PDF name, pages, heading, language, image ids… Used to show **sources** on the HMI. |
| **ChromaDB** | Local vector database. Holds each chunk’s **text + embedding + metadata**. No internet. |
| **Cosine distance** | How far two embeddings are. In this code, semantic score ≈ `1 - distance` (closer to 1 = more similar). |
| **Hybrid search** | Combine **semantic** (meaning), **lexical** (words/synonyms), and **exact** (codes like `E102`, `16 bar`). |
| **Recall@k** | Share of test questions where the expected passage is among the top *k* search hits. Measures search, not Qwen’s wording. |
| **OCR** | Reading text from a **scanned image** page. Optional, prep PC only (`--ocr`). Native PDFs do not need it. |
| **HMI** | The industrial operator screen (Debian, ~8 GB RAM). It runs search + Qwen, not PDF ingestion. |
| **Temperature 0.0** | Qwen stays deterministic: same excerpts → same wording. Better fidelity to the manual. |
| **Prompt** | The instructions + excerpts sent to Qwen. It must not use general knowledge. |
| **Hallucination** | The model invents steps that are not in the excerpts. Thresholds exist so Qwen is **not** called if nothing is relevant enough. |
| **Hash (SHA-256)** | Fingerprint of a PDF file. If it changes, `--update` reindexes that document only. |

---

## ingest.py — build the document memory

**Role.** Read the PDF manuals, split them into chunks, compute an embedding (vector) for each chunk, store everything in ChromaDB. This runs on a **prep PC**, not on the HMI.

**When.** New PDF, changed manual, or new chunking settings. `llama-embed` (:8081) must be up. Qwen is not required.

### Pipeline for one PDF

1. **Identity** — from the filename (`298.410-fr-v1.5.0 …`): `document_id`, language (`fr` / `en`), version, type (`manual`, `quick_guide`, `profiling`).
2. **SHA-256 hash** of the file — detect changes (`--update`).
3. **Pages** — native text via PyMuPDF. Pages with almost no text plus images are flagged for OCR (`ocr_needed.json`). Real OCR (`--ocr`) runs only if Tesseract is installed.
4. **Images** — ids `document-p{page}-img{n}` in metadata (`image_ids`). No multimodal model. Optional captions: `image_captions.json`.
5. **Structure** — numbered headings (1, 1.1, 7.1…), NOTE / DANGER / IMPORTANT. A section **may span several pages**.
6. **Chunking** — `MIN_CHARS` / `MAX_CHARS` / `OVERLAP_CHARS` (`config.json`). Each chunk repeats `section_path`.
7. **Embedding** — batches to `POST http://localhost:8081/v1/embeddings`.
8. **Storage** — Chroma (`chroma_db/`) plus `chroma_db/manifest.json`.

Typical chunk metadata: `source`, `page`, `start_page`, `end_page`, `lang`, `heading`, `section_path`, `chunk_index`, `image_ids`, `document_hash`, `document_version`.

### Commands

```powershell
python ingest.py --update --lang fr      # default: new or changed PDFs
python ingest.py --update --lang en
python ingest.py --update --lang all     # FR + EN in the same index
python ingest.py --reset --lang all      # wipe Chroma, reindex everything
python ingest.py --report-ocr --lang all # list scanned pages, do not index
python ingest.py --ocr --update --lang fr
```

`--reset` without `--lang all` rebuilds the collection and only puts back the requested language: the other language is dropped.

`ingest.py` imports `embed_texts` and `open_collection` from `rag.py`.

---

## rag.py — retrieve and draft the answer

**Role.** Library used by `ask.py`, `ui.py`, `ingest.py`, and `evaluate.py`. Running `python rag.py` does nothing useful by itself.

Two runtime phases:

1. **Search** in Chroma (fast).
2. **Generation** with Qwen if excerpts are relevant enough (slow, ~2 GB RAM).

### Public functions

| Function | Role |
|---|---|
| `embed_texts(texts)` | POST embeddings :8081 |
| `open_collection(reset=False)` | Open / recreate the Chroma collection `manuals` |
| `search_chunks(question, …)` | Hybrid search, list of excerpts (or `[]`) |
| `ask_chat(question, excerpts)` | Call Qwen, return `{answer, sources, confidence}` |
| `answer_question(question, lang=…)` | Full HMI pipeline |

### Pipeline for one question (`answer_question`)

1. **Language filter** — `lang=fr` / `en` / `all`. No silent fallback to the other language.
2. **`enrich_query`** — add synonyms (`synonyms.json`) to the embedding text, prefix `query: `.
3. **`retrieve`** — top `FETCH_K` cosine neighbours + exact recall for codes (E102, 16 bar, 7.1…).
4. **Scores** — exact + lexical + semantic, merged by `combine_scores()` (weights in `config.json`).
5. **`rerank`** — identity today (hook for a future model).
6. **`select_context`** — thresholds `MIN_SEMANTIC_SCORE` / `MIN_FINAL_SCORE`. If nothing passes: **Qwen is not called**. Otherwise keep excerpts from the **same section family** (do not mix “power on” with “cable laying”).
7. **`ask_chat`** — prompt depends on question type (procedure, alarm, value…). `temperature = 0.0`.

`ask_chat` returns a **dict**, not a string: `result["answer"]`, `result["sources"]`, `result["confidence"]`.

Verbose logs go to `logs/rag.log` (`--debug` on `ask.py` / `evaluate.py`), not to the operator.

---

## evaluate.py — measure search quality

**Role.** Test **search only** (not Qwen). For each question in `evaluation.json` it calls `search_chunks(..., apply_threshold=False)` and checks whether the expected passage is in the top 1 / 3 / 5.

**When.** After changing chunking, synonyms, or weights, to see if Recall@k goes up or down. `llama-embed` must be up; Qwen is not required.

### `evaluation.json`

Each case:

```json
{
  "question": "Comment demarrer la machine OPTIJET ?",
  "expected_source": "298.410-fr",
  "expected_page": 16,
  "expected_heading": "7.1 Mise en marche",
  "lang": "fr"
}
```

A case succeeds if a hit has:

- `source` containing `expected_source`;
- a page inside `[start_page, end_page]`;
- `heading` / `section_path` containing `expected_heading`.

### Metrics

- **Recall@1** — the first excerpt is the right one.
- **Recall@3** / **Recall@5** — the right one is among the first 3 / 5.

Display: `OK` (Recall@1), `~` (found lower), `KO` (missing).

### Commands

```powershell
python evaluate.py
python evaluate.py --file evaluation.json --lang fr
python evaluate.py --lang en --debug
```

Add real operator questions to `evaluation.json` so the score matches the shop floor.

---

## Typical sequence

| Step | Script | Servers |
|---|---|---|
| 1. Index PDFs | `ingest.py` | embed :8081 |
| 2. Measure search | `evaluate.py` | embed :8081 |
| 3. Operator question | `ask.py` or `ui.py` | embed :8081 + chat :8080 |

`rag.py` is used in all three steps.
