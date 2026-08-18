from __future__ import annotations

import time
from pathlib import Path

import chromadb
import requests

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "manuals"

EMBED_URL = "http://localhost:8081/v1/embeddings"
CHAT_URL = "http://localhost:8080/v1/chat/completions"
HEALTH_EMBED = "http://localhost:8081/health"
HEALTH_CHAT = "http://localhost:8080/health"

SYSTEM_PROMPT = (
    "Tu es l'assistant technique de la machine. "
    "Réponds à l'opérateur uniquement avec l'extrait suivant du manuel."
)


def wait_for(url: str, name: str, timeout_s: int = 180) -> None:
    last_error = None
    for _ in range(timeout_s):
        try:
            response = requests.get(url, timeout=3)
            if response.status_code < 500:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"{name} n'est pas pret sur {url}: {last_error}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = requests.post(EMBED_URL, json={"input": texts}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    items = sorted(payload["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in items]


def enrich_query(question: str) -> str:
    cleaned = " ".join(question.split())
    if "optijet" not in cleaned.lower():
        cleaned = f"{cleaned} machine OPTIJET"
    return f"query: {cleaned}"


def embed_query(question: str) -> list[float]:
    return embed_texts([enrich_query(question)])[0]


def open_collection(reset: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def search_chunks(
    question: str,
    n_results: int = 5,
    lang: str = "fr",
    max_distance: float = 0.55,
    fetch_k: int = 12,
) -> list[dict]:
    collection = open_collection()
    if collection.count() == 0:
        raise RuntimeError("La base est vide. Lance d'abord: python ingest.py")

    query_kwargs: dict = {
        "query_embeddings": [embed_query(question)],
        "n_results": min(fetch_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if lang and lang != "all":
        query_kwargs["where"] = {"lang": lang}

    try:
        result = collection.query(**query_kwargs)
    except Exception:
        result = collection.query(
            query_embeddings=query_kwargs["query_embeddings"],
            n_results=query_kwargs["n_results"],
            include=query_kwargs["include"],
        )

    hits = []
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    for doc, meta, distance in zip(documents[0], metadatas[0], distances[0]):
        meta = meta or {}
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "lang": meta.get("lang", ""),
                "distance": distance,
            }
        )

    filtered = [hit for hit in hits if hit["distance"] <= max_distance]
    return (filtered or hits[:1])[:n_results]


def ask_chat(question: str, excerpts: list[dict]) -> str:
    excerpt_block = "\n\n".join(
        f"[{hit['source']} p.{hit['page']}]\n{hit['text']}" for hit in excerpts
    )
    user_prompt = (
        f"Extrait du manuel : {excerpt_block}\n\n"
        f"Question : {question}\n\n"
        "Reponds en francais, de facon courte et operationnelle. "
        "Si l'extrait ne contient pas la reponse, dis-le clairement."
    )
    response = requests.post(
        CHAT_URL,
        json={
            "model": "qwen",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 256,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
